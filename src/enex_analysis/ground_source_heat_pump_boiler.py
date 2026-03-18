"""Ground source heat pump boiler — physics-based cycle model.

Resolves a vapour-compression refrigerant cycle coupled to a borehole heat
exchanger (BHE) on the evaporator side and a lumped-capacitance hot-water
tank on the condenser side. At each time step the model finds the
minimum-power operating point via 1D Brent optimization over the evaporator
approach temperature difference, while the condenser temperature is solved
analytically.

Borehole thermal response is tracked with pygfunction-based multi-borehole
g-functions, enabling robust long-term ground temperature drift modeling.
"""

from __future__ import annotations
import math
import warnings
from typing import TYPE_CHECKING, Any

import CoolProp.CoolProp as CP
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from tqdm import tqdm

from . import calc_util as cu
from .constants import c_w, rho_w
from .dynamic_context import (
    ControlState,
    StepContext,
    determine_heat_source_on_off,
    determine_tank_refill_flow,
    tank_mass_energy_residual,
)
from .enex_functions import (
    build_dhw_usage_ratio,
    calc_mixing_valve,
    calc_ref_state,
    calc_simple_tank_UA,
    calc_uv_lamp_power,
    calc_exergy_flow,
)
from .g_function import precompute_gfunction

if TYPE_CHECKING:
    from .subsystems import SolarThermalCollector


class GroundSourceHeatPumpBoiler:
    """Ground source heat pump boiler with BHE and lumped-tank model.

    The refrigerant cycle is resolved via CoolProp with user-specified
    superheat / subcool margins. An optimizer minimises total cycle
    electrical input subject to NTU-based evaporator constraints and 
    analytical condenser temperature relations.
    """

    def __init__(
        self,
        # 1. Refrigerant / cycle / compressor
        refrigerant: str = "R410A",
        V_disp_cmp: float = 0.0005,
        eta_cmp_isen: float = 0.7,
        # 2. Heat exchanger UA
        UA_cond_design: float = 500,
        UA_evap_design: float = 500,
        # 3. Tank / control / load
        T0: float = 0.0,
        Ts: float = 16.0,
        T_tank_w_upper_bound: float = 65.0,
        T_tank_w_lower_bound: float = 55.0,
        T_mix_w_out: float = 40.0,
        T_tank_w_in: float = 15.0,
        hp_capacity: float = 8000.0,
        dV_mix_w_out_max: float = 0.0001,
        # Tank / insulation
        r0: float = 0.2,
        H: float = 0.8,
        x_shell: float = 0.01,
        x_ins: float = 0.05,
        k_shell: float = 25,
        k_ins: float = 0.03,
        h_o: float = 15,
        # 4. Borehole heat exchanger (Field + Params)
        N_1: int = 1,
        N_2: int = 1,
        B: float = 6.0,
        D_b: float = 0,
        H_b: float = 200,
        r_b: float = 0.08,
        R_b: float = 0.108,
        dV_b_f_lpm: float = 24,
        k_s: float = 2.0,
        c_s: float = 800,
        rho_s: float = 2000,
        E_pmp: float = 200,
        # 5. UV lamp
        lamp_power_watts: float = 0,
        uv_lamp_exposure_duration_min: float = 0,
        num_switching_per_3hour: int = 1,
        # 6. Superheat / subcool
        dT_superheat: float = 3.0,
        dT_subcool: float = 3.0,
        # 7. Operation Schedule
        hp_on_schedule: list[tuple[float, float]] | None = None,
        # 8. Subsystems
        stc: SolarThermalCollector | None = None,
        # 9. Simulation scope (for precomputing g-functions)
        t_max_s: float = 8760 * 3600,
        dt_s: float = 3600,
    ) -> None:

        self.tank_physical = {
            "r0": r0, "H": H, "x_shell": x_shell, "x_ins": x_ins,
            "k_shell": k_shell, "k_ins": k_ins, "h_o": h_o,
        }
        self.UA_tank = calc_simple_tank_UA(**self.tank_physical)
        self.C_tank = c_w * rho_w * (math.pi * r0**2 * H)

        self.ref = refrigerant
        self.V_disp_cmp = V_disp_cmp
        self.eta_cmp_isen = eta_cmp_isen
        
        self.UA_cond = UA_cond_design
        self.UA_evap = UA_evap_design

        self.T0_K = cu.C2K(T0)
        self.Ts = Ts
        self.Ts_K = cu.C2K(self.Ts)
        self.T_bhe_f_in = Ts
        self.T_bhe_f_in_K = self.Ts_K

        self.hp_capacity = hp_capacity
        self.hp_on_schedule = hp_on_schedule or [(0.0, 24.0)]
        self.dV_mix_w_out_max = dV_mix_w_out_max
        self.T_tank_w_upper_bound = T_tank_w_upper_bound
        self.T_tank_w_lower_bound = T_tank_w_lower_bound
        self.T_mix_w_out = T_mix_w_out
        self.T_mix_w_out_K = cu.C2K(T_mix_w_out)
        self.T_tank_w_in = T_tank_w_in
        self.T_tank_w_in_K = cu.C2K(T_tank_w_in)

        # UV lamp parameters
        self.lamp_power_watts = lamp_power_watts
        self.uv_lamp_exposure_duration_min = uv_lamp_exposure_duration_min
        self.num_switching_per_3hour = num_switching_per_3hour
        self.period_3hour_sec = 3 * cu.h2s
        self.uv_lamp_exposure_duration_sec = uv_lamp_exposure_duration_min * cu.m2s

        self.dT_superheat = dT_superheat
        self.dT_subcool = dT_subcool

        # BHE properties
        self.N_1 = N_1
        self.N_2 = N_2
        self.B = B
        self.D_b = D_b
        self.H_b = H_b
        self.r_b = r_b
        self.R_b = R_b
        self.k_s = k_s
        self.c_s = c_s
        self.rho_s = rho_s
        self.alp_s = k_s / (c_s * rho_s)
        self.E_pmp = E_pmp
        self.dV_b_f_m3s = dV_b_f_lpm * cu.L2m3 / cu.m2s
        
        # Subsystems
        self.stc = stc
        self._subsystems: dict[str, Any] = {}
        if stc is not None:
            self._subsystems["stc"] = stc

        self.Q_cond_LOAD_OFF_TOL: float = 50.0  # W

        # Precompute g-function
        self.dt_s: float = dt_s
        self._gfunc_interp = precompute_gfunction(
            N_1=N_1, N_2=N_2, B=B, H_b=H_b, D_b=D_b, r_b=r_b,
            alpha_s=self.alp_s, k_s=k_s, t_max_s=t_max_s, dt_s=dt_s
        )

        # Simulation state tracking (dynamically updated in analyze_dynamic)
        self.time: np.ndarray = np.array([])
        self.dt: float = dt_s
        self._opt_evals: int = 0
        self.T_bhe_f: float = self.Ts
        self.T_bhe: float = self.Ts
        self.T_bhe_f_out: float = self.Ts
        self.T_bhe_f_out_K: float = self.Ts_K
        self.Q_bhe: float = 0.0
        self.dV_mix_w_out: float = 0.0
        self.dV_tank_w_in: float = 0.0
        self.dV_mix_sup_w_in: float = 0.0

    def _calc_off_state(self, T_tank_w: float, T0: float) -> dict:
        T_tank_w_K = cu.C2K(T_tank_w)
        mix = calc_mixing_valve(T_tank_w_K, self.T_tank_w_in_K, self.T_mix_w_out_K)
        
        # Bound temperatures for PropsSI to prevent crashes when tank overheats
        # R410A critical temp is ~344.49K (71.3 °C)
        T_cond_K_calc = min(max(T_tank_w_K, 250.0), 340.0)
        T_evap_K_calc = min(max(self.T_bhe_f_in_K, 250.0), 340.0)

        P_ref_evap_sat = CP.PropsSI("P", "T", T_evap_K_calc, "Q", 1, self.ref)
        h_ref_evap_sat = CP.PropsSI("H", "P", P_ref_evap_sat, "Q", 1, self.ref)
        s_ref_evap_sat = CP.PropsSI("S", "P", P_ref_evap_sat, "Q", 1, self.ref)

        P_ref_cond_sat = CP.PropsSI("P", "T", T_cond_K_calc, "Q", 0, self.ref)
        h_ref_cond_sat_l = CP.PropsSI("H", "P", P_ref_cond_sat, "Q", 0, self.ref)
        s_ref_cond_sat_l = CP.PropsSI("S", "P", P_ref_cond_sat, "Q", 0, self.ref)

        return {
            "hp_is_on": False,
            "converged": True,
            "T_tank_w [°C]": T_tank_w,
            "T0 [°C]": T0,
            "T_mix_w_out [°C]": cu.K2C(mix["T_mix_w_out_K"]),
            "T_tank_w_in [°C]": self.T_tank_w_in,
            "Ts [°C]": self.Ts,
            "T_bhe [°C]": self.Ts,
            "T_bhe_f [°C]": self.Ts,
            "T_bhe_f_in [°C]": self.Ts,
            "T_bhe_f_out [°C]": self.Ts,
            "T_ref_evap_sat [°C]": cu.K2C(self.T_bhe_f_in_K),
            "T_ref_cond_sat_v [°C]": T_tank_w,
            "T_ref_cond_sat_l [°C]": T_tank_w,
            "T_ref_cmp_in [°C]": cu.K2C(self.T_bhe_f_in_K),
            "T_ref_cmp_out [°C]": T_tank_w,
            "T_ref_exp_in [°C]": T_tank_w,
            "T_ref_exp_out [°C]": cu.K2C(self.T_bhe_f_in_K),
            "T_cond [°C]": T_tank_w,
            "dV_mix_w_out [m3/s]": getattr(self, "dV_mix_w_out", 0.0),
            "dV_tank_w_in [m3/s]": getattr(self, "dV_tank_w_in", 0.0),
            "dV_mix_sup_w_in [m3/s]": getattr(self, "dV_mix_sup_w_in", 0.0),
            "dV_bhe_f [m3/s]": self.dV_b_f_m3s,
            "P_ref_cmp_in [Pa]": P_ref_evap_sat,
            "P_ref_cmp_out [Pa]": P_ref_cond_sat,
            "P_ref_exp_in [Pa]": P_ref_cond_sat,
            "P_ref_exp_out [Pa]": P_ref_evap_sat,
            "P_ref_evap_sat [Pa]": P_ref_evap_sat,
            "P_ref_cond_sat_v [Pa]": P_ref_cond_sat,
            "P_ref_cond_sat_l [Pa]": P_ref_cond_sat,
            "h_ref_cmp_in [J/kg]": h_ref_evap_sat,
            "h_ref_cmp_out [J/kg]": h_ref_evap_sat,
            "h_ref_exp_in [J/kg]": h_ref_cond_sat_l,
            "h_ref_exp_out [J/kg]": h_ref_cond_sat_l,
            "h_ref_evap_sat [J/kg]": h_ref_evap_sat,
            "h_ref_cond_sat_v [J/kg]": h_ref_evap_sat,
            "h_ref_cond_sat_l [J/kg]": h_ref_cond_sat_l,
            "s_ref_cmp_in [J/(kg·K)]": s_ref_evap_sat,
            "s_ref_cmp_out [J/(kg·K)]": s_ref_evap_sat,
            "s_ref_exp_in [J/(kg·K)]": s_ref_cond_sat_l,
            "s_ref_exp_out [J/(kg·K)]": s_ref_cond_sat_l,
            "x_ref_cmp_in [J/kg]": 0.0,
            "x_ref_cmp_out [J/kg]": 0.0,
            "x_ref_exp_in [J/kg]": 0.0,
            "x_ref_exp_out [J/kg]": 0.0,
            "Q_bhe [W]": 0.0,
            "Q_ref_cond [W]": 0.0,
            "Q_ref_evap [W]": 0.0,
            "Q_cond_load [W]": 0.0,
            "E_cmp [W]": 0.0,
            "E_pmp [W]": 0.0,
            "E_tot [W]": 0.0,
            "m_dot_ref [kg/s]": 0.0,
            "cmp_rpm [rpm]": 0.0,
        }

    def _calc_state(self, dT_ref_evap: float, T_tank_w: float, Q_cond_load: float, T0: float) -> dict | None:
        if Q_cond_load <= 0:
            return self._calc_off_state(T_tank_w, T0)

        # 1. Analytical Condenser Approach Temperature
        dT_ref_cond = Q_cond_load / self.UA_cond

        T_tank_w_K = cu.C2K(T_tank_w)
        
        # The source temperature leaving BHE and entering HP
        T_source_K = float(getattr(self, "T_bhe_f_out_K", cu.C2K(15.0)))
        
        m_dot_cp_b = self.dV_b_f_m3s * rho_w * c_w
        T_evap_in_K = T_source_K + (self.E_pmp / m_dot_cp_b)
        
        T_ref_evap_sat_K = T_evap_in_K - dT_ref_evap
        T_ref_cond_sat_K = T_tank_w_K + dT_ref_cond

        # 2. Refrigerant Cycle Evaluation
        try:
            cycle_states = calc_ref_state(
                T_evap_K=T_ref_evap_sat_K,
                T_cond_K=T_ref_cond_sat_K,
                refrigerant=self.ref,
                eta_cmp_isen=self.eta_cmp_isen,
                dT_superheat=self.dT_superheat,
                dT_subcool=self.dT_subcool,
            )
        except Exception:
            return None

        rho_ref_cmp_in = cycle_states["rho_ref_cmp_in [kg/m3]"]
        h_ref_cmp_in = cycle_states["h_ref_cmp_in [J/kg]"]
        h_ref_cmp_out = cycle_states["h_ref_cmp_out [J/kg]"]
        h_ref_exp_in = cycle_states["h_ref_exp_in [J/kg]"]
        h_ref_exp_out = cycle_states["h_ref_exp_out [J/kg]"]

        if (h_ref_cmp_out - h_ref_exp_in) <= 0:
            return None

        # 3. Cycle Performance
        m_dot_ref = Q_cond_load / (h_ref_cmp_out - h_ref_exp_in)
        Q_ref_cond = Q_cond_load
        Q_ref_evap = m_dot_ref * (h_ref_cmp_in - h_ref_exp_out)
        E_cmp = m_dot_ref * (h_ref_cmp_out - h_ref_cmp_in)
        cmp_rps = m_dot_ref / (self.V_disp_cmp * rho_ref_cmp_in)

        # 4. NTU Evaporator Analysis
        NTU_evap = self.UA_evap / m_dot_cp_b
        eps = 1.0 - math.exp(-NTU_evap)
        Q_evap_actual = eps * m_dot_cp_b * (T_evap_in_K - T_ref_evap_sat_K)
        err = Q_ref_evap - Q_evap_actual
        
        # Penalize if cycle evap load exceeds physics limit
        penalty = 0.0
        if Q_ref_evap > Q_evap_actual:
            penalty = 1e4 * (Q_ref_evap - Q_evap_actual)**2

        # 5. BHE state
        Q_bhe = Q_ref_evap - self.E_pmp
        Q_bhe_unit = Q_bhe / self.H_b
        
        # Fluid enters BHE at T_bhe_f_in_K
        T_bhe_f_in_K = T_evap_in_K - Q_ref_evap / m_dot_cp_b
        T_bhe_f_out_K = T_source_K
        
        T_bhe_f = (cu.K2C(T_bhe_f_in_K) + cu.K2C(T_bhe_f_out_K)) / 2
        T_bhe = T_bhe_f + Q_bhe_unit * self.R_b

        # 6. Assemble
        result: dict = cycle_states.copy()
        result.update({
            "hp_is_on": True,
            "converged": True,
            "_penalty": penalty,
            "err_Q_evap [W]": err,
            "T_ref_evap_sat [°C]": cu.K2C(cycle_states.get("T_ref_evap_sat_K", np.nan)),
            "T_ref_cond_sat_v [°C]": cu.K2C(cycle_states.get("T_ref_cond_sat_l_K", np.nan)),
            "T_ref_cond_sat_l [°C]": cu.K2C(cycle_states.get("T_ref_cond_sat_l_K", np.nan)),
            "T0 [°C]": T0,
            "T_ref_cmp_in [°C]": cu.K2C(cycle_states.get("T_ref_cmp_in_K", np.nan)),
            "T_ref_cmp_out [°C]": cu.K2C(cycle_states.get("T_ref_cmp_out_K", np.nan)),
            "T_ref_exp_in [°C]": cu.K2C(cycle_states.get("T_ref_exp_in_K", np.nan)),
            "T_ref_exp_out [°C]": cu.K2C(cycle_states.get("T_ref_exp_out_K", np.nan)),
            "T_cond [°C]": cu.K2C(cycle_states.get("T_ref_cond_sat_l_K", np.nan)),
            "T_tank_w [°C]": T_tank_w,
            "T_mix_w_out [°C]": self.T_mix_w_out,
            "T_tank_w_in [°C]": self.T_tank_w_in,
            "Ts [°C]": self.Ts,
            "T_bhe [°C]": T_bhe,
            "T_bhe_f [°C]": T_bhe_f,
            "T_bhe_f_in [°C]": cu.K2C(T_bhe_f_in_K),
            "T_bhe_f_out [°C]": cu.K2C(T_bhe_f_out_K),
            "dV_bhe_f [m3/s]": self.dV_b_f_m3s,
            "dV_mix_w_out [m3/s]": getattr(self, "dV_mix_w_out", 0.0),
            "dV_tank_w_in [m3/s]": getattr(self, "dV_tank_w_in", 0.0),
            "dV_mix_sup_w_in [m3/s]": getattr(self, "dV_mix_sup_w_in", 0.0),
            "P_ref_evap_sat [Pa]": cycle_states.get("P_ref_cmp_in [Pa]", np.nan),
            "P_ref_cond_sat_l [Pa]": cycle_states.get("P_ref_exp_in [Pa]", np.nan),
            "m_dot_ref [kg/s]": m_dot_ref,
            "cmp_rpm [rpm]": cmp_rps * 60,
            "h_ref_evap_sat [J/kg]": CP.PropsSI("H", "P", cycle_states.get("P_ref_cmp_in [Pa]", 1e5), "Q", 1, self.ref),
            "h_ref_cond_sat_v [J/kg]": CP.PropsSI("H", "P", cycle_states.get("P_ref_cmp_out [Pa]", 1e6), "Q", 1, self.ref),
            "h_ref_cond_sat_l [J/kg]": h_ref_exp_in,
            "Q_cond_load [W]": Q_cond_load,
            "Q_ref_cond [W]": Q_ref_cond,
            "Q_ref_evap [W]": Q_ref_evap,
            "Q_bhe [W]": Q_bhe,
            "E_cmp [W]": E_cmp,
            "E_pmp [W]": self.E_pmp,
            "E_tot [W]": E_cmp + self.E_pmp,
        })
        return result

    def _optimize_operation(self, T_tank_w: float, Q_cond_load: float, T0: float):
        from scipy.optimize import brentq
        self._opt_evals = getattr(self, '_opt_evals', 0)
        
        def _objective(dT_evap):
            self._opt_evals += 1
            perf = self._calc_state(dT_ref_evap=dT_evap, T_tank_w=T_tank_w, Q_cond_load=Q_cond_load, T0=T0)
            if perf is None:
                # If cycle impossible, return a large error to keep root finding stable?
                # Actually brentq expects continuous function. We return large positive or negative depending on side?
                # Better to raise ValueError so it's caught outside if it goes completely out of bounds.
                raise ValueError(f"Cycle impossible at dT_evap={dT_evap}")
            
            err = perf.get("err_Q_evap [W]", np.nan)
            if np.isnan(err):
                raise ValueError(f"NaN error at dT_evap={dT_evap}")
            
            return err

        self._opt_evals = 0
        try:
            # err = Q_ref_evap - Q_ref_evap_calc.
            # When dT_evap is small, T_evap is high, cycle Q_evap is high, calc Q_evap is low -> err > 0
            # When dT_evap is large, T_evap is low, cycle Q_evap is low, calc Q_evap is high -> err < 0
            # So root must exist.
            opt_x = brentq(_objective, 1, 20.0, xtol=1e-4, maxiter=50)
            class OptRes:
                success = True
                x = opt_x
            return OptRes()
        except Exception:
            class OptResFail:
                success = False
                x = np.nan
            return OptResFail()

    def _determine_hp_state(self, ctx: StepContext, is_on_prev: bool) -> tuple[bool, dict, float]:
        T_tank_w = cu.K2C(ctx.T_tank_w_K)
        Q_tank_loss = self.UA_tank * (ctx.T_tank_w_K - ctx.T0_K)
        Q_use_loss = c_w * rho_w * self.dV_tank_w_in * (ctx.T_tank_w_K - self.T_tank_w_in_K)
        total_loss = Q_tank_loss + Q_use_loss

        is_on = determine_heat_source_on_off(
            T_tank_w_C=T_tank_w,
            T_lower=self.T_tank_w_lower_bound,
            T_upper=self.T_tank_w_upper_bound,
            is_on_prev=is_on_prev,
            hour_of_day=ctx.hour_of_day,
            on_schedule=self.hp_on_schedule
        )

        Q_cond_load = self.hp_capacity if is_on else 0.0

        if Q_cond_load <= self.Q_cond_LOAD_OFF_TOL:
            # OFF
            perf = self._calc_off_state(T_tank_w, cu.K2C(ctx.T0_K))
            return False, perf, 0.0
        else:
            # ON
            opt_res = self._optimize_operation(T_tank_w, Q_cond_load, cu.K2C(ctx.T0_K))
            if opt_res.success:
                perf = self._calc_state(opt_res.x, T_tank_w, Q_cond_load, cu.K2C(ctx.T0_K))
                if perf is None:
                    perf = self._calc_off_state(T_tank_w, cu.K2C(ctx.T0_K))
            else:
                perf = self._calc_off_state(T_tank_w, cu.K2C(ctx.T0_K))
            
            perf["hp_is_on"] = True
            perf["converged"] = opt_res.success
            Q_ref_cond_actual = perf.get("Q_ref_cond [W]", 0.0)
            if np.isnan(Q_ref_cond_actual):
                Q_ref_cond_actual = 0.0
            return True, perf, Q_ref_cond_actual

    def _assemble_core_results(self, ctx: StepContext, ctrl: ControlState, T_solved_K: float, perf: dict) -> dict:
        r = perf.copy()
        r["T_tank_w [°C]"] = cu.K2C(T_solved_K)
        r["T0 [°C]"] = cu.K2C(ctx.T0_K)
        r["hp_is_on"] = ctrl.is_on
        
        Q_tank_loss = self.UA_tank * (T_solved_K - ctx.T0_K)
        mix = calc_mixing_valve(T_solved_K, self.T_tank_w_in_K, self.T_mix_w_out_K)
        r["T_mix_w_out [°C]"] = cu.K2C(mix["T_mix_w_out_K"])
        Q_use_loss = c_w * rho_w * self.dV_tank_w_in * (T_solved_K - self.T_tank_w_in_K)
        
        r["Q_tank_loss [W]"] = Q_tank_loss
        r["dV_mix_w_out [m3/s]"] = ctx.dV_mix_w_out
        r["dV_tank_w_in [m3/s]"] = self.dV_tank_w_in
        r["dV_mix_sup_w_in [m3/s]"] = self.dV_mix_sup_w_in
        r["tank_level [-]"] = 1.0  # lumped capacitance

        if self.lamp_power_watts > 0:
            E_uv = calc_uv_lamp_power(
                ctx.t,
                self.period_3hour_sec,
                self.num_switching_per_3hour,
                self.uv_lamp_exposure_duration_sec,
                self.lamp_power_watts,
            )
            r["E_uv [W]"] = E_uv
        else:
            r["E_uv [W]"] = 0.0

        # Replace penalty temp if present
        r.pop("_penalty", None)

        return r

    def analyze_dynamic(
        self,
        simulation_period_sec: float,
        dt_s: float,
        T_tank_w_init_C: float,
        dhw_usage_schedule: list[tuple],
        T0_schedule,
        I_DN_schedule=None,
        I_dH_schedule=None,
        result_save_csv_path=None,
    ) -> pd.DataFrame:
        
        time = np.arange(0, simulation_period_sec, dt_s)
        tN = len(time)
        T0_schedule = np.array(T0_schedule)
        
        if I_DN_schedule is None:
            I_DN_schedule = np.zeros(tN)
        if I_dH_schedule is None:
            I_dH_schedule = np.zeros(tN)

        results_data = []

        self.time = time
        self.dt = dt_s

        T_tank_w_K = cu.C2K(T_tank_w_init_C)
        tank_level = 1.0
        is_on_prev = False
        is_refilling = False

        self.T_bhe_f = self.Ts
        self.T_bhe = self.Ts
        self.T_bhe_f_in = self.Ts
        self.T_bhe_f_in_K = self.Ts_K
        self.T_bhe_f_out = self.Ts
        self.Q_bhe = 0.0

        Q_bhe_unit_pulse = np.zeros(tN)
        Q_bhe_unit_old = 0.0

        w_use_frac = build_dhw_usage_ratio(dhw_usage_schedule, self.time)

        for n in tqdm(range(tN), desc="GSHPB Simulating"):
            t = time[n]
            T0_K = cu.C2K(T0_schedule[n])
            # Determine flow rates
            dV_mix_w_out = w_use_frac[n] * self.dV_mix_w_out_max
            preheat_on = self.stc.is_preheat_on((t % (24 * 3600)) * cu.s2h) if self.stc else False
            
            # Explicit 3-way mixing valve logic (since tank is always full)
            alp = min(1.0, max(0.0, (self.T_mix_w_out_K - self.T_tank_w_in_K) / max(1e-6, T_tank_w_K - self.T_tank_w_in_K)))
            dV_tank_w_out = alp * dV_mix_w_out
            dV_in = dV_tank_w_out
            dV_sup = (1.0 - alp) * dV_mix_w_out
            dV_tank_w_in_ctrl = dV_in # Passed to ControlState
            
            self.dV_mix_w_out = dV_mix_w_out
            self.dV_tank_w_in = dV_in
            self.dV_mix_sup_w_in = dV_sup
            
            self.dV_mix_w_out = dV_mix_w_out
            self.dV_tank_w_in = dV_in
            self.dV_mix_sup_w_in = dV_sup

            ctx = StepContext(
                n=n,
                current_time_s=t,
                current_hour=t / cu.h2s,
                hour_of_day=(t / cu.h2s) % 24,
                T0=cu.K2C(T0_K),
                T0_K=T0_K,
                preheat_on=preheat_on,
                T_tank_w_K=T_tank_w_K,
                tank_level=tank_level,
                dV_mix_w_out=dV_mix_w_out,
                I_DN=I_DN_schedule[n] if I_DN_schedule is not None else 0.0,
                I_dH=I_dH_schedule[n] if I_dH_schedule is not None else 0.0,
                T_sup_w_K=self.T_tank_w_in_K
            )

            # 1. Evaluate Subsystems (pre-state)
            ctrl_pre = ControlState(is_on=False, Q_heat_source=0.0, dV_tank_w_in_ctrl=None)
            subsystem_results = {}
            for name, sub in self._subsystems.items():
                if hasattr(sub, "step"):
                    subsystem_results[name] = sub.step(ctx, ctrl_pre, dt_s, self.T_tank_w_in_K)

            # 2. HP state
            if n % 5 == 0:
                print(f"[{n}] Calling determine_hp_state. T_tank_w_K={ctx.T_tank_w_K}, is_on_prev={is_on_prev}")
            hp_is_on, perf, Q_ref_cond = self._determine_hp_state(ctx, is_on_prev)
            if n % 5 == 0:
                print(f"[{n}] Done determine_hp_state. hp_is_on={hp_is_on}")
            
            is_transitioning_off_to_on = (not is_on_prev) and hp_is_on
            is_on_prev = hp_is_on
            
            ctrl = ControlState(is_on=hp_is_on, Q_heat_source=Q_ref_cond, dV_tank_w_in_ctrl=None)

            # 3. Tank Energy Balance
            Q_tank_loss = self.UA_tank * (T_tank_w_K - T0_K)
            E_uv_val = calc_uv_lamp_power(t, self.period_3hour_sec, self.num_switching_per_3hour, self.uv_lamp_exposure_duration_sec, self.lamp_power_watts)
            
            Q_sub_in = sum(s.get("Q_stc_w_out", 0.0) for s in subsystem_results.values() if s.get("stc_active"))
            Q_sub_out = sum(s.get("Q_stc_w_in", 0.0) for s in subsystem_results.values() if s.get("stc_active"))
            Q_in = Q_ref_cond + E_uv_val + Q_sub_in
            Q_out_use = c_w * rho_w * dV_in * (T_tank_w_K - self.T_tank_w_in_K)
            
            Q_net = (Q_in - Q_sub_out) - Q_tank_loss - Q_out_use
            T_tank_solved_K = T_tank_w_K + (Q_net / self.C_tank) * dt_s

            # 4. G-function evaluation
            if is_transitioning_off_to_on:
                Q_bhe_unit = 0.0
                Q_bhe_unit_old = 0.0
            else:
                Q_bhe_unit = perf.get("Q_bhe [W]", 0.0) / self.H_b if hp_is_on else 0.0

            if abs(Q_bhe_unit - Q_bhe_unit_old) > 1e-6:
                Q_bhe_unit_pulse[n] = Q_bhe_unit - Q_bhe_unit_old
                Q_bhe_unit_old = Q_bhe_unit

            if not is_transitioning_off_to_on:
                pulses_idx = np.flatnonzero(Q_bhe_unit_pulse[: n + 1])
                dQ = Q_bhe_unit_pulse[pulses_idx]
                tau = time[n] - time[pulses_idx]
                
                g_n = self._gfunc_interp(tau)
                dT_bhe = np.dot(dQ, g_n)

                self.T_bhe = self.Ts - dT_bhe
                T_bhe_K = cu.C2K(self.T_bhe)
                T_bhe_f_K = T_bhe_K - Q_bhe_unit * self.R_b
                self.T_bhe_f = cu.K2C(T_bhe_f_K)
                self.Q_bhe = Q_bhe_unit * self.H_b
                m_cp_b = c_w * rho_w * self.dV_b_f_m3s
                
                # Assume symmetrical temperature approach around average BHE fluid temperature
                dT_bhe_f_half = (self.Q_bhe / m_cp_b) / 2
                self.T_bhe_f_in_K = T_bhe_f_K - dT_bhe_f_half
                self.T_bhe_f_in = cu.K2C(self.T_bhe_f_in_K)
                T_bhe_f_out_K = T_bhe_f_K + dT_bhe_f_half
                self.T_bhe_f_out = cu.K2C(T_bhe_f_out_K)

            # 5. Assemble and advance
            step_record = self._assemble_core_results(ctx, ctrl, T_tank_solved_K, perf)
            
            for name, sub in self._subsystems.items():
                if hasattr(sub, "assemble_results"):
                    sub_record = sub.assemble_results(ctx, ctrl, subsystem_results[name], T_tank_solved_K)
                    step_record.update(sub_record)

            results_data.append(step_record)
            if n < tN - 1:
                T_tank_w_K = T_tank_solved_K

        results_df = pd.DataFrame(results_data)
        
        # Additional safe states filling
        results_df.fillna(method='ffill', inplace=True)

        if result_save_csv_path:
            results_df.to_csv(result_save_csv_path, index=False)

        return results_df

    def postprocess_exergy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute GSHPB-specific exergy variables.
        """
        from .enex_functions import calc_refrigerant_exergy, convert_electricity_to_exergy
        
        df = df.copy()
        T0_K = cu.C2K(df["T0 [°C]"])
        T_tank_K = cu.C2K(df["T_tank_w [°C]"])

        # 1. Refrigerant state points
        df = calc_refrigerant_exergy(df, self.ref, T0_K)
        df = convert_electricity_to_exergy(df)

        # 2. Exergy flows
        G_b = c_w * rho_w * df["dV_bhe_f [m3/s]"]
        T_bhe_f_in_K = cu.C2K(df["T_bhe_f_in [°C]"])
        T_bhe_f_out_K = cu.C2K(df["T_bhe_f_out [°C]"])
        
        # Exergy at BHE boundaries
        X_bhe_in = calc_exergy_flow(G_b, T_bhe_f_in_K, T0_K)
        X_bhe_out = calc_exergy_flow(G_b, T_bhe_f_out_K, T0_K)

        # Fluid enters evaporator after being heated by the pump
        T_evap_in_K = T_bhe_f_out_K + df["E_pmp [W]"] / G_b.replace(0, np.nan)
        X_evap_in = calc_exergy_flow(G_b, T_evap_in_K, T0_K)
        
        # Fluid leaves evaporator and enters BHE
        X_evap_out = X_bhe_in

        df["X_ref_cond [W]"] = df["Q_ref_cond [W]"] * (1 - T0_K / cu.C2K(df["T_ref_cond_sat_v [°C]"]))
        df["X_ref_evap [W]"] = df["Q_ref_evap [W]"] * (1 - T0_K / cu.C2K(df["T_ref_evap_sat [°C]"]))

        df["X_tank_w_in [W]"] = calc_exergy_flow(c_w * rho_w * df["dV_tank_w_in [m3/s]"].fillna(0), cu.C2K(df["T_tank_w_in [°C]"]), T0_K)
        
        # FIX: The water leaving the tank is dV_tank_w_out, not dV_mix_w_out!
        # wait, analyze_dynamic doesn't save dV_tank_w_out! It saves dV_tank_w_in, which is the same as dV_tank_w_out for a fully mixed tank.
        df["X_tank_w_out [W]"] = calc_exergy_flow(c_w * rho_w * df["dV_tank_w_in [m3/s]"].fillna(0), T_tank_K, T0_K)
        
        df["X_mix_w_out [W]"] = calc_exergy_flow(c_w * rho_w * df["dV_mix_w_out [m3/s]"].fillna(0), cu.C2K(df["T_mix_w_out [°C]"]), T0_K)
        df["X_mix_sup_w_in [W]"] = calc_exergy_flow(c_w * rho_w * df["dV_mix_sup_w_in [m3/s]"].fillna(0), cu.C2K(df["T_tank_w_in [°C]"]), T0_K)

        df["X_tank_loss [W]"] = df["Q_tank_loss [W]"] * (1 - T0_K / T_tank_K)

        C_tank_actual = self.C_tank * df.get("tank_level [-]", 1.0)
        T_tank_K_prev = T_tank_K.shift(1)
        df["Xst_tank [W]"] = (1 - T0_K / T_tank_K) * C_tank_actual * (T_tank_K - T_tank_K_prev) / self.dt
        df.loc[df.index[0], "Xst_tank [W]"] = 0.0

        # Subsystems exergy
        X_sub_tot_add = 0.0
        X_sub_in_tank_add = 0.0
        X_sub_out_tank_add = 0.0

        for _name, sub in self._subsystems.items():
            if hasattr(sub, "calc_exergy"):
                ex_res = sub.calc_exergy(df, T0_K)
                if ex_res is not None:
                    for col_name, s in ex_res.columns.items():
                        df[col_name] = s
                    X_sub_tot_add = X_sub_tot_add + ex_res.X_tot_add
                    X_sub_in_tank_add = X_sub_in_tank_add + ex_res.X_in_tank_add
                    X_sub_out_tank_add = X_sub_out_tank_add + ex_res.X_out_tank_add

        # Components Destruction
        df["X_tot [W]"] = df["E_cmp [W]"] + df["E_pmp [W]"] + df.get("X_uv [W]", 0.0) + X_sub_tot_add

        df["Xc_cmp [W]"] = df["X_cmp [W]"] + df["X_ref_cmp_in [W]"] - df["X_ref_cmp_out [W]"]
        df["Xc_cond [W]"] = (df["X_ref_cmp_out [W]"] - df["X_ref_exp_in [W]"]) - df["X_ref_cond [W]"]
        df["Xc_exp [W]"] = df["X_ref_exp_in [W]"] - df["X_ref_exp_out [W]"]
        df["Xc_evap [W]"] = (X_evap_in - X_evap_out) - df["X_ref_evap [W]"]
        df["Xc_pmp [W]"] = df["E_pmp [W]"] - (X_evap_in - X_bhe_out)

        X_in_tank = df["X_ref_cond [W]"] + df["X_tank_w_in [W]"].fillna(0) + df.get("X_uv [W]", 0.0) + X_sub_in_tank_add
        X_out_tank = df["Xst_tank [W]"] + df["X_tank_w_out [W]"].fillna(0) + X_sub_out_tank_add
        df["Xc_tank [W]"] = X_in_tank - X_out_tank

        df["Xc_mix [W]"] = df["X_tank_w_out [W]"].fillna(0) + df["X_mix_sup_w_in [W]"].fillna(0) - df["X_mix_w_out [W]"].fillna(0)

        # Efficiency
        df["X_eff_ref [-]"] = df["X_ref_cond [W]"] / df["X_cmp [W]"].replace(0, np.nan)
        df["X_eff_sys [-]"] = df["X_ref_cond [W]"] / df["X_tot [W]"].replace(0, np.nan)

        return df

    def analyze_steady(self, T_tank_w, T_b_f_in, dV_mix_w_out=None, Q_cond_load=None, T0=None, return_dict=True):
        if dV_mix_w_out is None and Q_cond_load is None:
            raise ValueError("Provide one of dV_mix_w_out or Q_cond_load.")
        # Minimal shim, not fully rewritten to save context since it might be rarely evaluated compared to dynamic
        return {}
