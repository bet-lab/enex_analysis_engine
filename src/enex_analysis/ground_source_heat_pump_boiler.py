"""Ground source heat pump boiler — physics-based cycle model.

Resolves a vapour-compression refrigerant cycle coupled to a borehole heat
exchanger (BHE) on the evaporator side and a lumped-capacitance hot-water
tank on the condenser side.  At each time step the model finds the
minimum-power operating point via SLSQP optimisation over two temperature
differences (evaporator and condenser approach ΔT).

Borehole thermal response is tracked with the Finite Line Source (FLS)
g-function, enabling long-term ground temperature drift.
"""

# %%
import contextlib
import math
from dataclasses import dataclass

import CoolProp.CoolProp as CP
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from tqdm import tqdm

from . import calc_util as cu
from .constants import c_w, rho_w
from .enex_functions import (
    G_FLS,
    build_dhw_usage_ratio,
    calc_mixing_valve,
    calc_ref_state,
    calc_simple_tank_UA,
    calc_uv_lamp_power,
)


@dataclass
class GroundSourceHeatPumpBoiler:
    """Ground source heat pump boiler with BHE and lumped-tank model.

    The refrigerant cycle is resolved via CoolProp with user-specified
    superheat / subcool margins.  An SLSQP optimiser minimises total
    electrical input (``E_cmp + E_pmp``) subject to LMTD heat-exchanger
    constraints.
    """

    def __init__(
        self,
        # 1. Refrigerant / cycle / compressor -------------------------
        refrigerant="R410A",
        V_disp_cmp=0.0005,  # [m3] Compressor displacement
        eta_cmp_isen=0.7,  # [-]  Isentropic efficiency
        # 2. Heat exchanger -------------------------------------------
        UA_cond_design=500,  # [W/K] Condenser design UA
        UA_evap_design=500,  # [W/K] Evaporator design UA
        # 3. Tank / control / load ------------------------------------
        T0=0.0,  # [°C] Dead-state (ambient) temp
        Ts=16.0,  # [°C] Undisturbed ground temp
        T_tank_w_upper_bound=65.0,  # [°C] Tank upper setpoint
        T_tank_w_lower_bound=55.0,  # [°C] Tank lower setpoint
        T_mix_w_out=40.0,  # [°C] Service water delivery temp
        T_tank_w_in=15.0,  # [°C] Mains water supply temp
        hp_capacity=8000.0,  # [W]   HP rated capacity
        dV_mix_w_out_max=0.0001,  # [m3/s] Max service flow rate
        # Tank / insulation
        r0=0.2,  # [m]     Tank inner radius
        H=0.8,  # [m]     Tank height
        x_shell=0.01,  # [m]     Shell thickness
        x_ins=0.05,  # [m]     Insulation thickness
        k_shell=25,  # [W/m·K] Shell conductivity
        k_ins=0.03,  # [W/m·K] Insulation conductivity
        h_o=15,  # [W/m²·K] External convective coefficient
        # 4. Borehole heat exchanger ----------------------------------
        D_b=0,  # [m]     Borehole depth (reserved)
        H_b=200,  # [m]     Borehole effective length
        r_b=0.08,  # [m]     Borehole radius
        R_b=0.108,  # [m·K/W] Effective borehole thermal resistance
        # Ground-loop fluid
        dV_b_f_lpm=24,  # [L/min] Borehole fluid flow rate
        # Soil properties
        k_s=2.0,  # [W/m·K] Soil thermal conductivity
        c_s=800,  # [J/kg·K] Soil specific heat
        rho_s=2000,  # [kg/m3] Soil density
        # Circulation pump
        E_pmp=200,  # [W] Pump electrical power
        # 5. UV lamp --------------------------------------------------
        lamp_power_watts=0,  # [W]   Lamp power
        uv_lamp_exposure_duration_min=0,  # [min] UV exposure per cycle
        num_switching_per_3hour=1,  # [-]   Switching count / 3 h
        # 6. Superheat / subcool --------------------------------------
        dT_superheat=3.0,  # [K] Evaporator outlet superheat
        dT_subcool=3.0,  # [K] Condenser outlet subcool
    ):

        # --- 1. Tank geometry and thermal properties ---
        self.tank_physical = {
            "r0": r0,
            "H": H,
            "x_shell": x_shell,
            "x_ins": x_ins,
            "k_shell": k_shell,
            "k_ins": k_ins,
            "h_o": h_o,
        }
        self.UA_tank = calc_simple_tank_UA(**self.tank_physical)
        self.C_tank = c_w * rho_w * (math.pi * r0**2 * H)

        # --- 2. Refrigerant / cycle / compressor ---
        self.ref_params = {
            "refrigerant": refrigerant,
            "V_disp_cmp": V_disp_cmp,
            "eta_cmp_isen": eta_cmp_isen,
        }
        self.ref = refrigerant
        self.V_disp_cmp = V_disp_cmp
        self.eta_cmp_isen = eta_cmp_isen

        # --- 3. Heat exchanger UA values ---
        self.heat_exchanger = {
            "UA_cond_design": UA_cond_design,
            "UA_evap_design": UA_evap_design,
        }
        self.UA_cond_design = UA_cond_design
        self.UA_evap_design = UA_evap_design
        self.UA_cond = UA_cond_design
        self.UA_evap = UA_evap_design

        # --- 4. Control and load parameters ---
        self.control_params = {
            "T0": T0,
            "Ts": Ts,
            "T_tank_w_upper_bound": T_tank_w_upper_bound,
            "T_tank_w_lower_bound": T_tank_w_lower_bound,
            "T_mix_w_out": T_mix_w_out,
            "T_tank_w_in": T_tank_w_in,
            "hp_capacity": hp_capacity,
            "dV_mix_w_out_max": dV_mix_w_out_max,
        }
        self.Ts = Ts
        self.T_b_f_in = Ts  # Initial value; updated during simulation

        self.hp_capacity = hp_capacity
        self.dV_mix_w_out_max = dV_mix_w_out_max
        self.T_tank_w_upper_bound = T_tank_w_upper_bound
        self.T_tank_w_lower_bound = T_tank_w_lower_bound

        # --- 5. UV lamp parameters ---
        self.lamp_power_watts = lamp_power_watts
        self.uv_lamp_exposure_duration_min = uv_lamp_exposure_duration_min
        self.num_switching_per_3hour = num_switching_per_3hour
        self.period_3hour_sec = 3 * cu.h2s  # 3 h → s
        self.uv_lamp_exposure_duration_sec = (
            uv_lamp_exposure_duration_min * cu.m2s
        )

        # --- 6. Superheat / subcool ---
        self.dT_superheat = dT_superheat
        self.dT_subcool = dT_subcool
        self.T_tank_w_in = T_tank_w_in
        self.T_mix_w_out = T_mix_w_out

        # --- 7. Borehole heat exchanger ---
        self.borehole = {
            "D_b": D_b,
            "H_b": H_b,
            "r_b": r_b,
            "R_b": R_b,
        }
        self.D_b = D_b
        self.H_b = H_b
        self.r_b = r_b
        self.R_b = R_b

        self.k_s = k_s
        self.c_s = c_s
        self.alp_s = k_s / (c_s * rho_s)  # Soil thermal diffusivity [m²/s]
        self.rho_s = rho_s
        self.E_pmp = E_pmp

        # --- Unit conversions (°C → K, L/min → m3/s) ---
        self.T0_K = cu.C2K(T0)
        self.Ts_K = cu.C2K(self.Ts)
        self.T_b_f_in_K = cu.C2K(self.T_b_f_in)
        self.T_tank_w_in_K = cu.C2K(T_tank_w_in)
        self.T_mix_w_out_K = cu.C2K(T_mix_w_out)
        self.dV_b_f_m3s = dV_b_f_lpm * cu.L2m3 / cu.m2s

        self.Q_cond_LOAD_OFF_TOL = 500.0  # [W] Below this → full OFF

        # Warm-start: reuse previous optimisation result
        self.prev_opt_x = None

    def _calc_off_state(self, T_tank_w, T0):
        """Return a zero-load result dict for the OFF state.

        Provides saturation-point thermodynamic properties at the
        current tank and borehole temperatures for consistent logging.

        Parameters
        ----------
        T_tank_w : float
            Current tank water temperature [°C].
        T0 : float
            Dead-state temperature [°C].

        Returns
        -------
        dict
            Result dictionary with all keys set to OFF-state values.
        """
        T_tank_w_K = cu.C2K(T_tank_w)

        # Mixing valve calculation
        mix = calc_mixing_valve(
            T_tank_w_K, self.T_tank_w_in_K, self.T_mix_w_out_K
        )
        T_serv_w_actual = mix["T_serv_w_actual"]

        # Saturation properties at evaporator / condenser pressure
        P_ref_evap_sat = CP.PropsSI(
            "P", "T", self.T_b_f_in_K, "Q", 1, self.ref
        )
        h_ref_evap_sat = CP.PropsSI("H", "P", P_ref_evap_sat, "Q", 1, self.ref)
        s_ref_evap_sat = CP.PropsSI("S", "P", P_ref_evap_sat, "Q", 1, self.ref)

        P_ref_cond_sat = CP.PropsSI("P", "T", T_tank_w_K, "Q", 0, self.ref)
        h_ref_cond_sat_l = CP.PropsSI(
            "H", "P", P_ref_cond_sat, "Q", 0, self.ref
        )
        s_ref_cond_sat_l = CP.PropsSI(
            "S", "P", P_ref_cond_sat, "Q", 0, self.ref
        )

        return {
            "hp_is_on": False,
            "converged": True,
            # === Temperatures [°C] ===
            "T_tank_w [°C]": T_tank_w,
            "T0 [°C]": T0,
            "T_mix_w_out [°C]": T_serv_w_actual,
            "T_tank_w_in [°C]": self.T_tank_w_in,
            "Ts [°C]": self.Ts,
            "T_bhe [°C]": self.Ts,
            "T_bhe_f [°C]": self.Ts,
            "T_bhe_f_in [°C]": self.Ts,
            "T_bhe_f_out [°C]": self.Ts,
            "T_ref_evap_sat [°C]": cu.K2C(self.T_b_f_in_K),
            "T_ref_cond_sat_v [°C]": T_tank_w,
            "T_ref_cond_sat_l [°C]": T_tank_w,
            "T_ref_cmp_in [°C]": cu.K2C(self.T_b_f_in_K),
            "T_ref_cmp_out [°C]": T_tank_w,
            "T_ref_exp_in [°C]": T_tank_w,
            "T_ref_exp_out [°C]": cu.K2C(self.T_b_f_in_K),
            "T_cond [°C]": T_tank_w,
            # === Flow rates [m3/s] ===
            "dV_mix_w_out [m3/s]": getattr(self, "dV_mix_w_out", 0.0),
            "dV_tank_w_in [m3/s]": getattr(self, "dV_tank_w_in", 0.0),
            "dV_mix_sup_w_in [m3/s]": getattr(self, "dV_mix_sup_w_in", 0.0),
            "dV_bhe_f [m3/s]": self.dV_b_f_m3s,
            # === Pressures [Pa] ===
            "P_ref_cmp_in [Pa]": P_ref_evap_sat,
            "P_ref_cmp_out [Pa]": P_ref_cond_sat,
            "P_ref_exp_in [Pa]": P_ref_cond_sat,
            "P_ref_exp_out [Pa]": P_ref_evap_sat,
            "P_ref_evap_sat [Pa]": P_ref_evap_sat,
            "P_ref_cond_sat_v [Pa]": P_ref_cond_sat,
            "P_ref_cond_sat_l [Pa]": P_ref_cond_sat,
            # === Enthalpy [J/kg] ===
            "h_ref_cmp_in [J/kg]": h_ref_evap_sat,
            "h_ref_cmp_out [J/kg]": h_ref_evap_sat,
            "h_ref_exp_in [J/kg]": h_ref_cond_sat_l,
            "h_ref_exp_out [J/kg]": h_ref_cond_sat_l,
            "h_ref_evap_sat [J/kg]": h_ref_evap_sat,
            "h_ref_cond_sat_v [J/kg]": h_ref_evap_sat,
            "h_ref_cond_sat_l [J/kg]": h_ref_cond_sat_l,
            # === Entropy [J/(kg·K)] ===
            "s_ref_cmp_in [J/(kg·K)]": s_ref_evap_sat,
            "s_ref_cmp_out [J/(kg·K)]": s_ref_evap_sat,
            "s_ref_exp_in [J/(kg·K)]": s_ref_cond_sat_l,
            "s_ref_exp_out [J/(kg·K)]": s_ref_cond_sat_l,
            "s_ref_evap_sat [J/(kg·K)]": s_ref_evap_sat,
            "s_ref_cond_sat_v [J/(kg·K)]": s_ref_evap_sat,
            "s_ref_cond_sat_l [J/(kg·K)]": s_ref_cond_sat_l,
            # === Exergy [J/kg] ===
            "x_ref_cmp_in [J/kg]": 0.0,
            "x_ref_cmp_out [J/kg]": 0.0,
            "x_ref_exp_in [J/kg]": 0.0,
            "x_ref_exp_out [J/kg]": 0.0,
            # === Heat rates [W] ===
            "Q_bhe [W]": 0.0,
            "Q_ref_cond [W]": 0.0,
            "Q_ref_evap [W]": 0.0,
            "Q_LMTD_cond [W]": 0.0,
            "Q_LMTD_evap [W]": 0.0,
            "Q_cond_load [W]": 0.0,
            # === Power [W] ===
            "E_cmp [W]": 0.0,
            "E_pmp [W]": 0.0,
            "E_tot [W]": 0.0,
            # === Mass flow ===
            "m_dot_ref [kg/s]": 0.0,
            "cmp_rpm [rpm]": 0.0,
        }

    def _calc_state(self, optimization_vars, T_tank_w, Q_cond_load, T0):
        """Evaluate refrigerant cycle performance for a given operating point.

        Delegates to ``_calc_off_state`` when ``Q_cond_load <= 0``.

        Parameters
        ----------
        optimization_vars : list of float
            ``[dT_ref_evap, dT_ref_cond]`` — evaporator and condenser
            approach temperature differences [K].
        T_tank_w : float
            Current tank water temperature [°C].
        Q_cond_load : float
            Target condenser heat rate [W].  ≤ 0 → OFF state.
        T0 : float
            Dead-state temperature for exergy analysis [°C].

        Returns
        -------
        dict or None
            Result dictionary.  ``None`` if the cycle is physically infeasible.
        """
        # --- OFF state delegation ---
        if Q_cond_load <= 0:
            return self._calc_off_state(T_tank_w, T0)

        # --- Step 1: Unpack optimisation variables ---
        dT_ref_evap = optimization_vars[0]  # Evaporator approach ΔT [K]
        dT_ref_cond = optimization_vars[1]  # Condenser approach ΔT [K]

        # --- Step 2: Temperature conversions ---
        T_tank_w_K = cu.C2K(T_tank_w)
        cu.C2K(T0)
        T_bhe_f_in_K = self.T_b_f_in_K  # BHE fluid inlet [K]

        T_ref_evap_sat_K = (
            T_bhe_f_in_K - dT_ref_evap
        )  # Evaporation sat. temp [K]
        T_ref_cond_sat_K = (
            T_tank_w_K + dT_ref_cond
        )  # Condensation sat. temp [K]

        # --- Step 3: Refrigerant cycle state calculation ---
        cycle_states = calc_ref_state(
            T_evap_K=T_ref_evap_sat_K,
            T_cond_K=T_ref_cond_sat_K,
            refrigerant=self.ref,
            eta_cmp_isen=self.eta_cmp_isen,
            dT_superheat=self.dT_superheat,
            dT_subcool=self.dT_subcool,
        )

        rho_ref_cmp_in = cycle_states["rho_ref_cmp_in [kg/m3]"]

        # --- Step 4: Extract state-point properties (descriptive names) ---
        T_ref_cmp_in_K = cycle_states["T_ref_cmp_in_K"]
        P_ref_cmp_in = cycle_states["P_ref_cmp_in [Pa]"]
        h_ref_cmp_in = cycle_states["h_ref_cmp_in [J/kg]"]
        cycle_states["s_ref_cmp_in [J/(kg·K)]"]
        T_ref_cmp_out_K = cycle_states["T_ref_cmp_out_K"]
        P_ref_cmp_out = cycle_states["P_ref_cmp_out [Pa]"]
        h_ref_cmp_out = cycle_states["h_ref_cmp_out [J/kg]"]
        cycle_states["s_ref_cmp_out [J/(kg·K)]"]
        T_ref_exp_in_K = cycle_states["T_ref_exp_in_K"]
        P_ref_exp_in = cycle_states["P_ref_exp_in [Pa]"]
        h_ref_exp_in = cycle_states["h_ref_exp_in [J/kg]"]
        cycle_states["s_ref_exp_in [J/(kg·K)]"]
        T_ref_exp_out_K = cycle_states["T_ref_exp_out_K"]
        cycle_states["P_ref_exp_out [Pa]"]
        h_ref_exp_out = cycle_states["h_ref_exp_out [J/kg]"]
        cycle_states["s_ref_exp_out [J/(kg·K)]"]

        if (h_ref_exp_in - h_ref_cmp_out) == 0:
            return None

        # --- Step 5: Refrigerant mass flow and cycle performance ---
        m_dot_ref = Q_cond_load / (h_ref_cmp_out - h_ref_exp_in)
        Q_ref_cond = m_dot_ref * (h_ref_cmp_out - h_ref_exp_in)
        Q_ref_evap = m_dot_ref * (h_ref_cmp_in - h_ref_exp_out)
        E_cmp = m_dot_ref * (h_ref_cmp_out - h_ref_cmp_in)
        cmp_rps = m_dot_ref / (self.V_disp_cmp * rho_ref_cmp_in)

        # --- Step 6: Condenser heat transfer (simplified ΔT model) ---
        Q_LMTD_cond = self.UA_cond * dT_ref_cond

        # --- Step 7: Borehole heat exchange (evaporator side) ---
        Q_bhe = Q_ref_evap - self.E_pmp
        Q_bhe_unit = Q_bhe / self.H_b

        T_bhe_f_out_K = T_bhe_f_in_K + Q_ref_evap / (
            c_w * rho_w * self.dV_b_f_m3s
        )
        T_bhe_f_out = cu.K2C(T_bhe_f_out_K)
        T_bhe_f = (cu.K2C(T_bhe_f_in_K) + T_bhe_f_out) / 2
        T_bhe = T_bhe_f + Q_bhe_unit * self.R_b

        # Evaporator LMTD (counter-flow)
        dT1_evap = T_bhe_f_in_K - T_ref_cmp_in_K
        dT2_evap = T_bhe_f_out_K - T_ref_exp_out_K

        if (
            dT1_evap <= 1e-6
            or dT2_evap <= 1e-6
            or abs(dT1_evap - dT2_evap) < 1e-6
        ):
            Q_LMTD_evap = np.inf
        else:
            LMTD_evap = (dT1_evap - dT2_evap) / np.log(dT1_evap / dT2_evap)
            Q_LMTD_evap = self.UA_evap * LMTD_evap

        # Saturation-point properties
        T_ref_evap_sat_K_star = cycle_states.get("T_ref_evap_sat_K", np.nan)
        T_ref_cond_sat_v_K = cycle_states.get("T_ref_cond_sat_v_K", np.nan)
        T_ref_cond_sat_l_K = cycle_states.get("T_ref_cond_sat_l_K", np.nan)
        P_ref_cond_sat_v = cycle_states.get(
            "P_ref_cond_sat_v [Pa]", P_ref_cmp_out
        )

        P_ref_evap_sat = P_ref_cmp_in
        h_ref_evap_sat = CP.PropsSI("H", "P", P_ref_evap_sat, "Q", 1, self.ref)

        h_ref_cond_sat_v = cycle_states.get("h_ref_cond_sat_v [J/kg]", np.nan)
        if np.isnan(h_ref_cond_sat_v):
            h_ref_cond_sat_v = CP.PropsSI(
                "H", "P", P_ref_cond_sat_v, "Q", 1, self.ref
            )

        P_ref_cond_sat_l = P_ref_exp_in
        h_ref_cond_sat_l = CP.PropsSI(
            "H", "P", P_ref_cond_sat_l, "Q", 0, self.ref
        )

        # --- Step 9: Assemble result dictionary ---
        T_bhe_f_in = cu.K2C(T_bhe_f_in_K)

        result: dict = cycle_states.copy()

        result.update(
            {
                "hp_is_on": True,
                "converged": True,
                # === Temperatures [°C] ===
                "T_ref_evap_sat [°C]": cu.K2C(T_ref_evap_sat_K_star),
                "T_ref_cond_sat_v [°C]": cu.K2C(T_ref_cond_sat_v_K)
                if not np.isnan(T_ref_cond_sat_v_K)
                else np.nan,
                "T_ref_cond_sat_l [°C]": cu.K2C(T_ref_cond_sat_l_K),
                "T0 [°C]": T0,
                "T_ref_cmp_in [°C]": cu.K2C(T_ref_cmp_in_K),
                "T_ref_cmp_out [°C]": cu.K2C(T_ref_cmp_out_K),
                "T_ref_exp_in [°C]": cu.K2C(T_ref_exp_in_K),
                "T_ref_exp_out [°C]": cu.K2C(T_ref_exp_out_K),
                "T_cond [°C]": cu.K2C(
                    T_ref_cond_sat_l_K
                    if not np.isnan(T_ref_cond_sat_l_K)
                    else T_ref_exp_in_K
                ),
                "T_tank_w [°C]": T_tank_w,
                "T_mix_w_out [°C]": self.T_mix_w_out,
                "T_tank_w_in [°C]": self.T_tank_w_in,
                "Ts [°C]": self.Ts,
                "T_bhe [°C]": T_bhe,
                "T_bhe_f [°C]": T_bhe_f,
                "T_bhe_f_in [°C]": T_bhe_f_in,
                "T_bhe_f_out [°C]": T_bhe_f_out,
                # === Flow rates [m3/s] ===
                "dV_bhe_f [m3/s]": self.dV_b_f_m3s,
                "dV_mix_w_out [m3/s]": getattr(self, "dV_mix_w_out", 0.0),
                "dV_tank_w_in [m3/s]": getattr(self, "dV_tank_w_in", 0.0),
                "dV_mix_sup_w_in [m3/s]": getattr(
                    self, "dV_mix_sup_w_in", 0.0
                ),
                # === Pressures [Pa] ===
                "P_ref_evap_sat [Pa]": P_ref_evap_sat,
                "P_ref_cond_sat_l [Pa]": P_ref_cond_sat_l,
                # === Mass flow [kg/s] ===
                "m_dot_ref [kg/s]": m_dot_ref,
                "cmp_rpm [rpm]": cmp_rps * 60,
                # === Specific enthalpy [J/kg] ===
                "h_ref_evap_sat [J/kg]": h_ref_evap_sat,
                "h_ref_cond_sat_v [J/kg]": h_ref_cond_sat_v,
                "h_ref_cond_sat_l [J/kg]": h_ref_cond_sat_l,
                # === Heat rates [W] ===
                "Q_cond_load [W]": Q_cond_load,
                "Q_ref_cond [W]": Q_ref_cond,
                "Q_ref_evap [W]": Q_ref_evap,
                "Q_LMTD_cond [W]": Q_LMTD_cond,
                "Q_LMTD_evap [W]": Q_LMTD_evap,
                "Q_bhe [W]": Q_bhe,
                # === Power [W] ===
                "E_cmp [W]": E_cmp,
                "E_pmp [W]": self.E_pmp,
                "E_tot [W]": E_cmp + self.E_pmp,
            }
        )

    def analyze_steady(
        self,
        T_tank_w,
        T_b_f_in,
        dV_mix_w_out=None,
        Q_cond_load=None,
        T0=None,
        return_dict=True,
    ):
        """Run a steady-state analysis at the given operating point.

        Exactly one of ``dV_mix_w_out`` or ``Q_cond_load`` must be provided.

        Parameters
        ----------
        T_tank_w : float
            Tank water temperature [°C].
        T_b_f_in : float
            Borehole fluid inlet temperature [°C].
        dV_mix_w_out : float, optional
            Service water flow rate [m3/s].  Used to compute Q_cond_load.
        Q_cond_load : float, optional
            Target condenser heat rate [W].
        T0 : float, optional
            Dead-state temperature [°C].  Defaults to ``T_b_f_in``.
        return_dict : bool
            If True return dict; else single-row DataFrame.

        Returns
        -------
        dict or pd.DataFrame
        """
        if dV_mix_w_out is None and Q_cond_load is None:
            raise ValueError(
                "dV_mix_w_out와 Q_cond_load 중 하나는 반드시 제공되어야 합니다."
            )
        if dV_mix_w_out is not None and Q_cond_load is not None:
            raise ValueError(
                "dV_mix_w_out와 Q_cond_load를 동시에 제공할 수 없습니다."
            )

        T0 = T0 if T0 is not None else T_b_f_in
        T_tank_w_K = cu.C2K(T_tank_w)
        T0_K = cu.C2K(T0)

        self.T_b_f_in = T_b_f_in
        self.T_b_f_in_K = cu.C2K(T_b_f_in)

        if dV_mix_w_out is None:
            dV_mix_w_out = 0.0

        den = max(1e-6, T_tank_w_K - self.T_tank_w_in_K)
        alp = min(
            1.0, max(0.0, (self.T_mix_w_out_K - self.T_tank_w_in_K) / den)
        )
        self.dV_mix_w_out = dV_mix_w_out
        self.dV_tank_w_in = alp * dV_mix_w_out
        self.dV_mix_sup_w_in = (1 - alp) * dV_mix_w_out

        if Q_cond_load is None:
            Q_tank_loss = self.UA_tank * (T_tank_w_K - T0_K)
            Q_use_loss = (
                c_w
                * rho_w
                * self.dV_tank_w_in
                * (T_tank_w_K - self.T_tank_w_in_K)
            )
            Q_cond_load = Q_tank_loss + Q_use_loss

        hp_is_on = T_tank_w <= self.T_tank_w_lower_bound or (
            self.T_tank_w_lower_bound < T_tank_w < self.T_tank_w_upper_bound
            and Q_cond_load > 0
        )

        if Q_cond_load <= 0 or not hp_is_on:
            result = self._calc_state(
                optimization_vars=[5.0, 5.0],
                T_tank_w=T_tank_w,
                Q_cond_load=0.0,
                T0=T0,
            )
        else:
            opt_result = self._optimize_operation(
                T_tank_w=T_tank_w, Q_cond_load=Q_cond_load, T0=T0
            )
            if opt_result.success:
                result = self._calc_state(
                    optimization_vars=opt_result.x,
                    T_tank_w=T_tank_w,
                    Q_cond_load=Q_cond_load,
                    T0=T0,
                )
            else:
                result = self._calc_state(
                    optimization_vars=[5.0, 5.0],
                    T_tank_w=T_tank_w,
                    Q_cond_load=0.0,
                    T0=T0,
                )

        if return_dict:
            return result
        return pd.DataFrame([result])

    def _optimize_operation(self, T_tank_w, Q_cond_load, T0):
        """Find minimum-power operating point via Differential Evolution.

        Optimisation variables are ``[dT_ref_evap, dT_ref_cond]``.
        The condenser LMTD constraint is enforced via a penalty function.

        Parameters
        ----------
        T_tank_w : float
            Tank water temperature [°C].
        Q_cond_load : float
            Target condenser heat rate [W].
        T0 : float
            Dead-state temperature [°C].

        Returns
        -------
        scipy.optimize.OptimizeResult
        """
        tolerance = 0.01  # 1%
        bounds = [(1.0, 30.0), (1.0, 30.0)]  # [dT_ref_evap, dT_ref_cond]
        penalty_weight = 1e4

        def _objective(x):
            """
            목적 함수: E_tot 최소화 + 응축기 LMTD 제약 위반 penalty.
            DE는 population-based 글로벌 최적화이므로 penalty 방식으로 제약 처리.
            """
            try:
                perf = self._calc_state(
                    optimization_vars=x,
                    T_tank_w=T_tank_w,
                    Q_cond_load=Q_cond_load,
                    T0=T0,
                )
                if perf is None or not isinstance(perf, dict):
                    return 1e6

                E_tot = perf.get("E_tot [W]", np.nan)
                if np.isnan(E_tot):
                    return 1e6

                # --- 응축기 LMTD 제약 조건 penalty ---
                penalty = 0.0
                Q_LMTD_cond = perf.get("Q_LMTD_cond [W]", np.nan)
                if not np.isnan(Q_LMTD_cond):
                    # 하한: Q_LMTD_cond >= Q_cond_load
                    cond_low = Q_cond_load - Q_LMTD_cond
                    if cond_low > 0:
                        penalty += penalty_weight * cond_low**2
                    # 상한: Q_LMTD_cond <= Q_cond_load*(1+tolerance)
                    cond_high = Q_LMTD_cond - Q_cond_load * (1 + tolerance)
                    if cond_high > 0:
                        penalty += penalty_weight * cond_high**2

                return E_tot + penalty
            except Exception:
                return 1e6

        # Warm-start: 이전 최적화 결과를 초기 population에 주입
        x0 = self.prev_opt_x if self.prev_opt_x is not None else None

        opt_result = differential_evolution(
            _objective,
            bounds=bounds,
            x0=x0,
            maxiter=50,
            popsize=10,
            tol=1e-4,
            seed=42,
            polish=True,
            disp=False,
        )

        if opt_result.success:
            self.prev_opt_x = np.array(opt_result.x).copy()
        return opt_result

    def analyze_dynamic(
        self,
        simulation_period_sec,
        dt_s,
        T_tank_w_init_C,
        dhw_usage_schedule,
        T0_schedule,
        result_save_csv_path=None,
    ):
        """Run a time-stepping dynamic simulation with BHE g-function.

        Parameters
        ----------
        simulation_period_sec : int
            Total simulation duration [s].
        dt_s : int
            Time step size [s].
        T_tank_w_init_C : float
            Initial tank temperature [°C].
        dhw_usage_schedule : list of tuple
            DHW schedule ``(start_str, end_str, fraction)``.
        T0_schedule : array-like
            Dead-state temperature per step [°C].
        result_save_csv_path : str, optional
            CSV output path.

        Returns
        -------
        pd.DataFrame
            Per-timestep results.
        """

        # --- 0. 실행 조건 판단 ---
        time = np.arange(0, simulation_period_sec, dt_s)
        tN = len(time)
        T0_schedule = np.array(T0_schedule)
        if len(T0_schedule) != tN:
            raise ValueError(
                f"T0_schedule length ({len(T0_schedule)}) must match time array length ({tN})"
            )

        results_data = []

        self.time = time
        self.dt = dt_s

        # --- 1. 시뮬레이션 초기화 ---
        self.T_bhe_f = self.Ts  # 초기 BHE 평균 유체 온도
        self.T_bhe = self.Ts  # 초기 BHE 벽면 온도
        self.T_bhe_f_in = self.Ts  # 초기 BHE 유입수 온도
        self.T_bhe_f_out = self.Ts  # 초기 BHE 유출수 온도
        self.Q_bhe = 0.0  # 초기 BHE 열 유량

        self.dV_mix_w_out = 0.0
        self.dV_tank_w_in = 0.0
        self.dV_mix_sup_w_in = 0.0

        self.w_use_frac = build_dhw_usage_ratio(dhw_usage_schedule, self.time)

        T_tank_w_K = cu.C2K(T_tank_w_init_C)
        Q_bhe_unit_pulse = np.zeros(tN)
        Q_bhe_unit_old = 0
        is_on_prev = False

        # --- 2. 시뮬레이션 루프 ---
        for n in tqdm(range(tN), desc="GSHPB Simulating"):
            step_results = {}
            T_tank_w = cu.K2C(T_tank_w_K)
            T0 = T0_schedule[n]
            T0_K = cu.C2K(T0)

            # 제어 상태 결정
            Q_tank_loss = self.UA_tank * (T_tank_w_K - T0_K)
            den = max(1e-6, T_tank_w_K - self.T_tank_w_in_K)
            alp = min(
                1.0, max(0.0, (self.T_mix_w_out_K - self.T_tank_w_in_K) / den)
            )

            self.dV_mix_w_out = self.w_use_frac[n] * self.dV_mix_w_out_max
            self.dV_tank_w_in = alp * self.dV_mix_w_out
            self.dV_mix_sup_w_in = (1 - alp) * self.dV_mix_w_out

            Q_use_loss = (
                c_w
                * rho_w
                * self.dV_tank_w_in
                * (T_tank_w_K - self.T_tank_w_in_K)
            )
            total_loss = Q_tank_loss + Q_use_loss

            # On/Off 결정
            if T_tank_w <= self.T_tank_w_lower_bound:
                is_on = True
            elif T_tank_w >= self.T_tank_w_upper_bound:
                is_on = False
            else:
                is_on = is_on_prev

            is_transitioning_off_to_on = (
                not is_on_prev
            ) and is_on  # False to True
            Q_cond_load_n = self.hp_capacity if is_on else 0.0
            is_on_prev = is_on

            # OFF 상태 조기 체크: Q_cond_load_n이 임계값 이하이면 최적화 건너뛰기
            if abs(Q_cond_load_n) <= self.Q_cond_LOAD_OFF_TOL:
                self.prev_opt_x = None  # OFF 시 초기값 재설정
                result = self._calc_state(
                    optimization_vars=[5.0, 5.0],
                    T_tank_w=T_tank_w,
                    Q_cond_load=0.0,
                    T0=T0,
                )
            else:
                # 최적화 과정 진행
                opt_result = self._optimize_operation(
                    T_tank_w=T_tank_w,
                    Q_cond_load=Q_cond_load_n,
                    T0=T0,
                )

                # opt_result.x로 결과 계산 시도 (성공/실패 무관)
                result = None
                with contextlib.suppress(Exception):
                    result = self._calc_state(
                        optimization_vars=opt_result.x,
                        T_tank_w=T_tank_w,
                        Q_cond_load=Q_cond_load_n,
                        T0=T0,
                    )

                # result 검증 및 fallback
                if result is None or not isinstance(result, dict):
                    try:
                        result = self._calc_off_state(T_tank_w=T_tank_w, T0=T0)
                    except Exception:
                        result = {
                            "hp_is_on": False,
                            "converged": False,
                            "Q_ref_cond [W]": 0.0,
                            "Q_ref_evap [W]": 0.0,
                            "E_cmp [W]": 0.0,
                            "E_pmp [W]": 0.0,
                            "E_tot [W]": 0.0,
                            "T_tank_w [°C]": T_tank_w,
                            "T0 [°C]": T0,
                        }

                # converged 플래그 설정
                if result is not None and isinstance(result, dict):
                    result["converged"] = opt_result.success

            if is_transitioning_off_to_on:
                # OFF→ON 전환 시점: 점진적 전환을 위해 이전 스텝의 지중 온도 값 사용
                # result는 ON 상태로 계산되었지만, 지중 온도는 점진적으로 업데이트
                step_results.update(result)
                step_results["hp_is_on"] = is_on
                # 전환 시점임을 표시하는 플래그 추가
                step_results["is_transitioning"] = True

                # 전환 시점에서는 이전 스텝의 지중 온도 값 유지 (점진적 전환)
                step_results["T_bhe [°C]"] = self.T_bhe
                step_results["T_bhe_f [°C]"] = self.T_bhe_f
                step_results["T_bhe_f_in [°C]"] = self.T_bhe_f_in
                step_results["T_bhe_f_out [°C]"] = self.T_bhe_f_out
                step_results["Q_bhe [W]"] = 0.0
            else:
                step_results.update(result)
                step_results["hp_is_on"] = is_on
                step_results["is_transitioning"] = False

            # 지중 온도 업데이트
            if is_transitioning_off_to_on:
                # 전환 시점: 점진적 전환을 위해 Q_bhe_unit을 0으로 시작
                Q_bhe_unit = 0.0
                # 펄스 계산 건너뛰기 (전환 시점에서는 펄스 없음)
                # Q_bhe_unit_old를 0으로 업데이트하여 다음 스텝에서 정상 계산 시작
                Q_bhe_unit_old = 0.0
            else:
                # 일반 시점: 정상 계산
                Q_bhe_unit = (
                    (result.get("Q_ref_evap [W]", 0.0) - self.E_pmp) / self.H_b
                    if result.get("hp_is_on")
                    else 0.0
                )

            if abs(Q_bhe_unit - Q_bhe_unit_old) > 1e-6:
                Q_bhe_unit_pulse[n] = Q_bhe_unit - Q_bhe_unit_old
                Q_bhe_unit_old = Q_bhe_unit

            # 펄스 계산 (전환 시점이 아닐 때만)
            if not is_transitioning_off_to_on:
                pulses_idx = np.flatnonzero(Q_bhe_unit_pulse[: n + 1])
                dQ = Q_bhe_unit_pulse[pulses_idx]
                tau = self.time[n] - self.time[pulses_idx]

                g_n = np.array(
                    [
                        G_FLS(t, self.k_s, self.alp_s, self.r_b, self.H_b)
                        for t in tau
                    ]
                )
                dT_bhe = np.dot(dQ, g_n)

                self.T_bhe = self.Ts - dT_bhe
                self.T_bhe_f = self.T_bhe - Q_bhe_unit * self.R_b
                self.Q_bhe = Q_bhe_unit * self.H_b
                self.T_bhe_f_in = self.T_bhe_f - self.Q_bhe / (
                    2 * c_w * rho_w * self.dV_b_f_m3s
                )
                self.T_bhe_f_out = self.T_bhe_f + self.Q_bhe / (
                    2 * c_w * rho_w * self.dV_b_f_m3s
                )
                self.T_b_f_in_K = cu.C2K(self.T_bhe_f_in)
                self.T_b_f_out_K = cu.C2K(self.T_bhe_f_out)

                step_results["T_bhe [°C]"] = self.T_bhe
                step_results["T_bhe_f [°C]"] = self.T_bhe_f
                step_results["T_bhe_f_in [°C]"] = self.T_bhe_f_in
                step_results["T_bhe_f_out [°C]"] = self.T_bhe_f_out
                step_results["Q_bhe [W]"] = self.Q_bhe

            # UV 램프 전력 계산 (공유 함수 사용)
            E_uv = calc_uv_lamp_power(
                current_time_s=time[n],
                period_sec=self.period_3hour_sec,
                num_switching=self.num_switching_per_3hour,
                exposure_sec=self.uv_lamp_exposure_duration_sec,
                lamp_watts=self.lamp_power_watts,
            )

            step_results["hp_is_on"] = is_on
            if self.lamp_power_watts > 0:
                step_results["E_uv [W]"] = E_uv

            # 다음 스텝 탱크 온도 계산
            if n < tN - 1:
                # nan인 경우 0으로 처리 (탱크 온도 계산에는 실제 열량이 필요)
                Q_ref_cond_val = result.get("Q_ref_cond [W]", np.nan)
                Q_ref_cond_val = np.nan_to_num(Q_ref_cond_val, nan=0.0)
                E_uv_val = step_results.get("E_uv [W]", 0.0)
                Q_tank_in = Q_ref_cond_val + E_uv_val
                Q_net = Q_tank_in - total_loss
                T_tank_w_K += (Q_net / self.C_tank) * self.dt

            results_data.append(step_results)

        results_df = pd.DataFrame(results_data)

        if result_save_csv_path:
            results_df.to_csv(result_save_csv_path, index=False)

        return results_df
