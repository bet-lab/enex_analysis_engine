"""Air source heat pump boiler — physics-based cycle model.

Resolves a vapour-compression refrigerant cycle coupled to an outdoor-air
evaporator with a VSD fan and a lumped-capacitance hot-water tank.
At each time step the model finds the minimum-power operating point
(compressor + fan) via bounded 1-D optimisation (Brent's method) over
the evaporator approach temperature difference.  The condenser approach
temperature is determined analytically from the target heat load.

Optional subsystems (injected via constructor):
- ``SolarThermalCollector`` — tank-circuit or mains-preheat placement
- (future) ``PVPanel`` — photovoltaic integration

Tank-level management and UV disinfection are built-in features
configured through constructor parameters.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from tqdm import tqdm
import CoolProp.CoolProp as CP

from . import calc_util as cu
from .constants import *
from .enex_functions import *
from .dynamic_context import (
    ControlState,
    StepContext,
    Subsystem,
    determine_hp_on_off,
    determine_tank_refill_flow,
    tank_mass_energy_residual,
)
from .subsystems import SolarThermalCollector


@dataclass
class AirSourceHeatPumpBoiler:
    """Air source heat pump boiler with outdoor-air evaporator.

    The refrigerant cycle is resolved via CoolProp with
    user-specified superheat / subcool margins.  The condenser
    approach temperature is determined analytically
    (``dT_ref_cond = Q_cond_target / UA_cond``), and a bounded
    1-D optimiser (Brent's method) minimises total electrical
    input (``E_cmp + E_ou_fan``) over the evaporator approach.
    """

    def __init__(
        self,

        # 1. Refrigerant / cycle / compressor -----------
        ref: str = 'R134a',
        V_disp_cmp: float = 0.0002,
        eta_cmp_isen: float = 0.8,
        dT_superheat: float = 3.0,
        dT_subcool: float = 3.0,

        # 2. Heat exchanger -----------------------------
        UA_cond_design: float = 2000.0,
        UA_evap_design: float = 1000.0,

        # 3. Outdoor unit fan ---------------------------
        dV_ou_fan_a_design: float = 1.5,
        dP_ou_fan_design: float = 90.0,
        A_cross_ou: float = 0.25 ** 2 * np.pi,
        eta_ou_fan_design: float = 0.6,

        # 4. Tank / control / load ----------------------
        T_tank_w_upper_bound: float = 65.0,
        T_tank_w_lower_bound: float = 60.0,
        T_mix_w_out: float = 40.0,
        T_sup_w: float = 15.0,
        hp_capacity: float = 15000.0,
        dV_mix_w_out_max: float = 0.0045,

        # Tank insulation
        r0: float = 0.2,
        H: float = 1.2,
        x_shell: float = 0.005,
        x_ins: float = 0.05,
        k_shell: float = 25,
        k_ins: float = 0.03,
        h_o: float = 15,

        # 5. UV lamp ------------------------------------
        lamp_power_watts: float = 0,
        uv_lamp_exposure_duration_min: float = 0,
        num_switching_per_3hour: int = 1,

        # 6. Tank water level management ----------------
        tank_always_full: bool = True,
        tank_level_lower_bound: float = 0.5,
        tank_level_upper_bound: float = 1.0,
        dV_tank_w_in_refill: float = 0.001,
        prevent_simultaneous_flow: bool = False,

        # 7. HP operating schedule ----------------------
        hp_on_schedule: list[tuple[float, float]]
            | None = None,

        # 8. Subsystems (class-based injection) ---------
        stc: SolarThermalCollector | None = None,

        # ASHRAE 90.1-2022 VSD coefficients
        vsd_coeffs_ou: dict | None = None,
    ):
        if hp_on_schedule is None:
            hp_on_schedule = [(0.0, 24.0)]
        if vsd_coeffs_ou is None:
            vsd_coeffs_ou = {
                'c1': 0.0013, 'c2': 0.1470,
                'c3': 0.9506, 'c4': -0.0998, 'c5': 0.0,
            }

        # --- 1. Refrigerant / cycle / compressor ---
        self.ref: str = ref
        self.V_disp_cmp: float = V_disp_cmp
        self.eta_cmp_isen: float = eta_cmp_isen
        self.dT_superheat: float = dT_superheat
        self.dT_subcool: float = dT_subcool
        self.hp_capacity: float = hp_capacity

        # --- 2. Heat exchanger UA ---
        self.UA_cond_design: float = UA_cond_design
        self.UA_evap_design: float = UA_evap_design

        # --- 3. Outdoor unit fan ---
        self.dV_ou_fan_a_design: float = dV_ou_fan_a_design
        self.dP_ou_fan_design: float = dP_ou_fan_design
        self.eta_ou_fan_design: float = eta_ou_fan_design
        self.A_cross_ou: float = A_cross_ou
        self.E_ou_fan_design: float = (
            dV_ou_fan_a_design * dP_ou_fan_design
            / eta_ou_fan_design
        )
        self.vsd_coeffs_ou: dict = vsd_coeffs_ou
        self.fan_params_ou: dict = {
            'fan_design_flow_rate': dV_ou_fan_a_design,
            'fan_design_power': self.E_ou_fan_design,
        }

        # --- 4. Tank geometry and thermal props ---
        self.tank_physical: dict = {
            'r0': r0, 'H': H,
            'x_shell': x_shell, 'x_ins': x_ins,
            'k_shell': k_shell, 'k_ins': k_ins,
            'h_o': h_o,
        }
        self.UA_tank: float = calc_simple_tank_UA(
            **self.tank_physical,
        )
        self.V_tank_full: float = math.pi * r0 ** 2 * H
        self.C_tank: float = c_w * rho_w * self.V_tank_full

        self.dV_mix_w_out_max: float = dV_mix_w_out_max
        self.T_tank_w_upper_bound: float = T_tank_w_upper_bound
        self.T_tank_w_lower_bound: float = T_tank_w_lower_bound
        self.T_sup_w: float = T_sup_w
        self.T_sup_w_K: float = cu.C2K(T_sup_w)
        self.T_tank_w_in: float = T_sup_w
        self.T_mix_w_out: float = T_mix_w_out
        self.T_tank_w_in_K: float = cu.C2K(T_sup_w)
        self.T_mix_w_out_K: float = cu.C2K(T_mix_w_out)

        # --- 5. UV lamp ---
        self.lamp_power_watts: float = lamp_power_watts
        self.uv_lamp_exposure_duration_min: float = (
            uv_lamp_exposure_duration_min
        )
        self.num_switching_per_3hour: int = (
            num_switching_per_3hour
        )
        self.period_3hour_sec: float = 3 * cu.h2s
        self.uv_lamp_exposure_duration_sec: float = (
            uv_lamp_exposure_duration_min * cu.m2s
        )

        # --- 6. Tank water level management ---
        self.tank_always_full: bool = tank_always_full
        self.tank_level_lower_bound: float = (
            tank_level_lower_bound
        )
        self.tank_level_upper_bound: float = (
            tank_level_upper_bound
        )
        self.dV_tank_w_in_refill: float = dV_tank_w_in_refill
        self.prevent_simultaneous_flow: bool = (
            prevent_simultaneous_flow
        )

        # --- 7. HP operating schedule ---
        self.hp_on_schedule: list[tuple[float, float]] = (
            hp_on_schedule
        )

        # --- 8. Subsystems ---
        self.stc: SolarThermalCollector | None = stc
        self._subsystems: dict[str, Subsystem] = {}
        if stc is not None:
            self._subsystems['stc'] = stc

        # Flow-rate sync variables
        self.dV_tank_w_in: float = 0.0
        self.dV_tank_w_out: float = 0.0
        self.dV_mix_sup_w_in: float = 0.0
        self.dV_mix_w_out: float = 0.0

    # =============================================================
    # Refrigerant cycle physics (ASHP-specific)
    # =============================================================

    def _calc_state(
        self,
        dT_ref_evap: float,
        T_tank_w: float,
        Q_cond_target: float,
        T0: float,
    ) -> dict | None:
        """Evaluate refrigerant cycle at a given operating point.

        Parameters
        ----------
        dT_ref_evap : float
            Evaporator approach ΔT [K].
        T_tank_w : float
            Tank water temperature [°C].
        Q_cond_target : float
            Target condenser heat rate [W].
        T0 : float
            Dead-state / outdoor-air temperature [°C].

        Returns
        -------
        dict | None
            Cycle performance dictionary; ``None`` if infeasible.
        """
        dT_ref_cond: float = (
            Q_cond_target / self.UA_cond_design
            if Q_cond_target > 0 else 0.0
        )

        T_tank_w_K: float = cu.C2K(T_tank_w)
        T0_K: float = cu.C2K(T0)

        T_evap_sat_K: float = T0_K - dT_ref_evap
        T_cond_sat_K: float = T_tank_w_K + dT_ref_cond

        is_active: bool = Q_cond_target > 0.0

        cs: dict = calc_ref_state(
            T_evap_K=T_evap_sat_K,
            T_cond_K=T_cond_sat_K,
            refrigerant=self.ref,
            eta_cmp_isen=self.eta_cmp_isen,
            T0_K=T0_K,
            mode='heating',
            dT_superheat=self.dT_superheat,
            dT_subcool=self.dT_subcool,
            is_active=is_active,
        )

        m_dot_ref: float = (
            Q_cond_target / (cs['h2'] - cs['h3'])
            if is_active else 0.0
        )
        Q_ref_cond: float = (
            m_dot_ref * (cs['h2'] - cs['h3'])
            if is_active else 0.0
        )
        Q_ref_evap: float = (
            m_dot_ref * (cs['h1'] - cs['h4'])
            if is_active else 0.0
        )
        E_cmp: float = (
            m_dot_ref * (cs['h2'] - cs['h1'])
            if is_active else 0.0
        )
        cmp_rps: float = (
            m_dot_ref / (self.V_disp_cmp * cs['rho'])
            if is_active else 0.0
        )

        HX_perf_ou: dict = calc_HX_perf_for_target_heat(
            Q_ref_target=(
                Q_ref_evap if is_active else 0.0
            ),
            T_ou_a_in_C=T0,
            T1_star_K=cs['T1_star_K'],
            T3_star_K=cs['T3_star_K'],
            A_cross=self.A_cross_ou,
            UA_design=self.UA_evap_design,
            dV_fan_design=self.dV_ou_fan_a_design,
            is_active=is_active,
        )

        if HX_perf_ou.get('converged', True) is False:
            return {
                'converged': False,
                '_hx_diag': HX_perf_ou,
            }

        dV_ou_a: float = HX_perf_ou['dV_fan']
        v_ou_a: float = (
            dV_ou_a / self.A_cross_ou
            if is_active else 0.0
        )
        T_ou_a_mid: float = HX_perf_ou['T_ou_a_mid']
        Q_ou_a: float = HX_perf_ou['Q_ou_air']

        E_ou_fan: float = calc_fan_power_from_dV_fan(
            dV_fan=dV_ou_a,
            fan_params=self.fan_params_ou,
            vsd_coeffs=self.vsd_coeffs_ou,
            is_active=is_active,
        )

        T_ou_a_out: float = (
            T_ou_a_mid
            + E_ou_fan / (c_a * rho_a * dV_ou_a)
            if is_active else T0
        )
        eta_ou_fan: float = (
            self.eta_ou_fan_design
            * dV_ou_a / E_ou_fan * 100
            if is_active else 0.0
        )

        UA_cond: float = self.UA_cond_design

        dV_tank_w_out: float = self.dV_tank_w_out
        dV_tank_w_in: float = self.dV_tank_w_in
        dV_mix_sup_w_in: float = self.dV_mix_sup_w_in
        dV_mix_w_out_val: float = self.dV_mix_w_out

        if dV_mix_w_out_val == 0:
            T_mix_w_out_val: float = np.nan
            T_mix_w_out_val_K: float = np.nan
        else:
            mix: dict = calc_mixing_valve(
                T_tank_w_K,
                self.T_sup_w_K,
                self.T_mix_w_out_K,
            )
            T_mix_w_out_val = mix['T_mix_w_out']
            T_mix_w_out_val_K = mix['T_mix_w_out_K']

        Q_tank_w_in: float = calc_energy_flow(
            G=c_w * rho_w * dV_tank_w_in,
            T=self.T_tank_w_in_K, T0=T0_K,
        )
        Q_tank_w_out: float = calc_energy_flow(
            G=c_w * rho_w * dV_tank_w_out,
            T=T_tank_w_K, T0=T0_K,
        )
        Q_mix_sup_w_in: float = calc_energy_flow(
            G=c_w * rho_w * dV_mix_sup_w_in,
            T=self.T_sup_w_K, T0=T0_K,
        )
        Q_mix_w_out: float = calc_energy_flow(
            G=c_w * rho_w * dV_mix_w_out_val,
            T=T_mix_w_out_val_K, T0=T0_K,
        )

        result: dict = {
            'hp_is_on': (Q_cond_target > 0),
            'converged': True,

            # Temperatures [°C]
            'T_ref_evap_sat [°C]': cu.K2C(
                cs['T1_star_K'],
            ),
            'T_ref_cond_sat_v [°C]': cu.K2C(
                cs['T2_star_K'],
            ),
            'T_ref_cond_sat_l [°C]': cu.K2C(
                cs['T3_star_K'],
            ),
            'T_ou_a_in [°C]': T0,
            'T_ou_a_mid [°C]': T_ou_a_mid,
            'T_ou_a_out [°C]': T_ou_a_out,
            'T_ref_cmp_in [°C]': cu.K2C(cs['T1_K']),
            'T_ref_cmp_out [°C]': cu.K2C(cs['T2_K']),
            'T_ref_exp_in [°C]': cu.K2C(cs['T3_K']),
            'T_ref_exp_out [°C]': cu.K2C(cs['T4_K']),
            'T_tank_w [°C]': T_tank_w,
            'T_sup_w [°C]': self.T_sup_w,
            'T_tank_w_in [°C]': self.T_tank_w_in,
            'T_mix_w_out [°C]': T_mix_w_out_val,
            'T0 [°C]': T0,

            # Volume flow rates [m3/s]
            'dV_ou_a [m3/s]': dV_ou_a,
            'v_ou_a [m/s]': v_ou_a,
            'dV_mix_w_out [m3/s]': (
                dV_mix_w_out_val
                if dV_mix_w_out_val > 0 else np.nan
            ),
            'dV_tank_w_out [m3/s]': (
                dV_tank_w_out
                if dV_tank_w_out > 0 else np.nan
            ),
            'dV_tank_w_in [m3/s]': (
                dV_tank_w_in
                if dV_tank_w_in > 0 else np.nan
            ),
            'dV_mix_sup_w_in [m3/s]': (
                dV_mix_sup_w_in
                if dV_mix_sup_w_in > 0 else np.nan
            ),

            # Pressures [Pa]
            'P_ref_cmp_in [Pa]': cs['P1'],
            'P_ref_cmp_out [Pa]': cs['P2'],
            'P_ref_exp_in [Pa]': cs['P3'],
            'P_ref_exp_out [Pa]': cs['P4'],
            'P_ref_evap_sat [Pa]': (
                cs['P1'] if is_active else np.nan
            ),
            'P_ref_cond_sat_v [Pa]': (
                cs['P2'] if is_active else np.nan
            ),
            'P_ref_cond_sat_l [Pa]': (
                cs['P3'] if is_active else np.nan
            ),
            'dP_ou_fan_static [Pa]': (
                self.dP_ou_fan_design
                - 0.5 * rho_a * v_ou_a ** 2
            ),
            'dP_ou_fan_dynamic [Pa]': (
                0.5 * rho_a * v_ou_a ** 2
            ),

            # Mass flow rate [kg/s]
            'm_dot_ref [kg/s]': m_dot_ref,

            # Compressor speed [rpm]
            'cmp_rpm [rpm]': cmp_rps * 60,

            # Specific enthalpy [J/kg]
            'h_ref_cmp_in [J/kg]': cs['h1'],
            'h_ref_cmp_out [J/kg]': cs['h2'],
            'h_ref_exp_in [J/kg]': cs['h3'],
            'h_ref_exp_out [J/kg]': cs['h4'],
            'h_ref_evap_sat [J/kg]': (
                cs.get('h1_star', np.nan)
                if is_active else np.nan
            ),
            'h_ref_cond_sat_v [J/kg]': (
                cs.get('h2_star', np.nan)
                if is_active else np.nan
            ),
            'h_ref_cond_sat_l [J/kg]': (
                cs.get('h3_star', np.nan)
                if is_active else np.nan
            ),

            # Energy rates [W]
            'E_ou_fan [W]': E_ou_fan,
            'Q_ref_evap [W]': Q_ref_evap,
            'Q_ou_a [W]': Q_ou_a,
            'E_cmp [W]': E_cmp,
            'Q_ref_cond [W]': Q_ref_cond,
            'Q_tank_w_in [W]': Q_tank_w_in,
            'Q_tank_w_out [W]': Q_tank_w_out,
            'Q_mix_sup_w_in [W]': Q_mix_sup_w_in,
            'Q_mix_w_out [W]': Q_mix_w_out,
            'E_tot [W]': E_cmp + E_ou_fan,
        }

        return result

    def _optimize_operation(
        self,
        T_tank_w: float,
        Q_cond_target: float,
        T0: float,
    ):
        """Find min-power operating point (Brent 1-D).

        Parameters
        ----------
        T_tank_w : float
            Tank water temperature [°C].
        Q_cond_target : float
            Target condenser heat rate [W].
        T0 : float
            Dead-state temperature [°C].

        Returns
        -------
        scipy.optimize.OptimizeResult
        """

        def _objective(dT_ref_evap: float) -> float:
            try:
                perf: dict | None = self._calc_state(
                    dT_ref_evap=dT_ref_evap,
                    T_tank_w=T_tank_w,
                    Q_cond_target=Q_cond_target,
                    T0=T0,
                )
                if perf is None or not isinstance(perf, dict):
                    return 1e6
                E_tot: float = perf.get(
                    'E_tot [W]', np.nan,
                )
                return E_tot if not np.isnan(E_tot) else 1e6
            except Exception:
                return 1e6

        return minimize_scalar(
            _objective,
            bounds=(5.0, 15.0),
            method='bounded',
            options={'maxiter': 200, 'xatol': 1e-6},
        )

    # =============================================================
    # Steady-state analysis
    # =============================================================

    def analyze_steady(
        self,
        T_tank_w: float,
        T0: float,
        dV_mix_w_out: float | None = None,
        Q_cond_target: float | None = None,
        return_dict: bool = True,
    ) -> dict | pd.DataFrame:
        """Run a steady-state analysis.

        Exactly one of ``dV_mix_w_out`` or ``Q_cond_target``
        must be provided.

        Parameters
        ----------
        T_tank_w : float
            Tank water temperature [°C].
        T0 : float
            Dead-state temperature [°C].
        dV_mix_w_out : float | None
            Service water flow rate [m³/s].
        Q_cond_target : float | None
            Target condenser heat rate [W].
        return_dict : bool
            If True return dict; else single-row DataFrame.

        Returns
        -------
        dict | pd.DataFrame
        """
        if dV_mix_w_out is None and Q_cond_target is None:
            raise ValueError(
                "Either dV_mix_w_out or Q_cond_target "
                "must be provided.",
            )
        if (
            dV_mix_w_out is not None
            and Q_cond_target is not None
        ):
            raise ValueError(
                "Cannot provide both dV_mix_w_out "
                "and Q_cond_target.",
            )

        T_tank_w_K: float = cu.C2K(T_tank_w)
        T0_K: float = cu.C2K(T0)

        if dV_mix_w_out is None:
            dV_mix_w_out = 0.0

        Q_tank_loss: float = (
            self.UA_tank * (T_tank_w_K - T0_K)
        )
        den: float = max(
            1e-6, T_tank_w_K - self.T_sup_w_K,
        )
        alp: float = min(
            1.0,
            max(
                0.0,
                (self.T_mix_w_out_K - self.T_sup_w_K)
                / den,
            ),
        )

        self.dV_mix_w_out = dV_mix_w_out
        self.dV_tank_w_out = alp * dV_mix_w_out
        self.dV_mix_sup_w_in = (1 - alp) * dV_mix_w_out

        if Q_cond_target is None:
            Q_tank_w_use: float = (
                c_w * rho_w * self.dV_tank_w_out
                * (T_tank_w_K - self.T_sup_w_K)
            )
            Q_cond_target = Q_tank_loss + Q_tank_w_use

        if T_tank_w <= self.T_tank_w_lower_bound:
            hp_is_on: bool = True
        elif T_tank_w > self.T_tank_w_upper_bound:
            hp_is_on = False
        else:
            hp_is_on = Q_cond_target > 0

        if Q_cond_target <= 0 or not hp_is_on:
            result: dict | None = self._calc_state(
                dT_ref_evap=5.0,
                T_tank_w=T_tank_w,
                Q_cond_target=0.0,
                T0=T0,
            )
        else:
            opt_result = self._optimize_operation(
                T_tank_w=T_tank_w,
                Q_cond_target=Q_cond_target,
                T0=T0,
            )
            result = None
            try:
                result = self._calc_state(
                    dT_ref_evap=opt_result.x,
                    T_tank_w=T_tank_w,
                    T0=T0,
                    Q_cond_target=Q_cond_target,
                )
            except Exception:
                pass

            if result is None or not isinstance(
                result, dict,
            ):
                try:
                    result = self._calc_state(
                        dT_ref_evap=5.0,
                        T_tank_w=T_tank_w,
                        Q_cond_target=0.0,
                        T0=T0,
                    )
                except Exception:
                    result = {
                        'hp_is_on': False,
                        'converged': False,
                        'Q_ref_cond [W]': 0.0,
                        'Q_ref_evap [W]': 0.0,
                        'E_cmp [W]': 0.0,
                        'E_ou_fan [W]': 0.0,
                        'E_tot [W]': 0.0,
                        'T_tank_w [°C]': T_tank_w,
                        'T0 [°C]': T0,
                    }

            if result is not None and isinstance(
                result, dict,
            ):
                if (
                    'opt_result' in locals()
                    and hasattr(opt_result, 'success')
                ):
                    result['converged'] = opt_result.success
                    if not result['converged']:
                        print("Optimization failed")

        if return_dict:
            return result
        return pd.DataFrame([result])

    # =============================================================
    # Private helpers for analyze_dynamic
    # =============================================================

    def _determine_hp_state(
        self,
        ctx: StepContext,
        hp_is_on_prev: bool,
    ) -> tuple[bool, dict, float]:
        """HP on/off + cycle optimisation for one step.

        Parameters
        ----------
        ctx : StepContext
            Current-step immutable context.
        hp_is_on_prev : bool
            HP state at previous step.

        Returns
        -------
        tuple[bool, dict, float]
            ``(hp_is_on, hp_result, Q_ref_cond)``.
        """
        T_tank_w: float = cu.K2C(ctx.T_tank_w_K)

        hp_is_on: bool = determine_hp_on_off(
            T_tank_w_C=T_tank_w,
            T_lower=self.T_tank_w_lower_bound,
            T_upper=self.T_tank_w_upper_bound,
            hp_is_on_prev=hp_is_on_prev,
            hour_of_day=ctx.hour_of_day,
            hp_on_schedule=self.hp_on_schedule,
        )

        Q_cond_target: float = (
            self.hp_capacity if hp_is_on else 0.0
        )

        # Mixing valve flows for _calc_state
        den: float = max(
            1e-6, ctx.T_tank_w_K - self.T_sup_w_K,
        )
        alp: float = min(
            1.0,
            max(
                0.0,
                (self.T_mix_w_out_K - self.T_sup_w_K)
                / den,
            ),
        )
        self.dV_mix_w_out = ctx.dV_mix_w_out
        self.dV_tank_w_out = alp * ctx.dV_mix_w_out
        self.dV_mix_sup_w_in = (
            (1 - alp) * ctx.dV_mix_w_out
        )

        if Q_cond_target == 0:
            hp_result: dict = self._calc_state(
                5.0, T_tank_w, 0.0, ctx.T0,
            )
        else:
            opt = self._optimize_operation(
                T_tank_w, Q_cond_target, ctx.T0,
            )
            hp_result = self._calc_state(
                opt.x, T_tank_w, Q_cond_target, ctx.T0,
            )
            if (
                not opt.success
                or hp_result is None
                or hp_result.get('converged') is False
            ):
                self._print_opt_failure(
                    ctx, T_tank_w, Q_cond_target,
                    opt, hp_result,
                )
                raise ValueError(
                    f"Optimization failed at step "
                    f"{ctx.n} "
                    f"(hour={ctx.hour_of_day:.2f}h): "
                    f"T_tank_w={T_tank_w:.1f}°C, "
                    f"T0={ctx.T0:.1f}°C, "
                    f"Q_target={Q_cond_target:.0f}W",
                )

        return (
            hp_is_on,
            hp_result,
            hp_result.get('Q_ref_cond [W]', 0.0),
        )

    def _print_opt_failure(
        self,
        ctx: StepContext,
        T_tank_w: float,
        Q_cond_target: float,
        opt,
        hp_result: dict | None,
    ) -> None:
        """Print detailed optimisation failure diagnostics."""
        print(f"\n{'=' * 70}")
        print(
            f"[HP OPTIMIZATION FAILED] Step n={ctx.n}, "
            f"hour_of_day={ctx.hour_of_day:.2f}h",
        )
        print(f"{'=' * 70}")
        print(f"  Operating conditions:")
        print(f"    T_tank_w     = {T_tank_w:.2f} °C")
        print(f"    T0 (outdoor) = {ctx.T0:.2f} °C")
        print(
            f"    Q_cond_target= {Q_cond_target:.1f} W",
        )
        print(f"  Optimizer result:")
        print(f"    opt.success  = {opt.success}")
        print(f"    opt.x        = {opt.x}")
        dT_cond: float = (
            Q_cond_target / self.UA_cond_design
        )
        print(f"    dT_ref_cond  = {dT_cond:.4f} K")
        print(f"    opt.fun      = {opt.fun:.2f}")
        print(f"    opt.message  = {opt.message}")
        hx_diag: dict = (
            hp_result.get('_hx_diag', {})
            if hp_result else {}
        )
        if hx_diag:
            print(f"  HX bracket failure diagnostics:")
            print(
                f"    Q_ref_target = "
                f"{hx_diag.get('Q_ref_target', np.nan):.1f}"
                f" W",
            )
        print(f"  Suggested fixes:")
        print(f"    ↑ dV_ou_fan_a_design")
        print(f"    ↑ UA_evap_design")
        print(f"    ↓ hp_capacity")
        print(f"{'=' * 70}\n")

    def _assemble_core_results(
        self,
        ctx: StepContext,
        ctrl: ControlState,
        T_solved_K: float,
        level_solved: float,
        ier: int,
    ) -> dict:
        """Build HP-core result dict at solved state.

        Subsystem results are appended separately by
        each subsystem's ``assemble_results()``.
        """
        den: float = max(
            1e-6, T_solved_K - self.T_sup_w_K,
        )
        alp: float = min(
            1.0,
            max(
                0.0,
                (self.T_mix_w_out_K - self.T_sup_w_K)
                / den,
            ),
        )
        dV_tank_w_out: float = alp * ctx.dV_mix_w_out
        dV_tank_w_in: float = (
            dV_tank_w_out
            if ctrl.dV_tank_w_in_ctrl is None
            else ctrl.dV_tank_w_in_ctrl
        )

        self.dV_tank_w_out = dV_tank_w_out
        self.dV_tank_w_in = dV_tank_w_in
        self.dV_mix_w_out = ctx.dV_mix_w_out
        self.dV_mix_sup_w_in = (
            (1 - alp) * ctx.dV_mix_w_out
        )

        T_mix_w_out_val: float = (
            calc_mixing_valve(
                T_solved_K,
                self.T_sup_w_K,
                self.T_mix_w_out_K,
            )['T_mix_w_out']
            if ctx.dV_mix_w_out > 0 else np.nan
        )

        r: dict = {}
        r.update(ctrl.hp_result)
        r.update({
            'hp_is_on': ctrl.hp_is_on,
            'Q_tank_loss [W]': (
                self.UA_tank * (T_solved_K - ctx.T0_K)
            ),
            'T_tank_w [°C]': cu.K2C(T_solved_K),
            'T_mix_w_out [°C]': T_mix_w_out_val,
        })

        if self.lamp_power_watts > 0:
            r['E_uv [W]'] = ctx.E_uv

        if (
            not self.tank_always_full
            or (
                self.tank_always_full
                and self.prevent_simultaneous_flow
            )
        ):
            r['tank_level [-]'] = level_solved

        return r

    # =============================================================
    # Main dynamic simulation
    # =============================================================

    def analyze_dynamic(
        self,
        simulation_period_sec: int,
        dt_s: int,
        T_tank_w_init_C: float,
        dhw_usage_schedule,
        T0_schedule,
        I_DN_schedule=None,
        I_dH_schedule=None,
        tank_level_init: float = 1.0,
        result_save_csv_path: str | None = None,
    ) -> pd.DataFrame:
        """Run a time-stepping dynamic simulation.

        Fully implicit scheme: ``fsolve`` solves for
        ``[T_next, level_next]`` each timestep.

        Parameters
        ----------
        simulation_period_sec : int
            Total simulation duration [s].
        dt_s : int
            Time step size [s].
        T_tank_w_init_C : float
            Initial tank temperature [°C].
        dhw_usage_schedule : array-like or list
            DHW usage schedule.
        T0_schedule : array-like
            Outdoor temperature per step [°C].
        I_DN_schedule : array-like | None
            Direct-normal irradiance per step [W/m²].
        I_dH_schedule : array-like | None
            Diffuse-horizontal irradiance [W/m²].
        tank_level_init : float
            Initial fractional tank level (0–1).
        result_save_csv_path : str | None
            Optional CSV output path.

        Returns
        -------
        pd.DataFrame
            Per-timestep result DataFrame.
        """
        from scipy.optimize import fsolve

        time: np.ndarray = np.arange(
            0, simulation_period_sec, dt_s,
        )
        tN: int = len(time)

        T0_schedule = np.array(T0_schedule)
        if len(T0_schedule) != tN:
            raise ValueError(
                f"T0_schedule length ({len(T0_schedule)})"
                f" != time length ({tN})",
            )
        if (
            I_DN_schedule is not None
            and len(I_DN_schedule) != tN
        ):
            raise ValueError(
                f"I_DN_schedule length "
                f"({len(I_DN_schedule)}) != tN ({tN})",
            )
        if (
            I_dH_schedule is not None
            and len(I_dH_schedule) != tN
        ): 
            raise ValueError(
                f"I_dH_schedule length "
                f"({len(I_dH_schedule)}) != tN ({tN})",
            )

        self.time: np.ndarray = time
        self.dt: int = dt_s

        # DHW schedule handling
        self.w_use_frac = build_dhw_usage_ratio(dhw_usage_schedule, time)

        T_tank_w_K: float = cu.C2K(T_tank_w_init_C)
        tank_level: float = tank_level_init
        is_refilling: bool = False
        hp_is_on_prev: bool = False
        results_data: list[dict] = []

        for n in tqdm(range(tN), desc="ASHPB Simulating"):
            t_s: float = time[n]
            hr: float = t_s * cu.s2h
            hour_of_day: float = (
                (t_s % (24 * cu.h2s)) * cu.s2h
            )

            ctx: StepContext = StepContext(
                n=n,
                current_time_s=t_s,
                current_hour=hr,
                hour_of_day=hour_of_day,
                T0=T0_schedule[n],
                T0_K=cu.C2K(T0_schedule[n]),
                preheat_on=preheat_on,
                T_tank_w_K=T_tank_w_K,
                tank_level=tank_level,
                dV_mix_w_out=(self.w_use_frac[n] * self.dV_mix_w_out_max),
                E_uv=calc_uv_lamp_power(
                    t_s,
                    self.period_3hour_sec,
                    self.num_switching_per_3hour,
                    self.uv_lamp_exposure_duration_sec,
                    self.lamp_power_watts,
                ),
                I_DN=(I_DN_schedule[n] if use_stc else 0.0),
                I_dH=(I_dH_schedule[n] if use_stc else 0.0),
            )

            # --- Phase A: control decisions ---
            hp_is_on, hp_result, Q_ref_cond = (
                self._determine_hp_state(ctx, hp_is_on_prev)
            )
            hp_is_on_prev = hp_is_on

            dV_tank_w_in_ctrl, is_refilling = (
                determine_tank_refill_flow(
                    dt                        = dt_s,
                    tank_level                = ctx.tank_level,
                    dV_tank_w_out             = self.dV_tank_w_out,
                    V_tank_full               = self.V_tank_full,
                    tank_always_full          = self.tank_always_full,
                    prevent_simultaneous_flow = self.prevent_simultaneous_flow,
                    tank_level_lower_bound    = self.tank_level_lower_bound,
                    tank_level_upper_bound    = self.tank_level_upper_bound,
                    dV_tank_w_in_refill       = self.dV_tank_w_in_refill,
                    is_refilling              = is_refilling,
                    use_stc                   = use_stc,
                    stc_placement             = stc_placement,
                    preheat_on                = ctx.preheat_on,
                )
            )

            ctrl: ControlState = ControlState(
                hp_is_on=hp_is_on,
                hp_result=hp_result,
                Q_ref_cond=Q_ref_cond,
                dV_tank_w_in_ctrl=dV_tank_w_in_ctrl,
            )

            # --- Phase A-2: subsystem step ---
            sub_states: dict[str, dict] = {}
            for name, sub in self._subsystems.items():
                sub_states[name] = sub.step(
                    ctx, ctrl, dt_s,
                    self.T_tank_w_in_K,
                )

            # --- Phase B: implicit solve ---
            sol, _info, ier, _msg = fsolve(
                tank_mass_energy_residual,
                [ctx.T_tank_w_K, ctx.tank_level],
                args=(
                    ctx, ctrl, dt_s,
                    self.T_tank_w_in_K,
                    self.T_sup_w_K,
                    self.T_mix_w_out_K,
                    self.C_tank,
                    self.UA_tank,
                    self.V_tank_full,
                    self._subsystems,
                    sub_states,
                ),
                full_output=True,
            )

            T_tank_w_K = sol[0]
            tank_level = max(0.001, min(1.0, sol[1]))

            # --- Phase C: core + subsystem results ---
            r: dict = self._assemble_core_results(
                ctx, ctrl, T_tank_w_K,
                tank_level, ier,
            )
            for name, sub in self._subsystems.items():
                r.update(sub.assemble_results(
                    ctx, ctrl,
                    sub_states[name], T_tank_w_K,
                ))
            results_data.append(r)

        results_df: pd.DataFrame = pd.DataFrame(
            results_data,
        )
        results_df = self.postprocess_exergy(results_df)
        if result_save_csv_path:
            results_df.to_csv(
                result_save_csv_path, index=False,
            )
        return results_df

    # =============================================================
    # Exergy post-processing (ASHP-specific)
    # =============================================================

    def postprocess_exergy(
        self, df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute exergy variables for simulation results.

        Delegates to the standalone ``postprocess_exergy`` in
        ``enex_functions`` with instance-specific parameters.

        Parameters
        ----------
        df : pd.DataFrame
            Result DataFrame from ``analyze_dynamic()``.

        Returns
        -------
        pd.DataFrame
            DataFrame with exergy columns appended.
        """
        from .enex_functions import (
            postprocess_exergy as _postprocess_exergy,
        )
        return _postprocess_exergy(
            df, self.ref, self.C_tank,
            self.dt, self.T_sup_w,
        )
