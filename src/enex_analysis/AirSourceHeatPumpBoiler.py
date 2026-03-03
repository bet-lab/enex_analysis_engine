"""Air source heat pump boiler — physics-based cycle model.

Resolves a vapour-compression refrigerant cycle coupled to an outdoor-air
evaporator with a VSD fan and a lumped-capacitance hot-water tank.
At each time step the model finds the minimum-power operating point
(compressor + fan) via bounded 1-D optimisation (Brent's method) over
the evaporator approach temperature difference.  The condenser approach
temperature is determined analytically from the target heat load.

Optional subsystems include:
- Solar Thermal Collector (STC) with tank-circuit or mains-preheat placement
- UV disinfection lamp with periodic switching
- Tank water-level management
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


@dataclass
class StepContext:
    """Per-timestep immutable context (time, environment, demand)."""
    n: int
    current_time_s: float
    current_hour: float
    hour_of_day: float   # 0 ~ 24, repeats each day (for HP schedule / preheat)
    T0: float
    T0_K: float
    preheat_on: bool
    T_tank_w_K: float
    tank_level: float
    dV_mix_w_out: float
    E_uv: float
    I_DN: float = 0.0
    I_dH: float = 0.0


@dataclass
class ControlState:
    """Control decisions and HP cycle results for one timestep."""
    hp_is_on: bool
    hp_result: dict
    Q_ref_cond: float
    dV_tank_w_in_ctrl: object   # float or None (None = always-full sentinel)
    stc_active: bool
    E_stc_pump: float
    T_tank_w_in_heated_K: float
    stc_result: dict
    T_stc_w_out_K_mp: float


@dataclass
class AirSourceHeatPumpBoiler:
    """Air source heat pump boiler with outdoor-air evaporator and VSD fan.

    The refrigerant cycle is resolved via CoolProp with user-specified
    superheat / subcool margins.  The condenser approach temperature is
    determined analytically (``dT_ref_cond = Q_cond_target / UA_cond``),
    and a bounded 1-D optimiser (Brent's method) minimises total electrical
    input (``E_cmp + E_ou_fan``) over the evaporator approach temperature.
    """
    def __init__(
        self,

        # 1. Refrigerant / cycle / compressor -------------------------
        ref         = 'R134a',
        V_disp_cmp  = 0.0002,            # [m³] Compressor displacement
        eta_cmp_isen = 0.8,              # [-]  Isentropic efficiency

        dT_superheat = 3.0,              # [K] Evaporator outlet superheat
        dT_subcool   = 3.0,              # [K] Condenser outlet subcool

        # 2. Heat exchanger -------------------------------------------
        UA_cond_design = 2000.0,         # [W/K] Condenser design UA
        UA_evap_design = 1000.0,         # [W/K] Evaporator design UA

        # 3. Outdoor unit fan -----------------------------------------
        dV_ou_fan_a_design = 1.5,                 # [m³/s] Design airflow
        dP_ou_fan_design   = 90.0,                # [Pa]   Design static pressure
        A_cross_ou         = 0.25 ** 2 * np.pi,   # [m²] Cross-section area
        eta_ou_fan_design  = 0.6,                 # [-] Fan design efficiency

        # 4. Tank / control / load ------------------------------------
        T_tank_w_upper_bound = 65.0,     # [°C] Tank upper setpoint
        T_tank_w_lower_bound = 60.0,     # [°C] Tank lower setpoint
        T_mix_w_out          = 40.0,     # [°C] Service water delivery temp
        T_tank_w_in          = 15.0,     # [°C] Mains water supply temp

        hp_capacity      = 15000.0,   # [W]     HP rated capacity
        dV_mix_w_out_max = 0.0045,    # [m³/s]  Max service flow rate

        # Tank insulation
        r0      = 0.2,     # [m]     Tank inner radius
        H       = 1.2,     # [m]     Tank height
        x_shell = 0.005,   # [m]     Shell thickness
        x_ins   = 0.05,    # [m]     Insulation thickness
        k_shell = 25,      # [W/m·K] Shell conductivity
        k_ins   = 0.03,    # [W/m·K] Insulation conductivity
        h_o     = 15,      # [W/m²·K] External convective coefficient
        
        # 5. UV lamp --------------------------------------------------
        lamp_power_watts              = 0,   # [W] Lamp power
        uv_lamp_exposure_duration_min = 0,   # [min] UV exposure per cycle
        num_switching_per_3hour       = 1,   # [-] Switching count / 3 h
        
        # 6. Tank water level management ------------------------------
        tank_always_full          = True,
        tank_level_lower_bound    = 0.5,
        tank_level_upper_bound    = 1.0,
        dV_tank_w_in_refill       = 0.001,   # [m³/s] Refill flow rate
        prevent_simultaneous_flow = False,
        
        # 7. HP operating schedule ------------------------------------
        hp_on_schedule = [(0.0, 24.0)],  # (start_h, end_h) active windows
        
        # 8. Solar Thermal Collector (STC) ----------------------------
        A_stc      = 0.0,    # [m²] Collector area (0 = disabled)
        stc_tilt   = 35.0,   # [°] Collector tilt from horizontal
        stc_azimuth= 180.0,  # [°] Collector azimuth (180 = south)
        A_stc_pipe = 2.0,    # [m²] Pipe surface area
        alpha_stc  = 0.95,   # [-] Absorptivity
        h_o_stc    = 15,     # [W/m²·K] External convective coeff
        h_r_stc    = 2,      # [W/m²·K] Radiative coeff
        k_ins_stc  = 0.03,   # [W/m·K]  Insulation conductivity
        x_air_stc  = 0.01,   # [m] Air gap thickness
        x_ins_stc  = 0.05,   # [m] Insulation thickness
        
        # 9. STC pump -------------------------------------------------
        preheat_start_hour = 6,
        preheat_end_hour = 18,
        dV_stc_w = 0.001,                # [m³/s] STC loop flow rate
        E_stc_pump = 50.0,               # [W] STC pump power
        
        # 10. STC placement -------------------------------------------
        stc_placement = 'tank_circuit',  # 'tank_circuit' | 'mains_preheat'
        
        # ASHRAE 90.1-2022 VSD coefficients (p. 325)
        vsd_coeffs_ou = {
            'c1': 0.0013, 'c2': 0.1470, 'c3': 0.9506,
            'c4': -0.0998, 'c5': 0.0,
        },
    
        ):

        # --- 1. Refrigerant / cycle / compressor ---
        self.ref = ref
        self.V_disp_cmp = V_disp_cmp
        self.eta_cmp_isen = eta_cmp_isen
        self.dT_superheat = dT_superheat
        self.dT_subcool = dT_subcool
        self.hp_capacity = hp_capacity

        # --- 2. Heat exchanger UA ---
        self.UA_cond_design = UA_cond_design
        self.UA_evap_design = UA_evap_design

        # --- 3. Outdoor unit fan ---
        self.dV_ou_fan_a_design = dV_ou_fan_a_design
        self.dP_ou_fan_design = dP_ou_fan_design
        self.eta_ou_fan_design = eta_ou_fan_design
        self.A_cross_ou = A_cross_ou
        self.E_ou_fan_design = (self.dV_ou_fan_a_design * self.dP_ou_fan_design) / self.eta_ou_fan_design
        self.vsd_coeffs_ou = vsd_coeffs_ou
        self.fan_params_ou = {
            'fan_design_flow_rate': self.dV_ou_fan_a_design,
            'fan_design_power': self.E_ou_fan_design
        }

        # --- 4. Tank geometry and thermal properties ---
        self.tank_physical = {
            'r0': r0, 'H': H, 'x_shell': x_shell, 'x_ins': x_ins,
            'k_shell': k_shell, 'k_ins': k_ins, 'h_o': h_o,
        }
        self.UA_tank = calc_simple_tank_UA(**self.tank_physical)
        self.V_tank_full = math.pi * r0**2 * H             # [m³]
        self.C_tank = c_w * rho_w * self.V_tank_full       # [J/K]

        self.dV_mix_w_out_max = dV_mix_w_out_max
        self.T_tank_w_upper_bound = T_tank_w_upper_bound
        self.T_tank_w_lower_bound = T_tank_w_lower_bound
        self.T_tank_w_in = T_tank_w_in
        self.T_mix_w_out = T_mix_w_out
        self.T_tank_w_in_K = cu.C2K(T_tank_w_in)
        self.T_mix_w_out_K = cu.C2K(T_mix_w_out)

        # --- 5. UV lamp ---
        self.lamp_power_watts = lamp_power_watts
        self.uv_lamp_exposure_duration_min = uv_lamp_exposure_duration_min
        self.num_switching_per_3hour = num_switching_per_3hour
        self.period_3hour_sec = 3 * cu.h2s
        self.uv_lamp_exposure_duration_sec = uv_lamp_exposure_duration_min * cu.m2s

        # --- 6. Tank water level management ---
        self.tank_always_full = tank_always_full
        self.tank_level_lower_bound = tank_level_lower_bound
        self.tank_level_upper_bound = tank_level_upper_bound
        self.dV_tank_w_in_refill = dV_tank_w_in_refill
        self.prevent_simultaneous_flow = prevent_simultaneous_flow

        # --- 7. HP operating schedule ---
        self.hp_on_schedule = hp_on_schedule

        # --- 8–10. STC + STC pump + placement ---
        self._setup_stc_parameters(
            A_stc, stc_tilt, stc_azimuth,
            A_stc_pipe, alpha_stc, h_o_stc, h_r_stc, k_ins_stc, x_air_stc, x_ins_stc,
            preheat_start_hour, preheat_end_hour, dV_stc_w, E_stc_pump, stc_placement
        )



        # Flow-rate synchronisation variables
        self.dV_tank_w_in = 0.0
        self.dV_tank_w_out = 0.0
        self.dV_mix_w_in_sup = 0.0
        self.dV_mix_w_out = 0.0

    def _calc_state(self, dT_ref_evap, T_tank_w, Q_cond_target, T0):
        """Evaluate refrigerant cycle performance for a given operating point.

        Called repeatedly by the optimiser.  The condenser approach
        temperature is determined analytically as
        ``dT_ref_cond = Q_cond_target / UA_cond``, so only the evaporator
        approach temperature remains as the free variable.

        Parameters
        ----------
        dT_ref_evap : float
            Evaporator approach temperature difference [K].
        T_tank_w : float
            Current tank water temperature [°C].
        Q_cond_target : float
            Target condenser heat rate [W].
        T0 : float
            Dead-state / outdoor-air temperature [°C].

        Returns
        -------
        dict or None
            Cycle performance dictionary.  ``None`` if infeasible.
        """
        dT_ref_cond = Q_cond_target / self.UA_cond_design if Q_cond_target > 0 else 0.0

        T_tank_w_K = cu.C2K(T_tank_w)
        T0_K = cu.C2K(T0)

        T_evap_sat_K = T0_K - dT_ref_evap
        T_cond_sat_K = T_tank_w_K + dT_ref_cond

        is_active = (Q_cond_target > 0.0)

        # Cycle state calculation (superheat / subcool applied)
        cs = calc_ref_state(
            T_evap_K     = T_evap_sat_K,
            T_cond_K     = T_cond_sat_K,
            refrigerant  = self.ref,
            eta_cmp_isen = self.eta_cmp_isen,
            T0_K         = T0_K,
            mode         = 'heating',
            dT_superheat = self.dT_superheat,
            dT_subcool   = self.dT_subcool,
            is_active    = is_active,
        )

        # Refrigerant mass flow and performance
        m_dot_ref  = Q_cond_target / (cs['h2'] - cs['h3']) if is_active else 0.0
        Q_ref_cond = m_dot_ref * (cs['h2'] - cs['h3']) if is_active else 0.0
        Q_ref_evap = m_dot_ref * (cs['h1'] - cs['h4']) if is_active else 0.0
        E_cmp      = m_dot_ref * (cs['h2'] - cs['h1']) if is_active else 0.0
        cmp_rps    = m_dot_ref / (self.V_disp_cmp * cs['rho']) if is_active else 0.0

        # Outdoor unit HX performance
        HX_perf_ou = calc_HX_perf_for_target_heat(
            Q_ref_target  = Q_ref_evap if is_active else 0.0,
            T_ou_a_in_C   = T0,
            T1_star_K     = cs['T1_star_K'],
            T3_star_K     = cs['T3_star_K'],
            A_cross       = self.A_cross_ou,
            UA_design     = self.UA_evap_design,
            dV_fan_design = self.dV_ou_fan_a_design,
            is_active     = is_active,
        )

        if HX_perf_ou.get('converged', True) is False:
            return {'converged': False, '_hx_diag': HX_perf_ou}

        dV_ou_a  = HX_perf_ou['dV_fan']
        v_ou_a   = dV_ou_a / self.A_cross_ou if is_active else 0.0
        UA_evap  = HX_perf_ou['UA']
        T_ou_a_mid = HX_perf_ou['T_ou_a_mid']
        Q_ou_a   = HX_perf_ou['Q_ou_air']

        # Fan power
        E_ou_fan = calc_fan_power_from_dV_fan(
            dV_fan     = dV_ou_a,
            fan_params = self.fan_params_ou,
            vsd_coeffs = self.vsd_coeffs_ou,
            is_active  = is_active,
        )

        T_ou_a_out = T_ou_a_mid + E_ou_fan / (c_a * rho_a * dV_ou_a) if is_active else T0
        eta_ou_fan = self.eta_ou_fan_design * dV_ou_a / E_ou_fan * 100 if is_active else 0.0

        # Condenser heat transfer (simplified: Q = UA * dT)
        UA_cond  = self.UA_cond_design
        Q_cond_w = UA_cond * dT_ref_cond

        # Mixing valve
        dV_tank_w_out   = self.dV_tank_w_out
        dV_tank_w_in    = self.dV_tank_w_in
        dV_mix_w_in_sup = self.dV_mix_w_in_sup
        dV_mix_w_out    = self.dV_mix_w_out

        if dV_mix_w_out == 0:
            T_mix_w_out_val = np.nan
            T_mix_w_out_val_K = np.nan
        else:
            mix = calc_mixing_valve(T_tank_w_K, self.T_tank_w_in_K, self.T_mix_w_out_K)
            T_mix_w_out_val = mix['T_mix_w_out']
            T_mix_w_out_val_K = mix['T_mix_w_out_K']

        Q_tank_w_in  = calc_energy_flow(G=c_w * rho_w * dV_tank_w_in, T=self.T_tank_w_in_K, T0=T0_K)
        Q_tank_w_out = calc_energy_flow(G=c_w * rho_w * dV_tank_w_out, T=T_tank_w_K, T0=T0_K)
        Q_mix_w_in_sup = calc_energy_flow(G=c_w * rho_w * dV_mix_w_in_sup, T=self.T_tank_w_in_K, T0=T0_K)
        Q_mix_w_out  = calc_energy_flow(G=c_w * rho_w * dV_mix_w_out, T=T_mix_w_out_val_K, T0=T0_K)

        result = {
            'hp_is_on': (Q_cond_target > 0),
            'converged': True,

            # === Temperatures [°C] ===
            'T_ref_evap_sat [°C]': cu.K2C(cs['T1_star_K']),
            'T_ref_cond_sat_v [°C]': cu.K2C(cs['T2_star_K']),
            'T_ref_cond_sat_l [°C]': cu.K2C(cs['T3_star_K']),

            'T_ou_a_in [°C]': T0,
            'T_ou_a_out [°C]': T_ou_a_out,
            'T_ref_cmp_in [°C]': cu.K2C(cs['T1_K']),
            'T_ref_cmp_out [°C]': cu.K2C(cs['T2_K']),
            'T_ref_exp_in [°C]': cu.K2C(cs['T3_K']),
            'T_ref_exp_out [°C]': cu.K2C(cs['T4_K']),
            'T_tank_w [°C]': T_tank_w,
            'T_tank_w_in [°C]': self.T_tank_w_in,
            'T_mix_w_out [°C]': T_mix_w_out_val,
            'T0 [°C]': T0,

            # === Volume flow rates [m3/s] ===
            'dV_ou_a [m3/s]': dV_ou_a,
            'v_ou_a [m/s]': v_ou_a,
            'dV_mix_w_out [m3/s]': dV_mix_w_out if dV_mix_w_out > 0 else np.nan,
            'dV_tank_w_in [m3/s]': dV_tank_w_in if dV_tank_w_in > 0 else np.nan,
            'dV_mix_w_in_sup [m3/s]': dV_mix_w_in_sup if dV_mix_w_in_sup > 0 else np.nan,

            # === Pressures [Pa] ===
            'P_ref_cmp_in [Pa]': cs['P1'],
            'P_ref_cmp_out [Pa]': cs['P2'],
            'P_ref_exp_in [Pa]': cs['P3'],
            'P_ref_exp_out [Pa]': cs['P4'],
            'P_ref_evap_sat [Pa]': cs['P1'] if is_active else np.nan,
            'P_ref_cond_sat_v [Pa]': cs['P2'] if is_active else np.nan,
            'P_ref_cond_sat_l [Pa]': cs['P3'] if is_active else np.nan,
            'dP_ou_fan_static [Pa]': self.dP_ou_fan_design - 1/2 * rho_a * v_ou_a**2,
            'dP_ou_fan_dynamic [Pa]': 1/2 * rho_a * v_ou_a**2,

            # === Mass flow rate [kg/s] ===
            'm_dot_ref [kg/s]': m_dot_ref,

            # === Compressor speed [rpm] ===
            'cmp_rpm [rpm]': cmp_rps * 60,

            # === Specific enthalpy [J/kg] ===
            'h_ref_cmp_in [J/kg]': cs['h1'],
            'h_ref_cmp_out [J/kg]': cs['h2'],
            'h_ref_exp_in [J/kg]': cs['h3'],
            'h_ref_exp_out [J/kg]': cs['h4'],
            'h_ref_evap_sat [J/kg]': cs.get('h1_star', np.nan) if is_active else np.nan,
            'h_ref_cond_sat_v [J/kg]': cs.get('h2_star', np.nan) if is_active else np.nan,
            'h_ref_cond_sat_l [J/kg]': cs.get('h3_star', np.nan) if is_active else np.nan,

            # === Energy rates [W] ===
            'E_ou_fan [W]': E_ou_fan,
            'Q_ref_evap [W]': Q_ref_evap,
            'Q_ou_a [W]': Q_ou_a,
            'E_cmp [W]': E_cmp,
            'Q_cond_target [W]': Q_cond_target,
            'Q_ref_cond [W]': Q_ref_cond,
            'Q_cond_w [W]': Q_cond_w,
            'Q_tank_w_in [W]': Q_tank_w_in,
            'Q_tank_w_out [W]': Q_tank_w_out,
            'Q_mix_w_in_sup [W]': Q_mix_w_in_sup,
            'Q_mix_w_out [W]': Q_mix_w_out,
            'E_tot [W]': E_cmp + E_ou_fan,

            # === Efficiency ===
            'eta_ou_fan [%]': eta_ou_fan,
        }

        return result

    def _optimize_operation(self, T_tank_w, Q_cond_target, T0):
        """Find minimum-power operating point via Brent's bounded 1-D method.

        The condenser approach temperature is analytically determined
        (``dT_ref_cond = Q_cond_target / UA_cond``), so only ``dT_ref_evap``
        remains as the optimisation variable.

        Parameters
        ----------
        T_tank_w : float
            Tank water temperature [°C].
        Q_cond_target : float
            Target condenser heat rate [W].
        T0 : float
            Dead-state / outdoor-air temperature [°C].

        Returns
        -------
        scipy.optimize.OptimizeResult
            Contains ``x`` (optimal ``dT_ref_evap``, scalar) and ``success``.
        """

        def _objective(dT_ref_evap):
            """Objective: minimise E_tot = E_cmp + E_ou_fan."""
            try:
                perf = self._calc_state(
                    dT_ref_evap=dT_ref_evap,
                    T_tank_w=T_tank_w,
                    Q_cond_target=Q_cond_target,
                    T0=T0
                )
                if perf is None or not isinstance(perf, dict):
                    return 1e6

                E_tot = perf.get("E_tot [W]", np.nan)
                return E_tot if not np.isnan(E_tot) else 1e6
            except Exception:
                return 1e6

        return minimize_scalar(
            _objective,
            bounds=(5.0, 15.0),
            method='bounded',
            options={'maxiter': 200, 'xatol': 1e-6},
        )

    def analyze_steady(
        self,
        T_tank_w,
        T0,
        dV_mix_w_out=None,
        Q_cond_target=None,
        return_dict=True
        ):
        """Run a steady-state analysis at the given operating point.

        Exactly one of ``dV_mix_w_out`` or ``Q_cond_target`` must be provided.

        Parameters
        ----------
        T_tank_w : float
            Tank water temperature [°C].
        T0 : float
            Dead-state / outdoor-air temperature [°C].
        dV_mix_w_out : float, optional
            Service water flow rate [m³/s].
        Q_cond_target : float, optional
            Target condenser heat rate [W].
        return_dict : bool
            If True return dict; else single-row DataFrame.

        Returns
        -------
        dict or pd.DataFrame
        """
        if dV_mix_w_out is None and Q_cond_target is None:
            raise ValueError("Either dV_mix_w_out or Q_cond_target must be provided.")
        if dV_mix_w_out is not None and Q_cond_target is not None:
            raise ValueError("Cannot provide both dV_mix_w_out and Q_cond_target.")

        T_tank_w_K = cu.C2K(T_tank_w)
        T0_K = cu.C2K(T0)

        if dV_mix_w_out is None:
            dV_mix_w_out = 0.0

        Q_tank_loss = self.UA_tank * (T_tank_w_K - T0_K)
        den = max(1e-6, T_tank_w_K - self.T_tank_w_in_K)
        alp = min(1.0, max(0.0, self.T_mix_w_out_K - self.T_tank_w_in_K) / den)

        self.dV_mix_w_out = dV_mix_w_out
        self.dV_tank_w_out = alp * dV_mix_w_out
        self.dV_mix_w_in_sup = (1 - alp) * dV_mix_w_out

        if Q_cond_target is None:
            Q_tank_w_use = c_w * rho_w * self.dV_tank_w_out * (T_tank_w_K - self.T_tank_w_in_K)
            Q_cond_target = Q_tank_loss + Q_tank_w_use

        if T_tank_w <= self.T_tank_w_lower_bound:
            hp_is_on = True
        elif T_tank_w > self.T_tank_w_upper_bound:
            hp_is_on = False
        else:
            hp_is_on = Q_cond_target > 0

        if Q_cond_target <= 0 or not hp_is_on:
            result = self._calc_state(
                dT_ref_evap=5.0,
                T_tank_w=T_tank_w,
                Q_cond_target=0.0,
                T0=T0
            )
        else:
            opt_result = self._optimize_operation(
                T_tank_w=T_tank_w,
                Q_cond_target=Q_cond_target,
                T0=T0
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

            if result is None or not isinstance(result, dict):
                try:
                    result = self._calc_state(
                        dT_ref_evap=5.0,
                        T_tank_w=T_tank_w,
                        Q_cond_target=0.0,
                        T0=T0
                    )
                except Exception:
                    result = {
                        'hp_is_on': False,
                        'converged': False,
                        'Q_cond_target [W]': Q_cond_target,
                        'Q_ref_cond [W]': 0.0,
                        'Q_ref_evap [W]': 0.0,
                        'E_cmp [W]': 0.0,
                        'E_ou_fan [W]': 0.0,
                        'E_tot [W]': 0.0,
                        'T_tank_w [°C]': T_tank_w,
                        'T0 [°C]': T0
                    }

            if result is not None and isinstance(result, dict):
                if 'opt_result' in locals() and hasattr(opt_result, 'success'):
                    result['converged'] = opt_result.success
                    if result['converged'] is False:
                        print(f"Optimization failed")

        if return_dict:
            return result
        else:
            return pd.DataFrame([result])

    # =================================================================
    # Private helpers for analyze_dynamic
    # =================================================================

    def _determine_hp_state(self, ctx, hp_is_on_prev):
        """Determine HP on/off via hysteresis and run cycle optimisation."""
        T_tank_w = cu.K2C(ctx.T_tank_w_K)
        if T_tank_w <= self.T_tank_w_lower_bound:
            hp_is_on = True
        elif T_tank_w >= self.T_tank_w_upper_bound:
            hp_is_on = False
        else:
            hp_is_on = hp_is_on_prev
        hp_is_on = hp_is_on and check_hp_schedule_active(ctx.hour_of_day, self.hp_on_schedule)

        Q_cond_target = self.hp_capacity if hp_is_on else 0.0

        # Set mixing valve flows for _calc_state
        den = max(1e-6, ctx.T_tank_w_K - self.T_tank_w_in_K)
        alp = min(1.0, max(0.0, (self.T_mix_w_out_K - self.T_tank_w_in_K) / den))
        self.dV_mix_w_out = ctx.dV_mix_w_out
        self.dV_tank_w_out = alp * ctx.dV_mix_w_out
        self.dV_mix_w_in_sup = (1 - alp) * ctx.dV_mix_w_out

        if Q_cond_target == 0:
            hp_result = self._calc_state(5.0, T_tank_w, 0.0, ctx.T0)
        else:
            opt = self._optimize_operation(T_tank_w, Q_cond_target, ctx.T0)
            hp_result = self._calc_state(opt.x, T_tank_w, Q_cond_target, ctx.T0)
            if not opt.success or hp_result is None or hp_result.get('converged') is False:
                print(f"\n{'='*70}")
                print(f"[HP OPTIMIZATION FAILED] Step n={ctx.n}, hour_of_day={ctx.hour_of_day:.2f}h")
                print(f"{'='*70}")
                print(f"  Operating conditions:")
                print(f"    T_tank_w     = {T_tank_w:.2f} °C")
                print(f"    T0 (outdoor) = {ctx.T0:.2f} °C")
                print(f"    Q_cond_target= {Q_cond_target:.1f} W  (hp_capacity)")
                print(f"  Optimizer result:")
                print(f"    opt.success  = {opt.success}")
                print(f"    opt.x        = {opt.x}  [dT_ref_evap]")
                print(f"    dT_ref_cond  = {Q_cond_target / self.UA_cond_design:.4f} K  (analytically determined)")
                print(f"    opt.fun      = {opt.fun:.2f}")
                print(f"    opt.message  = {opt.message}")
                print(f"  Current design parameters:")
                print(f"    hp_capacity        = {self.hp_capacity:.0f} W")
                print(f"    dV_ou_fan_a_design = {self.dV_ou_fan_a_design:.4f} m³/s")
                print(f"    UA_evap_design     = {self.UA_evap_design:.1f} W/K")
                print(f"    UA_cond_design     = {self.UA_cond_design:.1f} W/K")
                print(f"    A_cross_ou         = {self.A_cross_ou:.4f} m²")
                # HX bracket failure diagnostics
                hx_diag = hp_result.get('_hx_diag', {}) if hp_result else {}
                if hx_diag:
                    print(f"  HX bracket failure diagnostics:")
                    print(f"    Q_ref_target  = {hx_diag.get('Q_ref_target', np.nan):.1f} W")
                    print(f"    Q @ dV_min    = {hx_diag.get('Q_at_dV_min', np.nan):.1f} W  (dV={hx_diag.get('dV_min', np.nan):.4f} m³/s)")
                    print(f"    Q @ dV_max    = {hx_diag.get('Q_at_dV_max', np.nan):.1f} W  (dV={hx_diag.get('dV_max', np.nan):.4f} m³/s)")
                    print(f"    → {hx_diag.get('hint', '')}")
                elif hp_result is None:
                    print(f"  _calc_state returned None")
                print(f"  Suggested fixes:")
                print(f"    ↑ dV_ou_fan_a_design (increase outdoor fan airflow)")
                print(f"    ↑ UA_evap_design     (increase evaporator heat transfer)")
                print(f"    ↓ hp_capacity        (reduce target heating capacity)")
                print(f"{'='*70}\n")
                raise ValueError(
                    f"Optimization failed at step {ctx.n} (hour_of_day={ctx.hour_of_day:.2f}h): "
                    f"T_tank_w={T_tank_w:.1f}°C, T0={ctx.T0:.1f}°C, Q_target={Q_cond_target:.0f}W"
                )

        return hp_is_on, hp_result, hp_result.get('Q_ref_cond [W]', 0.0)

    def _determine_refill_flow(self, ctx, is_refilling, use_stc):
        """Determine refill flow rate from current level and control mode.

        Returns ``(dV_tank_w_in_ctrl, is_refilling)``.
        ``dV_tank_w_in_ctrl`` is ``None`` for tank_always_full (no PSF),
        signalling that inflow = outflow (resolved inside residual).
        """
        dt = self.dt
        lv = ctx.tank_level
        if not self.tank_always_full or (self.tank_always_full and self.prevent_simultaneous_flow):
            lv = max(0.0, ctx.tank_level - (self.dV_tank_w_out * dt) / self.V_tank_full)

        dV_tank_w_in = 0.0
        if self.tank_always_full and self.prevent_simultaneous_flow:
            if self.dV_tank_w_out > 0:
                is_refilling = False
            elif lv < 1.0:
                req = (1.0 - lv) * self.V_tank_full
                if self.dV_tank_w_in_refill * dt <= req:
                    dV_tank_w_in = self.dV_tank_w_in_refill
        elif self.tank_always_full:
            return None, is_refilling   # sentinel
        else:
            lo, hi = self.tank_level_lower_bound, self.tank_level_upper_bound
            if use_stc and self.stc_placement == 'mains_preheat' and ctx.preheat_on:
                lo, hi = 1.0, 1.0
            if not is_refilling and lv < lo - 1e-6:
                is_refilling = True
            if is_refilling:
                req = (hi - lv) * self.V_tank_full
                if self.dV_tank_w_in_refill * dt <= req:
                    dV_tank_w_in = self.dV_tank_w_in_refill
                if (lv + dV_tank_w_in * dt / self.V_tank_full) >= hi - 1e-6:
                    is_refilling = False

        return dV_tank_w_in, is_refilling

    def _tank_mass_energy_residual(self, x, ctx, ctrl, dt, stc_kwargs, use_stc):
        """Energy and mass balance residuals evaluated at T^{n+1}.

        The 3-way mixing valve ratio  α(T) = (T_mix - T_in) / (T - T_in)
        makes the outflow a nonlinear function of T^{n+1}, requiring an
        iterative solver.  If α were frozen at time n the residual becomes
        linear in T and admits a direct algebraic solution.
        """
        T_next, level_next = x

        den = max(1e-6, T_next - self.T_tank_w_in_K)
        alp = min(1.0, max(0.0, (self.T_mix_w_out_K - self.T_tank_w_in_K) / den))
        dV_tank_w_out = alp * ctx.dV_mix_w_out
        dV_tank_w_in = dV_tank_w_out if ctrl.dV_tank_w_in_ctrl is None else ctrl.dV_tank_w_in_ctrl

        r_mass = level_next - ctx.tank_level - (dV_tank_w_in - dV_tank_w_out) * dt / self.V_tank_full

        C_curr = self.C_tank * max(0.001, ctx.tank_level)
        C_next = self.C_tank * max(0.001, level_next)
        Q_loss = self.UA_tank * (T_next - ctx.T0_K)

        # Open-system enthalpy flow: Q_flow_net = G_in * T_in - G_out * T_out
        # When dV_tank_w_in = dV_tank_w_out (constant mass): reduces to G*(T_tank_w_in_heated - T_next)
        # When dV_tank_w_in ≠ dV_tank_w_out (variable mass): properly accounts for outflow enthalpy
        Q_flow_net = c_w * rho_w * (dV_tank_w_in * ctrl.T_tank_w_in_heated_K - dV_tank_w_out * T_next)

        Q_stc_net = 0.0
        if use_stc and self.stc_placement == 'tank_circuit' and ctrl.stc_active:
            stc_r = calc_stc_performance(
                I_DN_stc=ctx.I_DN, I_dH_stc=ctx.I_dH,
                T_stc_w_in_K=T_next, T0_K=ctx.T0_K,
                dV_stc=self.dV_stc_w, is_active=True, **stc_kwargs)
            Q_stc_net = stc_r.get('Q_stc_w_out', 0.0) - stc_r.get('Q_stc_w_in', 0.0)

        Q_total = ctrl.Q_ref_cond + ctx.E_uv + ctrl.E_stc_pump + Q_stc_net + Q_flow_net
        r_energy = C_next * T_next - C_curr * ctx.T_tank_w_K - dt * (Q_total - Q_loss)

        return [r_energy, r_mass]

    def _assemble_step_results(self, ctx, ctrl, T_solved_K, level_solved,
                               ier, stc_kwargs, use_stc):
        """Recompute reporting quantities at solved state and assemble dict."""
        den = max(1e-6, T_solved_K - self.T_tank_w_in_K)
        alp = min(1.0, max(0.0, (self.T_mix_w_out_K - self.T_tank_w_in_K) / den))
        dV_tank_w_out = alp * ctx.dV_mix_w_out
        dV_tank_w_in = dV_tank_w_out if ctrl.dV_tank_w_in_ctrl is None else ctrl.dV_tank_w_in_ctrl

        self.dV_tank_w_out = dV_tank_w_out
        self.dV_tank_w_in = dV_tank_w_in
        self.dV_mix_w_out = ctx.dV_mix_w_out
        self.dV_mix_w_in_sup = (1 - alp) * ctx.dV_mix_w_out

        T_mix_w_out_val = (calc_mixing_valve(T_solved_K, self.T_tank_w_in_K, self.T_mix_w_out_K)['T_mix_w_out']
                  if ctx.dV_mix_w_out > 0 else np.nan)

        # STC at solved T for reporting
        Q_stc_w_out, Q_stc_w_in = 0.0, 0.0
        T_stc_w_out_K, T_stc_w_final_K = np.nan, np.nan
        stc_result = ctrl.stc_result

        if use_stc and self.stc_placement == 'tank_circuit':
            stc_result = calc_stc_performance(
                I_DN_stc=ctx.I_DN, I_dH_stc=ctx.I_dH,
                T_stc_w_in_K=T_solved_K, T0_K=ctx.T0_K,
                dV_stc=self.dV_stc_w, is_active=ctrl.stc_active, **stc_kwargs)
            T_stc_w_out_K = stc_result['T_stc_w_out_K']
            T_stc_w_final_K = stc_result.get('T_stc_w_final_K', T_stc_w_out_K)
            Q_stc_w_out = stc_result.get('Q_stc_w_out', 0.0)
            Q_stc_w_in = stc_result.get('Q_stc_w_in', 0.0)
        elif use_stc and self.stc_placement == 'mains_preheat':
            T_stc_w_out_K = ctrl.T_stc_w_out_K_mp
            Q_stc_w_out = stc_result.get('Q_stc_w_out', 0.0)
            Q_stc_w_in = stc_result.get('Q_stc_w_in', 0.0)

        r = {}
        r.update(ctrl.hp_result)
        r.update({
            'hp_is_on': ctrl.hp_is_on,
            'Q_tank_loss [W]': self.UA_tank * (T_solved_K - ctx.T0_K),
            'T_tank_w [°C]': cu.K2C(T_solved_K),
            'T_mix_w_out [°C]': T_mix_w_out_val,
            'implicit_converged': (ier == 1),
        })
        if self.lamp_power_watts > 0:
            r['E_uv [W]'] = ctx.E_uv
        if not self.tank_always_full or (self.tank_always_full and self.prevent_simultaneous_flow):
            r['tank_level [-]'] = level_solved
        if use_stc:
            r.update({
                'stc_active [-]': ctrl.stc_active,
                'I_DN_stc [W/m2]': ctx.I_DN, 'I_dH_stc [W/m2]': ctx.I_dH,
                'I_sol_stc [W/m2]': stc_result.get('I_sol_stc', np.nan),
                'Q_sol_stc [W]': stc_result.get('Q_sol_stc', np.nan),
                'Q_stc_w_out [W]': Q_stc_w_out, 'Q_stc_w_in [W]': Q_stc_w_in,
                'Q_l_stc [W]': stc_result.get('Q_l_stc', np.nan),
                'T_stc_w_out [°C]': cu.K2C(T_stc_w_out_K) if not np.isnan(T_stc_w_out_K) else np.nan,
                'T_stc_w_in [°C]': cu.K2C(T_solved_K),
                'T_stc [°C]': cu.K2C(stc_result.get('T_stc_K', np.nan)),
                'E_stc_pump [W]': ctrl.E_stc_pump,
            })
            if self.stc_placement == 'tank_circuit':
                r['T_stc_w_final [°C]'] = cu.K2C(T_stc_w_final_K) if not np.isnan(T_stc_w_final_K) else np.nan
        else:
            r['stc_active [-]'] = False
        return r

    # =================================================================
    # Main dynamic simulation
    # =================================================================

    def analyze_dynamic(
        self, simulation_period_sec, dt_s, T_tank_w_init_C,
        schedule_entries, T0_schedule,
        I_DN_schedule=None, I_dH_schedule=None,
        tank_level_init=1.0, result_save_csv_path=None):
        """Run a time-stepping dynamic simulation (fully implicit scheme).

        At each timestep ``fsolve`` solves for ``[T_next, level_next]``
        such that the coupled energy / mass balance residuals vanish.
        """
        from scipy.optimize import fsolve

        time = np.arange(0, simulation_period_sec, dt_s)
        tN = len(time)

        # Convert schedules to arrays for length checks and further use
        T0_schedule = np.array(T0_schedule)

        # Check lengths of array inputs
        if len(T0_schedule) != tN: raise ValueError(f"T0_schedule length ({len(T0_schedule)}) != time length ({tN})")
        if I_DN_schedule is not None and len(I_DN_schedule) != tN: raise ValueError(f"I_DN_schedule length ({len(I_DN_schedule)}) != time length ({tN})")
        if I_dH_schedule is not None and len(I_dH_schedule) != tN: raise ValueError(f"I_dH_schedule length ({len(I_dH_schedule)}) != time length ({tN})")

        use_stc = (self.A_stc > 0) and (I_DN_schedule is not None) and (I_dH_schedule is not None)
        self.time, self.dt = time, dt_s

        # schedule_entries: accept pre-computed numpy array (O(N)) or
        #   list of (start, end, frac) interval tuples (converted via build_schedule_ratios)
        if isinstance(schedule_entries, np.ndarray) and schedule_entries.ndim == 1:
            if len(schedule_entries) != tN:
                raise ValueError(f"schedule_entries array length ({len(schedule_entries)}) != time length ({tN})")
            self.w_use_frac = schedule_entries.astype(float)
        else:
            self.w_use_frac = build_schedule_ratios(schedule_entries, time)
        stc_kwargs = dict(
            A_stc_pipe=self.A_stc_pipe, alpha_stc=self.alpha_stc,
            h_o_stc=self.h_o_stc, h_r_stc=self.h_r_stc,
            k_ins_stc=self.k_ins_stc, x_air_stc=self.x_air_stc,
            x_ins_stc=self.x_ins_stc, E_pump=self.E_stc_pump,
        )

        _STC_OFF = {'stc_active': False, 'stc_result': {},
                    'T_stc_w_out_K': np.nan, 'T_stc_w_final_K': np.nan,
                    'Q_stc_w_out': 0.0, 'Q_stc_w_in': 0.0, 'E_stc_pump': 0.0}

        T_tank_w_K = cu.C2K(T_tank_w_init_C)
        tank_level = tank_level_init
        is_refilling = False
        hp_is_on_prev = False
        results_data = []

        for n in tqdm(range(tN), desc="ASHPB Simulating"):
            # --- Build context ---
            t_s = time[n]
            hr = t_s * cu.s2h
            hour_of_day = (t_s % (24 * cu.h2s)) * cu.s2h
            ctx = StepContext(
                n=n, current_time_s=t_s, current_hour=hr, hour_of_day=hour_of_day,
                T0=T0_schedule[n], T0_K=cu.C2K(T0_schedule[n]),
                preheat_on=(hour_of_day >= self.preheat_start_hour and hour_of_day < self.preheat_end_hour),
                T_tank_w_K=T_tank_w_K, tank_level=tank_level,
                dV_mix_w_out=self.w_use_frac[n] * self.dV_mix_w_out_max,
                E_uv=calc_uv_lamp_power(t_s, self.period_3hour_sec,
                    self.num_switching_per_3hour,
                    self.uv_lamp_exposure_duration_sec, self.lamp_power_watts),
                I_DN=I_DN_schedule[n] if use_stc else 0.0,
                I_dH=I_dH_schedule[n] if use_stc else 0.0,
            )

            # --- Phase A: control decisions ---
            hp_is_on, hp_result, Q_ref_cond = self._determine_hp_state(ctx, hp_is_on_prev)
            hp_is_on_prev = hp_is_on

            dV_tank_w_in_ctrl, is_refilling = self._determine_refill_flow(ctx, is_refilling, use_stc)

            if use_stc:
                dV_stc_w_feed = dV_tank_w_in_ctrl if dV_tank_w_in_ctrl is not None else self.dV_tank_w_out
                stc_state = self._calculate_stc_dynamic(
                    I_DN=ctx.I_DN, I_dH=ctx.I_dH,
                    T_tank_w_K=ctx.T_tank_w_K, T0_K=ctx.T0_K,
                    preheat_on=ctx.preheat_on, dV_tank_w_in=dV_stc_w_feed)
            else:
                stc_state = _STC_OFF

            T_tank_w_in_heated_K = (stc_state['T_stc_w_out_K']
                          if use_stc and self.stc_placement == 'mains_preheat' and stc_state['stc_active']
                          else self.T_tank_w_in_K)

            ctrl = ControlState(
                hp_is_on=hp_is_on, hp_result=hp_result, Q_ref_cond=Q_ref_cond,
                dV_tank_w_in_ctrl=dV_tank_w_in_ctrl,
                stc_active=stc_state['stc_active'], E_stc_pump=stc_state['E_stc_pump'],
                T_tank_w_in_heated_K=T_tank_w_in_heated_K, stc_result=stc_state['stc_result'],
                T_stc_w_out_K_mp=stc_state.get('T_stc_w_out_K', self.T_tank_w_in_K))

            # --- Phase B: implicit solve ---
            sol, _info, ier, _msg = fsolve(
                self._tank_mass_energy_residual,
                [ctx.T_tank_w_K, ctx.tank_level],
                args=(ctx, ctrl, dt_s, stc_kwargs, use_stc),
                full_output=True)

            T_tank_w_K = sol[0]
            tank_level = max(0.001, min(1.0, sol[1]))

            # --- Phase C: results ---
            results_data.append(
                self._assemble_step_results(ctx, ctrl, T_tank_w_K, tank_level,
                                            ier, stc_kwargs, use_stc))

        results_df = pd.DataFrame(results_data)
        results_df = self.postprocess_exergy(results_df)
        if result_save_csv_path:
            results_df.to_csv(result_save_csv_path, index=False)
        return results_df

    def postprocess_exergy(self, df: pd.DataFrame) -> pd.DataFrame:
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
        from .enex_functions import postprocess_exergy as _postprocess_exergy
        return _postprocess_exergy(df, self.ref, self.C_tank, self.dt, self.T_tank_w_in)

    def _setup_stc_parameters(
        self, A_stc, stc_tilt, stc_azimuth,
        A_stc_pipe, alpha_stc, h_o_stc, h_r_stc, k_ins_stc, x_air_stc, x_ins_stc,
        preheat_start_hour, preheat_end_hour, dV_stc_w, E_stc_pump, stc_placement
    ):
        """Initialise Solar Thermal Collector (STC) parameters.

        Separated from ``__init__`` to reduce constructor complexity.
        """
        self.A_stc = A_stc
        self.stc_tilt = stc_tilt
        self.stc_azimuth = stc_azimuth
        self.A_stc_pipe = A_stc_pipe
        self.alpha_stc = alpha_stc
        self.h_o_stc = h_o_stc
        self.h_r_stc = h_r_stc
        self.k_ins_stc = k_ins_stc
        self.x_air_stc = x_air_stc
        self.x_ins_stc = x_ins_stc

        self.preheat_start_hour = preheat_start_hour
        self.preheat_end_hour = preheat_end_hour
        self.dV_stc_w = dV_stc_w
        self.E_stc_pump = E_stc_pump

        self.stc_placement = stc_placement

    def _calculate_stc_dynamic(
        self, 
        I_DN, I_dH, 
        T_tank_w_K, T0_K, 
        preheat_on, 
        dV_tank_w_in
    ):
        """Compute STC performance and state for a single timestep.

        Handles both ``tank_circuit`` and ``mains_preheat`` placement
        modes, including probing calculations for activation decisions.

        Parameters
        ----------
        I_DN : float
            Direct-normal irradiance [W/m²].
        I_dH : float
            Diffuse-horizontal irradiance [W/m²].
        T_tank_w_K : float
            Current tank water temperature [K].
        T0_K : float
            Dead-state temperature [K].
        preheat_on : bool
            Whether the preheat window is active.
        dV_tank_w_in : float
            Current refill flow rate [m³/s].

        Returns
        -------
        dict
            Keys: ``stc_active``, ``stc_result``, ``T_stc_w_out_K``,
            ``T_stc_w_final_K``, ``Q_stc_w_out``, ``Q_stc_w_in``,
            ``E_stc_pump``.
        """
        stc_active = False
        stc_result = {}

        T_stc_w_out_K = np.nan
        T_stc_w_final_K = np.nan
        Q_stc_w_out = 0.0
        Q_stc_w_in = 0.0
        E_stc_pump_val = 0.0

        stc_kwargs = dict(
            A_stc_pipe=self.A_stc_pipe, alpha_stc=self.alpha_stc,
            h_o_stc=self.h_o_stc, h_r_stc=self.h_r_stc,
            k_ins_stc=self.k_ins_stc, x_air_stc=self.x_air_stc,
            x_ins_stc=self.x_ins_stc,
            E_pump=self.E_stc_pump,
        )

        if self.stc_placement == 'tank_circuit':
            # Probing
            stc_result_test = calc_stc_performance(
                I_DN_stc=I_DN, I_dH_stc=I_dH,
                T_stc_w_in_K=T_tank_w_K, T0_K=T0_K,
                dV_stc=self.dV_stc_w,
                is_active=True,
                **stc_kwargs,
            )

            stc_active = preheat_on and stc_result_test['T_stc_w_out_K'] > T_tank_w_K

            if stc_active:
                stc_result = stc_result_test
                E_stc_pump_val = self.E_stc_pump
            else:
                stc_result = calc_stc_performance(
                    I_DN_stc=I_DN, I_dH_stc=I_dH,
                    T_stc_w_in_K=T_tank_w_K, T0_K=T0_K,
                    dV_stc=self.dV_stc_w,
                    is_active=False,
                    **stc_kwargs,
                )
                E_stc_pump_val = 0.0

            T_stc_w_out_K = stc_result['T_stc_w_out_K']
            T_stc_w_final_K = stc_result.get('T_stc_w_final_K', T_stc_w_out_K)
            Q_stc_w_out   = stc_result.get('Q_stc_w_out', 0.0)
            Q_stc_w_in    = stc_result.get('Q_stc_w_in', 0.0)

        elif self.stc_placement == 'mains_preheat':
            if preheat_on and dV_tank_w_in > 0:
                stc_result_test = calc_stc_performance(
                    I_DN_stc=I_DN, I_dH_stc=I_dH,
                    T_stc_w_in_K=self.T_tank_w_in_K, T0_K=T0_K,
                    dV_stc=dV_tank_w_in,
                    is_active=True,
                    **stc_kwargs,
                )

                if stc_result_test['T_stc_w_out_K'] > self.T_tank_w_in_K:
                    stc_active = True
                    stc_result = stc_result_test
                else:
                    stc_active = False
                    stc_result = calc_stc_performance(
                        I_DN_stc=I_DN, I_dH_stc=I_dH,
                        T_stc_w_in_K=self.T_tank_w_in_K, T0_K=T0_K,
                        dV_stc=dV_tank_w_in,
                        is_active=False,
                        **stc_kwargs,
                    )
            else:
                stc_active = False
                stc_result = calc_stc_performance(
                    I_DN_stc=I_DN, I_dH_stc=I_dH,
                    T_stc_w_in_K=self.T_tank_w_in_K, T0_K=T0_K,
                    dV_stc=1.0,
                    is_active=False,
                    **stc_kwargs,
                )

            T_stc_w_out_K = stc_result['T_stc_w_out_K']
            Q_stc_w_out = stc_result.get('Q_stc_w_out', 0.0)
            Q_stc_w_in = stc_result.get('Q_stc_w_in', 0.0)

            if stc_active:
                E_stc_pump_val = self.E_stc_pump
            else:
                E_stc_pump_val = 0.0

        return {
            'stc_active': stc_active,
            'stc_result': stc_result,
            'T_stc_w_out_K': T_stc_w_out_K,
            'T_stc_w_final_K': T_stc_w_final_K,
            'Q_stc_w_out': Q_stc_w_out,
            'Q_stc_w_in': Q_stc_w_in,
            'E_stc_pump': E_stc_pump_val
        }
