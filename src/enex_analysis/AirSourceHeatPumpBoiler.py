"""Air source heat pump boiler — physics-based cycle model.

Resolves a vapour-compression refrigerant cycle coupled to an outdoor-air
evaporator with a VSD fan and a lumped-capacitance hot-water tank.
At each time step the model finds the minimum-power operating point
(compressor + fan) via Differential Evolution optimisation over two approach
temperature differences.

Optional subsystems include:
- Solar Thermal Collector (STC) with tank-circuit or mains-preheat placement
- UV disinfection lamp with periodic switching
- Tank water-level management
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from tqdm import tqdm
import CoolProp.CoolProp as CP

from . import calc_util as cu
from .constants import *
from .enex_functions import *


@dataclass
class AirSourceHeatPumpBoiler:
    """Air source heat pump boiler with outdoor-air evaporator and VSD fan.

    The refrigerant cycle is resolved via CoolProp with user-specified
    superheat / subcool margins.  A Differential Evolution optimiser
    minimises total electrical input (``E_cmp + E_ou_fan``) subject to
    condenser and evaporator heat-transfer constraints.
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
            A_stc, A_stc_pipe, alpha_stc, h_o_stc, h_r_stc, k_ins_stc, x_air_stc, x_ins_stc,
            preheat_start_hour, preheat_end_hour, dV_stc_w, E_stc_pump, stc_placement
        )

        # Warm-start: reuse previous optimisation result
        self.prev_opt_x = None

        # Flow-rate synchronisation variables
        self.dV_tank_w_in = 0.0
        self.dV_tank_w_out = 0.0
        self.dV_mix_w_in_sup = 0.0
        self.dV_mix_w_out = 0.0

    def _calc_state(self, optimization_vars, T_tank_w, Q_cond_load, T0):
        """Evaluate refrigerant cycle performance for a given operating point.

        Called repeatedly by the optimiser.  The method resolves the full
        cycle (States 1–4 with superheat/subcool), computes required refrigerant
        flow for the target ``Q_cond_load``, evaluates outdoor HX airflow and
        fan power, and assembles a result dictionary.

        Parameters
        ----------
        optimization_vars : list of float
            ``[dT_ref_cond, dT_ref_evap]`` — condenser and evaporator
            approach temperature differences [K].
        T_tank_w : float
            Current tank water temperature [°C].
        Q_cond_load : float
            Target condenser heat rate [W].
        T0 : float
            Dead-state / outdoor-air temperature [°C].

        Returns
        -------
        dict or None
            Cycle performance dictionary.  ``None`` if infeasible.
        """
        dT_ref_cond = optimization_vars[0]
        dT_ref_evap = optimization_vars[1]

        T_tank_w_K = cu.C2K(T_tank_w)
        T0_K = cu.C2K(T0)

        T_evap_sat_K = T0_K - dT_ref_evap
        T_cond_sat_K = T_tank_w_K + dT_ref_cond

        is_active = (Q_cond_load > 0.0)

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
        m_dot_ref  = Q_cond_load / (cs['h2'] - cs['h3'])
        Q_ref_cond = m_dot_ref * (cs['h2'] - cs['h3'])
        Q_ref_evap = m_dot_ref * (cs['h1'] - cs['h4'])
        E_cmp      = m_dot_ref * (cs['h2'] - cs['h1'])
        cmp_rps    = m_dot_ref / (self.V_disp_cmp * cs['rho'])

        # Outdoor unit HX performance
        HX_perf_ou = calc_HX_perf_for_target_heat(
            Q_ref_target  = Q_ref_evap if is_active else 0.0,
            T_a_ou_in_C   = T0,
            T1_star_K     = cs['T1_star_K'],
            T3_star_K     = cs['T3_star_K'],
            A_cross       = self.A_cross_ou,
            UA_design     = self.UA_evap_design,
            dV_fan_design = self.dV_ou_fan_a_design,
            is_active     = is_active,
        )

        if HX_perf_ou.get('converged', True) is False:
            return None

        dV_ou_a_fan = HX_perf_ou['dV_fan']
        v_ou_a_fan  = dV_ou_a_fan / self.A_cross_ou
        UA_evap     = HX_perf_ou['UA']
        T_a_ou_mid  = HX_perf_ou['T_a_ou_mid']
        Q_ou_a      = HX_perf_ou['Q_ou_air']

        # Fan power
        E_ou_fan = calc_fan_power_from_dV_fan(
            dV_fan     = dV_ou_a_fan,
            fan_params = self.fan_params_ou,
            vsd_coeffs = self.vsd_coeffs_ou,
            is_active  = is_active,
        )

        T_a_ou_out = T_a_ou_mid + E_ou_fan / (c_a * rho_a * dV_ou_a_fan)
        fan_eff = self.eta_ou_fan_design * dV_ou_a_fan / E_ou_fan * 100

        # Condenser heat transfer (simplified: Q = UA * dT)
        UA_cond  = self.UA_cond_design
        Q_tank_w = UA_cond * dT_ref_cond

        # Mixing valve
        dV_tank_w_out   = self.dV_tank_w_out
        dV_tank_w_in    = self.dV_tank_w_in
        dV_mix_w_in_sup = self.dV_mix_w_in_sup
        dV_mix_w_out    = self.dV_mix_w_out

        if dV_mix_w_out == 0:
            T_serv_w_actual = np.nan
            T_serv_w_actual_K = np.nan
        else:
            mix = calc_mixing_valve(T_tank_w_K, self.T_tank_w_in_K, self.T_mix_w_out_K)
            T_serv_w_actual = mix['T_serv_w_actual']
            T_serv_w_actual_K = mix['T_serv_w_actual_K']

        Q_tank_w_in  = calc_energy_flow(G=c_w * rho_w * dV_tank_w_in, T=self.T_tank_w_in_K, T0=T0_K)
        Q_tank_w_out = calc_energy_flow(G=c_w * rho_w * dV_tank_w_out, T=T_tank_w_K, T0=T0_K)
        Q_mix_w_in_sup = calc_energy_flow(G=c_w * rho_w * dV_mix_w_in_sup, T=self.T_tank_w_in_K, T0=T0_K)
        Q_mix_w_out  = calc_energy_flow(G=c_w * rho_w * dV_mix_w_out, T=T_serv_w_actual_K, T0=T0_K)

        result = {
            'hp_is_on': (Q_cond_load > 0),
            'converged': True,

            # === Temperatures [°C] ===
            'T_ref_evap_sat [°C]': cu.K2C(cs['T1_star_K']),
            'T_ref_cond_sat_v [°C]': cu.K2C(cs['T2_star_K']),
            'T_ref_cond_sat_l [°C]': cu.K2C(cs['T3_star_K']),

            'T_a_ou_in [°C]': T0,
            'T_a_ou_out [°C]': T_a_ou_out,
            'T_ref_cmp_in [°C]': cu.K2C(cs['T1_K']),
            'T_ref_cmp_out [°C]': cu.K2C(cs['T2_K']),
            'T_ref_exp_in [°C]': cu.K2C(cs['T3_K']),
            'T_ref_exp_out [°C]': cu.K2C(cs['T4_K']),
            'T_tank_w [°C]': T_tank_w,
            'T_tank_w_in [°C]': self.T_tank_w_in,
            'T_mix_w_out [°C]': T_serv_w_actual,
            'T0 [°C]': T0,

            # === Volume flow rates [m3/s] ===
            'dV_ou_a_fan [m3/s]': dV_ou_a_fan,
            'v_ou_a_fan [m/s]': v_ou_a_fan,
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
            'dP_ou_fan_static [Pa]': self.dP_ou_fan_design - 1/2 * rho_a * v_ou_a_fan**2,
            'dP_ou_fan_dynamic [Pa]': 1/2 * rho_a * v_ou_a_fan**2,

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
            'Q_cond_load [W]': Q_cond_load,
            'Q_ref_cond [W]': Q_ref_cond,
            'Q_tank_w [W]': Q_tank_w,
            'Q_tank_w_in [W]': Q_tank_w_in,
            'Q_tank_w_out [W]': Q_tank_w_out,
            'Q_mix_w_in_sup [W]': Q_mix_w_in_sup,
            'Q_mix_w_out [W]': Q_mix_w_out,
            'E_tot [W]': E_cmp + E_ou_fan,

            # === Efficiency ===
            'fan_eff [%]': fan_eff,
        }

        return result

    def _optimize_operation(self, T_tank_w, Q_cond_load, T0):
        """Find minimum-power operating point via Differential Evolution.

        Optimisation variables are ``[dT_ref_cond, dT_ref_evap]``.
        Condenser and evaporator heat transfer feasibility constraints
        are enforced via a penalty function.

        Parameters
        ----------
        T_tank_w : float
            Tank water temperature [°C].
        Q_cond_load : float
            Target condenser heat rate [W].
        T0 : float
            Dead-state / outdoor-air temperature [°C].

        Returns
        -------
        scipy.optimize.OptimizeResult
            Contains ``x`` (optimal variables) and ``success`` flag.
        """
        tolerance = 0.001  # 0.1%
        bounds = [(1.0, 30.0), (1.0, 30.0)]
        penalty_weight = 1e4

        def _objective(x):
            """Objective: minimise E_tot with penalty for constraint violations."""
            try:
                perf = self._calc_state(
                    optimization_vars=x,
                    T_tank_w=T_tank_w,
                    Q_cond_load=Q_cond_load,
                    T0=T0
                )
                if perf is None or not isinstance(perf, dict):
                    return 1e6

                E_tot = perf.get("E_tot [W]", np.nan)
                if np.isnan(E_tot):
                    return 1e6

                penalty = 0.0

                # Condenser constraint: Q_tank_w ∈ [Q_cond_load, Q_cond_load*(1+tol)]
                Q_tank_w = perf.get("Q_tank_w [W]", np.nan)
                if not np.isnan(Q_tank_w):
                    cond_low = Q_cond_load - Q_tank_w
                    if cond_low > 0:
                        penalty += penalty_weight * cond_low**2
                    cond_high = Q_tank_w - Q_cond_load * (1 + tolerance)
                    if cond_high > 0:
                        penalty += penalty_weight * cond_high**2

                # Evaporator constraint: Q_ou_a ∈ [Q_ref_evap, Q_ref_evap*(1+tol)]
                Q_ou_a = perf.get("Q_ou_a [W]", np.nan)
                Q_ref_evap = perf.get("Q_ref_evap [W]", np.nan)
                if not np.isnan(Q_ou_a) and not np.isnan(Q_ref_evap):
                    evap_low = Q_ref_evap - Q_ou_a
                    if evap_low > 0:
                        penalty += penalty_weight * evap_low**2
                    evap_high = Q_ou_a - Q_ref_evap * (1 + tolerance)
                    if evap_high > 0:
                        penalty += penalty_weight * evap_high**2

                return E_tot + penalty
            except Exception:
                return 1e6

        x0 = self.prev_opt_x if self.prev_opt_x is not None else None

        opt_result = differential_evolution(
            _objective,
            bounds=bounds,
            x0=x0,
            maxiter=1000,
            popsize=10,
            tol=1e-4,
            seed=42,
            disp=False,
        )

        if opt_result.success:
            self.prev_opt_x = opt_result.x

        return opt_result

    def analyze_steady(
        self,
        T_tank_w,
        T0,
        dV_w_serv=None,
        Q_cond_load=None,
        return_dict=True
        ):
        """Run a steady-state analysis at the given operating point.

        Exactly one of ``dV_w_serv`` or ``Q_cond_load`` must be provided.

        Parameters
        ----------
        T_tank_w : float
            Tank water temperature [°C].
        T0 : float
            Dead-state / outdoor-air temperature [°C].
        dV_w_serv : float, optional
            Service water flow rate [m³/s].
        Q_cond_load : float, optional
            Target condenser heat rate [W].
        return_dict : bool
            If True return dict; else single-row DataFrame.

        Returns
        -------
        dict or pd.DataFrame
        """
        if dV_w_serv is None and Q_cond_load is None:
            raise ValueError("Either dV_w_serv or Q_cond_load must be provided.")
        if dV_w_serv is not None and Q_cond_load is not None:
            raise ValueError("Cannot provide both dV_w_serv and Q_cond_load.")

        T_tank_w_K = cu.C2K(T_tank_w)
        T0_K = cu.C2K(T0)

        if dV_w_serv is None:
            dV_w_serv = 0.0

        Q_tank_loss = self.UA_tank * (T_tank_w_K - T0_K)
        den = max(1e-6, T_tank_w_K - self.T_tank_w_in_K)
        alp = min(1.0, max(0.0, self.T_mix_w_out_K - self.T_tank_w_in_K) / den)

        self.dV_mix_w_out = dV_w_serv
        self.dV_tank_w_out = alp * dV_w_serv
        self.dV_mix_w_in_sup = (1 - alp) * dV_w_serv

        if Q_cond_load is None:
            Q_use_loss = c_w * rho_w * self.dV_tank_w_out * (T_tank_w_K - self.T_tank_w_in_K)
            Q_cond_load = Q_tank_loss + Q_use_loss

        if T_tank_w <= self.T_tank_w_lower_bound:
            hp_is_on = True
        elif T_tank_w > self.T_tank_w_upper_bound:
            hp_is_on = False
        else:
            hp_is_on = Q_cond_load > 0

        if Q_cond_load <= 0 or not hp_is_on:
            result = self._calc_state(
                optimization_vars=[5.0, 5.0],
                T_tank_w=T_tank_w,
                Q_cond_load=0.0,
                T0=T0
            )
        else:
            opt_result = self._optimize_operation(
                T_tank_w=T_tank_w,
                Q_cond_load=Q_cond_load,
                T0=T0
            )
            result = None
            try:
                result = self._calc_state(
                    optimization_vars=opt_result.x,
                    T_tank_w=T_tank_w,
                    T0=T0,
                    Q_cond_load=Q_cond_load,
                )
            except Exception:
                pass

            if result is None or not isinstance(result, dict):
                try:
                    result = self._calc_state(
                        optimization_vars=[5.0, 5.0],
                        T_tank_w=T_tank_w,
                        Q_cond_load=0.0,
                        T0=T0
                    )
                except Exception:
                    result = {
                        'hp_is_on': False,
                        'converged': False,
                        'Q_cond_load [W]': Q_cond_load,
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

    def analyze_dynamic(
        self, 
        simulation_period_sec, 
        dt_s, 
        T_tank_w_init_C,
        schedule_entries,
        T0_schedule,
        I_DN_schedule=None,
        I_dH_schedule=None,
        tank_level_init = 1.0,    
        result_save_csv_path=None,
        ):
        """Run a time-stepping dynamic simulation (fully implicit scheme).

        At each timestep, ``scipy.optimize.fsolve`` solves for
        ``[T_tank_next, tank_level_next]`` such that the energy and
        mass balance residuals vanish.  All temperature-dependent terms
        (tank loss, refill mixing, STC gain) are evaluated at the
        unknown next-step state, providing unconditional stability.

        Parameters
        ----------
        simulation_period_sec : int
            Total simulation duration [s].
        dt_s : int
            Time step size [s].
        T_tank_w_init_C : float
            Initial tank temperature [°C].
        schedule_entries : list of tuple
            DHW schedule ``(start_str, end_str, fraction)``.
        T0_schedule : array-like
            Dead-state temperature per step [°C].
        I_DN_schedule : array-like, optional
            Direct-normal irradiance per step [W/m²] (required if STC active).
        I_dH_schedule : array-like, optional
            Diffuse-horizontal irradiance per step [W/m²].
        tank_level_init : float
            Initial tank water level [0–1].
        result_save_csv_path : str, optional
            CSV output path.

        Returns
        -------
        pd.DataFrame
            Per-timestep results.
        """
        from scipy.optimize import fsolve

        time = np.arange(0, simulation_period_sec, dt_s)
        tN = len(time)
        T0_schedule = np.array(T0_schedule)
        if len(T0_schedule) != tN:
            raise ValueError(f"T0_schedule length ({len(T0_schedule)}) must match time array length ({tN})")

        use_stc = (self.A_stc > 0) and (I_DN_schedule is not None) and (I_dH_schedule is not None)
        if use_stc:
            I_DN_schedule = np.array(I_DN_schedule)
            I_dH_schedule = np.array(I_dH_schedule)
            if len(I_DN_schedule) != tN:
                raise ValueError(f"I_DN_schedule length ({len(I_DN_schedule)}) must match time array length ({tN})")
            if len(I_dH_schedule) != tN:
                raise ValueError(f"I_dH_schedule length ({len(I_dH_schedule)}) must match time array length ({tN})")

        results_data = []

        self.time = time
        self.dt = dt_s

        self.dV_mix_w_out = 0.0
        self.dV_tank_w_out = 0.0
        self.dV_mix_w_in_sup = 0.0

        self.w_use_frac = build_schedule_ratios(schedule_entries, self.time)

        T_tank_w_K = cu.C2K(T_tank_w_init_C)
        tank_level = tank_level_init
        is_refilling = False
        hp_is_on_prev = False

        # STC kwargs (shared across timesteps)
        stc_kwargs = dict(
            A_stc_pipe=self.A_stc_pipe, alpha_stc=self.alpha_stc,
            h_o_stc=self.h_o_stc, h_r_stc=self.h_r_stc,
            k_ins_stc=self.k_ins_stc, x_air_stc=self.x_air_stc,
            x_ins_stc=self.x_ins_stc, E_pump=self.E_stc_pump,
        )

        C_full = self.C_tank  # full-tank thermal capacitance [J/K]

        for n in tqdm(range(tN), desc="ASHPB Simulating"):
            # =================================================================
            # PHASE A: CONTROL DECISIONS (evaluated at current state)
            # =================================================================
            current_time_s = time[n]
            current_hour = current_time_s * cu.s2h
            T0 = T0_schedule[n]
            T0_K = cu.C2K(T0)

            preheat_on = (
                current_hour >= self.preheat_start_hour
                and current_hour < self.preheat_end_hour
            )

            T_tank_w = cu.K2C(T_tank_w_K)

            # --- UV lamp (time-based) ---
            E_uv = calc_uv_lamp_power(
                current_time_s, self.period_3hour_sec,
                self.num_switching_per_3hour,
                self.uv_lamp_exposure_duration_sec,
                self.lamp_power_watts,
            )

            # --- Demand (schedule-based) ---
            dV_mix_w_out_n = self.w_use_frac[n] * self.dV_mix_w_out_max

            # --- HP control (hysteresis at current T) ---
            if T_tank_w <= self.T_tank_w_lower_bound:
                hp_is_on = True
            elif T_tank_w >= self.T_tank_w_upper_bound:
                hp_is_on = False
            else:
                hp_is_on = hp_is_on_prev

            hp_is_on_schedule = check_hp_schedule_active(current_hour, self.hp_on_schedule)
            hp_is_on = hp_is_on and hp_is_on_schedule
            hp_is_on_prev = hp_is_on

            Q_cond_load_n = self.hp_capacity if hp_is_on else 0.0

            # --- HP cycle (quasi-steady at current T) ---
            # Set mixing valve flows for _calc_state
            den_curr = max(1e-6, T_tank_w_K - self.T_tank_w_in_K)
            alp_curr = min(1.0, max(0.0, (self.T_mix_w_out_K - self.T_tank_w_in_K) / den_curr))
            self.dV_mix_w_out = dV_mix_w_out_n
            self.dV_tank_w_out = alp_curr * dV_mix_w_out_n
            self.dV_mix_w_in_sup = (1 - alp_curr) * dV_mix_w_out_n

            if Q_cond_load_n == 0:
                hp_result = self._calc_state(
                    optimization_vars=[5.0, 5.0],
                    T_tank_w=T_tank_w, Q_cond_load=0.0, T0=T0
                )
            else:
                opt_result = self._optimize_operation(
                    T_tank_w=T_tank_w, Q_cond_load=Q_cond_load_n, T0=T0
                )
                hp_result = self._calc_state(
                    optimization_vars=opt_result.x,
                    T_tank_w=T_tank_w, Q_cond_load=Q_cond_load_n, T0=T0
                )
                if not opt_result.success or hp_result is None:
                    for k, v in hp_result.items():
                        if isinstance(v, float): hp_result[k] = round(v, 2)
                    raise ValueError(f"Optimization failed at timestep {n}: {hp_result}")
                hp_result['converged'] = opt_result.success

            Q_ref_cond = hp_result.get('Q_ref_cond [W]', 0.0)

            # --- Refill control decision (based on current level) ---
            # Project level after outflow to decide refilling
            level_after_outflow = tank_level
            if not self.tank_always_full or (self.tank_always_full and self.prevent_simultaneous_flow):
                level_after_outflow = max(0.0, tank_level - (self.dV_tank_w_out * dt_s) / self.V_tank_full)

            dV_tank_w_in_ctrl = 0.0  # Default: no refill

            if self.tank_always_full and self.prevent_simultaneous_flow:
                tank_outlet_exist = self.dV_tank_w_out > 0
                if tank_outlet_exist:
                    dV_tank_w_in_ctrl = 0.0
                    is_refilling = False
                else:
                    if level_after_outflow < 1.0:
                        req_vol = (1.0 - level_after_outflow) * self.V_tank_full
                        if self.dV_tank_w_in_refill * dt_s <= req_vol:
                            dV_tank_w_in_ctrl = self.dV_tank_w_in_refill
                    else:
                        dV_tank_w_in_ctrl = 0.0

            elif self.tank_always_full:
                dV_tank_w_in_ctrl = None  # Sentinel: dV_in = dV_out (resolved inside residual)

            elif not self.tank_always_full:
                target_lower = self.tank_level_lower_bound
                target_upper = self.tank_level_upper_bound
                if use_stc and self.stc_placement == 'mains_preheat' and preheat_on:
                    target_lower = 1.0
                    target_upper = 1.0
                if not is_refilling:
                    if level_after_outflow < target_lower - 1e-6:
                        is_refilling = True
                if is_refilling:
                    req_vol = (target_upper - level_after_outflow) * self.V_tank_full
                    if self.dV_tank_w_in_refill * dt_s <= req_vol:
                        dV_tank_w_in_ctrl = self.dV_tank_w_in_refill
                    if (level_after_outflow + dV_tank_w_in_ctrl * dt_s / self.V_tank_full) >= target_upper - 1e-6:
                        is_refilling = False
                else:
                    dV_tank_w_in_ctrl = 0.0

            # --- STC control decision (probe at current T) ---
            stc_active = False
            E_stc_pump_pwr = 0.0
            stc_result_probe = {}
            T_stc_w_out_K_mp = self.T_tank_w_in_K  # mains_preheat default

            if use_stc:
                I_DN_n = I_DN_schedule[n]
                I_dH_n = I_dH_schedule[n]

                if self.stc_placement == 'tank_circuit':
                    stc_probe = calc_stc_performance(
                        I_DN_stc=I_DN_n, I_dH_stc=I_dH_n,
                        T_stc_w_in_K=T_tank_w_K, T0_K=T0_K,
                        dV_stc=self.dV_stc_w, is_active=True,
                        **stc_kwargs,
                    )
                    stc_active = preheat_on and stc_probe['T_stc_w_out_K'] > T_tank_w_K
                    E_stc_pump_pwr = self.E_stc_pump if stc_active else 0.0
                    stc_result_probe = stc_probe

                elif self.stc_placement == 'mains_preheat':
                    dV_refill_for_stc = dV_tank_w_in_ctrl if dV_tank_w_in_ctrl is not None else self.dV_tank_w_out
                    if preheat_on and dV_refill_for_stc > 0:
                        stc_probe = calc_stc_performance(
                            I_DN_stc=I_DN_n, I_dH_stc=I_dH_n,
                            T_stc_w_in_K=self.T_tank_w_in_K, T0_K=T0_K,
                            dV_stc=dV_refill_for_stc, is_active=True,
                            **stc_kwargs,
                        )
                        stc_active = stc_probe['T_stc_w_out_K'] > self.T_tank_w_in_K
                        stc_result_probe = stc_probe
                    else:
                        stc_active = False
                        stc_result_probe = calc_stc_performance(
                            I_DN_stc=I_DN_n, I_dH_stc=I_dH_n,
                            T_stc_w_in_K=self.T_tank_w_in_K, T0_K=T0_K,
                            dV_stc=1.0, is_active=False,
                            **stc_kwargs,
                        )

                    if stc_active:
                        T_stc_w_out_K_mp = stc_result_probe['T_stc_w_out_K']
                        E_stc_pump_pwr = self.E_stc_pump
                    else:
                        T_stc_w_out_K_mp = self.T_tank_w_in_K
                        E_stc_pump_pwr = 0.0

            # --- Refill source temperature ---
            if use_stc and self.stc_placement == 'mains_preheat' and stc_active:
                T_refill_K = T_stc_w_out_K_mp
            else:
                T_refill_K = self.T_tank_w_in_K

            # =================================================================
            # PHASE B: IMPLICIT SOLVE — fsolve([T_next, level_next])
            # =================================================================
            T_n = T_tank_w_K
            level_n = tank_level

            # Capture loop-local variables for closure
            _dV_mix = dV_mix_w_out_n
            _dV_in_ctrl = dV_tank_w_in_ctrl
            _Q_hp = Q_ref_cond
            _E_uv = E_uv
            _E_pump = E_stc_pump_pwr
            _T_refill = T_refill_K
            _stc_active = stc_active

            def residual_func(x):
                T_next = x[0]
                level_next = x[1]

                # Mixing valve at T_next
                den = max(1e-6, T_next - self.T_tank_w_in_K)
                alp = min(1.0, max(0.0, (self.T_mix_w_out_K - self.T_tank_w_in_K) / den))
                dV_out = alp * _dV_mix

                # Refill flow
                if _dV_in_ctrl is None:
                    # tank_always_full: inflow matches outflow
                    dV_in = dV_out
                else:
                    dV_in = _dV_in_ctrl

                # --- Mass residual ---
                r_mass = level_next - level_n - (dV_in - dV_out) * dt_s / self.V_tank_full

                # --- Energy terms at T_next ---
                C_curr = C_full * max(0.001, level_n)
                C_next = C_full * max(0.001, level_next)

                Q_loss = self.UA_tank * (T_next - T0_K)
                Q_refill = c_w * rho_w * dV_in * (_T_refill - T_next)

                # STC energy (tank_circuit: evaluate at T_next)
                Q_stc_net = 0.0
                if use_stc and self.stc_placement == 'tank_circuit' and _stc_active:
                    stc_r = calc_stc_performance(
                        I_DN_stc=I_DN_n, I_dH_stc=I_dH_n,
                        T_stc_w_in_K=T_next, T0_K=T0_K,
                        dV_stc=self.dV_stc_w, is_active=True,
                        **stc_kwargs,
                    )
                    Q_stc_net = stc_r.get('Q_stc_w_out', 0.0) - stc_r.get('Q_stc_w_in', 0.0)

                Q_total_gain = _Q_hp + _E_uv + _E_pump + Q_stc_net + Q_refill

                # Energy residual: C_next*T_next - C_curr*T_n = dt*(Q_gain - Q_loss)
                r_energy = C_next * T_next - C_curr * T_n - dt_s * (Q_total_gain - Q_loss)

                return [r_energy, r_mass]

            # Solve
            x0 = [T_n, level_n]
            sol, info, ier, msg = fsolve(residual_func, x0, full_output=True)

            T_tank_w_K = sol[0]
            tank_level = max(0.001, min(1.0, sol[1]))

            # =================================================================
            # PHASE C: POST-SOLVE RESULTS
            # =================================================================
            T_tank_w_solved = cu.K2C(T_tank_w_K)

            # Recompute quantities at solved state for reporting
            den_sol = max(1e-6, T_tank_w_K - self.T_tank_w_in_K)
            alp_sol = min(1.0, max(0.0, (self.T_mix_w_out_K - self.T_tank_w_in_K) / den_sol))
            dV_out_sol = alp_sol * dV_mix_w_out_n
            dV_in_sol = dV_out_sol if dV_tank_w_in_ctrl is None else dV_tank_w_in_ctrl

            # Update instance flow-rate attributes
            self.dV_tank_w_out = dV_out_sol
            self.dV_tank_w_in = dV_in_sol
            self.dV_mix_w_out = dV_mix_w_out_n
            self.dV_mix_w_in_sup = (1 - alp_sol) * dV_mix_w_out_n

            Q_tank_loss = self.UA_tank * (T_tank_w_K - T0_K)
            Q_refill_net = c_w * rho_w * dV_in_sol * (T_refill_K - T_tank_w_K)
            Q_use_loss = c_w * rho_w * dV_out_sol * (T_tank_w_K - T0_K)
            Q_tank_w_in = c_w * rho_w * dV_in_sol * (T_refill_K - T0_K)

            # Mixing valve result at solved T
            if dV_mix_w_out_n > 0:
                mix_sol = calc_mixing_valve(T_tank_w_K, self.T_tank_w_in_K, self.T_mix_w_out_K)
                T_serv_w_actual = mix_sol['T_serv_w_actual']
            else:
                T_serv_w_actual = np.nan

            # STC results at solved T (for reporting)
            Q_stc_w_out = 0.0
            Q_stc_w_in = 0.0
            T_stc_w_out_K = np.nan
            T_stc_w_final_K = np.nan
            stc_result = stc_result_probe

            if use_stc and self.stc_placement == 'tank_circuit':
                if stc_active:
                    stc_result = calc_stc_performance(
                        I_DN_stc=I_DN_n, I_dH_stc=I_dH_n,
                        T_stc_w_in_K=T_tank_w_K, T0_K=T0_K,
                        dV_stc=self.dV_stc_w, is_active=True,
                        **stc_kwargs,
                    )
                else:
                    stc_result = calc_stc_performance(
                        I_DN_stc=I_DN_n, I_dH_stc=I_dH_n,
                        T_stc_w_in_K=T_tank_w_K, T0_K=T0_K,
                        dV_stc=self.dV_stc_w, is_active=False,
                        **stc_kwargs,
                    )
                T_stc_w_out_K = stc_result['T_stc_w_out_K']
                T_stc_w_final_K = stc_result.get('T_stc_w_final_K', T_stc_w_out_K)
                Q_stc_w_out = stc_result.get('Q_stc_w_out', 0.0)
                Q_stc_w_in = stc_result.get('Q_stc_w_in', 0.0)
            elif use_stc and self.stc_placement == 'mains_preheat':
                T_stc_w_out_K = T_stc_w_out_K_mp
                stc_result = stc_result_probe
                Q_stc_w_out = stc_result.get('Q_stc_w_out', 0.0)
                Q_stc_w_in = stc_result.get('Q_stc_w_in', 0.0)

            # --- Assemble step_results ---
            step_results = {}
            step_results.update(hp_result)
            step_results['hp_is_on'] = hp_is_on
            step_results['Q_tank_loss [W]'] = Q_tank_loss
            step_results['T_tank_w [°C]'] = T_tank_w_solved
            step_results['T_mix_w_out [°C]'] = T_serv_w_actual
            step_results['implicit_converged'] = (ier == 1)

            if self.lamp_power_watts > 0:
                step_results['E_uv [W]'] = E_uv

            if not self.tank_always_full or (self.tank_always_full and self.prevent_simultaneous_flow):
                step_results['tank_level [-]'] = tank_level

            if use_stc:
                step_results.update({
                    'stc_active [-]': stc_active,
                    'I_DN_stc [W/m2]': I_DN_schedule[n],
                    'I_dH_stc [W/m2]': I_dH_schedule[n],
                    'I_sol_stc [W/m2]': stc_result.get('I_sol_stc', np.nan),
                    'Q_sol_stc [W]': stc_result.get('Q_sol_stc', np.nan),
                    'Q_stc_w_out [W]': Q_stc_w_out,
                    'Q_stc_w_in [W]': Q_stc_w_in,
                    'Q_l_stc [W]': stc_result.get('Q_l_stc', np.nan),
                    'T_stc_w_out [°C]': cu.K2C(T_stc_w_out_K) if not np.isnan(T_stc_w_out_K) else np.nan,
                    'T_stc_w_in [°C]': cu.K2C(T_tank_w_K),
                    'T_stc [°C]': cu.K2C(stc_result.get('T_stc_K', np.nan)),
                    'E_stc_pump [W]': E_stc_pump_pwr,
                })
                if self.stc_placement == 'tank_circuit':
                    step_results['T_stc_w_final [°C]'] = cu.K2C(T_stc_w_final_K) if not np.isnan(T_stc_w_final_K) else np.nan
            else:
                step_results['stc_active [-]'] = False

            results_data.append(step_results)

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
        self, A_stc, A_stc_pipe, alpha_stc, h_o_stc, h_r_stc, k_ins_stc, x_air_stc, x_ins_stc,
        preheat_start_hour, preheat_end_hour, dV_stc_w, E_stc_pump, stc_placement
    ):
        """Initialise Solar Thermal Collector (STC) parameters.

        Separated from ``__init__`` to reduce constructor complexity.
        """
        self.A_stc = A_stc
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
        dV_tank_refill
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
        dV_tank_refill : float
            Current refill flow rate [m³/s].

        Returns
        -------
        dict
            Keys: ``stc_active``, ``stc_result``, ``T_stc_w_out_K``,
            ``T_stc_w_final_K``, ``Q_stc_w_out``, ``Q_stc_w_in``,
            ``E_stc_pump_pwr``.
        """
        stc_active = False
        stc_result = {}

        T_stc_w_out_K = np.nan
        T_stc_w_final_K = np.nan
        Q_stc_w_out = 0.0
        Q_stc_w_in = 0.0
        E_stc_pump_pwr = 0.0

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
                E_stc_pump_pwr = self.E_stc_pump
            else:
                stc_result = calc_stc_performance(
                    I_DN_stc=I_DN, I_dH_stc=I_dH,
                    T_stc_w_in_K=T_tank_w_K, T0_K=T0_K,
                    dV_stc=self.dV_stc_w,
                    is_active=False,
                    **stc_kwargs,
                )
                E_stc_pump_pwr = 0.0

            T_stc_w_out_K = stc_result['T_stc_w_out_K']
            T_stc_w_final_K = stc_result.get('T_stc_w_final_K', T_stc_w_out_K)
            Q_stc_w_out   = stc_result.get('Q_stc_w_out', 0.0)
            Q_stc_w_in    = stc_result.get('Q_stc_w_in', 0.0)

        elif self.stc_placement == 'mains_preheat':
            if preheat_on and dV_tank_refill > 0:
                stc_result_test = calc_stc_performance(
                    I_DN_stc=I_DN, I_dH_stc=I_dH,
                    T_stc_w_in_K=self.T_tank_w_in_K, T0_K=T0_K,
                    dV_stc=dV_tank_refill,
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
                        dV_stc=dV_tank_refill,
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
                E_stc_pump_pwr = self.E_stc_pump
            else:
                E_stc_pump_pwr = 0.0

        return {
            'stc_active': stc_active,
            'stc_result': stc_result,
            'T_stc_w_out_K': T_stc_w_out_K,
            'T_stc_w_final_K': T_stc_w_final_K,
            'Q_stc_w_out': Q_stc_w_out,
            'Q_stc_w_in': Q_stc_w_in,
            'E_stc_pump_pwr': E_stc_pump_pwr
        }
