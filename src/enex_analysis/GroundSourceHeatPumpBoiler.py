"""Ground source heat pump boiler — physics-based cycle model.

Resolves a vapour-compression refrigerant cycle coupled to a borehole heat
exchanger (BHE) on the evaporator side and a lumped-capacitance hot-water
tank on the condenser side.  At each time step the model finds the
minimum-power operating point via SLSQP optimisation over two temperature
differences (evaporator and condenser approach ΔT).

Borehole thermal response is tracked with the Finite Line Source (FLS)
g-function, enabling long-term ground temperature drift.
"""

#%%
import numpy as np
import math
from . import calc_util as cu
from dataclasses import dataclass
from scipy.optimize import minimize
from .enex_functions import (
    calc_ref_state,
)
import CoolProp.CoolProp as CP
from tqdm import tqdm
import pandas as pd

from .constants import *
from .enex_functions import *

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
        refrigerant    = 'R410A',
        V_disp_cmp     = 0.0005,        # [m³] Compressor displacement
        eta_cmp_isen   = 0.7,           # [-]  Isentropic efficiency

        # 2. Heat exchanger -------------------------------------------
        UA_cond_design = 500,           # [W/K] Condenser design UA
        UA_evap_design = 500,           # [W/K] Evaporator design UA

        # 3. Tank / control / load ------------------------------------
        T0                     = 0.0,   # [°C] Dead-state (ambient) temp
        Ts                     = 16.0,  # [°C] Undisturbed ground temp
        T_tank_w_upper_bound   = 65.0,  # [°C] Tank upper setpoint
        T_tank_w_lower_bound   = 55.0,  # [°C] Tank lower setpoint
        T_mix_w_out            = 40.0,  # [°C] Service water delivery temp
        T_tank_w_in            = 15.0,  # [°C] Mains water supply temp

        hp_capacity            = 8000.0,   # [W]   HP rated capacity
        dV_mix_w_out_max       = 0.0001,   # [m³/s] Max service flow rate

        # Tank / insulation
        r0       = 0.2,    # [m]     Tank inner radius
        H        = 0.8,    # [m]     Tank height
        x_shell  = 0.01,   # [m]     Shell thickness
        x_ins    = 0.05,   # [m]     Insulation thickness
        k_shell  = 25,     # [W/m·K] Shell conductivity
        k_ins    = 0.03,   # [W/m·K] Insulation conductivity
        h_o      = 15,     # [W/m²·K] External convective coefficient

        # 4. Borehole heat exchanger ----------------------------------
        D_b = 0,           # [m]     Borehole depth (reserved)
        H_b = 200,         # [m]     Borehole effective length
        r_b = 0.08,        # [m]     Borehole radius
        R_b = 0.108,       # [m·K/W] Effective borehole thermal resistance

        # Ground-loop fluid
        dV_b_f_lpm = 24,   # [L/min] Borehole fluid flow rate

        # Soil properties
        k_s   = 2.0,       # [W/m·K] Soil thermal conductivity
        c_s   = 800,       # [J/kg·K] Soil specific heat
        rho_s = 2000,      # [kg/m³] Soil density

        # Circulation pump
        E_pmp = 200,       # [W] Pump electrical power

        # 5. UV lamp --------------------------------------------------
        lamp_power_watts = 0,                # [W]   Lamp power
        uv_lamp_exposure_duration_min = 0,    # [min] UV exposure per cycle
        num_switching_per_3hour = 1,          # [-]   Switching count / 3 h

        # 6. Superheat / subcool --------------------------------------
        dT_superheat = 3.0,  # [K] Evaporator outlet superheat
        dT_subcool   = 3.0,  # [K] Condenser outlet subcool
        ):

        # --- 1. Tank geometry and thermal properties ---
        self.tank_physical = {
            'r0': r0, 'H': H, 'x_shell': x_shell, 'x_ins': x_ins,
            'k_shell': k_shell, 'k_ins': k_ins, 'h_o': h_o,
        }
        self.UA_tank = calc_simple_tank_UA(**self.tank_physical)
        self.C_tank = c_w * rho_w * (math.pi * r0**2 * H)

        # --- 2. Refrigerant / cycle / compressor ---
        self.ref_params = {
            'refrigerant': refrigerant,
            'V_disp_cmp': V_disp_cmp,
            'eta_cmp_isen': eta_cmp_isen,
        }
        self.ref = refrigerant
        self.V_disp_cmp = V_disp_cmp
        self.eta_cmp_isen = eta_cmp_isen

        # --- 3. Heat exchanger UA values ---
        self.heat_exchanger = {
            'UA_cond_design': UA_cond_design,
            'UA_evap_design': UA_evap_design,
        }
        self.UA_cond_design = UA_cond_design
        self.UA_evap_design = UA_evap_design
        self.UA_cond = UA_cond_design
        self.UA_evap = UA_evap_design

        # --- 4. Control and load parameters ---
        self.control_params = {
            'T0': T0, 'Ts': Ts,
            'T_tank_w_upper_bound': T_tank_w_upper_bound,
            'T_tank_w_lower_bound': T_tank_w_lower_bound,
            'T_mix_w_out': T_mix_w_out,
            'T_tank_w_in': T_tank_w_in,
            'hp_capacity': hp_capacity,
            'dV_mix_w_out_max': dV_mix_w_out_max,
        }
        self.Ts = Ts
        self.T_b_f_in = Ts   # Initial value; updated during simulation

        self.hp_capacity = hp_capacity
        self.dV_mix_w_out_max = dV_mix_w_out_max
        self.T_tank_w_upper_bound = T_tank_w_upper_bound
        self.T_tank_w_lower_bound = T_tank_w_lower_bound

        # --- 5. UV lamp parameters ---
        self.lamp_power_watts = lamp_power_watts
        self.uv_lamp_exposure_duration_min = uv_lamp_exposure_duration_min
        self.num_switching_per_3hour = num_switching_per_3hour
        self.period_3hour_sec = 3 * cu.h2s                                # 3 h → s
        self.uv_lamp_exposure_duration_sec = uv_lamp_exposure_duration_min * cu.m2s

        # --- 6. Superheat / subcool ---
        self.dT_superheat = dT_superheat
        self.dT_subcool = dT_subcool
        self.T_tank_w_in = T_tank_w_in
        self.T_mix_w_out = T_mix_w_out

        # --- 7. Borehole heat exchanger ---
        self.borehole = {
            'D_b': D_b, 'H_b': H_b, 'r_b': r_b, 'R_b': R_b,
        }
        self.D_b = D_b
        self.H_b = H_b
        self.r_b = r_b
        self.R_b = R_b

        self.k_s    = k_s
        self.c_s    = c_s
        self.alp_s  = k_s / (c_s * rho_s)   # Soil thermal diffusivity [m²/s]
        self.rho_s  = rho_s
        self.E_pmp  = E_pmp

        # --- Unit conversions (°C → K, L/min → m³/s) ---
        self.T0_K       = cu.C2K(T0)
        self.Ts_K       = cu.C2K(self.Ts)
        self.T_b_f_in_K = cu.C2K(self.T_b_f_in)
        self.T_tank_w_in_K  = cu.C2K(T_tank_w_in)
        self.T_mix_w_out_K = cu.C2K(T_mix_w_out)
        self.dV_b_f_m3s = dV_b_f_lpm * cu.L2m3/cu.m2s  

        self.Q_cond_LOAD_OFF_TOL = 500.0   # [W] Below this → full OFF

        # Warm-start: reuse previous optimisation result
        self.prev_opt_x = None
        
    def _calc_state(self, optimization_vars, T_tank_w, Q_cond_load, T0):
        """Evaluate refrigerant cycle performance for a given operating point.

        Handles both ON and OFF states in a single entry point.
        When ``Q_cond_load <= 0`` the method returns a zero-load result
        with saturation-point thermodynamic properties.

        Parameters
        ----------
        optimization_vars : list of float
            ``[dT_ref_HX, dT_ref_cond]`` — evaporator and condenser
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
            Result dictionary with cycle state points, heat rates,
            electrical inputs, and borehole temperatures.
            ``None`` if the cycle is physically infeasible.
        """
        # OFF 상태: Q_cond_load <= 0
        if Q_cond_load <= 0:
            T_tank_w_K = cu.C2K(T_tank_w)
            den = max(1e-6, T_tank_w_K - self.T_tank_w_in_K)
            alp = min(1.0, max(0.0, (self.T_mix_w_out_K - self.T_tank_w_in_K) / den))
            if alp >= 1.0:
                T_serv_w_actual = T_tank_w
            else:
                T_serv_w_actual_K = alp * T_tank_w_K + (1 - alp) * self.T_tank_w_in_K
                T_serv_w_actual = cu.K2C(T_serv_w_actual_K)
            P1_off = CP.PropsSI('P', 'T', self.T_b_f_in_K, 'Q', 1, self.ref)
            h1_off = CP.PropsSI('H', 'P', P1_off, 'Q', 1, self.ref)
            s1_off = CP.PropsSI('S', 'P', P1_off, 'Q', 1, self.ref)
            P3_off = CP.PropsSI('P', 'T', T_tank_w_K, 'Q', 0, self.ref)
            h3_off = CP.PropsSI('H', 'P', P3_off, 'Q', 0, self.ref)
            s3_off = CP.PropsSI('S', 'P', P3_off, 'Q', 0, self.ref)
            return {
                'hp_is_on': False,
                'converged': True,
                'T_tank_w [°C]': T_tank_w,
                'T0 [°C]': T0,
                'T_mix_w_out [°C]': T_serv_w_actual,
                'T_tank_w_in [°C]': self.T_tank_w_in,
                'Ts [°C]': self.Ts,
                'T_b [°C]': self.Ts,
                'T_b_f [°C]': self.Ts,
                'T_b_f_in [°C]': self.Ts,
                'T_b_f_out [°C]': self.Ts,
                'dV_mix_w_out [m3/s]': self.dV_mix_w_out if hasattr(self, 'dV_mix_w_out') else 0.0,
                'dV_tank_w_in [m3/s]': self.dV_tank_w_in if hasattr(self, 'dV_tank_w_in') else 0.0,
                'dV_mix_w_in_sup [m3/s]': self.dV_mix_w_in_sup if hasattr(self, 'dV_mix_w_in_sup') else 0.0,
                'dV_b_f [m3/s]': self.dV_b_f_m3s,
                'P_ref_cmp_in [Pa]': P1_off,
                'P_ref_cmp_out [Pa]': P3_off,
                'P_ref_exp_in [Pa]': P3_off,
                'P_ref_exp_out [Pa]': P1_off,
                'P_ref_evap_sat [Pa]': P1_off,
                'P_ref_cond_sat_v [Pa]': P3_off,
                'P_ref_cond_sat_l [Pa]': P3_off,
                'h_ref_cmp_in [J/kg]': h1_off,
                'h_ref_cmp_out [J/kg]': h1_off,
                'h_ref_exp_in [J/kg]': h3_off,
                'h_ref_exp_out [J/kg]': h3_off,
                'h_ref_evap_sat [J/kg]': h1_off,
                'h_ref_cond_sat_v [J/kg]': h1_off,
                'h_ref_cond_sat_l [J/kg]': h3_off,
                's_ref_cmp_in [J/(kg·K)]': s1_off,
                's_ref_cmp_out [J/(kg·K)]': s1_off,
                's_ref_exp_in [J/(kg·K)]': s3_off,
                's_ref_exp_out [J/(kg·K)]': s3_off,
                's_ref_evap_sat [J/(kg·K)]': s1_off,
                's_ref_cond_sat_v [J/(kg·K)]': s1_off,
                's_ref_cond_sat_l [J/(kg·K)]': s3_off,
                'x1 [J/kg]': 0.0, 'x2 [J/kg]': 0.0, 'x3 [J/kg]': 0.0, 'x4 [J/kg]': 0.0,
                'T_ref_evap_sat [°C]': cu.K2C(self.T_b_f_in_K),
                'T_ref_cond_sat_v [°C]': T_tank_w,
                'T_ref_cond_sat_l [°C]': T_tank_w,
                'T_ref_cmp_in [°C]': cu.K2C(self.T_b_f_in_K),
                'T_ref_cmp_out [°C]': T_tank_w,
                'T_ref_exp_in [°C]': T_tank_w,
                'T_ref_exp_out [°C]': cu.K2C(self.T_b_f_in_K),
                'T_cond [°C]': T_tank_w,
                'Q_b [W]': 0.0, 'Q_ref_cond [W]': 0.0, 'Q_ref_evap [W]': 0.0,
                'Q_LMTD_cond [W]': 0.0, 'Q_LMTD_evap [W]': 0.0, 'Q_cond_load [W]': 0.0,
                'E_cmp [W]': 0.0, 'E_pmp [W]': 0.0, 'E_tot [W]': 0.0,
                'm_dot_ref [kg/s]': 0.0, 'cmp_rpm [rpm]': 0.0,
            }

        # --- Step 1: Unpack optimisation variables ---
        dT_ref_HX = optimization_vars[0]       # Evaporator approach ΔT [K]
        dT_ref_cond = optimization_vars[1]     # Condenser approach ΔT [K]

        # --- Step 2: Temperature conversions and evap/cond temperatures ---
        T_tank_w_K = cu.C2K(T_tank_w)
        T0_K = cu.C2K(T0)
        T_b_f_in_K = self.T_b_f_in_K           # BHE fluid inlet [K] (from previous step)

        T_evap_K = T_b_f_in_K - dT_ref_HX      # Evaporation temperature [K]
        T_cond_K = T_tank_w_K + dT_ref_cond    # Condensation temperature [K]
        
        # --- Step 3: Common cycle state calculation ---
        cycle_states = calc_ref_state(
            T_evap_K=T_evap_K, T_cond_K=T_cond_K,
            refrigerant=self.ref, eta_cmp_isen=self.eta_cmp_isen,
            T0_K=T0_K, P0=101325,
            dT_superheat=self.dT_superheat, dT_subcool=self.dT_subcool
        )

        rho_ref_cmp_in = cycle_states['rho']  # Density at compressor inlet

        # --- Step 4: Extract state-point properties ---
        T1_K = cycle_states['T1_K'];  P1 = cycle_states['P1']
        h1 = cycle_states['h1'];      s1 = cycle_states['s1']
        T2_K = cycle_states['T2_K'];  P2 = cycle_states['P2']
        h2 = cycle_states['h2'];      s2 = cycle_states['s2']
        T3_K = cycle_states['T3_K'];  P3 = cycle_states['P3']
        h3 = cycle_states['h3'];      s3 = cycle_states['s3']
        T4_K = cycle_states['T4_K'];  P4 = cycle_states['P4']
        h4 = cycle_states['h4'];      s4 = cycle_states['s4']

        if (h3 - h2) == 0:  # Guard against division by zero
            return None

        # --- Step 5: Refrigerant mass flow and cycle performance ---
        m_dot_ref = Q_cond_load / (h2 - h3)   # [kg/s]
        Q_ref_cond = m_dot_ref * (h2 - h3)    # Condenser heat [W]
        Q_ref_evap = m_dot_ref * (h1 - h4)    # Evaporator heat (from ground) [W]
        E_cmp = m_dot_ref * (h2 - h1)         # Compressor power [W]
        cmp_rps = m_dot_ref / (self.V_disp_cmp * rho_ref_cmp_in)  # [rev/s]
        
        # --- Step 6: Condenser heat transfer (simplified ΔT model) ---
        Q_LMTD_cond = self.UA_cond * dT_ref_cond

        # --- Step 7: Borehole heat exchange (evaporator side) ---
        Q_b = Q_ref_evap - self.E_pmp           # Net ground heat extraction [W]
        Q_b_unit = Q_b / self.H_b               # Per-unit-length heat rate [W/m]

        # BHE fluid outlet temperature (energy conservation)
        T_b_f_out_K = T_b_f_in_K + Q_ref_evap / (c_w * rho_w * self.dV_b_f_m3s)
        T_b_f_out = cu.K2C(T_b_f_out_K)

        # Mean fluid temperature and borehole wall temperature
        T_b_f = (cu.K2C(T_b_f_in_K) + T_b_f_out) / 2
        T_b = T_b_f + Q_b_unit * self.R_b

        # Evaporator LMTD (counter-flow: ground fluid vs refrigerant)
        dT1_HX = T_b_f_in_K - T1_K              # Fluid inlet − refrigerant outlet
        dT2_HX = T_b_f_out_K - T4_K             # Fluid outlet − refrigerant inlet

        if dT1_HX <= 1e-6 or dT2_HX <= 1e-6 or abs(dT1_HX - dT2_HX) < 1e-6:
            Q_LMTD_evap = np.inf                 # Physically infeasible
        else:
            LMTD_HX = (dT1_HX - dT2_HX) / np.log(dT1_HX / dT2_HX)
            Q_LMTD_evap = self.UA_evap * LMTD_HX

        # --- Step 8: Specific exergy at each state point ---
        P0 = 101325
        h0 = CP.PropsSI('H', 'T', T0_K, 'P', P0, self.ref)
        s0 = CP.PropsSI('S', 'T', T0_K, 'P', P0, self.ref)

        x1 = (h1-h0) - T0_K*(s1 - s0)
        x2 = (h2-h0) - T0_K*(s2 - s0)
        x3 = (h3-h0) - T0_K*(s3 - s0)
        x4 = (h4-h0) - T0_K*(s4 - s0)

        # Saturation-point properties
        T1_star_K = cycle_states.get('T1_star_K', np.nan)
        T2_star_K = cycle_states.get('T2_star_K', np.nan)
        T3_star_K = cycle_states.get('T3_star_K', np.nan)
        P2_star = cycle_states.get('P2_star', P2)

        P1_star = P1
        h1_star = CP.PropsSI('H', 'P', P1_star, 'Q', 1, self.ref)
        s1_star = CP.PropsSI('S', 'P', P1_star, 'Q', 1, self.ref)
        x1_star = (h1_star-h0) - T0_K*(s1_star - s0)

        h2_star = cycle_states.get('h2_star', np.nan)
        s2_star = cycle_states.get('s2_star', np.nan)
        if np.isnan(h2_star) or np.isnan(s2_star):
            h2_star = CP.PropsSI('H', 'P', P2_star, 'Q', 1, self.ref)
            s2_star = CP.PropsSI('S', 'P', P2_star, 'Q', 1, self.ref)
        x2_star = (h2_star-h0) - T0_K*(s2_star - s0)

        P3_star = P3
        h3_star = CP.PropsSI('H', 'P', P3_star, 'Q', 0, self.ref)
        s3_star = CP.PropsSI('S', 'P', P3_star, 'Q', 0, self.ref)
        x3_star = (h3_star-h0) - T0_K*(s3_star - s0)

        # --- Step 9: Assemble result dictionary ---
        T_b_f_in = cu.K2C(T_b_f_in_K)
        
        result = {
            'hp_is_on': True,
            'converged': True,
            
            # === [온도: °C] =======================================
            # Saturation Points (ASHPB naming)
            'T_ref_evap_sat [°C]': cu.K2C(T1_star_K),
            'T_ref_cond_sat_v [°C]': cu.K2C(T2_star_K),
            'T_ref_cond_sat_l [°C]': cu.K2C(T3_star_K),
            
            # Actual Points (ASHPB naming)
            'T0 [°C]': T0,
            'T_ref_cmp_in [°C]': cu.K2C(T1_K),
            'T_ref_cmp_out [°C]': cu.K2C(T2_K),
            'T_ref_exp_in [°C]': cu.K2C(T3_K),
            'T_ref_exp_out [°C]': cu.K2C(T4_K),
            'T_cond [°C]': cu.K2C(T3_star_K if not np.isnan(T3_star_K) else T3_K),  # 대표 응축 온도는 포화 온도로 표시
            'T_tank_w [°C]': T_tank_w,
            'T_mix_w_out [°C]': self.T_mix_w_out,
            'T_tank_w_in [°C]': self.T_tank_w_in,
            'Ts [°C]': self.Ts,
            'T_b [°C]': T_b,
            'T_b_f [°C]': T_b_f,
            'T_b_f_in [°C]': T_b_f_in,
            'T_b_f_out [°C]': T_b_f_out,
            
            # === [체적유량: m3/s] ==================================
            'dV_b_f [m3/s]': self.dV_b_f_m3s,
            'dV_mix_w_out [m3/s]': self.dV_mix_w_out if hasattr(self, 'dV_mix_w_out') else 0.0,
            'dV_tank_w_in [m3/s]': self.dV_tank_w_in if hasattr(self, 'dV_tank_w_in') else 0.0,
            'dV_mix_w_in_sup [m3/s]': self.dV_mix_w_in_sup if hasattr(self, 'dV_mix_w_in_sup') else 0.0,
            
            # === [압력: Pa] ========================================
            'P_ref_cmp_in [Pa]': P1,
            'P_ref_cmp_out [Pa]': P2,
            'P_ref_exp_in [Pa]': P3,
            'P_ref_exp_out [Pa]': P4,
            'P_ref_evap_sat [Pa]': P1_star,
            'P_ref_cond_sat_v [Pa]': P2_star,
            'P_ref_cond_sat_l [Pa]': P3_star,
            
            # === [질량유량: kg/s] ==================================
            'm_dot_ref [kg/s]': m_dot_ref,
            
            # === [rpm] =============================================
            'cmp_rpm [rpm]': cmp_rps * 60,
            
            # === [엔탈피: J/kg] ====================================
            'h_ref_cmp_in [J/kg]': h1,
            'h_ref_cmp_out [J/kg]': h2,
            'h_ref_exp_in [J/kg]': h3,
            'h_ref_exp_out [J/kg]': h4,
            'h_ref_evap_sat [J/kg]': h1_star,
            'h_ref_cond_sat_v [J/kg]': h2_star,
            'h_ref_cond_sat_l [J/kg]': h3_star,
            
            # === [엔트로피: J/(kg·K)] ==============================
            's_ref_cmp_in [J/(kg·K)]': s1,
            's_ref_cmp_out [J/(kg·K)]': s2,
            's_ref_exp_in [J/(kg·K)]': s3,
            's_ref_exp_out [J/(kg·K)]': s4,
            's_ref_evap_sat [J/(kg·K)]': s1_star,
            's_ref_cond_sat_v [J/(kg·K)]': s2_star,
            's_ref_cond_sat_l [J/(kg·K)]': s3_star,
            
            # === [엑서지 단위: J/kg] ===============================
            'x1 [J/kg]': x1,
            'x2 [J/kg]': x2,
            'x3 [J/kg]': x3,
            'x4 [J/kg]': x4,
            'x1_star [J/kg]': x1_star,
            'x2_star [J/kg]': x2_star,
            'x3_star [J/kg]': x3_star,
            
            # === [에너지/열량: W] ==================================
            'Q_cond_load [W]': Q_cond_load,
            'Q_ref_cond [W]': Q_ref_cond,
            'Q_ref_evap [W]': Q_ref_evap,
            'Q_LMTD_cond [W]': Q_LMTD_cond,
            'Q_LMTD_evap [W]': Q_LMTD_evap,
            'Q_b [W]': Q_b,
            
            # === [전력: W] =========================================
            'E_cmp [W]': E_cmp,
            'E_pmp [W]': self.E_pmp,
            'E_tot [W]': E_cmp + self.E_pmp,
        }
        
        return result
    
    def analyze_steady(
        self,
        T_tank_w,
        T_b_f_in,
        dV_mix_w_out=None,
        Q_cond_load=None,
        T0=None,
        return_dict=True
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
            Service water flow rate [m³/s].  Used to compute Q_cond_load.
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
            raise ValueError("dV_mix_w_out와 Q_cond_load 중 하나는 반드시 제공되어야 합니다.")
        if dV_mix_w_out is not None and Q_cond_load is not None:
            raise ValueError("dV_mix_w_out와 Q_cond_load를 동시에 제공할 수 없습니다.")
        
        T0 = T0 if T0 is not None else T_b_f_in
        T_tank_w_K = cu.C2K(T_tank_w)
        T0_K = cu.C2K(T0)
        
        self.T_b_f_in = T_b_f_in
        self.T_b_f_in_K = cu.C2K(T_b_f_in)
        
        if dV_mix_w_out is None:
            dV_mix_w_out = 0.0
        
        den = max(1e-6, T_tank_w_K - self.T_tank_w_in_K)
        alp = min(1.0, max(0.0, (self.T_mix_w_out_K - self.T_tank_w_in_K) / den))
        self.dV_mix_w_out = dV_mix_w_out
        self.dV_tank_w_in = alp * dV_mix_w_out
        self.dV_mix_w_in_sup = (1 - alp) * dV_mix_w_out
        
        if Q_cond_load is None:
            Q_tank_loss = self.UA_tank * (T_tank_w_K - T0_K)
            Q_use_loss = c_w * rho_w * self.dV_tank_w_in * (T_tank_w_K - self.T_tank_w_in_K)
            Q_cond_load = Q_tank_loss + Q_use_loss
        
        hp_is_on = (T_tank_w <= self.T_tank_w_lower_bound or
                    (self.T_tank_w_lower_bound < T_tank_w < self.T_tank_w_upper_bound and Q_cond_load > 0))
        
        if Q_cond_load <= 0 or not hp_is_on:
            result = self._calc_state(optimization_vars=[5.0, 5.0], T_tank_w=T_tank_w, Q_cond_load=0.0, T0=T0)
        else:
            opt_result = self._optimize_operation(T_tank_w=T_tank_w, Q_cond_load=Q_cond_load, T0=T0)
            if opt_result.success:
                result = self._calc_state(
                    optimization_vars=opt_result.x,
                    T_tank_w=T_tank_w,
                    Q_cond_load=Q_cond_load,
                    T0=T0
                )
            else:
                result = self._calc_state(optimization_vars=[5.0, 5.0], T_tank_w=T_tank_w, Q_cond_load=0.0, T0=T0)
        
        if return_dict:
            return result
        return pd.DataFrame([result])
    
    def _optimize_operation(self, T_tank_w, Q_cond_load, T0):
        """Find minimum-power operating point via SLSQP.

        Optimisation variables are ``[dT_ref_HX, dT_ref_cond]``.
        The condenser LMTD constraint ensures heat-exchanger feasibility.

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
        # tolerance 변수 정의
        tolerance = 0.01  # 1%
        
        # 최적화 변수 경계 조건 및 초기 추정값 설정 (prev_opt_x 재사용)
        bounds = [(1.0, 30.0), (1.0, 30.0)]  # [dT_ref_HX, dT_ref_cond]
        initial_guess = self.prev_opt_x if self.prev_opt_x is not None else [5.0, 5.0]
        
        # 응축기 LMTD 제약 조건 함수 (하한): Q_LMTD_cond - Q_cond_load >= 0
        def _cond_LMTD_constraint_low(x):
            """
            응축기 LMTD 제약 조건 함수 (하한): Q_LMTD_cond - Q_cond_load >= 0
            perf가 None이면 (수렴 실패 등) 큰 제약 조건 위반 값을 반환하여
            최적화 알고리즘이 해당 변수 조합을 피하고 다른 조합을 시도하도록 함.
            """
            try:
                perf = self._calc_state(
                    optimization_vars=x,
                    T_tank_w=T_tank_w,
                    Q_cond_load=Q_cond_load,
                    T0=T0
                )
                if perf is None or not isinstance(perf, dict):
                    return -1e6  # ineq 제약이므로 음수 반환
                
                if "Q_LMTD_cond [W]" not in perf or np.isnan(perf["Q_LMTD_cond [W]"]):
                    return -1e6
                
                # 제약 조건: Q_LMTD_cond - Q_cond_load >= 0
                return perf["Q_LMTD_cond [W]"] - Q_cond_load
            except Exception as e:
                return -1e6
        
        # 응축기 LMTD 제약 조건 함수 (상한): Q_cond_load*(1+tolerance) - Q_LMTD_cond >= 0
        def _cond_LMTD_constraint_high(x):
            """
            응축기 LMTD 제약 조건 함수 (상한): Q_cond_load*(1+tolerance) - Q_LMTD_cond >= 0
            perf가 None이면 (수렴 실패 등) 큰 제약 조건 위반 값을 반환하여
            최적화 알고리즘이 해당 변수 조합을 피하고 다른 조합을 시도하도록 함.
            """
            try:
                perf = self._calc_state(
                    optimization_vars=x,
                    T_tank_w=T_tank_w,
                    Q_cond_load=Q_cond_load,
                    T0=T0
                )
                if perf is None or not isinstance(perf, dict):
                    return -1e6  # ineq 제약이므로 음수 반환
                
                if "Q_LMTD_cond [W]" not in perf or np.isnan(perf["Q_LMTD_cond [W]"]):
                    return -1e6
                
                # 제약 조건: Q_cond_load*(1+tolerance) - Q_LMTD_cond >= 0
                return Q_cond_load * (1 + tolerance) - perf["Q_LMTD_cond [W]"]
            except Exception as e:
                return -1e6
        
        # 제약 조건: 응축기만 두 개의 ineq 제약 (증발기 제약조건 제거)
        const_funcs = [
            {'type': 'ineq', 'fun': _cond_LMTD_constraint_low},   # Q_LMTD_cond - Q_cond_load >= 0
            {'type': 'ineq', 'fun': _cond_LMTD_constraint_high},  # Q_cond_load*(1+tolerance) - Q_LMTD_cond >= 0
        ]
        
        # 목적 함수: E_tot (총 전력 소비) 최소화
        def _objective(x):  # x = [dT_ref_HX, dT_ref_cond]
            """
            목적 함수: E_tot (총 전력 소비) 최소화.
            perf가 None이면 (수렴 실패 등) 큰 penalty 값을 반환하여
            최적화 알고리즘이 해당 변수 조합을 피하고 다른 조합을 시도하도록 함.
            """
            try:
                perf = self._calc_state(
                    optimization_vars=x,
                    T_tank_w=T_tank_w,
                    Q_cond_load=Q_cond_load,
                    T0=T0
                )
                if perf is None or not isinstance(perf, dict):
                    return 1e6
                
                if "E_tot [W]" not in perf or np.isnan(perf["E_tot [W]"]):
                    return 1e6
                
                # 목적 함수: E_tot 최소화
                return perf["E_tot [W]"]
            except Exception as e:
                return 1e6
        
        # SLSQP 옵션
        if True:  # SLSQP only
            options = {
                'disp': False,
                'maxiter': 100,
                'ftol': 10,      # 함수 값 수렴 허용 오차
                'eps': 0.01,      # 유한 차분 근사 스텝 크기
            }
        # 최적화 실행
        opt_result = minimize(
            _objective,           # 목적 함수 (E_tot = E_cmp + E_pmp 최소화)
            initial_guess,        # 초기 추정값
            method='SLSQP',
            bounds=bounds,
            constraints=const_funcs,
            options=options
        )
        
        if opt_result.success:
            self.prev_opt_x = np.array(opt_result.x).copy()
        return opt_result
    
    def analyze_dynamic(
        self, 
        simulation_period_sec, 
        dt_s, 
        T_tank_w_init_C,
        schedule_entries,
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
        schedule_entries : list of tuple
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
            raise ValueError(f"T0_schedule length ({len(T0_schedule)}) must match time array length ({tN})")
        
        results_data = []
        
        self.time = time
        self.dt = dt_s
        
        # --- 1. 시뮬레이션 초기화 ---
        self.T_b_f = self.Ts # 초기 지중열 교환기 유출수 온도
        self.T_b = self.Ts   # 초기 지중 온도
        self.T_b_f_in = self.Ts # 초기 지중열 교환기 유입수 온도
        self.T_b_f_out = self.Ts # 초기 지중열 교환기 유출수 온도
        self.Q_b = 0.0 # 초기 지중열 교환기 열 유량
        
        self.dV_mix_w_out = 0.0 
        self.dV_tank_w_in = 0.0 
        self.dV_mix_w_in_sup = 0.0 
        
        self.w_use_frac = build_schedule_ratios(schedule_entries, self.time)
        
        T_tank_w_K = cu.C2K(T_tank_w_init_C)
        Q_b_unit_pulse = np.zeros(tN)
        Q_b_unit_old = 0
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
            alp = min(1.0, max(0.0, (self.T_mix_w_out_K - self.T_tank_w_in_K) / den))

            self.dV_mix_w_out = self.w_use_frac[n] * self.dV_mix_w_out_max 
            self.dV_tank_w_in = alp * self.dV_mix_w_out
            self.dV_mix_w_in_sup = (1 - alp) * self.dV_mix_w_out 

            Q_use_loss = c_w * rho_w * self.dV_tank_w_in * (T_tank_w_K - self.T_tank_w_in_K)
            total_loss = Q_tank_loss + Q_use_loss
            
            # On/Off 결정
            if T_tank_w <= self.T_tank_w_lower_bound: is_on = True
            elif T_tank_w >= self.T_tank_w_upper_bound: is_on = False
            else: is_on = is_on_prev
            
            is_transitioning_off_to_on = (not is_on_prev) and is_on # False to True
            Q_cond_load_n = self.hp_capacity if is_on else 0.0
            is_on_prev = is_on
            
            # OFF 상태 조기 체크: Q_cond_load_n이 임계값 이하이면 최적화 건너뛰기
            if abs(Q_cond_load_n) <= self.Q_cond_LOAD_OFF_TOL:
                self.prev_opt_x = None  # OFF 시 초기값 재설정
                result = self._calc_state(
                    optimization_vars=[5.0, 5.0],
                    T_tank_w=T_tank_w,
                    Q_cond_load=0.0,
                    T0=T0
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
                try:
                    result = self._calc_state(
                        optimization_vars=opt_result.x,
                        T_tank_w=T_tank_w,
                        Q_cond_load=Q_cond_load_n,
                        T0=T0
                    )
                except Exception:
                    pass
                
                # result 검증 및 fallback
                if result is None or not isinstance(result, dict):
                    try:
                        result = self._calc_off_state(
                            T_tank_w=T_tank_w,
                            T0=T0
                        )
                    except Exception:
                        result = {
                            'hp_is_on': False,
                            'converged': False,
                            'Q_ref_cond [W]': 0.0,
                            'Q_ref_evap [W]': 0.0,
                            'E_cmp [W]': 0.0,
                            'E_pmp [W]': 0.0,
                            'E_tot [W]': 0.0,
                            'T_tank_w [°C]': T_tank_w,
                            'T0 [°C]': T0
                        }
                
                # converged 플래그 설정
                if result is not None and isinstance(result, dict):
                    result['converged'] = opt_result.success
            
            if is_transitioning_off_to_on:
                # OFF→ON 전환 시점: 점진적 전환을 위해 이전 스텝의 지중 온도 값 사용
                # result는 ON 상태로 계산되었지만, 지중 온도는 점진적으로 업데이트
                step_results.update(result)
                step_results['hp_is_on'] = is_on
                # 전환 시점임을 표시하는 플래그 추가
                step_results['is_transitioning'] = True
                
                # 전환 시점에서는 이전 스텝의 지중 온도 값 유지 (점진적 전환)
                step_results['T_b [°C]'] = self.T_b  # 이전 스텝 값
                step_results['T_b_f [°C]'] = self.T_b_f
                step_results['T_b_f_in [°C]'] = self.T_b_f_in
                step_results['T_b_f_out [°C]'] = self.T_b_f_out
                step_results['Q_b [W]'] = 0.0  # 전환 시점에서는 0으로 시작
            else:
                step_results.update(result)
                step_results['hp_is_on'] = is_on
                step_results['is_transitioning'] = False

            # 지중 온도 업데이트
            if is_transitioning_off_to_on:
                # 전환 시점: 점진적 전환을 위해 Q_b_unit을 0으로 시작
                Q_b_unit = 0.0
                # 펄스 계산 건너뛰기 (전환 시점에서는 펄스 없음)
                # Q_b_unit_old를 0으로 업데이트하여 다음 스텝에서 정상 계산 시작
                Q_b_unit_old = 0.0
            else:
                # 일반 시점: 정상 계산
                Q_b_unit = (result.get('Q_ref_evap [W]', 0.0) - self.E_pmp) / self.H_b if result.get('hp_is_on') else 0.0
            
            if abs(Q_b_unit - Q_b_unit_old) > 1e-6: # 만약 Q_b이 이전 스텝과 일정 수준 이상 차이가 난다면 펄스가 나타난 것으로 간주
                Q_b_unit_pulse[n] = Q_b_unit - Q_b_unit_old # 펄스는 이전 값과의 차이
                Q_b_unit_old = Q_b_unit # 업데이트
        
            # 펄스 계산 (전환 시점이 아닐 때만)
            if not is_transitioning_off_to_on:
                pulses_idx = np.flatnonzero(Q_b_unit_pulse[:n+1])
                dQ = Q_b_unit_pulse[pulses_idx]
                tau = self.time[n] - self.time[pulses_idx]
                
                # g-function 계산은 여전히 루프가 필요
                g_n = np.array([G_FLS(t, self.k_s, self.alp_s, self.r_b, self.H_b) for t in tau])
                dT_b = np.dot(dQ, g_n)
                
                self.T_b = self.Ts - dT_b
                self.T_b_f = self.T_b - Q_b_unit * self.R_b
                self.Q_b = Q_b_unit * self.H_b
                self.T_b_f_in  = self.T_b_f - self.Q_b / (2 * c_w * rho_w * self.dV_b_f_m3s) # °C
                self.T_b_f_out = self.T_b_f + self.Q_b / (2 * c_w * rho_w * self.dV_b_f_m3s) # °C
                self.T_b_f_in_K  = cu.C2K(self.T_b_f_in)
                self.T_b_f_out_K = cu.C2K(self.T_b_f_out)
            
                # step_results에 반영
                step_results['T_b [°C]'] = self.T_b
                step_results['T_b_f [°C]'] = self.T_b_f
                step_results['T_b_f_in [°C]'] = self.T_b_f_in
                step_results['T_b_f_out [°C]'] = self.T_b_f_out
                step_results['Q_b [W]'] = self.Q_b
            
            # UV 램프 전력 계산 (is_on과 무관)
            E_uv = 0
            if (
                self.num_switching_per_3hour > 0
                and self.lamp_power_watts > 0
            ):
                time_in_period = time[n] % self.period_3hour_sec
                interval = (
                    self.period_3hour_sec
                    - self.num_switching_per_3hour * self.uv_lamp_exposure_duration_sec
                ) / (self.num_switching_per_3hour + 1)
                for i in range(self.num_switching_per_3hour):
                    start_time = interval * (i + 1) + i * self.uv_lamp_exposure_duration_sec
                    if start_time <= time_in_period < start_time + self.uv_lamp_exposure_duration_sec:
                        E_uv = self.lamp_power_watts
                        break
            
            step_results['hp_is_on'] = is_on
            if self.lamp_power_watts > 0:
                step_results['E_uv [W]'] = E_uv
            
            # 다음 스텝 탱크 온도 계산
            if n < tN - 1:
                # nan인 경우 0으로 처리 (탱크 온도 계산에는 실제 열량이 필요)
                Q_ref_cond_val = result.get('Q_ref_cond [W]', np.nan)
                Q_ref_cond_val = np.nan_to_num(Q_ref_cond_val, nan=0.0)
                E_uv_val = step_results.get('E_uv [W]', 0.0)
                Q_tank_in = Q_ref_cond_val + E_uv_val
                Q_net = Q_tank_in - total_loss
                T_tank_w_K += (Q_net / self.C_tank) * self.dt
            
            results_data.append(step_results)
            
        results_df = pd.DataFrame(results_data)

        if result_save_csv_path:
            results_df.to_csv(result_save_csv_path, index=False)

        return results_df

