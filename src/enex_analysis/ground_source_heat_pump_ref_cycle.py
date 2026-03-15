"""
Ground Source Heat Pump module with detailed refrigerant cycle for Exergy Analysis.

Sub-systems (6):
    1. Indoor unit          – Room air ↔ refrigerant heat exchanger
    2. Compressor           – Refrigerant compression (State 1 → 2)
    3. Expansion valve      – Isenthalpic expansion (State 3 → 4)
    4. Refrigerant-to-water heat exchanger (RWHX) – Refrigerant ↔ ground loop water
    5. Ground heat exchanger (GHX) – Borehole heat transfer
    6. Ground              – Undisturbed soil thermal response

Refrigerant cycle state points:
    State 1: Compressor inlet  (superheated vapour after evaporation)
    State 2: Compressor outlet (superheated vapour after compression)
    State 3: Expansion valve inlet  (subcooled liquid after condensation)
    State 4: Expansion valve outlet (two-phase mixture after expansion)

Ground-loop fluid direction convention:
    f_in  → fluid entering the ground (from RWHX to borehole)
    f_out → fluid leaving the ground  (from borehole to RWHX)

T_evap / T_cond optimisation:
    Rather than fixing T_evap (cooling) or T_cond (heating) as a user
    parameter, the code minimises total electrical input (E_cmp + E_fan)
    subject to the delivered load constraint.  The indoor-unit coil is
    modelled with the ε-NTU method (C_r = 0 limit, i.e. condensing /
    evaporating refrigerant at constant temperature).

    Cooling: outer loop → T_f_in (RWHX↔GHX coupling)
             inner opt  → T_evap* = argmin E_cmp(T_evap) + E_fan(T_evap)
    Heating: outer loop → T_f_out (RWHX↔GHX coupling)
             inner opt  → T_cond* = argmin E_cmp(T_cond) + E_fan(T_cond)
"""

import math
from dataclasses import dataclass

import numpy as np
import pygfunction as gt
from CoolProp.CoolProp import PropsSI
from scipy.optimize import brentq, minimize_scalar

from . import calc_util as cu
from .components.fan import Fan
from .constants import c_a, c_w, k_w, rho_a, rho_w


# ---------------------------------------------------------------------------
# Helper: refrigerant flow exergy [W]
# ---------------------------------------------------------------------------
def _ref_flow_exergy(m_r: float, h: float, s: float,
                     h0: float, s0: float, T0_K: float) -> float:
    """Thermodynamic flow exergy of a refrigerant stream.
    X = m_r * [(h - h0) - T0 * (s - s0)]  [W]
    """
    return m_r * ((h - h0) - T0_K * (s - s0))


# ---------------------------------------------------------------------------
# Helper: solve for air-side capacity rate C_a = m_a*c_a [W/K]
#         from the ε-NTU equation for a phase-change HX (C_r = 0 limit).
#
#   Q = C_a * (1 - exp(-UA/C_a)) * |T_air_in - T_ref|
#   → solve for NTU = UA/C_a, then C_a = UA/NTU
# ---------------------------------------------------------------------------
def _solve_Ca_from_NTU(UA: float, Q: float, T_air_in_K: float,
                       T_ref_K: float) -> float:
    """Return C_a [W/K] = m_a*c_a for the indoor coil.

    Parameters
    ----------
    UA       : indoor-unit overall heat-transfer conductance [W/K]
    Q        : required heat-transfer rate (positive) [W]
    T_air_in_K : air inlet temperature [K]
    T_ref_K  : refrigerant saturation temperature [K] (T_evap cooling / T_cond heating)

    Returns
    -------
    C_a : air-side capacity rate [W/K], raises ValueError when infeasible.
    """
    dT = abs(T_air_in_K - T_ref_K)
    if dT < 1e-6:
        raise ValueError("T_air_in == T_ref: zero driving force in indoor coil.")

    capacity_limit = UA * dT   # maximum Q achievable as C_a → ∞ (NTU → 0)
    if Q >= capacity_limit:
        raise ValueError(
            f"Q ({Q:.1f} W) ≥ UA·ΔT ({capacity_limit:.1f} W): "
            f"indoor coil cannot deliver load at T_ref={T_ref_K - 273.15:.1f} °C."
        )

    # f(NTU) = (1 - exp(-NTU))/NTU  - Q/(UA*dT) = 0
    rhs = Q / (UA * dT)  # in (0, 1)

    def _f(ntu):
        if ntu < 1e-10:
            return 1.0 - rhs          # lim_{NTU→0} = 1
        return (1.0 - math.exp(-ntu)) / ntu - rhs

    # (1-exp(-NTU))/NTU is monotonically decreasing from 1 to 0
    # so there is exactly one root in (0, ∞)
    ntu_sol = brentq(_f, 1e-8, 500.0, xtol=1e-8, maxiter=200)
    return UA / ntu_sol   # C_a = UA / NTU


# ---------------------------------------------------------------------------
# Cooling mode
# ---------------------------------------------------------------------------
@dataclass
class GroundSourceHeatPump_cooling_RefCycle:
    """Ground source heat pump – cooling mode with detailed refrigerant cycle.

    The evaporating temperature T_evap is NOT a fixed input.  Instead it is
    determined each time step by minimising E_cmp + E_fan subject to the
    required cooling load, given the condensing temperature derived from the
    ground-loop coupling.

    Call ``system_update()`` each time step.
    """

    def __post_init__(self):
        # ------------------------------------------------------------------
        # Time
        # ------------------------------------------------------------------
        self.time = 0.0  # [h]

        # ------------------------------------------------------------------
        # Refrigerant
        # ------------------------------------------------------------------
        self.ref = "R32"
        self.SH = 5.0            # superheat at compressor inlet [K]
        self.SC = 5.0            # subcooling at expansion valve inlet [K]
        self.eta_is = 0.70       # compressor isentropic efficiency [-]
        self.eta_el = 0.95       # electro-mechanical efficiency [-]

        # ------------------------------------------------------------------
        # Operating conditions
        # ------------------------------------------------------------------
        self.T0 = 35.0           # dead-state temperature [°C]
        self.T_a_room = 22.0     # room air temperature [°C]
        self.T_g = 16.0          # undisturbed ground temperature [°C]

        # ------------------------------------------------------------------
        # Load
        # ------------------------------------------------------------------
        self.Q_r_iu = 6000.0     # cooling capacity target [W]

        # ------------------------------------------------------------------
        # Indoor unit heat exchanger
        # ------------------------------------------------------------------
        self.UA_iu = 2000.0      # indoor-unit coil conductance [W/K]
        self.T_evap_min = 0.0    # lower bound for T_evap to avoid icing [°C]
        self.fan_iu = Fan().fan1

        # ------------------------------------------------------------------
        # Borehole / ground parameters
        # ------------------------------------------------------------------
        self.D_b = 2.0
        self.H_b = 200.0
        self.r_b = 0.08
        self.R_b = 0.108

        # ------------------------------------------------------------------
        # Ground loop fluid
        # ------------------------------------------------------------------
        self.dV_f = 24.0         # [L/min]

        # ------------------------------------------------------------------
        # Ground thermal properties
        # ------------------------------------------------------------------
        self.k_g = 2.0
        self.c_g = 800.0
        self.rho_g = 2000.0

        # ------------------------------------------------------------------
        # Pump power
        # ------------------------------------------------------------------
        self.E_pmp = 200.0

        # ------------------------------------------------------------------
        # Temporal superposition (pygfunction)
        # ------------------------------------------------------------------
        self.Q_bh_history = [0.0]
        self.sim_hours = 8760
        self.dt_hours = 1
        self.dt_sec = self.dt_hours * 3600.0

        borehole = gt.boreholes.Borehole(
            H=self.H_b, D=self.D_b, r_b=self.r_b, x=0.0, y=0.0
        )
        n_steps = int(self.sim_hours / self.dt_hours)
        time_array = np.arange(1, n_steps + 1) * self.dt_sec
        alpha = self.k_g / (self.rho_g * self.c_g)
        self.g_func_list = gt.gfunction.gFunction(
            [borehole], alpha, time=time_array
        ).gFunc

    # -----------------------------------------------------------------------
    def system_update(self):
        """Advance simulation by one time step.

        Algorithm overview
        ------------------
        A. Unit conversions & dead-state refrigerant properties
        B. Temporal superposition – pre-compute history term
        C. Outer loop: converge T_f_in (RWHX ↔ GHX coupling)
           C1. T_cond = T_f_in + 5 K  (RWHX approach-temperature)
           C2. Inner optimisation: T_evap* = argmin (E_cmp + E_fan)
               C2a. For each T_evap candidate:
                    - ε-NTU → solve for required airflow dV_iu
                    - Fan curve → E_fan
                    - CoolProp cycle → State 1–4, m_r, E_cmp
               C2b. minimize_scalar over T_evap ∈ [T_evap_min, T_a_room-1]
           C3. Borehole superposition → T_b → T_f → T_f_in (new)
           C4. Convergence check
        D. Store history, advance timer
        E. Exergy calculations for all 6 sub-systems
        """

        # ------------------------------------------------------------------
        # A. Unit conversions & dead-state
        # ------------------------------------------------------------------
        dV_f_m3s = self.dV_f * cu.s2m * cu.L2m3

        T0_K     = cu.C2K(self.T0)
        T_g_K    = cu.C2K(self.T_g)
        T_room_K = cu.C2K(self.T_a_room)
        T_a_iu_in_K = T_room_K

        p0 = 101325.0
        h0 = PropsSI("H", "T", T0_K, "P", p0, self.ref)
        s0 = PropsSI("S", "T", T0_K, "P", p0, self.ref)

        # ------------------------------------------------------------------
        # B. Temporal superposition – historical T_b effect
        # ------------------------------------------------------------------
        T_b_history_effect = 0.0
        for i in range(1, len(self.Q_bh_history)):
            delta_Q = self.Q_bh_history[i] - self.Q_bh_history[i - 1]
            elapsed_steps = len(self.Q_bh_history) - i
            idx = elapsed_steps
            g_val = (self.g_func_list[idx]
                     if idx < len(self.g_func_list)
                     else self.g_func_list[-1])
            T_b_history_effect += delta_Q * g_val

        # ------------------------------------------------------------------
        # C2 helper: evaluate one refrigerant cycle given T_evap_C, T_cond_K
        # ------------------------------------------------------------------
        def _cycle(T_evap_C: float, T_cond_K: float):
            """Return (E_cmp, E_fan, m_r, dV_iu, p1,h1,s1,T1_K,
                                              p2,h2,s2,T2_K,
                                              p3,h3,s3,T3_K,
                                              p4,h4,s4,T4_K)
            Raises ValueError if infeasible.
            """
            T_evap_K = cu.C2K(T_evap_C)
            if T_evap_K >= T_cond_K - 1.0:
                raise ValueError("T_evap >= T_cond: infeasible cycle.")

            # State 1: compressor inlet (superheated)
            p1 = PropsSI("P", "T", T_evap_K, "Q", 1, self.ref)
            T1_K = T_evap_K + self.SH
            h1 = PropsSI("H", "T", T1_K, "P", p1, self.ref)
            s1 = PropsSI("S", "T", T1_K, "P", p1, self.ref)

            # State 2: compressor outlet
            p2 = PropsSI("P", "T", T_cond_K, "Q", 1, self.ref)
            h2s = PropsSI("H", "P", p2, "S", s1, self.ref)
            h2 = h1 + (h2s - h1) / self.eta_is
            T2_K = PropsSI("T", "P", p2, "H", h2, self.ref)
            s2 = PropsSI("S", "P", p2, "H", h2, self.ref)

            # State 3: expansion valve inlet (subcooled)
            p3 = p2
            T3_K = T_cond_K - self.SC
            h3 = PropsSI("H", "T", T3_K, "P", p3, self.ref)
            s3 = PropsSI("S", "T", T3_K, "P", p3, self.ref)

            # State 4: expansion valve outlet (isenthalpic)
            p4 = p1
            h4 = h3
            T4_K = PropsSI("T", "P", p4, "H", h4, self.ref)
            s4 = PropsSI("S", "P", p4, "H", h4, self.ref)

            # Mass flow rate from indoor load: Q = m_r * (h1 - h4)
            dh_evap = h1 - h4
            if dh_evap < 1.0:
                raise ValueError("h1 - h4 too small.")
            m_r = self.Q_r_iu / dh_evap

            # Compressor electrical power
            E_cmp = m_r * (h2 - h1) / self.eta_el

            # Indoor unit: ε-NTU to find airflow dV_iu
            # Cooling: refrigerant (evaporator) is T_evap, air enters at T_room
            # Feasibility: Q_r_iu < UA_iu * (T_room - T_evap)
            C_a = _solve_Ca_from_NTU(
                self.UA_iu, self.Q_r_iu, T_a_iu_in_K, T_evap_K
            )
            m_a = C_a / c_a
            dV_iu = m_a / rho_a
            E_fan = Fan().get_power(self.fan_iu, dV_iu)

            return (E_cmp, E_fan, m_r, dV_iu,
                    p1, h1, s1, T1_K,
                    p2, h2, s2, T2_K,
                    p3, h3, s3, T3_K,
                    p4, h4, s4, T4_K)

        # ------------------------------------------------------------------
        # C. Outer loop: T_f_in convergence
        # ------------------------------------------------------------------
        _LARGE = 1e12
        max_outer = 30
        tol_outer = 1e-3
        T_f_in_K = T_g_K

        # variables set inside loop, needed after
        T_evap_opt_C = None
        _cycle_result = None

        for _outer in range(max_outer):

            # --------------------------------------------------------------
            # C1. Condensing temperature from RWHX approach assumption
            # --------------------------------------------------------------
            T_cond_K = T_f_in_K + 5.0

            # --------------------------------------------------------------
            # C2. Inner optimisation: find T_evap* minimising E_cmp + E_fan
            #
            #   Feasibility bound (upper limit on T_evap):
            #     Q_r_iu < UA_iu * (T_room - T_evap)
            #     → T_evap < T_room - Q_r_iu/UA_iu
            #   Safety margin of 0.5 K applied.
            # --------------------------------------------------------------
            T_evap_ub_C = (T_a_iu_in_K - self.Q_r_iu / self.UA_iu
                           - 0.5) - 273.15
            # Also cap below T_cond (cycle must be feasible)
            T_evap_ub_C = min(T_evap_ub_C, T_cond_K - 1.0 - 273.15)
            T_evap_lb_C = self.T_evap_min

            if T_evap_lb_C >= T_evap_ub_C:
                # Window collapsed – fall back to midpoint
                T_evap_opt_C = 0.5 * (T_evap_lb_C + T_evap_ub_C)
            else:
                def _objective(T_evap_C: float) -> float:
                    try:
                        E_cmp, E_fan, *_ = _cycle(T_evap_C, T_cond_K)
                        return E_cmp + E_fan
                    except (ValueError, Exception):
                        return _LARGE

                result = minimize_scalar(
                    _objective,
                    bounds=(T_evap_lb_C, T_evap_ub_C),
                    method="bounded",
                    options={"xatol": 1e-3, "maxiter": 100},
                )
                T_evap_opt_C = result.x

            try:
                _cycle_result = _cycle(T_evap_opt_C, T_cond_K)
            except ValueError:
                # If still infeasible, step inward
                T_evap_opt_C = T_evap_lb_C + 0.1
                _cycle_result = _cycle(T_evap_opt_C, T_cond_K)

            (E_cmp, E_fan_iu, m_r, dV_iu,
             p1, h1, s1, T1_K,
             p2, h2, s2, T2_K,
             p3, h3, s3, T3_K,
             p4, h4, s4, T4_K) = _cycle_result

            # --------------------------------------------------------------
            # C3. Ground-loop heat balance
            # --------------------------------------------------------------
            Q_r_rwhx = m_r * (h2 - h3)                    # heat rejected to water
            Q_bh = (Q_r_rwhx + self.E_pmp) / self.H_b     # [W/m] > 0 in cooling

            g_i = self.g_func_list[0]
            T_b_K = (T_g_K
                     + T_b_history_effect
                     + (Q_bh - self.Q_bh_history[-1]) * g_i)

            T_f_K = T_b_K + Q_bh * self.R_b

            dT_f = Q_bh * self.H_b / (2.0 * c_w * rho_w * dV_f_m3s)
            T_f_in_new_K = T_f_K + dT_f   # hot end  (entering borehole)
            T_f_out_K    = T_f_K - dT_f   # cool end (leaving borehole)

            # --------------------------------------------------------------
            # C4. Convergence on T_f_in
            # --------------------------------------------------------------
            if abs(T_f_in_new_K - T_f_in_K) < tol_outer:
                T_f_in_K = T_f_in_new_K
                break
            T_f_in_K = T_f_in_new_K

        # ------------------------------------------------------------------
        # D. Store history & advance timer
        # ------------------------------------------------------------------
        self.Q_bh_history.append(Q_bh)
        self.time += self.dt_hours

        # Convenience attributes
        self.T_evap_opt_C = T_evap_opt_C
        self.m_r = m_r
        self.E_cmp = E_cmp
        self.E_fan_iu = E_fan_iu
        self.dV_iu = dV_iu
        self.Q_r_rwhx = Q_r_rwhx
        self.Q_bh = Q_bh
        self.COP = self.Q_r_iu / E_cmp

        self.p1, self.T1_K, self.h1, self.s1 = p1, T1_K, h1, s1
        self.p2, self.T2_K, self.h2, self.s2 = p2, T2_K, h2, s2
        self.p3, self.T3_K, self.h3, self.s3 = p3, T3_K, h3, s3
        self.p4, self.T4_K, self.h4, self.s4 = p4, T4_K, h4, s4

        self.T_b_K    = T_b_K
        self.T_f_K    = T_f_K
        self.T_f_in_K  = T_f_in_K
        self.T_f_out_K = T_f_out_K

        # ------------------------------------------------------------------
        # E. Exergy calculations
        # ------------------------------------------------------------------

        # E1. Refrigerant flow exergies
        X1 = _ref_flow_exergy(m_r, h1, s1, h0, s0, T0_K)
        X2 = _ref_flow_exergy(m_r, h2, s2, h0, s0, T0_K)
        X3 = _ref_flow_exergy(m_r, h3, s3, h0, s0, T0_K)
        X4 = _ref_flow_exergy(m_r, h4, s4, h0, s0, T0_K)

        # E2. Air exergy
        T_a_iu_out_K = T_a_iu_in_K - self.Q_r_iu / (c_a * rho_a * dV_iu)

        def _air_exergy(T_K):
            return (c_a * rho_a * dV_iu *
                    ((T_K - T0_K) - T0_K * math.log(T_K / T0_K)))

        X_a_iu_in  = _air_exergy(T_a_iu_in_K)
        X_a_iu_out = _air_exergy(T_a_iu_out_K)

        # E3. Ground-loop fluid exergies
        def _fluid_exergy(T_K):
            return (c_w * rho_w * dV_f_m3s *
                    ((T_K - T0_K) - T0_K * math.log(T_K / T0_K)))

        X_f_in  = _fluid_exergy(T_f_in_K)   # entering borehole (from RWHX, hot)
        X_f_out = _fluid_exergy(T_f_out_K)  # leaving borehole  (to RWHX, cool)

        # E4. Ground / borehole
        X_g = (1.0 - T0_K / T_g_K) * (-Q_bh * self.H_b)
        X_b = (1.0 - T0_K / T_b_K) * (-Q_bh * self.H_b)

        # E5. Sub-system balances
        # (1) Indoor unit – cooling: ref evaporates (State 4→1)
        X_in_iu  = self.E_fan_iu + X4 + X_a_iu_in
        X_out_iu = X1 + X_a_iu_out
        X_c_iu   = X_in_iu - X_out_iu

        # (2) Compressor – State 1→2
        X_in_cmp  = E_cmp + X1
        X_out_cmp = X2
        X_c_cmp   = X_in_cmp - X_out_cmp

        # (3) Expansion valve – State 3→4 (isenthalpic)
        X_in_exp  = X3
        X_out_exp = X4
        X_c_exp   = X_in_exp - X_out_exp

        # (4) RWHX – cooling: condenser (State 2→3)
        #   cool fluid from borehole (f_out) → heated fluid to borehole (f_in)
        X_in_rwhx  = X2 + X_f_out
        X_out_rwhx = X3 + X_f_in
        X_c_rwhx   = X_in_rwhx - X_out_rwhx

        # (5) GHX – borehole wall exergy X_b as thermal boundary input
        X_in_ghx  = self.E_pmp + X_f_in + X_b
        X_out_ghx = X_f_out
        X_c_ghx   = X_in_ghx - X_out_ghx

        # (6) Ground
        X_in_g  = X_g
        X_out_g = X_b
        X_c_g   = X_in_g - X_out_g

        # E6. System exergy efficiency
        self.X_eff = (X_a_iu_in - X_a_iu_out) / (E_cmp + self.E_fan_iu + self.E_pmp)

        # Store
        self.X1, self.X2, self.X3, self.X4 = X1, X2, X3, X4
        self.X_a_iu_in, self.X_a_iu_out = X_a_iu_in, X_a_iu_out
        self.X_f_in, self.X_f_out = X_f_in, X_f_out
        self.X_g, self.X_b = X_g, X_b

        self.exergy_bal = {
            "indoor unit": {
                "in":  {"E_fan_iu": self.E_fan_iu, "X4": X4, "X_a_iu_in": X_a_iu_in},
                "out": {"X1": X1, "X_a_iu_out": X_a_iu_out},
                "con": {"X_c_iu": X_c_iu},
            },
            "compressor": {
                "in":  {"E_cmp": E_cmp, "X1": X1},
                "out": {"X2": X2},
                "con": {"X_c_cmp": X_c_cmp},
            },
            "expansion valve": {
                "in":  {"X3": X3},
                "out": {"X4": X4},
                "con": {"X_c_exp": X_c_exp},
            },
            "refrigerant-to-water heat exchanger": {
                "in":  {"X2": X2, "X_f_out": X_f_out},
                "out": {"X3": X3, "X_f_in": X_f_in},
                "con": {"X_c_rwhx": X_c_rwhx},
            },
            "ground heat exchanger": {
                "in":  {"E_pmp": self.E_pmp, "X_f_in": X_f_in, "X_b": X_b},
                "out": {"X_f_out": X_f_out},
                "con": {"X_c_ghx": X_c_ghx},
            },
            "ground": {
                "in":  {"X_in_g": X_in_g},
                "out": {"X_out_g": X_out_g},
                "con": {"X_c_g": X_c_g},
            },
        }


# ---------------------------------------------------------------------------
# Heating mode
# ---------------------------------------------------------------------------
@dataclass
class GroundSourceHeatPump_heating_RefCycle:
    """Ground source heat pump – heating mode with detailed refrigerant cycle.

    In heating mode:
        Indoor unit  → condenser  (State 2→3, refrigerant heats room air)
        RWHX         → evaporator (State 4→1, refrigerant absorbs from ground)

    T_evap is derived from the outer-loop variable T_f_out (borehole outlet):
        T_evap = T_f_out - 5 K

    T_cond is optimised in an inner loop to minimise E_cmp + E_fan for the
    given heating load.

    Call ``system_update()`` each time step.
    """

    def __post_init__(self):
        self.time = 0.0

        # Refrigerant
        self.ref = "R32"
        self.SH = 5.0
        self.SC = 5.0
        self.eta_is = 0.70
        self.eta_el = 0.95

        # Operating conditions
        self.T0 = 0.0
        self.T_a_room = 20.0
        self.T_g = 15.0

        # Load
        self.Q_r_iu = 8000.0     # heating capacity [W]

        # Indoor unit
        self.UA_iu = 2000.0      # indoor-unit coil conductance [W/K]
        self.T_cond_max = 65.0   # upper bound for T_cond [°C]
        self.fan_iu = Fan().fan1

        # Borehole
        self.D_b = 2.0
        self.H_b = 200.0
        self.r_b = 0.08
        self.R_b = 0.108

        # Ground loop fluid
        self.dV_f = 24.0

        # Ground thermal properties
        self.k_g = 2.0
        self.c_g = 800.0
        self.rho_g = 2000.0

        # Pump power
        self.E_pmp = 200.0

        # Temporal superposition
        self.Q_bh_history = [0.0]
        self.sim_hours = 8760
        self.dt_hours = 1
        self.dt_sec = self.dt_hours * 3600.0

        borehole = gt.boreholes.Borehole(
            H=self.H_b, D=self.D_b, r_b=self.r_b, x=0.0, y=0.0
        )
        n_steps = int(self.sim_hours / self.dt_hours)
        time_array = np.arange(1, n_steps + 1) * self.dt_sec
        alpha = self.k_g / (self.rho_g * self.c_g)
        self.g_func_list = gt.gfunction.gFunction(
            [borehole], alpha, time=time_array
        ).gFunc

    # -----------------------------------------------------------------------
    def system_update(self):
        """Advance simulation by one time step.

        Algorithm overview
        ------------------
        A. Unit conversions & dead-state refrigerant properties
        B. Temporal superposition – pre-compute history term
        C. Outer loop: converge T_f_out (RWHX ↔ GHX coupling)
           C1. T_evap = T_f_out - 5 K  (RWHX approach-temperature)
           C2. Inner optimisation: T_cond* = argmin (E_cmp + E_fan)
               C2a. For each T_cond candidate:
                    - ε-NTU → solve for required airflow dV_iu (heating: condenser)
                    - Fan curve → E_fan
                    - CoolProp cycle → State 1–4, m_r, E_cmp
               C2b. minimize_scalar over T_cond ∈ [T_a_room+1, T_cond_max]
           C3. Borehole superposition → T_b → T_f → T_f_out (new)
           C4. Convergence check
        D. Store history, advance timer
        E. Exergy calculations for all 6 sub-systems
        """

        # ------------------------------------------------------------------
        # A. Unit conversions & dead-state
        # ------------------------------------------------------------------
        dV_f_m3s = self.dV_f * cu.s2m * cu.L2m3

        T0_K     = cu.C2K(self.T0)
        T_g_K    = cu.C2K(self.T_g)
        T_room_K = cu.C2K(self.T_a_room)
        T_a_iu_in_K = T_room_K

        p0 = 101325.0
        h0 = PropsSI("H", "T", T0_K, "P", p0, self.ref)
        s0 = PropsSI("S", "T", T0_K, "P", p0, self.ref)

        # ------------------------------------------------------------------
        # B. Temporal superposition
        # ------------------------------------------------------------------
        T_b_history_effect = 0.0
        for i in range(1, len(self.Q_bh_history)):
            delta_Q = self.Q_bh_history[i] - self.Q_bh_history[i - 1]
            elapsed_steps = len(self.Q_bh_history) - i
            idx = elapsed_steps
            g_val = (self.g_func_list[idx]
                     if idx < len(self.g_func_list)
                     else self.g_func_list[-1])
            T_b_history_effect += delta_Q * g_val

        # ------------------------------------------------------------------
        # C2 helper: evaluate one refrigerant cycle given T_evap_K, T_cond_C
        # ------------------------------------------------------------------
        def _cycle(T_evap_K: float, T_cond_C: float):
            """Return (E_cmp, E_fan, m_r, dV_iu, p1…s4) or raises ValueError."""
            T_cond_K = cu.C2K(T_cond_C)
            if T_evap_K >= T_cond_K - 1.0:
                raise ValueError("T_evap >= T_cond.")

            # State 1: compressor inlet
            p1 = PropsSI("P", "T", T_evap_K, "Q", 1, self.ref)
            T1_K = T_evap_K + self.SH
            h1 = PropsSI("H", "T", T1_K, "P", p1, self.ref)
            s1 = PropsSI("S", "T", T1_K, "P", p1, self.ref)

            # State 2: compressor outlet
            p2 = PropsSI("P", "T", T_cond_K, "Q", 1, self.ref)
            h2s = PropsSI("H", "P", p2, "S", s1, self.ref)
            h2 = h1 + (h2s - h1) / self.eta_is
            T2_K = PropsSI("T", "P", p2, "H", h2, self.ref)
            s2 = PropsSI("S", "P", p2, "H", h2, self.ref)

            # State 3: expansion valve inlet
            p3 = p2
            T3_K = T_cond_K - self.SC
            h3 = PropsSI("H", "T", T3_K, "P", p3, self.ref)
            s3 = PropsSI("S", "T", T3_K, "P", p3, self.ref)

            # State 4: expansion valve outlet (isenthalpic)
            p4 = p1
            h4 = h3
            T4_K = PropsSI("T", "P", p4, "H", h4, self.ref)
            s4 = PropsSI("S", "P", p4, "H", h4, self.ref)

            # Mass flow from heating load: Q_r_iu = m_r * (h2 - h3)
            dh_cond = h2 - h3
            if dh_cond < 1.0:
                raise ValueError("h2 - h3 too small.")
            m_r = self.Q_r_iu / dh_cond

            E_cmp = m_r * (h2 - h1) / self.eta_el

            # Indoor unit: ε-NTU for heating condenser
            # Refrigerant condenses at T_cond_K; air enters at T_room_K
            C_a = _solve_Ca_from_NTU(
                self.UA_iu, self.Q_r_iu, T_a_iu_in_K, T_cond_K
            )
            m_a = C_a / c_a
            dV_iu = m_a / rho_a
            E_fan = Fan().get_power(self.fan_iu, dV_iu)

            return (E_cmp, E_fan, m_r, dV_iu,
                    p1, h1, s1, T1_K,
                    p2, h2, s2, T2_K,
                    p3, h3, s3, T3_K,
                    p4, h4, s4, T4_K)

        # ------------------------------------------------------------------
        # C. Outer loop: T_f_out convergence
        # ------------------------------------------------------------------
        _LARGE = 1e12
        max_outer = 30
        tol_outer = 1e-3
        T_f_out_K = T_g_K   # initial guess (borehole outlet to RWHX)

        T_cond_opt_C = None
        _cycle_result = None

        for _outer in range(max_outer):

            # ---------------------------------------------------------------
            # C1. Evaporating temperature from RWHX approach assumption
            # ---------------------------------------------------------------
            T_evap_K = T_f_out_K - 5.0

            # ---------------------------------------------------------------
            # C2. Inner optimisation: T_cond* minimising E_cmp + E_fan
            #
            #   Feasibility bound (lower limit on T_cond):
            #     Q_r_iu < UA_iu * (T_cond - T_room)
            #     → T_cond > T_room + Q_r_iu/UA_iu + 0.5
            # ---------------------------------------------------------------
            T_cond_lb_C = (T_a_iu_in_K + self.Q_r_iu / self.UA_iu
                           + 0.5) - 273.15
            # Also ensure T_cond > T_evap + 1
            T_cond_lb_C = max(T_cond_lb_C, T_evap_K + 1.0 - 273.15)
            T_cond_ub_C = self.T_cond_max

            if T_cond_lb_C >= T_cond_ub_C:
                T_cond_opt_C = 0.5 * (T_cond_lb_C + T_cond_ub_C)
            else:
                def _objective(T_cond_C: float) -> float:
                    try:
                        E_cmp, E_fan, *_ = _cycle(T_evap_K, T_cond_C)
                        return E_cmp + E_fan
                    except (ValueError, Exception):
                        return _LARGE

                result = minimize_scalar(
                    _objective,
                    bounds=(T_cond_lb_C, T_cond_ub_C),
                    method="bounded",
                    options={"xatol": 1e-3, "maxiter": 100},
                )
                T_cond_opt_C = result.x

            try:
                _cycle_result = _cycle(T_evap_K, T_cond_opt_C)
            except ValueError:
                T_cond_opt_C = T_cond_lb_C + 0.1
                _cycle_result = _cycle(T_evap_K, T_cond_opt_C)

            (E_cmp, E_fan_iu, m_r, dV_iu,
             p1, h1, s1, T1_K,
             p2, h2, s2, T2_K,
             p3, h3, s3, T3_K,
             p4, h4, s4, T4_K) = _cycle_result

            # ---------------------------------------------------------------
            # C3. Ground-loop heat balance (heating: extraction → Q_bh < 0)
            # ---------------------------------------------------------------
            Q_r_rwhx = m_r * (h1 - h4)                        # absorbed by ref [W]
            Q_bh = -(Q_r_rwhx - self.E_pmp) / self.H_b       # [W/m] < 0 extraction

            g_i = self.g_func_list[0]
            T_b_K = (T_g_K
                     + T_b_history_effect
                     + (Q_bh - self.Q_bh_history[-1]) * g_i)

            T_f_K = T_b_K + Q_bh * self.R_b

            dT_f = Q_bh * self.H_b / (2.0 * c_w * rho_w * dV_f_m3s)
            T_f_out_new_K = T_f_K - dT_f   # warm end (leaving borehole to RWHX)
            T_f_in_K      = T_f_K + dT_f   # cool end (entering borehole from RWHX)

            # ---------------------------------------------------------------
            # C4. Convergence on T_f_out
            # ---------------------------------------------------------------
            if abs(T_f_out_new_K - T_f_out_K) < tol_outer:
                T_f_out_K = T_f_out_new_K
                break
            T_f_out_K = T_f_out_new_K

        # ------------------------------------------------------------------
        # D. Store history & advance timer
        # ------------------------------------------------------------------
        self.Q_bh_history.append(Q_bh)
        self.time += self.dt_hours

        self.T_cond_opt_C = T_cond_opt_C
        self.T_evap_K     = T_evap_K
        self.m_r = m_r
        self.E_cmp = E_cmp
        self.E_fan_iu = E_fan_iu
        self.dV_iu = dV_iu
        self.Q_r_rwhx = Q_r_rwhx
        self.Q_bh = Q_bh
        self.COP = self.Q_r_iu / E_cmp

        self.p1, self.T1_K, self.h1, self.s1 = p1, T1_K, h1, s1
        self.p2, self.T2_K, self.h2, self.s2 = p2, T2_K, h2, s2
        self.p3, self.T3_K, self.h3, self.s3 = p3, T3_K, h3, s3
        self.p4, self.T4_K, self.h4, self.s4 = p4, T4_K, h4, s4

        self.T_b_K    = T_b_K
        self.T_f_K    = T_f_K
        self.T_f_in_K  = T_f_in_K    # entering borehole (from RWHX, cool)
        self.T_f_out_K = T_f_out_K   # leaving borehole  (to RWHX, warm)

        # ------------------------------------------------------------------
        # E. Exergy calculations
        # ------------------------------------------------------------------

        # E1. Refrigerant
        X1 = _ref_flow_exergy(m_r, h1, s1, h0, s0, T0_K)
        X2 = _ref_flow_exergy(m_r, h2, s2, h0, s0, T0_K)
        X3 = _ref_flow_exergy(m_r, h3, s3, h0, s0, T0_K)
        X4 = _ref_flow_exergy(m_r, h4, s4, h0, s0, T0_K)

        # E2. Air exergy (heating: outlet warmer than inlet)
        T_a_iu_out_K = T_a_iu_in_K + self.Q_r_iu / (c_a * rho_a * dV_iu)

        def _air_exergy(T_K):
            return (c_a * rho_a * dV_iu *
                    ((T_K - T0_K) - T0_K * math.log(T_K / T0_K)))

        X_a_iu_in  = _air_exergy(T_a_iu_in_K)
        X_a_iu_out = _air_exergy(T_a_iu_out_K)

        # E3. Ground-loop fluid
        def _fluid_exergy(T_K):
            return (c_w * rho_w * dV_f_m3s *
                    ((T_K - T0_K) - T0_K * math.log(T_K / T0_K)))

        X_f_in  = _fluid_exergy(T_f_in_K)    # entering borehole (cool, from RWHX)
        X_f_out = _fluid_exergy(T_f_out_K)   # leaving borehole  (warm, to RWHX)

        # E4. Ground / borehole
        X_g = (1.0 - T0_K / T_g_K) * (Q_bh * self.H_b)
        X_b = (1.0 - T0_K / T_b_K) * (Q_bh * self.H_b)

        # E5. Sub-system balances
        # (1) Indoor unit – heating: condenser (State 2→3)
        X_in_iu  = self.E_fan_iu + X2 + X_a_iu_in
        X_out_iu = X3 + X_a_iu_out
        X_c_iu   = X_in_iu - X_out_iu

        # (2) Compressor
        X_in_cmp  = E_cmp + X1
        X_out_cmp = X2
        X_c_cmp   = X_in_cmp - X_out_cmp

        # (3) Expansion valve
        X_in_exp  = X3
        X_out_exp = X4
        X_c_exp   = X_in_exp - X_out_exp

        # (4) RWHX – heating: evaporator (State 4→1)
        #   warm fluid from borehole (f_out) enters; cooled fluid (f_in) leaves
        X_in_rwhx  = X4 + X_f_out
        X_out_rwhx = X1 + X_f_in
        X_c_rwhx   = X_in_rwhx - X_out_rwhx

        # (5) GHX
        X_in_ghx  = self.E_pmp + X_f_in + X_b
        X_out_ghx = X_f_out
        X_c_ghx   = X_in_ghx - X_out_ghx

        # (6) Ground
        X_in_g  = X_g
        X_out_g = X_b
        X_c_g   = X_in_g - X_out_g

        # E6. System exergy efficiency
        self.X_eff = (X_a_iu_out - X_a_iu_in) / (E_cmp + self.E_fan_iu + self.E_pmp)

        self.X1, self.X2, self.X3, self.X4 = X1, X2, X3, X4
        self.X_a_iu_in, self.X_a_iu_out = X_a_iu_in, X_a_iu_out
        self.X_f_in, self.X_f_out = X_f_in, X_f_out
        self.X_g, self.X_b = X_g, X_b

        self.exergy_bal = {
            "indoor unit": {
                "in":  {"E_fan_iu": self.E_fan_iu, "X2": X2, "X_a_iu_in": X_a_iu_in},
                "out": {"X3": X3, "X_a_iu_out": X_a_iu_out},
                "con": {"X_c_iu": X_c_iu},
            },
            "compressor": {
                "in":  {"E_cmp": E_cmp, "X1": X1},
                "out": {"X2": X2},
                "con": {"X_c_cmp": X_c_cmp},
            },
            "expansion valve": {
                "in":  {"X3": X3},
                "out": {"X4": X4},
                "con": {"X_c_exp": X_c_exp},
            },
            "refrigerant-to-water heat exchanger": {
                "in":  {"X4": X4, "X_f_out": X_f_out},
                "out": {"X1": X1, "X_f_in": X_f_in},
                "con": {"X_c_rwhx": X_c_rwhx},
            },
            "ground heat exchanger": {
                "in":  {"E_pmp": self.E_pmp, "X_f_in": X_f_in, "X_b": X_b},
                "out": {"X_f_out": X_f_out},
                "con": {"X_c_ghx": X_c_ghx},
            },
            "ground": {
                "in":  {"X_in_g": X_in_g},
                "out": {"X_out_g": X_out_g},
                "con": {"X_c_g": X_c_g},
            },
        }
