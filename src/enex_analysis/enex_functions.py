"""
Utility functions for energy, entropy, and exergy analysis.

This module contains helper functions organized into the following categories:

1. Friction and Flow Functions
   - darcy_friction_factor: Calculate Darcy friction factor
   - calc_Orifice_flow_coefficient: Calculate orifice flow coefficient
   - calc_boussinessq_mixing_flow: Calculate mixing flow based on Boussinesq approximation

2. Heat Transfer Functions
   - calc_h_vertical_plate: Natural convection heat transfer coefficient
   - calc_UA_tank_arr: Tank heat loss UA calculation
   - calc_lmtd_*: Log mean temperature difference calculations
   - calc_UA_from_dV_fan: Heat transfer coefficient from fan flow rate

3. Curve Fitting Functions
   - linear_function, quadratic_function, cubic_function, quartic_function

4. Exergy and Entropy Functions
   - generate_entropy_exergy_term: Calculate entropy and exergy terms
   - calc_exergy_flow: Calculate exergy flow rate due to material flow

5. G-function Calculations (Ground Source Heat Pumps)
   - f, chi, G_FLS: Helper functions for g-function calculation

6. TDMA Solver Functions
   - TDMA: Solve tri-diagonal matrix system
   - _add_loop_advection_terms: Add forced convection terms to TDMA coefficients

7. Heat Pump Cycle Functions
   - calculate_ASHP_*_COP: Air source heat pump COP calculations
   - calculate_GSHP_COP: Ground source heat pump COP calculation
   - calc_ref_state: Calculate refrigerant cycle states (with superheating/subcooling support)
   - find_ref_loop_optimal_operation: Find optimal operation point
8. Tank Functions
   - update_tank_temperature: Update tank temperature based on energy balance

9. Schedule Functions
   - _build_dhw_usage_ratio: Build schedule ratio array

10. Balance Printing Utilities
    - print_balance: Print energy/entropy/exergy balance
"""

import math

import numpy as np
from scipy.optimize import root_scalar

from . import calc_util as cu
from .constants import c_a, c_w, rho_a, rho_w
from .cop import (
    calc_ASHP_cooling_COP as calc_ASHP_cooling_COP,
)
from .cop import (
    calc_ASHP_heating_COP as calc_ASHP_heating_COP,
)
from .cop import (
    calc_GSHP_COP as calc_GSHP_COP,
)
from .dhw import (
    build_dhw_usage_ratio,
    calc_cold_water_temp,
    calc_total_water_use_from_schedule,
    make_dhw_schedule_from_Annex_42_profile,
)
from .dynamic_context import (
    determine_heat_source_on_off,
    determine_tank_refill_flow,
    tank_mass_energy_residual,
)
from .g_function import (
    G_FLS as G_FLS,
)
from .g_function import (
    air_dynamic_viscosity as air_dynamic_viscosity,
)
from .g_function import (
    air_prandtl_number as air_prandtl_number,
)
from .g_function import (
    chi as chi,
)
from .g_function import (
    f as f,
)
from .heat_transfer import (
    calc_h_vertical_plate,
    calc_LMTD_counter_flow,
    calc_LMTD_parallel_flow,
    calc_simple_tank_UA,
    calc_UA_tank_arr,
    darcy_friction_factor,
)
from .hx_fan import (
    calc_fan_power_from_dV_fan as calc_fan_power_from_dV_fan,
)
from .hx_fan import (
    calc_UA_from_dV_fan as calc_UA_from_dV_fan,
)
from .refrigerant import (
    calc_ref_state,
    create_lmtd_constraints,
    find_ref_loop_optimal_operation,
)
from .tdma import (
    TDMA as TDMA,
)
from .tdma import (
    _add_loop_advection_terms as _add_loop_advection_terms,
)
from .thermodynamics import (
    calc_energy_flow,
    calc_exergy_flow,
    calc_refrigerant_exergy,
    convert_electricity_to_exergy,
    generate_entropy_exergy_term,
)
from .uv_treatment import (
    calc_uv_exposure_time as calc_uv_exposure_time,
)
from .uv_treatment import (
    calc_uv_lamp_power as calc_uv_lamp_power,
)
from .uv_treatment import (
    get_uv_params_from_turbidity as get_uv_params_from_turbidity,
)
from .visualization import (
    plot_ph_diagram,
    plot_th_diagram,
    plot_ts_diagram,
    print_simulation_summary,
)
from .weather import (
    decompose_ghi_to_poa,
    load_kma_solar_csv,
    load_kma_T0_sol_hourly_csv,
)

try:
    import dartwork_mpl as dm
except ImportError:
    # dartwork_mpl이 없는 경우를 대비한 fallback
    dm = None


def linear_function(x, a, b):
    """Linear function: y = a*x + b"""
    return a * x + b


def quadratic_function(x, a, b, c):
    """Quadratic function: y = a*x² + b*x + c"""
    return a * x**2 + b * x + c


def cubic_function(x, a, b, c, d):
    """Cubic function: y = a*x³ + b*x² + c*x + d"""
    return a * x**3 + b * x**2 + c * x + d


def quartic_function(x, a, b, c, d, e):
    """Quartic function: y = a*x⁴ + b*x³ + c*x² + d*x + e"""
    return a * x**4 + b * x**3 + c * x**2 + d * x + e


def print_balance(balance, decimal=2):
    """
    Print energy, entropy, or exergy balance dictionary in a formatted way.

    This function prints balance information for subsystems, categorizing entries
    into in, out, consumed, and generated categories.

    Parameters
    ----------
    balance : dict
        Dictionary containing balance information for subsystems.
        Structure: {subsystem_name: {category: {symbol: value}}}
        Categories: 'in', 'out', 'con' (consumed), 'gen' (generated)
    decimal : int, optional
        Number of decimal places for output (default: 2)

    Returns
    -------
    None
        Only prints output

    Example
    -------
    >>> balance = {
    ...     "hot water tank": {
    ...         "in": {"E_heater": 5000.0},
    ...         "out": {"Q_w_tank": 4500.0, "Q_l_tank": 400.0},
    ...         "con": {"X_c_tank": 100.0}
    ...     }
    ... }
    >>> print_balance(balance)
    """
    total_length = 50

    balance_type = "energy"
    unit = "[W]"

    # Determine balance type and unit from dictionary structure
    for _subsystem, category_dict in balance.items():
        for category, _terms in category_dict.items():
            if "gen" in category:
                balance_type = "entropy"
                unit = "[W/K]"
            elif "con" in category:
                balance_type = "exergy"

    # Print balance for each subsystem
    for subsystem, category_dict in balance.items():
        text = f"{subsystem.upper()} {balance_type.upper()} BALANCE:"
        print(f"\n\n{text}" + "=" * (total_length - len(text)))

        for category, terms in category_dict.items():
            print(f"\n{category.upper()} ENTRIES:")

            for symbol, value in terms.items():
                print(f"{symbol}: {round(value, decimal)} {unit}")


# COP, G-function, Air property, and TDMA functions are now in dedicated
# modules.  Re-exported here for backward compatibility.
# See: cop.py, g_function.py, tdma.py


# ============================================================================
# Exergy and Entropy Functions
# ============================================================================


# ============================================================================
# Flow and Mixing Functions
# ============================================================================


def calc_mixing_valve(T_tank_w_K, T_tank_w_in_K, T_mix_w_out_K):
    """Calculate 3-way mixing valve output temperature and mixing ratio.

    Mixes hot tank water with cold mains water to achieve the target
    service temperature ``T_mix_w_out_K``.

    Parameters
    ----------
    T_tank_w_K : float
        Current tank water temperature [K].
    T_tank_w_in_K : float
        Mains (cold) water supply temperature [K].
    T_mix_w_out_K : float
        Target delivery temperature [K].

    Returns
    -------
    dict
        ``{'alp': float, 'T_mix_w_out': float, 'T_mix_w_out_K': float}``
        - ``alp``: hot-water fraction [0–1]
        - ``T_mix_w_out``: actual service temperature [°C]
        - ``T_mix_w_out_K``: actual service temperature [K]
    """
    den = max(1e-6, T_tank_w_K - T_tank_w_in_K)
    alp = min(1.0, max(0.0, (T_mix_w_out_K - T_tank_w_in_K) / den))

    T_mix_w_out_val_K = T_tank_w_K if alp >= 1.0 else alp * T_tank_w_K + (1 - alp) * T_tank_w_in_K

    T_mix_w_out_val = cu.K2C(T_mix_w_out_val_K)
    return {
        "alp": alp,
        "T_mix_w_out": T_mix_w_out_val,
        "T_mix_w_out_K": T_mix_w_out_val_K,
    }


# UV functions have been moved to uv_treatment.py.
# Re-exported above via ``from .uv_treatment import …``


def calc_Orifice_flow_coefficient(D0, D1):
    """
    Calculate the orifice flow coefficient based on diameters.

    Flow configuration:
    ---------------
     ->      |
     D0     D1 ->
     ->      |
    ---------------

    Parameters
    ----------
    D0 : float
        Pipe diameter [m]
    D1 : float
        Hole diameter [m]

    Returns
    -------
    C_d : float
        Orifice flow coefficient (dimensionless)

    Notes
    -----
    This is a simplified calculation. A more complete implementation
    should be based on physical equations.
    """
    m = D1 / D0  # Opening ratio
    return m**2


def calc_boussinessq_mixing_flow(T_upper, T_lower, A, dz, C_d=0.1):
    """
    Calculate mixing flow rate between two adjacent nodes based on Boussinesq approximation.

    Mixing occurs only when the lower node temperature is higher than the upper node,
    creating a gravitationally unstable condition.

    Parameters
    ----------
    T_upper : float
        Upper node temperature [K]
    T_lower : float
        Lower node temperature [K]
    A : float
        Tank cross-sectional area [m²]
    dz : float
        Node height [m]
    C_d : float, optional
        Flow coefficient (empirical constant), default 0.1

    Returns
    -------
    dV_mix : float
        Volumetric flow rate exchanged between nodes [m3/s]

    Notes
    -----
    TODO: C_d value should be calculated based on physical equations.
    """
    from .constants import beta, g

    if T_upper < T_lower:
        # Upper is colder (higher density) -> unstable -> mixing occurs
        delta_T = T_lower - T_upper
        dV_mix = C_d * A * math.sqrt(2 * g * beta * delta_T * dz)
        return dV_mix  # From top to bottom
    else:
        # Stable condition -> no mixing
        return 0.0


# ============================================================================
# Tank Heat Transfer Functions
# ============================================================================


# ============================================================================
# TDMA Solver Functions
# ============================================================================


# TDMA and advection-term functions have been moved to tdma.py.
# Re-exported above via ``from .tdma import …`` for backward compatibility.


# calc_UA_from_dV_fan has been moved to hx_fan.py.
# Re-exported above via ``from .hx_fan import …``


def calc_HX_perf_for_target_heat(
    Q_ref_target,
    T_ou_a_in_C,
    T_ref_evap_sat_K,
    T_ref_cond_sat_l_K,
    A_cross,
    UA_design,
    dV_fan_design,
    is_active=True,
    exponent=0.71,
):
    """
    Numerically solve for the air-side flow rate (fan airflow) required to achieve a target heat transfer rate in a heat exchanger, using a dynamically varying UA based on air velocity.

    This function determines the airflow that is needed to meet a specified heat transfer demand, accounting for dynamic changes in the overall heat transfer coefficient (UA) as a function of flow velocity based on the correlation for fin-and-tube heat exchangers by Wang et al. (2000) where UA ∝ velocity^0.71.

    Parameters
    ----------
    Q_ref_target : float
        Target heat transfer rate between refrigerant and air [W].
        Positive (+): Heat transferred from refrigerant to air (heating mode).
        Negative (−): Heat transferred from air to refrigerant (cooling mode).

    T_ou_a_in_C : float
        Inlet temperature of air [°C].

    T_ref_evap_sat_K : float
        Saturation temperature at evaporator (dew point, x=1) [K].
        Used as the constant-temperature side for evaporator heat exchange.

    T_ref_cond_sat_l_K : float
        Saturation temperature at condenser (bubble point, x=0) [K].
        Used as the constant-temperature side for condenser heat exchange.
        (Currently not used, reserved for future condenser calculations)

    A_cross : float
        Heat exchanger cross-sectional area for airflow [m²].

    UA_design : float
        Design overall heat transfer coefficient at design flow rate [W/K].

    dV_fan_design : float
        Design fan flow rate [m3/s]. Used for velocity normalization.
    is_active : bool, optional
        활성화 여부 (기본값: True)
        is_active=False일 때 nan 값으로 채워진 딕셔너리 반환

    Returns
    -------
    dict
        Dictionary containing:
            - dV_fan : Required air-side flow rate [m3/s]
            - UA : Actual heat exchanger overall heat transfer coefficient at solution point [W/K]
            - T_ou_a_mid_K : air temperature between heat exchanger and fan [K]
            - LMTD : Log-mean temperature difference at operating point [K]
            - Q_LMTD : Heat transfer rate at operating point [W]
            - epsilon : Effectiveness at operating point [–]
        Returns dict with all values as np.nan if is_active=False

    Notes
    -----
    - Air-side UA is dynamically updated using Dittus-Boelter scaling at each iterative guess.
    - LMTD is computed assuming one side stays at constant temperature (refrigerant avg).
    - The solution applies to both air-source heat pump condenser and evaporator, depending on Q_ref_target sign.
    """
    # is_active=False일 때 nan 값으로 채워진 딕셔너리 반환
    if not is_active:
        T_ou_a_in_K = cu.C2K(T_ou_a_in_C)
        return {
            "converged": True,
            "dV_fan": np.nan,
            "UA": np.nan,
            "T_ou_a_mid": np.nan,
            "Q_ou_air": np.nan,
            "epsilon": np.nan,
        }

    # All arguments are required. UA is always calculated using UA_design and velocity correction in this version.

    # Q_ref_target이 0에 가까우면 root_scalar 호출 없이 0 값 반환
    # bisect 메서드는 f(a)와 f(b)의 부호가 달라야 하므로, Q_ref_target=0일 때 실패함
    T_ou_a_in_K = cu.C2K(T_ou_a_in_C)
    if abs(Q_ref_target) < 1e-6:
        return {
            "converged": True,
            "dV_fan": 0.0,
            "UA": 0.0,
            "T_ou_a_mid_K": T_ou_a_in_K,  # 입구 온도와 동일 (열교환 없음)
            "Q_ou_air": 0.0,
            "epsilon": 0.0,
        }

    def _error_function(dV_fan):
        UA = calc_UA_from_dV_fan(dV_fan, dV_fan_design, A_cross, UA_design, exponent)
        epsilon = 1 - np.exp(-UA / (c_a * rho_a * dV_fan))
        # 증발기 계산이므로 T_ref_evap_sat_K 사용 (포화 증발 온도)
        T_ou_a_mid_K = T_ou_a_in_K - (T_ou_a_in_K - T_ref_evap_sat_K) * epsilon  # Heating assumption (Q_ref_target > 0)

        # [MODIFIED] LMTD 제거하고 공기 측 Q_air로 직접 계산
        Q_ou_air = c_a * rho_a * dV_fan * (T_ou_a_in_K - T_ou_a_mid_K)  # 흡열이므로 (입구 - 출구) * C_min
        # Heating 모드 기준: Refrigerant가 열 흡수, Air가 열 방출.
        # T_ou_a_in > T_ou_a_mid > T_ref_evap_sat_K
        # Q_ref_target > 0 (Refrigerant gains heat)
        # Q_air (Air loses heat) = m_dot * cp * (Tin - Tout) > 0

        return Q_ou_air - Q_ref_target

    dV_min = dV_fan_design * 0.1  # [m3/s]
    dV_max = dV_fan_design  # [m3/s]

    # --- Bracket validity check (avoid bisect ValueError) ---
    f_min = _error_function(dV_min)
    f_max = _error_function(dV_max)

    if f_min * f_max > 0:
        # Same sign → no root in [dV_min, dV_max]. Return best-effort result.
        # Pick the boundary closer to zero as fallback.
        dV_fallback = dV_min if abs(f_min) <= abs(f_max) else dV_max

        UA_fb = calc_UA_from_dV_fan(dV_fallback, dV_fan_design, A_cross, UA_design, exponent)
        eps_fb = 1 - np.exp(-UA_fb / (c_a * rho_a * dV_fallback))
        T_mid_fb = T_ref_evap_sat_K + eps_fb * (T_ou_a_in_K - T_ref_evap_sat_K)
        Q_fb = c_a * rho_a * dV_fallback * (T_ou_a_in_K - T_mid_fb)

        # Diagnostic hint (stored in return dict, not printed)
        if f_min > 0 and f_max > 0:
            hint = (
                "Q_achievable > Q_target at both bracket ends. "
                "Consider: ↓ dV_ou_fan_a_design or ↑ Q_ref_target via ↑ hp_capacity"
            )
        else:
            hint = (
                "Q_achievable < Q_target at both bracket ends. "
                "Consider: ↑ dV_ou_fan_a_design, ↑ UA_evap_design, or ↓ hp_capacity"
            )

        # Compute Q at both boundaries for diagnostics
        Q_at_dV_min = Q_ref_target + f_min  # f = Q_air - Q_target → Q_air = f + Q_target
        Q_at_dV_max = Q_ref_target + f_max

        return {
            "converged": False,
            "dV_fan": dV_fallback,
            "UA": UA_fb,
            "T_ou_a_mid": cu.K2C(T_mid_fb),
            "Q_ou_air": Q_fb,
            "epsilon": eps_fb,
            # Diagnostic fields for callers
            "Q_at_dV_min": Q_at_dV_min,
            "Q_at_dV_max": Q_at_dV_max,
            "Q_ref_target": Q_ref_target,
            "dV_min": dV_min,
            "dV_max": dV_max,
            "hint": hint,
        }

    sol = root_scalar(_error_function, bracket=[dV_min, dV_max], method="bisect")

    if sol.converged:
        # 수렴된 dV_fan 값을 사용하여 최종 값들 계산
        dV_fan_converged = sol.root
        UA = calc_UA_from_dV_fan(dV_fan_converged, dV_fan_design, A_cross, UA_design, exponent)
        epsilon = 1 - np.exp(-UA / (c_a * rho_a * dV_fan_converged))
        # 증발기 계산이므로 T_ref_evap_sat_K 사용 (포화 증발 온도)
        T_ou_a_mid_K = T_ref_evap_sat_K + epsilon * (
            T_ou_a_in_K - T_ref_evap_sat_K
        )  # Heating assumption (Q_ref_target > 0)

        Q_ou_air = c_a * rho_a * dV_fan_converged * (T_ou_a_in_K - T_ou_a_mid_K)

        return {
            "converged": True,  # 명시적으로 converged 플래그 추가
            "dV_fan": dV_fan_converged,
            "UA": UA,
            "T_ou_a_mid": cu.K2C(T_ou_a_mid_K),
            "Q_ou_air": Q_ou_air,
            "epsilon": epsilon,
        }
    else:
        return {
            "converged": False,
            "dV_fan": np.nan,
            "UA": np.nan,
            "T_ou_a_mid": np.nan,
            "Q_ou_air": np.nan,
            "epsilon": np.nan,
        }


# calc_fan_power_from_dV_fan and check_hp_schedule_active have been moved
# to hx_fan.py.  Re-exported above via ``from .hx_fan import …``


# get_uv_params_from_turbidity and calc_uv_exposure_time have been moved
# to uv_treatment.py.  Re-exported above via ``from .uv_treatment import …``


def update_tank_temperature(T_tank_w_K, Q_gain, UA_tank, T0_K, C_tank, dt):
    """Update tank temperature using the Crank-Nicolson implicit scheme.

    The governing ODE for a lumped-capacitance tank is:

        C dT/dt = Q_gain - UA (T - T0)

    Crank-Nicolson averages the loss term across both time levels:

        T^{n+1} = [(C/dt - UA/2) T^n + Q_gain + UA T0] / (C/dt + UA/2)

    This scheme is second-order accurate in time and unconditionally
    stable, eliminating the overshoot that Forward Euler can exhibit
    when dt is large relative to the thermal time constant C/UA.

    Parameters
    ----------
    T_tank_w_K : float
        Current tank temperature [K].
    Q_gain : float
        Total heat gain rate [W] (condenser, UV, STC, refill, etc.).
    UA_tank : float
        Overall tank heat-loss coefficient [W/K].
    T0_K : float
        Dead-state / ambient temperature [K].
    C_tank : float
        Tank thermal capacitance [J/K] (= c_w * rho_w * V_tank * level).
    dt : float
        Time step [s].

    Returns
    -------
    float
        Updated tank temperature [K].
    """
    a = C_tank / dt
    T_tank_w_K_new = ((a - UA_tank / 2) * T_tank_w_K + Q_gain + UA_tank * T0_K) / (a + UA_tank / 2)
    return T_tank_w_K_new


def calc_stc_performance(
    I_DN_stc,  # 직달일사 [W/m²]
    I_dH_stc,  # 확산일사 [W/m²]
    T_stc_w_in_K,  # STC 입수 온도 (저탕조 온도) [K]
    T0_K,  # 기준 온도 [K]
    A_stc_pipe,  # STC 파이프 면적 [m²]
    alpha_stc,  # 흡수율 [-]
    h_o_stc,  # 외부 대류 열전달계수 [W/m²K]
    h_r_stc,  # 공기층 복사 열전달계수 [W/m²K]
    k_ins_stc,  # 단열재 열전도도 [W/mK]
    x_air_stc,  # 공기층 두께 [m]
    x_ins_stc,  # 단열재 두께 [m]
    dV_stc,  # STC 유량 [m3/s]
    E_pump,  # 펌프 소모 전력 [W]
    is_active=True,  # 활성화 여부 (기본값: True)
):
    """
    Solar Thermal Collector (STC) 성능을 계산합니다.

    이 함수는 태양열 집열판의 열전달 분석을 수행합니다.
    enex_engine.py의 SolarAssistedGasBoiler.system_update() 로직을 참조하여 구현되었습니다.

    Parameters
    ----------
    I_DN_stc : float
        직달일사 [W/m²]
    I_dH_stc : float
        확산일사 [W/m²]
    T_stc_w_in_K : float
        STC 입수 온도 (저탕조 온도) [K]
    T0_K : float
        기준 온도 (환경 온도) [K]
    A_stc_pipe : float
        STC 파이프 면적 [m²]
    alpha_stc : float
        흡수율 [-]
    h_o_stc : float
        외부 대류 열전달계수 [W/m²K]
    h_r_stc : float
        공기층 복사 열전달계수 [W/m²K]
    k_ins_stc : float
        단열재 열전도도 [W/mK]
    x_air_stc : float
        공기층 두께 [m]
    x_ins_stc : float
        단열재 두께 [m]
    dV_stc : float
        STC 유량 [m3/s]
    E_pump : float
        펌프 소모 전력 [W]
    is_active : bool, optional
        활성화 여부 (기본값: True)
        is_active=False일 때 nan 값으로 채워진 딕셔너리 반환

    Returns
    -------
    dict
        계산 결과 딕셔너리:
        - I_sol_stc: 총 일사량 [W/m²]
        - Q_sol_stc: 태양열 흡수 열량 [W]
        - Q_stc_w_in: STC 입수 열량 [W]
        - Q_stc_w_out: STC 출수 열량 [W]
        - ksi_stc: 무차원 수 [-]
        - T_stc_w_out_K: STC 출수 온도 [K]
        - T_stc_w_final_K: 펌프 열 포함 최종 출수 온도 [K]
        - T_stc_w_in_K: STC 입수 온도 [K]
        - T_stc_K: 집열판 평균 온도 [K]
        - Q_l_stc: 집열판 열 손실 [W]
        Returns dict with all values as np.nan (except T_stc_w_out_K, T_stc_w_in_K = T_stc_w_in_K) if is_active=False

    Notes
    -----
    - 모든 변수명에 _stc 접미사를 사용하여 STC 관련 변수임을 명확히 구분합니다.
    - 열 손실은 Q_l_stc로 명명됩니다.
    - 엔트로피 및 엑서지 계산은 제거되었으며, CSV 파일 후처리를 통해 계산해야 합니다.
    """
    from .constants import k_a

    # U_stc 계산 (내부에서 계산)
    # Resistance [m²K/W]
    R_air_stc = x_air_stc / k_a
    R_ins_stc = x_ins_stc / k_ins_stc
    R_o_stc = 1 / h_o_stc
    R_r_stc = 1 / h_r_stc

    R1_stc = (R_r_stc * R_air_stc) / (R_r_stc + R_air_stc) + R_o_stc
    R2_stc = R_ins_stc + R_o_stc

    # U-value [W/m²K] (병렬)
    U1_stc = 1 / R1_stc
    U2_stc = 1 / R2_stc
    U_stc = U1_stc + U2_stc

    # is_active=False일 때 nan 값으로 채워진 딕셔너리 반환
    if not is_active:
        return {
            "I_sol_stc": np.nan,
            "Q_sol_stc": np.nan,
            "Q_stc_w_in": np.nan,
            "Q_stc_w_out": np.nan,
            "ksi_stc": np.nan,
            "T_stc_w_final_K": T_stc_w_in_K,  # 입수 온도와 동일
            "T_stc_w_out_K": T_stc_w_in_K,  # 입수 온도와 동일
            "T_stc_w_in_K": T_stc_w_in_K,
            "T_stc_K": np.nan,
            "Q_l_stc": np.nan,
        }

    # 총 일사량 계산
    I_sol_stc = I_DN_stc + I_dH_stc

    # 태양열 흡수 열량
    Q_sol_stc = I_sol_stc * A_stc_pipe * alpha_stc

    # Heat capacity flow rate
    G_stc = c_w * rho_w * dV_stc

    # 입수 열량 (기준 온도 기준) - calc_energy_flow 사용
    Q_stc_w_in = calc_energy_flow(G_stc, T_stc_w_in_K, T0_K)

    # 무차원 수 (효율 계수)
    ksi_stc = np.exp(-A_stc_pipe * U_stc / G_stc)

    # STC 출수 온도 계산
    T_stc_w_out_numerator = (
        T0_K
        + (
            Q_sol_stc
            + Q_stc_w_in
            + A_stc_pipe * U_stc * (ksi_stc * T_stc_w_in_K / (1 - ksi_stc))
            + A_stc_pipe * U_stc * T0_K
        )
        / G_stc
    )

    T_stc_w_out_denominator = 1 + (A_stc_pipe * U_stc) / ((1 - ksi_stc) * G_stc)

    T_stc_w_out_K = T_stc_w_out_numerator / T_stc_w_out_denominator
    T_stc_w_final_K = T_stc_w_out_K + E_pump / G_stc
    T_stc_K = 1 / (1 - ksi_stc) * T_stc_w_out_K - ksi_stc / (1 - ksi_stc) * T_stc_w_in_K

    # STC 출수 열량 - calc_energy_flow 사용
    Q_stc_w_out = calc_energy_flow(G_stc, T_stc_w_out_K, T0_K)

    # 집열판 열 손실
    Q_l_stc = A_stc_pipe * U_stc * (T_stc_K - T0_K)

    return {
        "I_sol_stc": I_sol_stc,
        "Q_sol_stc": Q_sol_stc,
        "Q_stc_w_in": Q_stc_w_in,
        "Q_stc_w_out": Q_stc_w_out,
        "ksi_stc": ksi_stc,
        "T_stc_w_final_K": T_stc_w_final_K,
        "T_stc_w_out_K": T_stc_w_out_K,
        "T_stc_w_in_K": T_stc_w_in_K,
        "T_stc_K": T_stc_K,
        "Q_l_stc": Q_l_stc,
    }
