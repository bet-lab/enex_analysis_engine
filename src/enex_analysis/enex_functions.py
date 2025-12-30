"""
Utility functions for energy, entropy, and exergy analysis.

This module contains helper functions for calculations including:
- Friction factor calculations
- Heat transfer coefficient calculations
- Curve fitting functions
- COP calculations for heat pumps
- G-function calculations for ground source heat pumps
- Balance printing utilities
"""

import numpy as np
import math
from scipy.optimize import curve_fit, root_scalar, minimize
from scipy import integrate
from scipy.special import erf
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
try:
    import dartwork_mpl as dm
except ImportError:
    # dartwork_mpl이 없는 경우를 대비한 fallback
    dm = None
from . import calc_util as cu
from .constants import SP, c_a, rho_a, c_w, rho_w


def darcy_friction_factor(Re, e_d):
    """
    Calculate the Darcy friction factor for given Reynolds number and relative roughness.
    
    Parameters
    ----------
    Re : float
        Reynolds number
    e_d : float
        Relative roughness (e/D)
    
    Returns
    -------
    float
        Darcy friction factor
    """
    # Laminar flow
    if Re < 2300:
        return 64 / Re
    # Turbulent flow
    else:
        return 0.25 / (math.log10(e_d / 3.7 + 5.74 / Re ** 0.9)) ** 2


def calc_h_vertical_plate(T_s, T_inf, L):
    """
    Calculate natural convection heat transfer coefficient for a vertical plate.
    
    This function calculates the heat transfer coefficient due to natural convection
    using the Churchill & Chu correlation.
    
    Parameters
    ----------
    T_s : float
        Surface temperature [K]
    T_inf : float
        Fluid temperature [K]
    L : float
        Characteristic length [m]
    
    Returns
    -------
    float
        Heat transfer coefficient [W/m²K]
    
    Note
    ----
    Uses Churchill & Chu correlation.
    Reference: https://doi.org/10.1016/0017-9310(75)90243-4
    """
    # Air properties @ 40°C
    nu = 1.6e-5  # Kinematic viscosity [m²/s]
    k_air = 0.027  # Thermal conductivity [W/m·K]
    Pr = 0.7  # Prandtl number
    beta = 1 / ((T_s + T_inf) / 2)  # Thermal expansion coefficient [1/K]
    g = 9.81  # Gravitational acceleration [m/s²]

    # Calculate Rayleigh number
    delta_T = T_s - T_inf
    Ra_L = g * beta * delta_T * L**3 / (nu**2) * Pr

    # Churchill & Chu correlation
    Nu_L = (0.825 + (0.387 * Ra_L**(1/6)) / (1 + (0.492/Pr)**(9/16))**(8/27))**2
    h_cp = Nu_L * k_air / L  # [W/m²K]
    
    return h_cp


def linear_function(x, a, b):
    """Linear function: y = a*x + b"""
    return a * x + b


def quadratic_function(x, a, b, c):
    """Quadratic function: y = a*x² + b*x + c"""
    return a * x ** 2 + b * x + c


def cubic_function(x, a, b, c, d):
    """Cubic function: y = a*x³ + b*x² + c*x + d"""
    return a * x ** 3 + b * x ** 2 + c * x + d


def quartic_function(x, a, b, c, d, e):
    """Quartic function: y = a*x⁴ + b*x³ + c*x² + d*x + e"""
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e


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
    for subsystem, category_dict in balance.items(): 
        for category, terms in category_dict.items():
            if "gen" in category:
                balance_type = "entropy"
                unit = "[W/K]"
            elif "con" in category:
                balance_type = "exergy"
    
    # Print balance for each subsystem
    for subsystem, category_dict in balance.items(): 
        text = f"{subsystem.upper()} {balance_type.upper()} BALANCE:"
        print(f'\n\n{text}'+'='*(total_length-len(text)))
        
        for category, terms in category_dict.items():
            print(f"\n{category.upper()} ENTRIES:")
            
            for symbol, value in terms.items():
                print(f"{symbol}: {round(value, decimal)} {unit}")


def calculate_ASHP_cooling_COP(T_a_int_out, T_a_ext_in, Q_r_int, Q_r_max, COP_ref):
    """
    Calculate the Coefficient of Performance (COP) for an Air Source Heat Pump (ASHP) in cooling mode.
    
    Reference: https://publications.ibpsa.org/proceedings/bs/2023/papers/bs2023_1118.pdf
    
    Parameters
    ----------
    T_a_int_out : float
        Indoor air temperature [K]
    T_a_ext_in : float
        Outdoor air temperature [K]
    Q_r_int : float
        Indoor heat load [W]
    Q_r_max : float
        Maximum cooling capacity [W]
    COP_ref : float
        Reference COP at standard conditions
    
    Returns
    -------
    float
        COP value
    
    Note
    ----
    COP is calculated based on:
    - PLR: Part Load Ratio
    - EIR: Energy input to cooling output ratio
    """
    PLR = Q_r_int / Q_r_max
    if PLR < 0.2:
        PLR = 0.2    
    EIR_by_T = 0.38 + 0.02 * cu.K2C(T_a_int_out) + 0.01 * cu.K2C(T_a_ext_in)
    EIR_by_PLR = 0.22 + 0.50 * PLR + 0.26 * PLR**2
    COP = PLR * COP_ref / (EIR_by_T * EIR_by_PLR)
    return COP


def calculate_ASHP_heating_COP(T0, Q_r_int, Q_r_max):
    """
    Calculate the Coefficient of Performance (COP) for an Air Source Heat Pump (ASHP) in heating mode.
    
    Reference: https://www.mdpi.com/2071-1050/15/3/1880
    
    Parameters
    ----------
    T0 : float
        Environmental temperature [K]
    Q_r_int : float
        Indoor heat load [W]
    Q_r_max : float
        Maximum heating capacity [W]
    
    Returns
    -------
    float
        COP value
    
    Note
    ----
    COP is calculated based on PLR (Part Load Ratio).
    """
    PLR = Q_r_int / Q_r_max
    if PLR < 0.2:
        PLR = 0.2    
    COP = -7.46 * (PLR - 0.0047 * cu.K2C(T0) - 0.477)**2 + 0.0941 * cu.K2C(T0) + 4.34
    return COP


def calculate_GSHP_COP(Tg, T_cond, T_evap, theta_hat):
    """
    Calculate the Carnot-based COP of a GSHP system using the modified formula.
    
    Reference: https://www.sciencedirect.com/science/article/pii/S0360544219304347?via%3Dihub
    
    Formula: COP = 1 / (1 - T0/T_cond + ΔT * θ̂ / T_cond)
    
    Parameters
    ----------
    Tg : float
        Undisturbed ground temperature [K]
    T_cond : float
        Condenser refrigerant temperature [K]
    T_evap : float
        Evaporator refrigerant temperature [K]
    theta_hat : float
        θ̂(x0, k_sb), dimensionless average fluid temperature
        Reference: Paper Fig 8, Table 1
    
    Returns
    -------
    float
        Modified Carnot-based COP. Returns NaN if denominator <= 0.
    
    Raises
    ------
    ValueError
        If T_cond <= T_evap (invalid for COP calculation)
    """
    # Temperature difference (ΔT = T0 - T1)
    if T_cond <= T_evap:
        raise ValueError("T_cond must be greater than T_evap for a valid COP calculation.")
    
    delta_T = Tg - T_evap

    # Compute COP using the modified Carnot expression
    denominator = 1 - (Tg / T_cond) + (delta_T / (T_cond * theta_hat))

    if denominator <= 0:
        return float('nan')  # Avoid division by zero or negative COP

    COP = 1 / denominator
    return COP


def f(x):
    """
    Helper function for G-function calculation.
    
    Parameters:
    -----------
    x : float
        Input value
    
    Returns:
    --------
    float
        f(x) = x*erf(x) - (1-exp(-x²))/√π
    """
    return x * erf(x) - (1 - np.exp(-x**2)) / SP


def chi(s, rb, H, z0=0):
    """
    Helper function for G-function calculation.
    
    Parameters:
    -----------
    s : float
        Integration variable
    rb : float
        Borehole radius [m]
    H : float
        Borehole height [m]
    z0 : float, optional
        Reference depth [m] (default: 0)
    
    Returns:
    --------
    float
        chi function value
    """
    h = H * s
    d = z0 * s
    
    temp = np.exp(-(rb * s)**2) / (h * s)
    Is = 2 * f(h) + 2 * f(h + 2*d) - f(2*h + 2*d) - f(2*d)
    
    return temp * Is


_g_func_cache = {}


def G_FLS(t, ks, as_, rb, H):
    """
    Calculate the g-function for finite line source (FLS) model.
    
    This function calculates the g-function used in ground source heat pump
    analysis. Results are cached for performance.
    
    Parameters:
    -----------
    t : float
        Time [s]
    ks : float
        Ground thermal conductivity [W/mK]
    as_ : float
        Ground thermal diffusivity [m²/s]
    rb : float
        Borehole radius [m]
    H : float
        Borehole height [m]
    
    Returns:
    --------
    float or array
        g-function value [mK/W]. Returns scalar for single time value,
        array for multiple time values.
    """
    key = (round(t, 0), round(ks, 2), round(as_, 6), round(rb, 2), round(H, 0))
    if key in _g_func_cache:
        return _g_func_cache[key]

    factor = 1 / (4 * np.pi * ks)
    
    lbs = 1 / np.sqrt(4 * as_ * t)
    
    # Handle scalar case: shape == (,)
    single = len(lbs.shape) == 0
    # Reshape to 1D array
    lbs = lbs.reshape(-1)
        
    # Pre-calculate integral from 0 to inf
    total = integrate.quad(chi, 0, np.inf, args=(rb, H))[0]
    # ODE initial value
    first = integrate.quad(chi, 0, lbs[0], args=(rb, H))[0]
   
    # Scipy ODE solver function form: dydx = f(y, x)
    def func(y, s):
        return chi(s, rb, H, z0=0)
    
    values = total - integrate.odeint(func, first, lbs)[:, 0]
    
    # For single time value, return first value as float
    if single:
        values = values[0]

    result = factor * values
    _g_func_cache[key] = result
    return result


def air_dynamic_viscosity(T_K):
    """
    Calculate air dynamic viscosity using Sutherland's formula.
    
    Parameters:
    -----------
    T_K : float
        Temperature [K]
    
    Returns:
    --------
    float
        Dynamic viscosity [Pa·s]
    
    Reference: Sutherland's formula for air
    mu = mu0 * (T/T0)^1.5 * (T0 + S) / (T + S)
    where mu0 = 1.716e-5 Pa·s at T0 = 273.15 K, S = 110.4 K
    """
    T0 = 273.15  # Reference temperature [K]
    mu0 = 1.716e-5  # Reference viscosity [Pa·s] at T0
    S = 110.4  # Sutherland constant [K] for air
    
    mu = mu0 * ((T_K / T0)**1.5) * ((T0 + S) / (T_K + S))
    return mu


def air_prandtl_number(T_K):
    """
    Calculate air Prandtl number.
    
    Parameters:
    -----------
    T_K : float
        Temperature [K]
    
    Returns:
    --------
    float
        Prandtl number [-]
    
    Note: Pr ≈ 0.71 for air at typical temperatures (20-50°C)
    Temperature dependence is weak, so using constant value.
    """
    # Pr = mu * cp / k
    # For air: Pr ≈ 0.71 (weak temperature dependence)
    return 0.71


def calc_simple_tank_UA(
        # Tank size [m]
        r0 = 0.2,
        H = 0.8,
        # Tank layer thickness [m]
        x_shell = 0.01,
        x_ins   = 0.10,
        # Tank thermal conductivity [W/mK]
        k_shell = 25,  
        k_ins   = 0.03,
        # External convective heat transfer coefficient [W/m²K]
        h_o     = 10,
        ):
    """
    Calculate simple tank UA value.
    
    Parameters:
    -----------
    r0 : float
        Tank radius [m]
    H : float
        Tank height [m]
    x_shell : float
        Shell thickness [m]
    x_ins : float
        Insulation thickness [m]
    k_shell : float
        Shell thermal conductivity [W/mK]
    k_ins : float
        Insulation thermal conductivity [W/mK]
    h_o : float
        External convective heat transfer coefficient [W/m²K]
    
    Returns:
    --------
    float
        Tank UA value [W/K]
    """
    r1 = r0 + x_shell
    r2 = r1 + x_ins
    
    # Tank surface areas [m²]
    A_side = 2 * math.pi * r2 * H
    A_base = math.pi * r0**2
    R_base_unit = x_shell / k_shell + x_ins / k_ins # [m2K/W]
    R_side_unit = math.log(r1 / r0) / (2 * math.pi * k_shell) + math.log(r2 / r1) / (2 * math.pi * k_ins) # [mK/W]
    
    # Thermal resistances [K/W]
    R_base = R_base_unit / A_base # [K/W]
    R_side = R_side_unit / H # [K/W]
    
    # Thermal resistances [K/W]
    R_base_ext = 1 / (h_o * A_base)
    R_side_ext = 1 / (h_o * A_side)

    # Total thermal resistances [K/W]
    R_base_tot = R_base + R_base_ext
    R_side_tot = R_side + R_side_ext

    # U-value [W/K]
    U_tank = 2/R_base_tot + 1/R_side_tot 
    return U_tank


def calc_lmtd_counter_flow(T_hot_in_K, T_hot_out_K, T_cold_in_K, T_cold_out_K):
    """
    Calculate LMTD for counter-flow heat exchanger.
    
    Parameters:
    -----------
    T_hot_in_K : float
        Hot fluid inlet temperature [K]
    T_hot_out_K : float
        Hot fluid outlet temperature [K]
    T_cold_in_K : float
        Cold fluid inlet temperature [K]
    T_cold_out_K : float
        Cold fluid outlet temperature [K]
    
    Returns:
    --------
    float
        LMTD [K]
    """
    # Counter-flow: hot inlet ↔ cold outlet, hot outlet ↔ cold inlet
    dT1 = T_hot_in_K - T_cold_out_K
    dT2 = T_hot_out_K - T_cold_in_K
    
    if dT1 <= 0 or dT2 <= 0:
        return np.nan
    
    if abs(dT1 - dT2) < 1e-4:
        return (dT1 + dT2) / 2
    else:
        return (dT1 - dT2) / np.log(dT1 / dT2)


def calc_lmtd_parallel_flow(T_hot_in_K, T_hot_out_K, T_cold_in_K, T_cold_out_K):
    """
    Calculate LMTD for parallel-flow heat exchanger.
    
    Parameters:
    -----------
    T_hot_in_K : float
        Hot fluid inlet temperature [K]
    T_hot_out_K : float
        Hot fluid outlet temperature [K]
    T_cold_in_K : float
        Cold fluid inlet temperature [K]
    T_cold_out_K : float
        Cold fluid outlet temperature [K]
    
    Returns:
    --------
    float
        LMTD [K]
    """
    # Parallel-flow: hot inlet ↔ cold inlet, hot outlet ↔ cold outlet
    dT1 = T_hot_in_K - T_cold_in_K
    dT2 = T_hot_out_K - T_cold_out_K
    
    if dT1 <= 0 or dT2 <= 0:
        return np.nan
    
    if abs(dT1 - dT2) < 1e-4:
        return (dT1 + dT2) / 2
    else:
        return (dT1 - dT2) / np.log(dT1 / dT2)


def calc_lmtd_fluid_and_constant_temp(T_constant_K, T_fluid_in_K, T_fluid_out_K):
    """
    Calculate LMTD when one fluid maintains constant temperature.
    
    One fluid maintains a constant temperature (e.g., during phase change),
    while the other fluid temperature changes from inlet to outlet.
    This applies to condensers, evaporators, or any heat exchanger where
    one fluid undergoes phase change or maintains constant temperature.
    
    Parameters:
    -----------
    T_constant_K : float
        Constant temperature of one fluid [K]
    T_fluid_in_K : float
        Inlet temperature of the other fluid [K]
    T_fluid_out_K : float
        Outlet temperature of the other fluid [K]
    
    Returns:
    --------
    float
        LMTD [K]
    
    Notes:
    ------
    - Since one fluid temperature is constant, LMTD is calculated in simplified form.
    - Q>0 (constant temp fluid releases heat): T_constant > T_fluid_in, T_constant > T_fluid_out
      → dT_in = T_constant - T_fluid_in, dT_out = T_constant - T_fluid_out
    - Q<0 (constant temp fluid absorbs heat): T_constant < T_fluid_in, T_constant < T_fluid_out
      → dT_in = T_fluid_in - T_constant, dT_out = T_fluid_out - T_constant
    """
    # Temperature difference calculation (maintain sign)
    dT_in = T_constant_K - T_fluid_in_K
    dT_out = T_constant_K - T_fluid_out_K
    
    # Physical validity check: dT_in and dT_out must have same sign
    if dT_in * dT_out <= 0:
        # Constant temperature is between fluid inlet and outlet (physically impossible)
        return np.nan
    
    # Calculate LMTD using absolute values
    dT_in_abs = abs(dT_in)
    dT_out_abs = abs(dT_out)
    
    if dT_in_abs <= 0 or dT_out_abs <= 0:
        return np.nan
    
    # LMTD calculation
    if abs(dT_in_abs - dT_out_abs) < 1e-4:
        return (dT_in_abs + dT_out_abs) / 2
    else:
        return (dT_in_abs - dT_out_abs) / np.log(dT_in_abs / dT_out_abs)


def calc_UA_from_dV_fan(dV_fan, dV_fan_design, A_cross, UA):
    """
    Calculate heat transfer coefficient based on Dittus-Boelter equation.
    
    This function calculates heat transfer coefficient based on air velocity.
    Dittus-Boelter equation: Nu = 0.023 * Re^0.8 * Pr^n
    Proportional to velocity^0.8.
    
    Parameters:
    -----------
    dV_fan : float
        Fan flow rate [m³/s]
    dV_fan_design : float
        Design fan flow rate [m³/s]
    A_cross : float
        Heat exchanger cross-sectional area [m²]
    UA : float
        Refrigerant-side resistance and correction factor [W/K]
        Coefficient for Dittus-Boelter equation, U = UA * V^0.8 form
    
    Returns:
    --------
    float
        Overall heat transfer coefficient U [W/K]
    
    Notes:
    ------
    - Velocity calculation: v = dV_fan / A_cross
    - Heat transfer coefficient: U = UA * v^0.8
    """
    # Velocity calculation
    v = dV_fan / A_cross if A_cross > 0 else 0
    v_design = dV_fan_design / A_cross if A_cross > 0 else 0
    return UA * (v / v_design) ** 0.8


def calc_HX_perf_for_target_heat(Q_ref_target, T_air_in_C, T_ref_avg_K, A_cross, UA_design, dV_fan_design):
    """
    Numerically solve for the air-side flow rate (fan airflow) required to achieve a target heat transfer rate in a heat exchanger, using a dynamically varying UA based on air velocity.

    This function determines the airflow that is needed to meet a specified heat transfer demand, accounting for dynamic changes in the overall heat transfer coefficient (UA) as a function of flow velocity using the Dittus-Boelter relationship (UA ∝ velocity^0.8).

    Parameters
    ----------
    Q_ref_target : float
        Target heat transfer rate between refrigerant and air [W].
        Positive (+): Heat transferred from refrigerant to air (heating mode).
        Negative (−): Heat transferred from air to refrigerant (cooling mode).

    T_air_in_C : float
        Inlet temperature of air [°C].

    T_ref_avg_K : float
        Average refrigerant temperature used as the constant-temperature side [K].
        Typically the mean of refrigerant inlet and outlet temperatures.

    A_cross : float
        Heat exchanger cross-sectional area for airflow [m²].

    UA_design : float
        Design overall heat transfer coefficient at design flow rate [W/K].

    dV_fan_design : float
        Design fan flow rate [m³/s]. Used for velocity normalization.

    Returns
    -------
    dict
        Dictionary containing:
            - dV_fan : Required air-side flow rate [m³/s]
            - UA : Actual heat exchanger overall heat transfer coefficient at solution point [W/K]
            - T_air_out_K : Outlet air temperature [K]
            - LMTD : Log-mean temperature difference at operating point [K]
            - Q_LMTD : Heat transfer rate at operating point [W]
            - epsilon : Effectiveness at operating point [–]

    Notes
    -----
    - Air-side UA is dynamically updated using Dittus-Boelter scaling at each iterative guess.
    - LMTD is computed assuming one side stays at constant temperature (refrigerant avg).
    - The solution applies to both air-source heat pump condenser and evaporator, depending on Q_ref_target sign.
    """
    # All arguments are required. UA is always calculated using UA_design and velocity correction in this version.
    
    T_air_in_K = cu.C2K(T_air_in_C)

    def _error_function(dV_fan):
        UA = calc_UA_from_dV_fan(dV_fan, dV_fan_design, A_cross, UA_design)
        epsilon = 1 - np.exp(-UA / (c_a * rho_a * dV_fan))
        T_air_out_K = T_air_in_K + epsilon * (T_ref_avg_K - T_air_in_K) # Heating assumption (Q_ref_target > 0)
            
        LMTD = calc_lmtd_fluid_and_constant_temp(
            T_constant_K  = T_ref_avg_K,
            T_fluid_in_K  = T_air_in_K,
            T_fluid_out_K = T_air_out_K
        )
        Q_LMTD = UA * LMTD # [W]
        return Q_LMTD - Q_ref_target

    dV_min = 0.1 # [m³/s]
    dV_max = dV_fan_design # [m³/s]
    sol = root_scalar(_error_function, bracket=[dV_min, dV_max], method='bisect')
    
    if sol.converged:
        # 수렴된 dV_fan 값을 사용하여 최종 값들 계산
        dV_fan_converged = sol.root
        UA = calc_UA_from_dV_fan(dV_fan_converged, dV_fan_design, A_cross, UA_design)
        epsilon = 1 - np.exp(-UA / (c_a * rho_a * dV_fan_converged))
        T_air_out_K = T_air_in_K + epsilon * (T_ref_avg_K - T_air_in_K)  # Heating assumption (Q_ref_target > 0)
        LMTD = calc_lmtd_fluid_and_constant_temp(
            T_constant_K  = T_ref_avg_K,
            T_fluid_in_K  = T_air_in_K,
            T_fluid_out_K = T_air_out_K
        )
        Q_LMTD = UA * LMTD
        return {
            'dV_fan': dV_fan_converged,
            'UA': UA,
            'T_air_out_K': T_air_out_K,
            'LMTD': LMTD,
            'Q_LMTD': Q_LMTD,
            'epsilon': epsilon,
            'converged': True  # 명시적으로 converged 플래그 추가
            }
    else:
        return {
            'converged': False,
            'dV_fan': np.nan,
            'UA': np.nan,
            'T_air_out_K': np.nan,
            'LMTD': np.nan,
            'Q_LMTD': np.nan,
            'epsilon': np.nan
        }


def calc_fan_power_from_dV_fan(dV_fan, fan_params, vsd_coeffs):
    """
    Calculate fan power using ASHRAE 90.1 VSD Curve.
    
    Parameters:
    -----------
    dV_fan : float
        Current flow rate [m³/s]
    fan_params : dict
        Fan parameters (fan_design_flow_rate, fan_design_power)
    vsd_coeffs : dict
        VSD Curve coefficients (c1~c5)
    
    Returns:
    --------
    float
        Fan power [W]
    """
    # Extract design parameters
    fan_design_flow_rate = fan_params.get('fan_design_flow_rate', None)
    fan_design_power = fan_params.get('fan_design_power', None)
    
    # Error if design parameters are missing
    if fan_design_flow_rate is None or fan_design_power is None:
        raise ValueError("fan_design_flow_rate and fan_design_power must be provided in fan_params")
    
    # Flow rate validation
    if dV_fan < 0:
        print(f"fan flow rate must be greater than 0: {dV_fan} [m³/s]")
        raise ValueError("fan flow rate must be greater than 0")
    
    # Extract VSD Curve coefficients
    c1 = vsd_coeffs.get('c1', 0.0013)
    c2 = vsd_coeffs.get('c2', 0.1470)
    c3 = vsd_coeffs.get('c3', 0.9506)
    c4 = vsd_coeffs.get('c4', -0.0998)
    c5 = vsd_coeffs.get('c5', 0.0)
    
    # Calculate flow fraction
    flow_fraction = dV_fan / fan_design_flow_rate
    
    # Calculate Part-load ratio: PLR = c1 + c2*x + c3*x² + c4*x³ + c5*x⁴
    x = flow_fraction
    PLR = c1 + c2*x + c3*x**2 + c4*x**3 + c5*x**4
    
    # Ensure PLR is not negative
    PLR = max(0.0, PLR)
    
    # Calculate fan power
    fan_power = fan_design_power * PLR
    
    return fan_power


def _build_schedule_ratios(entries, t_array):
    """
    Build schedule ratio array from schedule entries for each timestep (t_array).

    Parameters
    ----------
    entries : list of tuple
        Schedule entry list. Each item is (start_str, end_str, frac) format.
        - start_str, end_str: "H:M" or "H" format string (e.g., "6:00", "23:30", "24:00", etc.).
          "24:00" is specially handled as end of day (= 24*cu.h2s seconds).
        - frac: Usage ratio (float) for that interval. Clipped to 0.0 ~ 1.0 range.
        Intervals are treated as half-open [start, end).

    t_array : numpy.ndarray
        Timestep array in seconds (e.g., np.arange(0, sim_seconds, dt)). Each element
        is mapped to time within the same day using modulo operation with 24*cu.h2s.

    Returns
    -------
    numpy.ndarray
        Array with same shape as t_array. Schedule ratio (0.0 ~ 1.0) for each timestep.
        If multiple entries overlap, the value at that position is the maximum of the frac values.

    Summary
    -------
    - Time strings are converted to seconds internally using _time_str_to_sec.
    - If end is 24*cu.h2s (e.g., "24:00"), it is adjusted to last value before end of day.
    - If interval crosses midnight (e.g., start=23:00, end=02:00), OR mask is used to
      correctly cover intervals that cross midnight.
    - Ratio (frac) is clipped to 0~1 range using np.clip.
    - Return value is finally clipped to 0~1 range to guarantee bounds.

    Examples
    --------
    entries = [("6:00","7:00",0.5), ("6:30","8:00",0.8)]
    -> Interval 6:30~7:00 has max(0.5,0.8)=0.8 applied.

    Notes
    -----
    - t_array must be continuous time array in seconds, internally modulo operated with 24*cu.h2s.
    - When multiple entries exist at same time, priority is maximum frac value (merge instead of overwrite).
    """
    day = 24*cu.h2s
    secs_mod = (t_array % day)
    sched = np.zeros_like(t_array, dtype=float)

    def _time_str_to_sec(time_str):
        """
        Convert time string format (e.g., "H", "H:M") to integer seconds within day (0 ~ 86400).

        Parameters
        ----------
        time_str : str
            Time string in "H" or "H:M" format.
            - "H" represents hour, integer in 0~24 range.
            - "H:M" is format with hour and minute separated by colon (:).
                Hour is 0~24, minute is 0~59 integer.
            - "24:00" is specially handled as end of day (= 24*cu.h2s seconds).

        Behavior
        --------
        - Separate hour and minute, convert to seconds: seconds = (h % 24) * 3600 + m * 60
        - If input is "24:00" (string starts with '24' and h%24 == 0), return 24*cu.h2s.
        - Hour (h) is modulo operated with 24, so notation >= 24 is mapped to 0..23.
        
        Returns
        -------
        int
            Integer seconds within day (0 ~ 86400).
        """ 
        h, m = (time_str.split(':') + ['0'])[:2]
        h = int(h) % 24
        m = int(m)
        return 24*cu.h2s if (h == 0 and time_str.strip().startswith('24')) else h*cu.h2s + m*60
        
    # Process schedule entries
    for start_str, end_str, frac in entries:
        s = _time_str_to_sec(start_str)
        e = _time_str_to_sec(end_str)
        if e == 24*cu.h2s: e = 24*cu.h2s - 1e-9
        ratio = np.clip(frac, 0.0, 1.0)
        if s == e: continue
        
        if s < e:
            mask = (secs_mod >= s) & (secs_mod < e)
        else:
            mask = (secs_mod >= s) | (secs_mod < e)
        sched[mask] = np.maximum(sched[mask], ratio)

    return np.clip(sched, 0.0, 1.0)


# ============================================================================
# 히트펌프 사이클 성능 계산 및 최적 운전점 탐색 함수들
# (cycle_performance.py에서 이동)
# ============================================================================

def compute_refrigerant_thermodynamic_states(
    T_evap_K,  # 증발 온도 [K]
    T_cond_K,  # 응축 온도 [K]
    refrigerant,  # 냉매 이름
    eta_cmp_isen,  # 압축기 단열 효율
    T0_K=None,  # 기준 온도 [K] (엑서지 계산용, 선택적)
    P0=101325,  # 기준 압력 [Pa] (엑서지 계산용, 선택적)
    mode='heating',  # 작동 모드 ('heating' 또는 'cooling')
):
    """
    냉매 사이클의 State 1-4 열역학 물성치를 계산하는 공통 함수.
    
    이 함수는 히트펌프 사이클의 4개 주요 상태점을 계산합니다:
    
    난방 모드 (mode='heating'):
    - State 1: 압축기 입구 (증발기 출구, 저압 포화 증기)
    - State 2: 압축기 출구 (응축기 입구, 고압 과열 증기)
    - State 3: 응축기 출구 (팽창밸브 입구, 고압 포화 액체)
    - State 4: 팽창밸브 출구 (증발기 입구, 저압 액체+기체 혼합물)
    
    냉방 모드 (mode='cooling', 4-way 밸브로 인한 역순환):
    - State 1: 압축기 출구 (응축기 입구, 고압 과열 증기)
    - State 2: 압축기 입구 (증발기 출구, 저압 포화 증기)
    - State 3: 팽창밸브 출구 (증발기 입구, 저압 액체+기체 혼합물)
    - State 4: 응축기 출구 (팽창밸브 입구, 고압 포화 액체)
    
    알고리즘:
    1. 증발기와 응축기 압력 계산 (포화 압력)
    2. State 1: 저압 포화 증기 상태 계산
    3. State 2: 단열 압축 후 실제 압축(비단열) 계산
       - 등엔트로피 압축 후 엔탈피 계산 (이상적)
       - 압축기 효율을 고려한 실제 엔탈피 계산
    4. State 3: 고압 포화 액체 상태 계산
    5. State 4: 등엔탈피 팽창 (h4 = h3) 후 상태 계산
    
    호출 관계:
    - 호출자: _calculate_gshpb_next_step (DHW_main_engine.py)
    - 호출 함수: CoolProp.PropsSI (냉매 물성 계산)
    
    Args:
        - T_evap_K (float): 증발 온도 [K]
        - T_cond_K (float): 응축 온도 [K]
        - refrigerant (str): 냉매 이름 (CoolProp 형식, 예: 'R410A')
        - eta_cmp_isen (float): 압축기 단열 효율 [0-1]
            - 실제 압축 전력 = 이론 압축 전력 / eta_cmp_isen
        - T0_K (float, optional): 기준 온도 [K] (엑서지 계산용)
            - 제공되면 State 1-4의 엑서지 계산 수행
        - P0 (float, optional): 기준 압력 [Pa] (엑서지 계산용, 기본값: 101325)
        - mode (str, optional): 작동 모드 ('heating' 또는 'cooling', 기본값: 'heating')
            - 'heating': 난방 모드 (기본 계산, State 1=압축기 유입)
            - 'cooling': 냉방 모드 (4-way 밸브 역순환, State 2=압축기 유입으로 재매핑)
    
    Returns:
        dict: State 1-4의 물성치를 포함한 딕셔너리
        - P1, P2, P3, P4: 압력 [Pa] (모드에 따라 물리적 위치에 맞게 재매핑됨)
        - T1_K, T2_K, T3_K, T4_K: 온도 [K] (모드에 따라 물리적 위치에 맞게 재매핑됨)
        - h1, h2, h3, h4: 엔탈피 [J/kg] (모드에 따라 물리적 위치에 맞게 재매핑됨)
        - s1, s2, s3, s4: 엔트로피 [J/kgK] (모드에 따라 물리적 위치에 맞게 재매핑됨)
        - rho1: 압축기 유입 밀도 [kg/m³] (냉매 유량 계산에 사용)
        - x1, x2, x3, x4: 엑서지 [J/kg] (T0_K, P0가 제공된 경우, 모드에 따라 재매핑됨)
        - mode: 계산에 사용된 모드 ('heating' 또는 'cooling')
        
        물리적 위치 매핑:
        - 난방 모드: h1=압축기 유입, h2=압축기 유출, h3=응축기 출구, h4=팽창밸브 출구
        - 냉방 모드: h1=압축기 유출, h2=압축기 유입, h3=팽창밸브 출구, h4=응축기 출구 (4-way 밸브 역순환)
    
    Notes:
        - 엑서지 계산식: x = (h - h0) - T0 * (s - s0)
          여기서 (h0, s0)는 기준 상태(T0_K, P0)의 엔탈피와 엔트로피
        - State 2는 단열 효율을 고려한 실제 압축 과정을 반영
        - State 4는 등엔탈피 과정 (h4 = h3)으로 팽창밸브를 모델링
    """
    
    # 1단계: 증발기 및 응축기 압력 계산
    P1 = CP.PropsSI('P', 'T', T_evap_K, 'Q', 1, refrigerant)
    P3 = CP.PropsSI('P', 'T', T_cond_K, 'Q', 0, refrigerant)
    
    # 2. State 1 계산 - 압축기 입구 (저압 포화 증기)
    h1 = CP.PropsSI('H', 'P', P1, 'Q', 1, refrigerant)  # 엔탈피 [J/kg]
    s1 = CP.PropsSI('S', 'P', P1, 'Q', 1, refrigerant)  # 엔트로피 [J/kgK]
    rho = CP.PropsSI('D', 'P', P1, 'Q', 1, refrigerant)  # 밀도 [kg/m³] (냉매 유량 계산용)
    T1_K = T_evap_K  # 포화 증기이므로 증발 온도와 동일
    
    # 3. State 2 계산 - 압축기 출구 (고압 과열 증기)
    h2_isen = CP.PropsSI('H', 'P', P3, 'S', s1, refrigerant)  # 등엔트로피 압축 후 엔탈피
    
    h2 = h1 + (h2_isen - h1) / eta_cmp_isen
    T2_K = CP.PropsSI('T', 'P', P3, 'H', h2, refrigerant)  # 과열 온도
    P2 = P3  # 압력은 응축기 압력과 동일
    s2 = CP.PropsSI('S', 'P', P3, 'H', h2, refrigerant)  # 실제 엔트로피 (s1보다 큼)
    
    # 4. State 3 계산 - 응축기 출구 (고압 포화 액체)
    h3 = CP.PropsSI('H', 'P', P3, 'Q', 0, refrigerant)  # 포화 액체 엔탈피
    s3 = CP.PropsSI('S', 'P', P3, 'Q', 0, refrigerant)  # 포화 액체 엔트로피
    T3_K = T_cond_K  # 포화 액체이므로 응축 온도와 동일
    
    # 5. State 4 계산 - 팽창밸브 출구 (저압 액체+기체 혼합물)
    h4 = h3  # 등엔탈피 팽창
    P4 = P1  # 압력은 증발기 압력과 동일
    T4_K = CP.PropsSI('T', 'P', P1, 'H', h4, refrigerant)  # 저압에서 엔탈피 h4에 해당하는 온도
    s4 = CP.PropsSI('S', 'P', P1, 'H', h4, refrigerant)  # 팽창 후 엔트로피
    
    h0 = CP.PropsSI('H', 'T', T0_K, 'P', P0, refrigerant)
    s0 = CP.PropsSI('S', 'T', T0_K, 'P', P0, refrigerant)
    
    if mode == 'cooling':
        result = {
            # 냉방 모드 기준 물성치 (물리적 위치에 따라 재매핑)
            'P1': P2,
            'P2': P1,
            'P3': P4,
            'P4': P3,
            'T1_K': T2_K,
            'T2_K': T1_K,
            'T3_K': T4_K,
            'T4_K': T3_K,
            'h1': h2,
            'h2': h1,
            'h3': h4,
            'h4': h3,
            's1': s2,
            's2': s1,
            's3': s4,
            's4': s3,
            'rho': rho,
            'mode': 'cooling',
        }
    else:
        # 난방 모드: 기본 계산값 그대로 사용
        result = {
            'P1': P1,
            'P2': P2,
            'P3': P3,
            'P4': P4,
            'T1_K': T1_K,
            'T2_K': T2_K,
            'T3_K': T3_K,
            'T4_K': T4_K,
            'h1': h1,
            'h2': h2,
            'h3': h3,
            'h4': h4,
            's1': s1,
            's2': s2,
            's3': s3,
            's4': s4,
            'rho': rho,  # 압축기 유입 밀도 (State 1)
            'x1': (h1-h0) - T0_K*(s1 - s0),
            'x2': (h2-h0) - T0_K*(s2 - s0),
            'x3': (h3-h0) - T0_K*(s3 - s0),
            'x4': (h4-h0) - T0_K*(s4 - s0),
            'mode': 'heating',
        }
    
    return result


def create_lmtd_constraints():
    """
    LMTD(Log Mean Temperature Difference) 기반 제약 조건 함수들을 생성합니다.
    
    최적화 문제에서 열교환기의 실제 열전달량과 사이클 계산 열량이 일치해야 합니다:
    - Q_LMTD: LMTD와 UA값을 이용한 실제 열전달량 (Q = UA * LMTD)
    - Q_ref: 냉매 사이클 계산으로부터 얻은 열량 (Q = m_dot * (h_in - h_out))
    
    이 두 값이 같아야 물리적으로 일관된 해가 됩니다.
    
    호출 관계:
    - 호출자: find_ref_loop_optimal_operation (최적화 시 제약 조건으로 사용)
    - 사용: scipy.optimize.minimize의 제약 조건으로 전달
    
    물리적 의미:
    ──────────────────────────────────────────────────────────────────────────
    열교환기 모델링에서 두 가지 열량 계산 방법이 있습니다:
    
    1. LMTD 방법 (현실적 제약):
       Q_LMTD = UA * LMTD
       - UA: 총괄 열전달 계수 * 면적 [W/K]
       - LMTD: 대수 평균 온도차 [K]
       - 열교환기 물리적 특성 반영
    
    2. 냉매 사이클 방법 (이상적):
       Q_ref = m_dot * (h_in - h_out)
       - m_dot: 냉매 유량 [kg/s]
       - h_in, h_out: 입구/출구 엔탈피 [J/kg]
       - 에너지 보존 법칙 기반
    
    최적화 목표:
    Q_LMTD_cond - Q_ref_cond = 0  (응축기)
    Q_LMTD_evap - Q_ref_evap = 0  (증발기)
    
    이 제약 조건을 만족하는 최적 운전점에서:
    - 열교환기의 실제 열전달 능력과 사이클 열량이 일치
    - 물리적으로 현실 가능한 운전 조건
    
    Returns:
        list: 제약 조건 함수 리스트 (각 함수는 scipy.optimize.minimize에서 사용)
            - constraint_tank: 응축기(저탕조) 제약 조건
            - constraint_hx: 증발기(열교환기) 제약 조건
    
    Notes:
        - 제약 조건 함수는 최적화 중 performance 딕셔너리를 받아 제약 값 반환
        - 반환값이 0이 되어야 제약 조건 만족
        - perf가 None이면 큰 패널티(1e6) 반환하여 최적화에서 제외
    """
    def constraint_tank(perf):
        """
        응축기(저탕조) 제약 조건: Q_LMTD_cond - Q_ref_cond = 0
        
        응축기에서 냉매가 방출하는 열량(Q_ref_cond)과 응축기가 응축기  측 두 냉매 지점의 온도 값과 열전달계수에 기반해
        전달가능한 열량(Q_LMTD_cond)이 일치해야 합니다.
        이 제약 조건은 열교환기 물리적 특성과 사이클 열량의 일관성을 보장하는 제약 조건입니다.
        
        Args:
            perf (dict): 사이클 성능 결과 딕셔너리
                - Q_LMTD_cond (float): LMTD 기반 응축기 열량 [W]
                - Q_ref_cond (float): 사이클 계산 응축기 열량 [W]
        
        Returns:
            float: 제약 조건 값
                - 0: 제약 조건 만족
                - != 0: 제약 조건 불만족 (최적화 알고리즘이 이 값을 0에 가깝게 만들려고 시도)
                - 1e6: perf가 None인 경우 (물리적으로 불가능한 상태)
        """
        if perf is None:
            return 1e6  # 물리적으로 불가능한 상태에 대한 큰 패널티
        return perf.get('Q_LMTD_cond', 0) - perf.get('Q_ref_cond', 0)
    
    def constraint_hx(perf):
        """
        증발기(열교환기) 제약 조건: Q_LMTD_evap - Q_ref_evap = 0
        
        증발기에서 냉매가 흡수하는 열량(Q_ref_evap)과 증발기가 증발기 측 두 냉매 지점의 온도 값과 열전달계수에 기반해
        전달가능한 열량(Q_LMTD_evap)이 일치해야 합니다.
        일치해야 합니다. 이 제약 조건은 지중열교환기 물리적 특성과 사이클 열량의 일관성을 보장합니다.
        
        Args:
            perf (dict): 사이클 성능 결과 딕셔너리
                - Q_LMTD_evap (float): LMTD 기반 증발기 열량 [W]
                - Q_ref_evap (float): 사이클 계산 증발기 열량 [W]
        
        Returns:
            float: 제약 조건 값
                - 0: 제약 조건 만족
                - != 0: 제약 조건 불만족 (최적화 알고리즘이 이 값을 0에 가깝게 만들려고 시도)
                - 1e6: perf가 None인 경우 (물리적으로 불가능한 상태)
        """
        if perf is None:
            return 1e6  # 물리적으로 불가능한 상태에 대한 큰 패널티
        return perf.get('Q_LMTD_evap', 0) - perf.get('Q_ref_evap', 0)
    
    return [constraint_tank, constraint_hx]


def find_ref_loop_optimal_operation(
    calculate_performance_func,  # 사이클 성능 계산 함수 (사용자 정의)
    T_tank_w,  # 저탕조 온도 [°C]
    T_oa,  # 실외 공기 온도 [°C]
    Q_cond_load,  # 저탕조 목표 열 교환율 [W]
    Q_cond_LOAD_OFF_TOL=500.0,  # OFF 임계값 [W]
    bounds=None,  # 최적화 변수 경계 [(min, max), ...]
    initial_guess=None,  # 초기 추정값
    constraint_funcs=None,  # 제약 조건 함수 리스트
    off_result_formatter=None,  # OFF 상태 결과 포맷팅 함수
):
    """
    냉매 루프 최적 운전점을 찾는 함수.
    
    이 함수는 히트펌프 시스템에서 주어진 목표 열 교환율(Q_cond_load)을 만족하면서
    압축기 전력을 최소화하는 최적 운전점을 탐색합니다.
    
    최적화 문제:
    ──────────────────────────────────────────────────────────────────────────
    목적 함수 (minimize): E_cmp (압축기 전력 [W])
    
    최적화 변수:
    - optimization_vars[0]: dT_ref_HX (냉매-열교환기 온도차 [K])
    - optimization_vars[1]: dT_ref_tank (냉매-저탕조 온도차 [K])
    
    제약 조건:
    - Q_LMTD_cond - Q_ref_cond = 0  (응축기 열량 일치)
    - Q_LMTD_evap - Q_ref_evap = 0  (증발기 열량 일치)
    
    알고리즘:
    ──────────────────────────────────────────────────────────────────────────
    1. OFF 상태 판단: Q_cond_load가 임계값 이하이면 OFF 상태 처리
    2. 최적화 변수 초기화: bounds, initial_guess 설정
    3. 목적 함수 정의: E_cmp를 최소화
    4. 제약 조건 설정: LMTD 기반 제약 조건 추가
    5. SLSQP 알고리즘으로 최적화 실행
    6. 최적해 검증 및 결과 반환
    
    호출 관계:
    ──────────────────────────────────────────────────────────────────────────
    호출자: 
    AirSourceHeatPumpBoiler.run_simulation (DHW_main_engine.py) 또는
    GroundSourceHeatPumpBoiler.run_simulation (DHW_main_engine.py)
        ↓
    find_ref_loop_optimal_operation (본 함수)
        ├─ calculate_performance_func 호출 (최적화 반복 중 여러 번)
        │   └─ _calculate_gshpb_next_step (DHW_main_engine.py)
        │       └─ compute_refrigerant_thermodynamic_states 호출
        ├─ constraint_funcs 호출 (제약 조건 평가)
        │   └─ create_lmtd_constraints() 반환 함수들
        └─ off_result_formatter 호출 (OFF 상태 시)
            └─ _format_gshpb_off_results_dict (DHW_main_engine.py)
    
    데이터 흐름:
    ──────────────────────────────────────────────────────────────────────────
    [T_tank_w, Q_cond_load, optimization_vars]
        ↓
    calculate_performance_func
        ↓ [성능 딕셔너리: E_cmp, Q_LMTD_cond, Q_LMTD_evap, ...]
    최적화 알고리즘 (반복)
        ↓
    [최적 optimization_vars]
        ↓
    calculate_performance_func (최종 1회)
        ↓
    [최적 운전점 결과]
    
    Args:
        calculate_performance_func (callable): 
            사이클 성능 계산 함수 (사용자 정의).
            시그니처: calculate_performance_func(optimization_vars, T_tank_w, Q_cond_load) -> dict
            
            입력:
                - optimization_vars (list): 최적화 변수 배열
                  예: [dT_ref_HX, dT_ref_tank] (온도차 [K])
                - T_tank_w (float): 저탕조 온도 [°C]
                - Q_cond_load (float): 저탕조 목표 열 교환율 [W]
            
            출력 (dict):
                - E_cmp (float): 압축기 전력 [W] (목적 함수)
                - Q_LMTD_cond (float): LMTD 기반 응축기 열량 [W] (제약 조건)
                - Q_ref_cond (float): 사이클 계산 응축기 열량 [W] (제약 조건)
                - Q_LMTD_evap (float): LMTD 기반 증발기 열량 [W] (제약 조건)
                - Q_ref_evap (float): 사이클 계산 증발기 열량 [W] (제약 조건)
                - 기타 성능 데이터
        
        T_tank_w (float): 저탕조 목표 온도 [°C]
        Q_cond_load (float): 저탕조 목표 열 교환율 [W]
        Q_cond_LOAD_OFF_TOL (float): OFF 임계값 [W] (기본값: 500.0)
            |Q_cond_load| <= 이 값이면 히트펌프를 OFF 상태로 처리
        
        bounds (list, optional): 최적화 변수 경계 [(min, max), ...]
            기본값: [(0.1, 30.0), (0.1, 30.0)] (두 변수 모두 0.1~30.0 K)
        
        initial_guess (list, optional): 초기 추정값
            기본값: [5.0, 5.0] (온도차 5K로 시작)
        
        constraint_funcs (list, optional): 제약 조건 함수 리스트
            각 함수는 perf (dict)를 받아 제약 조건 값을 반환
            기본값: None (제약 조건 없음)
            예: create_lmtd_constraints() 반환값
        
        off_result_formatter (callable, optional): 
            OFF 상태 결과 포맷팅 함수.
            시그니처: off_result_formatter(T_tank_w, Q_cond_load, T_oa=T_oa) -> dict
            기본값: None (기본 OFF 로직 사용)
    
    Returns:
        dict: 최적화 결과 또는 OFF 상태 결과
            - 최적화 성공 시: calculate_performance_func의 반환값 (최적 운전점)
            - OFF 상태 시: off_result_formatter의 반환값 또는 기본 OFF 결과
            - 최적화 실패 시: None
    
    Notes:
        - SLSQP (Sequential Least Squares Programming) 알고리즘 사용
        - 최적화는 비선형 제약 조건을 포함한 비선형 최적화 문제
        - calculate_performance_func는 최적화 중 여러 번 호출됨 (수렴할 때까지)
        - 최종적으로는 최적해에서 한 번 더 호출하여 정확한 결과 반환
    """

    # 1단계: OFF 상태 판단 및 처리
    if abs(Q_cond_load) <= Q_cond_LOAD_OFF_TOL:
        if off_result_formatter is not None:
            return off_result_formatter(T_tank_w, Q_cond_load, T_oa=T_oa)
        else:
            try:
                dummy_vars = initial_guess if initial_guess is not None else [5.0, 5.0]
                result_template = calculate_performance_func(
                    optimization_vars=dummy_vars,
                    T_tank_w=T_tank_w,
                    Q_cond_load=0.0,
                    T_oa=T_oa,
                )
                if result_template is None:
                    return None
                
                # 모든 숫자 값을 0.0으로 설정 (히트펌프 OFF 상태)
                for key, value in result_template.items():
                    if isinstance(value, (int, float)):
                        result_template[key] = 0.0
                
                result_template['is_on'] = False  # OFF 상태 플래그 설정
                return result_template
            except Exception:
                return None
    

    # 제약 조건 함수 리스트 초기화
    if constraint_funcs is None:
        constraint_funcs = []
    
    # ============================================================
    # 3단계: 목적 함수 정의
    # ============================================================
    # 목적: 압축기 전력(E_cmp) 최소화
    # 이 함수는 최적화 알고리즘에 의해 반복 호출됨
    def objective(x):
        """
        목적 함수: 압축기 전력 최소화
        
        Args:
            x (array): 최적화 변수 [dT_ref_HX, dT_ref_tank]
        
        Returns:
            float: 압축기 전력 [W] (최소화 대상)
        """
        try:
            # 주어진 최적화 변수로 사이클 성능 계산
            perf = calculate_performance_func(
                optimization_vars=x,
                T_tank_w=T_tank_w,
                Q_cond_load=Q_cond_load
            )
            if perf is None:
                return 1e6  # 계산 실패 시 큰 패널티
            return perf.get("E_cmp", 1e6)  # 압축기 전력 반환
        except Exception:
            return 1e6  # 예외 발생 시 큰 패널티
    
    # 4단계: 제약 조건 설정
    cons = []
    for constraint_func in constraint_funcs:
        # 클로저를 사용하여 각 제약 조건 함수를 올바르게 바인딩
        def make_constraint(cf):
            """
            제약 조건 함수 래퍼
            
            Args:
                cf (callable): 원본 제약 조건 함수 (perf를 받아 제약 값 반환)
            
            Returns:
                callable: 최적화 변수를 받는 제약 조건 함수
            """
            def constraint(x):
                """
                최적화 변수를 받는 제약 조건 함수
                
                Args:
                    x (array): 최적화 변수 [dT_ref_HX, dT_ref_tank]
                
                Returns:
                    float: 제약 조건 값 (0이 되어야 함)
                """
                # 최적화 변수로 성능 계산
                perf = calculate_performance_func(
                    optimization_vars=x,
                    T_tank_w=T_tank_w,
                    Q_cond_load=Q_cond_load
                )
                if perf is None:
                    return 1e6  # 계산 실패 시 큰 패널티
                # 원본 제약 조건 함수 호출 (LMTD 기반 제약)
                return cf(perf)
            return constraint
        cons.append({'type': 'eq', 'fun': make_constraint(constraint_func)})
    
    # 5단계: 최적화 실행
    try:
        result = minimize(
            objective,              # 목적 함수 (E_cmp 최소화)
            initial_guess,          # 초기 추정값
            method='SLSQP',         # Sequential Least Squares Programming
            bounds=bounds,          # 변수 경계 조건
            constraints=cons if cons else None,  # 등식 제약 조건
            options={'disp': False}  # 상세 출력 비활성화
        )
        
        # 최적화 성공 여부 확인
        if result.success:
            # 최적해에서 최종 성능 계산 (정확한 결과 얻기 위해)
            optimal_vars = result.x
            final_performance = calculate_performance_func(
                optimization_vars=optimal_vars,
                T_tank_w=T_tank_w,
                Q_cond_load=Q_cond_load
            )
            return final_performance
        else:
            # 최적화 실패 (수렴하지 못함)
            print(f'최적화에 실패했습니다: {result.message}')
            return None
    except Exception as e:
        # 최적화 중 예외 발생
        print(f'최적화 중 오류 발생: {e}')
        return None


def plot_cycle_diagrams(
    result,
    refrigerant,
    show=True,
    time_step_annotation=None,
    save_path=None,
    show_temp_limits=False,
    T_tank_w=None,
    T_b_f_in=None,
    T_b_f_out=None,
    face_color='#F9F8F6',
):
    """
    계산된 사이클 상태(1,2,3,4)를 바탕으로 P-h 및 T-h 선도를 그립니다.
    
    이 함수는 히트펌프 사이클의 열역학적 상태를 시각화합니다:
    - P-h 선도 (Pressure-Enthalpy): 압력 대 엔탈피 (로그 스케일)
    - T-h 선도 (Temperature-Enthalpy): 온도 대 엔탈피
    
    호출 관계:
    - 호출자: GroundSourceHeatPumpBoiler.run_simulation (save_ph_diagram=True일 때)
    - 사용 데이터: compute_refrigerant_thermodynamic_states 결과
    
    플로팅 단계:
    ──────────────────────────────────────────────────────────────────────────
    1. 냉매 물성 데이터 준비 (포화선, 임계점)
    2. 사이클 상태점 데이터 추출 (State 1-4)
    3. Figure 및 Axes 생성 (1행 2열: P-h, T-h)
    4. 포화선 그리기 (액체선, 증기선)
    5. 사이클 경로 그리기 (State 1→2→3→4→1)
    6. 온도 제한선 표시 (선택적)
    7. 상태점 라벨링 (1, 2, 3, 4)
    8. 축 설정 및 저장/표시
    
    Args:
        result (dict): 사이클 성능 결과 딕셔너리
            - P1, P2, P3, P4 (float): 압력 [Pa]
            - h1, h2, h3, h4 (float): 엔탈피 [J/kg]
            - T1, T2, T3, T4 (float): 온도 [°C]
                주의: result 딕셔너리에는 'T1', 'T2' 등이 °C 단위로 저장되어야 함
        
        refrigerant (str): 냉매 이름 (CoolProp 형식, 예: 'R410A')
        
        show (bool, optional): 그래프를 화면에 표시할지 여부 (기본값: True)
        
        time_step_annotation (str, optional): 타임스텝 주석 텍스트
            예: "Time step: 100", "Day 1, Hour 12" 등
        
        save_path (str, optional): 그래프 저장 경로
            제공되면 이미지 파일로 저장 (PNG, DPI 400)
        
        show_temp_limits (bool, optional): 온도 제한선 표시 여부 (기본값: False)
            True일 때 T-h 선도에 저탕조 및 지열교환기 온도선 표시
        
        T_tank_w (float, optional): 저탕조 온도 [°C]
            show_temp_limits=True일 때 사용
        
        T_b_f_in (float, optional): 지열교환기 유입수 온도 [°C]
            show_temp_limits=True일 때 사용
        
        T_b_f_out (float, optional): 지열교환기 유출수 온도 [°C]
            show_temp_limits=True일 때 사용
        
        face_color (str, optional): 그래프 배경색 (기본값: '#F9F8F6')
    
    Returns:
        None
    
    Raises:
        ImportError: dartwork_mpl 모듈이 설치되지 않은 경우
    
    Notes:
        - P-h 선도는 압력 축이 로그 스케일
        - 사이클 경로는 점선으로 표시되고 각 상태점에 마커 표시
        - 동일 좌표의 상태점은 "1,2" 형태로 그룹 표시
    """
    # ============================================================
    # 0단계: 의존성 확인
    # ============================================================
    if dm is None:
        raise ImportError("dartwork_mpl 모듈이 필요합니다. pip install dartwork_mpl로 설치하세요.")
    
    # ============================================================
    # 1단계: 색상 및 축 범위 설정
    # ============================================================
    # 그래프 색상 정의
    color1 = 'dm.blue5'   # 포화 액체선
    color2 = 'dm.red5'    # 포화 증기선
    color3 = 'dm.black'   # 사이클 경로 마커

    # P-h 선도 축 범위 (압력: 로그 스케일)
    ymin1, ymax1, yint1 = 500, 10**4, 0  # 압력 [kPa]: 500 ~ 10000
    
    # T-h 선도 축 범위 (온도: 선형 스케일)
    ymin2, ymax2, yint2 = -20, 120, 20  # 온도 [°C]: -20 ~ 120
    
    # 공통 X축 범위 (엔탈피)
    xmin, xmax, xint = 0, 600, 100  # 엔탈피 [kJ/kg]: 0 ~ 600

    # ============================================================
    # 2단계: 냉매 물성 데이터 준비 (포화선 계산)
    # ============================================================
    # 임계점 계산 (참고용)
    T_critical = cu.K2C(CP.PropsSI('Tcrit', refrigerant))  # 임계 온도 [°C]
    P_critical = CP.PropsSI('Pcrit', refrigerant) / 1000  # 임계 압력 [kPa] (참고용)

    # 포화선 계산을 위한 온도 범위 설정
    # 최소 온도부터 임계 온도까지 200개 포인트
    temps = np.linspace(cu.K2C(CP.PropsSI('Tmin', refrigerant)) + 1, T_critical, 200)
    
    # 각 온도에서 포화 액체 및 포화 증기 엔탈피 계산
    # 단위 변환: J/kg → kJ/kg
    h_liq = [CP.PropsSI('H', 'T', cu.C2K(T), 'Q', 0, refrigerant) / 1000 for T in temps]  # 포화 액체선
    h_vap = [CP.PropsSI('H', 'T', cu.C2K(T), 'Q', 1, refrigerant) / 1000 for T in temps]  # 포화 증기선
    
    # 각 온도에서 포화 압력 계산
    # 단위 변환: Pa → kPa
    p_sat = [CP.PropsSI('P', 'T', cu.C2K(T), 'Q', 0, refrigerant) / 1000 for T in temps]

    # ============================================================
    # 3단계: 사이클 상태점 데이터 추출
    # ============================================================
    # State 1-4의 압력, 엔탈피, 온도 추출
    # 단위 변환: Pa → kPa, J/kg → kJ/kg
    p = np.array([result[f'P{i}'] for i in range(1, 5)]) * cu.Pa2kPa  # 압력 [kPa]
    h = np.array([result[f'h{i}'] for i in range(1, 5)]) * cu.J2kJ   # 엔탈피 [kJ/kg]
    T = np.array([result[f'T{i}'] for i in range(1, 5)])             # 온도 [°C]

    # ============================================================
    # 4단계: 사이클 경로 구성 (닫힌 경로)
    # ============================================================
    # State 1→2→3→4→1 순서로 닫힌 경로 만들기
    # 첫 상태점을 끝에 추가하여 경로를 닫음
    h_cycle = np.concatenate([h, h[:1]])  # [h1, h2, h3, h4, h1]
    p_cycle = np.concatenate([p, p[:1]])  # [p1, p2, p3, p4, p1]
    T_cycle = np.concatenate([T, T[:1]])  # [T1, T2, T3, T4, T1]

    # ============================================================
    # 5단계: Figure 및 Axes 생성
    # ============================================================
    # 선 두께 배열 (그래프 요소별로 다른 두께 사용)
    LW = np.arange(0.5, 3.0, 0.25)
    
    # 1행 2열 서브플롯 생성 (P-h 선도, T-h 선도)
    nrows, ncols = 1, 2
    fig, axes = plt.subplots(figsize=(dm.cm2in(16), dm.cm2in(6)), nrows=nrows, ncols=ncols)
    plt.subplots_adjust(wspace=0.5)  # 서브플롯 간 간격
    ax = axes.flatten()  # 1차원 배열로 변환

    # ============================================================
    # 6단계: 축별 메타데이터 설정
    # ============================================================
    # 각 서브플롯(idx=0: P-h, idx=1: T-h)의 축 라벨 및 스케일
    xlabels = ["Enthalpy [kJ/kg]", "Enthalpy [kJ/kg]"]  # 공통 X축: 엔탈피
    ylabels = ["Pressure (log scale) [kPa]", "Temperature [°C]"]  # Y축: 압력 또는 온도
    yscales = ["log", "linear"]  # Y축 스케일: 로그 또는 선형
    xlims   = [(xmin, xmax), (xmin, xmax)]  # X축 범위
    ylims   = [(ymin1, ymax1), (ymin2, ymax2)]  # Y축 범위

    # 포화선/사이클 Y데이터 선택자
    # idx=0 (P-h 선도): 포화선은 p_sat, 사이클은 p_cycle
    # idx=1 (T-h 선도): 포화선은 temps, 사이클은 T_cycle
    satY_list   = [p_sat, temps]  # 포화선 Y 데이터
    cycleY_list = [p_cycle, T_cycle]  # 사이클 경로 Y 데이터

    # 상태점 라벨 Y좌표 계산 함수 (축별로 다른 오프셋 적용)
    def state_y(idx, i):
        """상태점 라벨 Y좌표 계산"""
        if idx == 0:  # P-h 선도: 압력 위로 5% 오프셋
            return p[i] * 1.05
        else:  # T-h 선도: 온도 위로 고정 오프셋
            return T[i] + yint2 * 0.1

    # ============================================================
    # 7단계: 범례 스타일 설정
    # ============================================================
    # 공통 범례 스타일 (두 서브플롯 모두 동일)
    legend_kw = dict(
        loc='upper left',          # 범례 위치
        bbox_to_anchor=(0.0, 0.99),  # 범례 앵커 포인트
        handlelength=1.5,          # 범례 항목 길이
        labelspacing=0.5,          # 범례 항목 간 간격
        columnspacing=2,           # 범례 열 간 간격
        ncol=1,                    # 범례 열 수
        frameon=False,             # 범례 프레임 없음
        fontsize=dm.fs(-1.5)       # 범례 폰트 크기
    )

    # ============================================================
    # 8단계: 각 서브플롯에 그래프 그리기
    # ============================================================
    # 2중 for문으로 P-h 선도(idx=0)와 T-h 선도(idx=1) 생성
    for r in range(nrows):
        for c in range(ncols):
            idx = r * ncols + c  # 서브플롯 인덱스 (0 또는 1)
            axi = ax[idx]

            # --------------------------------------------
            # 8-1: 포화선 그리기
            # --------------------------------------------
            # 포화 액체선: 저압 영역에서 왼쪽 경계
            axi.plot(h_liq, satY_list[idx],  color=color1, label='Saturated liquid', 
                    linewidth=LW[2])
            
            # 포화 증기선: 저압 영역에서 오른쪽 경계
            axi.plot(h_vap, satY_list[idx],  color=color2, label='Saturated vapor',  
                    linewidth=LW[2])
            
            # --------------------------------------------
            # 8-2: 사이클 경로 그리기
            # --------------------------------------------
            # State 1→2→3→4→1 순서의 닫힌 경로
            # 점선(:)과 원형 마커(o)로 표시
            axi.plot(h_cycle, cycleY_list[idx], color='dm.gray5', label='Heat pump cycle',
                     markerfacecolor=color3, markeredgecolor=color3,
                     linewidth=LW[1], marker='o', linestyle=':', markersize=2)

            # --------------------------------------------
            # 8-3: 온도 제한선 표시 (T-h 선도만, 선택적)
            # --------------------------------------------
            # T-h 선도(idx=1)에만 온도 제한선 표시
            if show_temp_limits and (idx == 1):
                # 저탕조 온수 온도선 (빨간색)
                if T_tank_w is not None:
                    ax[1].axhline(y=T_tank_w, color='dm.red4', linestyle=':', linewidth=LW[1])
                    ax[1].text(xmin + 20, T_tank_w + 2,
                               f'Tank water: {T_tank_w:.1f} °C', ha='left', va='bottom', 
                               fontsize=dm.fs(-3), color='dm.red6')
                
                # 지열교환기 유입수 온도선 (파란색)
                if T_b_f_in is not None:
                    ax[1].axhline(y=T_b_f_in, color='dm.blue4', linestyle=':', linewidth=LW[1])
                    ax[1].text(xmin + 20, T_b_f_in - 2,
                               f'GHE inlet: {T_b_f_in:.1f} °C', ha='left', va='top', 
                               fontsize=dm.fs(-3), color='dm.blue6')
                
                # 지열교환기 유출수 온도선 (주황색)
                if T_b_f_out is not None:
                    ax[1].axhline(y=T_b_f_out, color='dm.orange4', linestyle=':', linewidth=LW[1])
                    ax[1].text(xmin + 20, T_b_f_out + 2,
                               f'GHE outlet: {T_b_f_out:.1f} °C', ha='left', va='bottom', 
                               fontsize=dm.fs(-3), color='dm.orange6')

            # --------------------------------------------
            # 8-4: 상태점 라벨 표시
            # --------------------------------------------
            # 동일 좌표의 상태점을 그룹핑하여 "1,2" 형태로 표시
            # x축: 엔탈피(h), y축: 압력(p) 또는 온도(T)
            xdata = h  # 모든 서브플롯에서 X축은 엔탈피
            ydata = p if idx == 0 else T  # Y축은 서브플롯에 따라 다름

            # 상태점 좌표 그룹핑 (소수점 3자리 반올림으로 동일 좌표 판단)
            groups = {}
            for i, (xi, yi) in enumerate(zip(xdata, ydata), start=1):
                key = (round(float(xi), 3), round(float(yi), 3))
                groups.setdefault(key, []).append(i)

            # 각 그룹에 대해 라벨 표시
            for (xg, yg), labels in groups.items():
                label_text = ",".join(map(str, labels))  # "1,2" 형태
                # 상태점 위로 3포인트 오프셋하여 라벨 표시
                axi.annotate(label_text, (xg, yg),
                             xytext=(0, 3), textcoords='offset points',
                             ha='center', va='bottom',
                             fontsize=dm.fs(-1.5))

            # --------------------------------------------
            # 8-5: 축 설정 및 범례 추가
            # --------------------------------------------
            # 축 라벨 설정
            axi.set_xlabel(xlabels[idx], fontsize=dm.fs(0), labelpad=6)
            axi.set_ylabel(ylabels[idx], fontsize=dm.fs(0), labelpad=6)
            
            # 눈금 표시 설정
            axi.tick_params(axis='both', which='major', labelsize=dm.fs(-1), pad=4)
            
            # Y축 스케일 설정 (로그 또는 선형)
            axi.set_yscale(yscales[idx])
            
            # 축 범위 설정
            axi.set_xlim(*xlims[idx])
            axi.set_ylim(*ylims[idx])
            
            # 범례 추가
            axi.legend(**legend_kw)
    
    # ============================================================
    # 9단계: 타임스텝 주석 및 레이아웃 조정
    # ============================================================
    # 타임스텝 주석이 제공된 경우 우측 상단에 표시
    if time_step_annotation is not None:
        fig.text(0.97, 0.92, 
                 time_step_annotation,
                 ha='right',
                 va='bottom',
                 fontweight='bold',
                 fontsize=dm.fs(0))
        # 주석 공간 확보를 위해 레이아웃 조정
        dm.simple_layout(fig, margins=(0.05, 0.05, 0.05, 0.05), bbox=(0, 1, 0, 0.95), verbose=False)
    else:
        # 주석 없는 경우 전체 영역 사용
        dm.simple_layout(fig, margins=(0.05, 0.05, 0.05, 0.05), bbox=(0, 1, 0, 1), verbose=False)
    
    # ============================================================
    # 10단계: 그래프 저장 및 표시
    # ============================================================
    # 이미지 파일로 저장 (선택적)
    if save_path is not None:
        # 서브플롯 배경을 투명하게 설정
        for axi in axes.flatten():
            axi.patch.set_alpha(0.0)
        # 고해상도 이미지로 저장 (DPI 400)
        plt.savefig(save_path, dpi=400, facecolor=face_color)
    
    # 화면에 표시 (선택적)
    if show:
        dm.save_and_show(fig)
    
    # 메모리 해제
    plt.close()


def update_tank_temperature(
    T_tank_w_K,      # 현재 탱크 온도 [K]
    Q_tank_in,       # 탱크 입력 열량 [W]
    total_loss,      # 총 손실 [W] (Q_tank_loss + Q_use_loss)
    C_tank,          # 탱크 열용량 [J/K]
    dt,              # 타임스텝 [s]
):
    """
    탱크 온도를 업데이트합니다.
    
    열량 밸런스 기반으로 다음 타임스텝의 탱크 온도를 계산합니다:
    Q_net = Q_tank_in - total_loss
    T_tank_w_K_new = T_tank_w_K + (Q_net / C_tank) * dt
    
    Parameters:
    -----------
    T_tank_w_K : float
        현재 탱크 온도 [K]
    Q_tank_in : float
        탱크 입력 열량 [W] (예: 응축기에서 전달되는 열량)
    total_loss : float
        총 손실 [W] (탱크 외벽 손실 + 급탕 사용 손실)
    C_tank : float
        탱크 열용량 [J/K] (C_tank = c_w * rho_w * V_tank)
    dt : float
        타임스텝 [s]
    
    Returns:
    --------
    float
        업데이트된 탱크 온도 [K]
    
    Notes:
    ------
    - 열량 밸런스: Q_net = Q_tank_in - total_loss
    - 온도 변화: dT = Q_net * dt / C_tank
    - 다음 타임스텝 온도: T_new = T_old + dT
    """
    Q_net = Q_tank_in - total_loss
    dT = (Q_net / C_tank) * dt
    T_tank_w_K_new = T_tank_w_K + dT
    return T_tank_w_K_new

