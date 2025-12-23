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
from scipy.optimize import curve_fit, root_scalar
from scipy import integrate
from scipy.special import erf
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


def calc_lmtd_constant_refrigerant_temp(T_ref_avg_K, T_air_in_K, T_air_out_K):
    """
    Calculate LMTD when refrigerant temperature is constant.
    
    The refrigerant maintains a constant average temperature, while air temperature
    changes from inlet to outlet. This applies to condensers or evaporators where
    the refrigerant undergoes phase change.
    
    Parameters:
    -----------
    T_ref_avg_K : float
        Refrigerant average temperature [K] (constant)
    T_air_in_K : float
        Air inlet temperature [K]
    T_air_out_K : float
        Air outlet temperature [K]
    
    Returns:
    --------
    float
        LMTD [K]
    
    Notes:
    ------
    - Since refrigerant temperature is constant, LMTD is calculated in simplified form.
    - Q>0 (refrigerant releases heat): T_ref_avg > T_air_in, T_ref_avg > T_air_out
      → dT_in = T_ref_avg - T_air_in, dT_out = T_ref_avg - T_air_out
    - Q<0 (refrigerant absorbs heat): T_ref_avg < T_air_in, T_ref_avg < T_air_out
      → dT_in = T_air_in - T_ref_avg, dT_out = T_air_out - T_ref_avg
    """
    # Temperature difference calculation (maintain sign)
    dT_in = T_ref_avg_K - T_air_in_K
    dT_out = T_ref_avg_K - T_air_out_K
    
    # Physical validity check: dT_in and dT_out must have same sign
    if dT_in * dT_out <= 0:
        # Refrigerant temperature is between air inlet and outlet (physically impossible)
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


def calculate_heat_transfer_coefficient(dV_fan, dV_fan_design, A_cross, UA):
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


def find_fan_airflow_for_heat_transfer(Q_ref_target, T_air_in_C, T_ref_in_K, T_ref_out_K, A_cross=None, UA_design=None, UA=None, dV_fan_design=None):
    """
    Calculate fan airflow rate to achieve target heat transfer.
    
    Considers dynamically changing heat transfer coefficient (UA) based on velocity.
    UA is proportional to velocity^0.8 according to Dittus-Boelter equation.
    
    Parameters:
    -----------
    Q_ref_target : float
        Target heat transfer rate [W]
    T_air_in_C : float
        Air inlet temperature [°C]
    T_ref_in_K : float
        Refrigerant inlet temperature [K]
    T_ref_out_K : float
        Refrigerant outlet temperature [K]
    A_cross : float, optional
        Heat exchanger cross-sectional area [m²]
        If None, UA_design/UA is used directly without velocity correction
    UA_design : float, optional
        Heat transfer coefficient at design flow rate [W/K]
        If None, UA parameter is used instead
    UA : float, optional
        Heat transfer coefficient [W/K] (alias for UA_design for backward compatibility)
        If provided, overrides UA_design
    dV_fan_design : float, optional
        Design fan flow rate [m³/s]
        Required if A_cross is provided
    
    Returns:
    --------
    float
        Required airflow rate [m³/s]
    """
    # Handle UA parameter (backward compatibility)
    if UA is not None:
        UA_design = UA
    
    if UA_design is None:
        raise ValueError("Either UA_design or UA must be provided")
    
    T_air_in_K = cu.C2K(T_air_in_C)

    def error_function(dV_fan):
        if dV_fan <= 0:
            return 1e8
        T_air_out_K = T_air_in_K + Q_ref_target / (c_a * rho_a * dV_fan)
        max_dT_air = 30.0
        if abs(T_air_out_K - T_air_in_K) > max_dT_air:
            return 1e8
        T_ref_avg_K = (T_ref_in_K + T_ref_out_K) / 2
        TEMP_TOLERANCE_K = 1e-6
        if Q_ref_target > 0:
            if not (T_ref_avg_K > max(T_air_in_K, T_air_out_K) - TEMP_TOLERANCE_K):
                return 1e8
        else:
            if not (T_ref_avg_K < min(T_air_in_K, T_air_out_K) + TEMP_TOLERANCE_K):
                return 1e8

        # Calculate UA based on whether A_cross is provided
        if A_cross is not None and dV_fan_design is not None:
            UA = calculate_heat_transfer_coefficient(dV_fan, dV_fan_design, A_cross, UA_design)
        else:
            # Use constant UA if A_cross is not provided
            UA = UA_design
            
        LMTD = calc_lmtd_constant_refrigerant_temp(
            T_ref_avg_K=T_ref_avg_K,
            T_air_in_K=T_air_in_K,
            T_air_out_K=T_air_out_K
        )
        if np.isnan(LMTD):
            return 1e8
        Q_calculated = UA * LMTD
        return Q_calculated - Q_ref_target

    max_dT_air = 30.0
    if abs(Q_ref_target) > 0:
        dV_min = abs(Q_ref_target) / (c_a * rho_a * max_dT_air)
        dV_min = max(0.01, dV_min)
    else:
        dV_min = 0.01
    dV_max = 10.0

    sol = root_scalar(error_function, bracket=[dV_min, dV_max], method='bisect')
    if sol.converged:
        return sol.root
    else:
        raise ValueError("Fan flow rate solution did not converge.")


def calculate_fan_power(dV_fan, fan_params, vsd_coeffs):
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
    if dV_fan <= 0:
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

