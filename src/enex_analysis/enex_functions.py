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

import numpy as np
import pandas as pd
from datetime import datetime
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


__all__ = [
    # Friction and Flow
    'darcy_friction_factor',
    'calc_h_vertical_plate',
    # Curve-fit helpers
    'linear_function',
    'quadratic_function',
    'cubic_function',
    'quartic_function',
    # Balance printing
    'print_balance',
    # Heat Pump COP
    'calc_ASHP_cooling_COP',
    'calc_ASHP_heating_COP',
    'calc_GSHP_COP',
    # G-function
    'f',
    'chi',
    'G_FLS',
    # Air properties
    'air_dynamic_viscosity',
    'air_prandtl_number',
    # Exergy / Entropy
    'generate_entropy_exergy_term',
    'calc_energy_flow',
    'calc_exergy_flow',
    # Flow and Mixing
    'calc_mixing_valve',
    'calc_uv_lamp_power',
    'calc_Orifice_flow_coefficient',
    'calc_boussinessq_mixing_flow',
    # Tank Heat Transfer
    'calc_UA_tank_arr',
    # TDMA
    'TDMA',
    # Tank UA
    'calc_simple_tank_UA',
    # LMTD
    'calc_LMTD_counter_flow',
    'calc_LMTD_parallel_flow',
    # HX / Fan
    'calc_UA_from_dV_fan',
    'calc_HX_perf_for_target_heat',
    'calc_fan_power_from_dV_fan',
    # Schedule / Control
    'check_hp_schedule_active',
    'build_dhw_usage_ratio',
    # UV
    'get_uv_params_from_turbidity',
    'calc_uv_exposure_time',
    # DHW
    'make_dhw_schedule_from_Annex_42_profile',
    'calc_total_water_use_from_schedule',
    # Cold water
    'calc_cold_water_temp',
    # Refrigerant cycle
    'calc_ref_state',
    'create_lmtd_constraints',
    'find_ref_loop_optimal_operation',
    # Tank temperature
    'update_tank_temperature',
    # Refrigerant exergy
    'calc_refrigerant_exergy',
    # Electricity → exergy
    'convert_electricity_to_exergy',
    # Data loading
    'load_kma_solar_csv',
    'load_kma_T0_sol_hourly_csv',
    'decompose_ghi_to_poa',
    # STC
    'calc_stc_performance',
    # Summary / Plotting
    'print_simulation_summary',
    'plot_th_diagram',
    'plot_ph_diagram',
    'plot_ts_diagram',
]


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

def calc_ASHP_cooling_COP(T_a_int_out, T_a_ext_in, Q_r_int, Q_r_max, COP_ref):
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
    if PLR > 1.0:
        PLR = 1.0
    EIR_by_T = 0.38 + 0.02 * cu.K2C(T_a_int_out) + 0.01 * cu.K2C(T_a_ext_in)
    EIR_by_PLR = 0.22 + 0.50 * PLR + 0.26 * PLR**2
    COP = PLR * COP_ref / (EIR_by_T * EIR_by_PLR)
    return COP

def calc_ASHP_heating_COP(T0, Q_r_int, Q_r_max):
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
    if PLR > 1.0:
        PLR = 1.0
    COP = -7.46 * (PLR - 0.0047 * cu.K2C(T0) - 0.477)**2 + 0.0941 * cu.K2C(T0) + 4.34
    return COP

def calc_GSHP_COP(Tg, T_cond, T_evap, theta_hat):
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

# ============================================================================
# Exergy and Entropy Functions
# ============================================================================

def generate_entropy_exergy_term(energy_term, Tsys, T0, fluid=None):
    """
    Calculate entropy and exergy terms based on energy term and temperatures.
    
    Parameters
    ----------
    energy_term : float
        The energy value for which entropy and exergy are to be calculated [W]
    Tsys : float
        The system temperature [K]
    T0 : float
        The reference (environment) temperature [K]
    fluid : optional
        If provided, modifies the entropy calculation using a logarithmic relation
    
    Returns
    -------
    tuple
        (entropy_term, exergy_term)
        - entropy_term (float): The calculated entropy term [W/K]
        - exergy_term (float): The calculated exergy term [W]
    
    Notes
    -----
    - For non-fluid systems: entropy_term = energy_term / Tsys
    - For fluid systems: entropy_term = energy_term * ln(Tsys/T0) / (Tsys - T0)
    - Exergy term: exergy_term = energy_term - entropy_term * T0
    - For cool exergy (non-fluid, Tsys < T0): exergy_term sign is reversed
    """
    entropy_term = energy_term / Tsys
    
    if fluid:
        if Tsys - T0 != 0:
            entropy_term = energy_term * math.log(Tsys/T0) / (Tsys - T0)
        elif Tsys - T0 == 0:
            entropy_term = 0
            
    exergy_term = energy_term - entropy_term * T0

    if not fluid and Tsys < T0:  # Cool exergy
        # For fluid, exergy term is always positive due to structure {(A-B)-ln(A/B)*B}
        # where A>0, B>0 always yields positive values
        exergy_term = -exergy_term
    return entropy_term, exergy_term

def calc_energy_flow(G, T, T0):
    """
    Calculate exergy flow rate due to material flow (advection).
    
    Formula: Xf = G * ((T - T0) - T0 * ln(T/T0))
    
    Parameters
    ----------
    G : float
        Heat capacity flow rate = specific heat × density × volumetric flow rate [W/K]
    T : float
        Flow temperature [K]
    T0 : float
        Reference (environment) temperature (T_dead_state) [K]
    
    Returns
    -------
    float
        Exergy flow rate [W]
        Returns np.nan if G == 0 (no flow)
    
    Notes
    -----
    This function calculates the exergy associated with a flowing stream
    of material at temperature T relative to the reference temperature T0.
    """
    if G == 0:
        return np.nan
    return G * (T - T0)

def calc_exergy_flow(G, T, T0):
    """
    Calculate exergy flow rate due to material flow (advection).
    
    Formula: Xf = G * ((T - T0) - T0 * ln(T/T0))
    
    Parameters
    ----------
    G : float, array-like, or pd.Series
        Heat capacity flow rate = specific heat × density × volumetric flow rate [W/K]
    T : float, array-like, or pd.Series
        Flow temperature [K]
    T0 : float, array-like, or pd.Series
        Reference (environment) temperature (T_dead_state) [K]
    
    Returns
    -------
    float, np.ndarray, or pd.Series
        Exergy flow rate [W]
        Returns np.nan (or array/Series with np.nan) if G == 0 (no flow)
        Return type matches input type: scalar -> scalar, Series -> Series, array -> array
    
    Notes
    -----
    This function calculates the exergy associated with a flowing stream
    of material at temperature T relative to the reference temperature T0.
    Supports vectorized operations for pandas Series and numpy arrays.
    """
    # Store original input types for return type matching
    G_input = G
    is_series = isinstance(G, pd.Series)
    is_scalar = not isinstance(G, (pd.Series, np.ndarray))
    
    # Convert to numpy arrays for computation
    G_arr = np.asarray(G) if not is_scalar else np.array([G])
    T_arr = np.asarray(T) if not isinstance(T, (pd.Series, np.ndarray)) else np.asarray(T)
    T0_arr = np.asarray(T0) if not isinstance(T0, (pd.Series, np.ndarray)) else np.asarray(T0)
    
    # Ensure all arrays have the same shape
    if G_arr.ndim == 0:
        G_arr = np.array([G_arr])
    if T_arr.ndim == 0:
        T_arr = np.array([T_arr])
    if T0_arr.ndim == 0:
        T0_arr = np.array([T0_arr])
    
    # Broadcast if needed
    G_arr, T_arr, T0_arr = np.broadcast_arrays(G_arr, T_arr, T0_arr)
    
    # Initialize result array with NaN
    result = np.full_like(G_arr, np.nan, dtype=np.float64)
    
    # Create mask for valid calculations
    mask = (G_arr != 0) & (~np.isnan(G_arr)) & (~np.isnan(T_arr)) & (~np.isnan(T0_arr)) & (T_arr > 0) & (T0_arr > 0)
    
    # Calculate exergy flow for valid elements
    result[mask] = G_arr[mask] * ((T_arr[mask] - T0_arr[mask]) - T0_arr[mask] * np.log(T_arr[mask] / T0_arr[mask]))
    
    # Return in original format
    if is_series:
        return pd.Series(result.flatten(), index=G_input.index)
    elif is_scalar:
        return result.item() if result.size == 1 else result[0]
    else:
        return result if result.ndim > 0 else result.item()


def calc_refrigerant_exergy(
    df: pd.DataFrame,
    ref: str,
    state_map: dict[int, tuple[str, str, str]],
    T0_K: pd.Series,
    P0: float = 101325,
) -> pd.DataFrame:
    """Calculate refrigerant state-point entropy and exergy.

    For each state in ``state_map``, compute specific entropy (*s*),
    specific exergy (*x* = (h − h0) − T0·(s − s0)), and exergy flow
    (*X* = ṁ · x) using CoolProp.  Dead-state properties (h0, s0)
    are evaluated at (T0, P0) for the given refrigerant.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing pressure ``[Pa]``, enthalpy ``[J/kg]``,
        and mass-flow ``m_dot_ref [kg/s]`` columns.
    ref : str
        CoolProp refrigerant identifier (e.g. ``'R410A'``).
    state_map : dict[int, tuple[str, str, str]]
        ``{state_num: (name, pressure_col, enthalpy_col)}``.
        Example::

            {1: ('cmp_in', 'P_ref_cmp_in [Pa]', 'h_ref_cmp_in [J/kg]')}

    T0_K : pd.Series
        Dead-state (environment) temperature per row [K].
    P0 : float
        Dead-state pressure [Pa] (default ``101325``).

    Returns
    -------
    pd.DataFrame
        ``df`` with columns added per state point:
        ``s_ref_{name} [J/(kg·K)]``, ``x_ref_{name} [J/kg]``,
        ``X_ref_{name} [W]``.

    Notes
    -----
    - Exergy equation: x = (h − h0) − T0·(s − s0)  [J/kg]
    - Exergy flow: X = ṁ · x  [W]
    - Rows with NaN pressure/enthalpy are silently skipped.
    """
    for idx in df.index:
        t0_k: float = T0_K.iloc[idx]
        try:
            h0: float = CP.PropsSI('H', 'T', t0_k, 'P', P0, ref)
            s0: float = CP.PropsSI('S', 'T', t0_k, 'P', P0, ref)
        except Exception:
            h0, s0 = np.nan, np.nan

        m_dot: float = (
            df.loc[idx, 'm_dot_ref [kg/s]']
            if 'm_dot_ref [kg/s]' in df.columns
            else np.nan
        )

        for _num, (name, P_col, h_col) in state_map.items():
            if P_col in df.columns and h_col in df.columns:
                P: float = df.loc[idx, P_col]
                h: float = df.loc[idx, h_col]
                try:
                    if not np.isnan(P) and not np.isnan(h):
                        s_val: float = CP.PropsSI(
                            'S', 'P', P, 'H', h, ref,
                        )
                        # Specific exergy [J/kg]
                        x_val: float = (
                            (h - h0) - t0_k * (s_val - s0)
                        )
                        # Exergy flow [W]
                        X_val: float = (
                            m_dot * x_val
                            if not np.isnan(m_dot)
                            else np.nan
                        )
                        df.loc[
                            idx,
                            f's_ref_{name} [J/(kg·K)]',
                        ] = s_val
                        df.loc[
                            idx,
                            f'x_ref_{name} [J/kg]',
                        ] = x_val
                        df.loc[
                            idx,
                            f'X_ref_{name} [W]',
                        ] = X_val
                except Exception:
                    pass

    return df


def convert_electricity_to_exergy(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Copy all electricity columns (``E_*``) to exergy columns (``X_*``).

    Electrical energy is 100 %% pure exergy, so ``X = E`` for all
    electricity-consumption columns.

    The function searches for columns matching the pattern
    ``E_xxx [W]`` and creates corresponding ``X_xxx [W]`` columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``E_xxx [W]`` columns.

    Returns
    -------
    pd.DataFrame
        ``df`` with ``X_xxx [W]`` columns added.
    """
    for col in df.columns:
        if col.startswith('E_') and col.endswith(' [W]'):
            # E_cmp [W] → X_cmp [W]
            x_col: str = 'X_' + col[2:]
            df[x_col] = df[col]
    return df


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

    if alp >= 1.0:
        T_mix_w_out_val_K = T_tank_w_K
    else:
        T_mix_w_out_val_K = alp * T_tank_w_K + (1 - alp) * T_tank_w_in_K

    T_mix_w_out_val = cu.K2C(T_mix_w_out_val_K)
    return {
        'alp': alp,
        'T_mix_w_out': T_mix_w_out_val,
        'T_mix_w_out_K': T_mix_w_out_val_K,
    }


def calc_uv_lamp_power(current_time_s, period_sec, num_switching, exposure_sec, lamp_watts):
    """Calculate UV lamp power at a given time instant.

    The lamp switches on ``num_switching`` times per ``period_sec``,
    each activation lasting ``exposure_sec``.

    Parameters
    ----------
    current_time_s : float
        Current simulation time [s].
    period_sec : float
        Switching period (e.g. 3 h → 10800 s).
    num_switching : int
        Number of on-cycles per period.
    exposure_sec : float
        Duration of each on-cycle [s].
    lamp_watts : float
        Rated lamp power [W].

    Returns
    -------
    float
        Instantaneous lamp power [W] (0 or ``lamp_watts``).
    """
    if num_switching <= 0 or lamp_watts <= 0:
        return 0.0

    time_in_period = current_time_s % period_sec
    interval = (period_sec - num_switching * exposure_sec) / (num_switching + 1)
    for i in range(num_switching):
        start_time = interval * (i + 1) + i * exposure_sec
        if start_time <= time_in_period < start_time + exposure_sec:
            return lamp_watts
    return 0.0

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
        Volumetric flow rate exchanged between nodes [m³/s]
    
    Notes
    -----
    TODO: C_d value should be calculated based on physical equations.
    """
    from .constants import g, beta
    
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

def calc_UA_tank_arr(r0, x_shell, x_ins, k_shell, k_ins, H, N, h_w, h_o):
    """
    Calculate overall heat-loss UA per vertical segment of a cylindrical tank.
    
    Heat loss occurs radially through the side and planarly through bottom/top.
    Side applies to all nodes; bottom/top add in parallel for node 1 and N.
    
    Parameters
    ----------
    r0 : float
        Inner radius of the tank [m]
    x_shell : float
        Thickness of the tank shell [m]
    x_ins : float
        Thickness of the insulation layer [m]
    k_shell : float
        Thermal conductivity of the tank shell material [W/mK]
    k_ins : float
        Thermal conductivity of the insulation material [W/mK]
    H : float
        Height of the tank [m]
    N : int
        Number of segments
    h_w : float
        Internal convective heat transfer coefficient [W/m²K]
    h_o : float
        External convective heat transfer coefficient [W/m²K]
    
    Returns
    -------
    UA_arr : np.ndarray
        Array of overall heat transfer coefficients for each segment [W/K]
    
    Notes
    -----
    - Side: convection (in/out) + cylindrical conduction (shell + insulation)
    - Bottom/Top: convection (in/out) + planar conduction (shell + insulation)
    - Middle nodes: side only
    - End nodes (1 and N): side || base (parallel)
    """
    dz = H / N
    r1 = r0 + x_shell
    r2 = r1 + x_ins

    # --- Areas ---
    # Side (per segment)
    A_side_in_seg = 2.0 * math.pi * r0 * dz   # Inner wetted area (for h_w)
    A_side_out_seg = 2.0 * math.pi * r2 * dz  # Outer area (for h_o)
    # Bases (single discs)
    A_base_in = math.pi * r0**2               # Internal disc area (for h_w)
    A_base_out = math.pi * r2**2              # External disc area (for h_o)

    # --- Side: convection (in/out) + cylindrical conduction (shell + insulation) ---
    # Conduction (cylindrical) per segment
    R_side_cond_shell = math.log(r1 / r0) / (2.0 * math.pi * k_shell * dz)
    R_side_cond_ins = math.log(r2 / r1) / (2.0 * math.pi * k_ins * dz)
    R_side_cond = R_side_cond_shell + R_side_cond_ins  # [K/W]

    R_side_w = 1.0 / (h_w * A_side_in_seg)    # [K/W]
    R_side_ext = 1.0 / (h_o * A_side_out_seg)  # [K/W]
    R_side_tot = R_side_w + R_side_cond + R_side_ext  # [K/W] (series)

    # --- Bottom/Top discs: convection (in/out) + planar conduction (shell + insulation) ---
    R_base_cond_shell = x_shell / (k_shell * A_base_in)   # [K/W] (inner metal plate)
    R_base_cond_ins = x_ins / (k_ins * A_base_out)       # [K/W] (outer insulation plate)
    R_base_cond = R_base_cond_shell + R_base_cond_ins

    R_base_w = 1.0 / (h_w * A_base_in)    # [K/W]
    R_base_ext = 1.0 / (h_o * A_base_out)  # [K/W]
    R_base_tot = R_base_w + R_base_cond + R_base_ext  # [K/W] (series through the base)

    # --- Equivalent node-to-ambient resistances ---
    # Middle nodes: side only
    R_mid = R_side_tot

    # Node 1 (bottom) and Node N (top): side || base
    R_end = (R_side_tot * R_base_tot) / (R_side_tot + R_base_tot)  # [K/W] (parallel)

    R_arr = np.array([R_end] + [R_mid]*(N-2) + [R_end], dtype=float)
    UA_arr = 1.0 / R_arr  # [W/K]
    return UA_arr

# ============================================================================
# TDMA Solver Functions
# ============================================================================

def TDMA(a, b, c, d):
    """
    Solve tri-diagonal matrix system using TDMA (Tri-Diagonal Matrix Algorithm).
    
    Reference: https://doi.org/10.1016/j.ijheatmasstransfer.2017.09.057 [Appendix B - Eq.(B7)]
    
    Parameters
    ----------
    a : np.ndarray
        Lower diagonal elements (length N-1)
    b : np.ndarray
        Main diagonal elements (length N)
    c : np.ndarray
        Upper diagonal elements (length N-1)
    d : np.ndarray
        Right-hand side vector (length N)
    
    Returns
    -------
    np.ndarray
        Solution vector (next time step temperatures)
    
    Notes
    -----
    If boundary conditions are not None, additional thermal resistances
    are added to the leftmost and rightmost columns, and surface temperatures
    are recalculated considering boundary layer thermal resistance.
    """
    n = len(b)

    A_mat = np.zeros((n, n))
    np.fill_diagonal(A_mat[1:], a[1:])
    np.fill_diagonal(A_mat, b)
    np.fill_diagonal(A_mat[:, 1:], c[:-1])
    A_inv = np.linalg.inv(A_mat)

    T_new = np.dot(A_inv, d).flatten()  # Flatten the result to 1D array
    return T_new

def _add_loop_advection_terms(a, b, c, d, in_idx, out_idx, G_loop, T_loop_in):
    """
    Add forced convection terms for a specified range (in_idx -> out_idx) to TDMA coefficients.
    
    Indices are 0-based (node 1 -> idx 0).
    Direction: in_idx > out_idx means 'upward' (bottom→top), otherwise 'downward' (top→bottom).
    
    Parameters
    ----------
    a, b, c, d : np.ndarray
        TDMA coefficient arrays (modified in-place)
    in_idx : int
        Inlet node index (0-based)
    out_idx : int
        Outlet node index (0-based)
    G_loop : float
        Heat capacity flow rate [W/K]
    T_loop_in : float
        Inlet stream temperature [K]
    
    Notes
    -----
    This function modifies the TDMA coefficients to account for directed advection
    across a node range in either direction.
    """
    # Invalid case: ignore
    if G_loop <= 0 or in_idx == out_idx:
        print("Warning: negative loop flow rate or identical in/out loop nodes.")
        return

    # Inlet node (common)
    b[in_idx] += G_loop
    d[in_idx] += G_loop * T_loop_in  # Inlet stream temperature
    
    # Upward: in(N side) -> ... -> out(1 side)
    if in_idx > out_idx:
        # Internal nodes in path (out_idx+1 .. in_idx-1)
        for k in range(in_idx - 1, out_idx, -1):
            b[k] += G_loop
            c[k] -= G_loop
        # Outlet node (out_idx)
        b[out_idx] += G_loop
        c[out_idx] -= G_loop

    # Downward: in(1 side) -> ... -> out(N side)
    else:
        for k in range(in_idx + 1, out_idx):
            a[k] -= G_loop 
            b[k] += G_loop
        # Outlet node (out_idx)
        a[out_idx] -= G_loop
        b[out_idx] += G_loop

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

def calc_LMTD_counter_flow(T_hot_in_K, T_hot_out_K, T_cold_in_K, T_cold_out_K):
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

def calc_LMTD_parallel_flow(T_hot_in_K, T_hot_out_K, T_cold_in_K, T_cold_out_K):
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

def calc_HX_perf_for_target_heat(Q_ref_target, T_ou_a_in_C, T_ref_evap_sat_K, T_ref_cond_sat_l_K, A_cross, UA_design, dV_fan_design, is_active=True):
    """
    Numerically solve for the air-side flow rate (fan airflow) required to achieve a target heat transfer rate in a heat exchanger, using a dynamically varying UA based on air velocity.

    This function determines the airflow that is needed to meet a specified heat transfer demand, accounting for dynamic changes in the overall heat transfer coefficient (UA) as a function of flow velocity using the Dittus-Boelter relationship (UA ∝ velocity^0.8).

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
        Design fan flow rate [m³/s]. Used for velocity normalization.
    is_active : bool, optional
        활성화 여부 (기본값: True)
        is_active=False일 때 nan 값으로 채워진 딕셔너리 반환

    Returns
    -------
    dict
        Dictionary containing:
            - dV_fan : Required air-side flow rate [m³/s]
            - UA : Actual heat exchanger overall heat transfer coefficient at solution point [W/K]
            - T_ou_a_out_K : Outlet air temperature [K]
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
            'converged': True,
            'dV_fan': np.nan,
            'UA': np.nan,
            'T_ou_a_mid': np.nan,
            'Q_ou_air': np.nan,
            'epsilon': np.nan,
        }
    
    # All arguments are required. UA is always calculated using UA_design and velocity correction in this version.
    
    # Q_ref_target이 0에 가까우면 root_scalar 호출 없이 0 값 반환
    # bisect 메서드는 f(a)와 f(b)의 부호가 달라야 하므로, Q_ref_target=0일 때 실패함
    T_ou_a_in_K = cu.C2K(T_ou_a_in_C)
    if abs(Q_ref_target) < 1e-6:
        return {
            'converged': True,
            'dV_fan': 0.0,
            'UA': 0.0,
            'T_ou_a_mid_K': T_ou_a_in_K,  # 입구 온도와 동일 (열교환 없음)
            'Q_ou_air': 0.0,
            'epsilon': 0.0,
        }
    
    def _error_function(dV_fan):
        UA = calc_UA_from_dV_fan(dV_fan, dV_fan_design, A_cross, UA_design)
        epsilon = (1 - np.exp(-UA / (c_a * rho_a * dV_fan)))
        # 증발기 계산이므로 T_ref_evap_sat_K 사용 (포화 증발 온도)
        T_ou_a_mid_K = T_ou_a_in_K - (T_ou_a_in_K - T_ref_evap_sat_K) * epsilon # Heating assumption (Q_ref_target > 0)
        
        # [MODIFIED] LMTD 제거하고 공기 측 Q_air로 직접 계산
        Q_ou_air = c_a * rho_a * dV_fan * (T_ou_a_in_K - T_ou_a_mid_K) # 흡열이므로 (입구 - 출구) * C_min
        # Heating 모드 기준: Refrigerant가 열 흡수, Air가 열 방출. 
        # T_ou_a_in > T_ou_a_mid > T_ref_evap_sat_K
        # Q_ref_target > 0 (Refrigerant gains heat)
        # Q_air (Air loses heat) = m_dot * cp * (Tin - Tout) > 0
        
        return Q_ou_air - Q_ref_target

    dV_min = dV_fan_design * 0.1 # [m³/s]
    dV_max = dV_fan_design # [m³/s]

    # --- Bracket validity check (avoid bisect ValueError) ---
    f_min = _error_function(dV_min)
    f_max = _error_function(dV_max)

    if f_min * f_max > 0:
        # Same sign → no root in [dV_min, dV_max]. Return best-effort result.
        # Pick the boundary closer to zero as fallback.
        if abs(f_min) <= abs(f_max):
            dV_fallback = dV_min
            closer_end = 'dV_min'
        else:
            dV_fallback = dV_max
            closer_end = 'dV_max'

        UA_fb = calc_UA_from_dV_fan(dV_fallback, dV_fan_design, A_cross, UA_design)
        eps_fb = 1 - np.exp(-UA_fb / (c_a * rho_a * dV_fallback))
        T_mid_fb = T_ref_evap_sat_K + eps_fb * (T_ou_a_in_K - T_ref_evap_sat_K)
        Q_fb = c_a * rho_a * dV_fallback * (T_ou_a_in_K - T_mid_fb)

        # Diagnostic hint (stored in return dict, not printed)
        if f_min > 0 and f_max > 0:
            hint = ("Q_achievable > Q_target at both bracket ends. "
                    "Consider: ↓ dV_ou_fan_a_design or ↑ Q_ref_target via ↑ hp_capacity")
        else:
            hint = ("Q_achievable < Q_target at both bracket ends. "
                    "Consider: ↑ dV_ou_fan_a_design, ↑ UA_evap_design, or ↓ hp_capacity")

        # Compute Q at both boundaries for diagnostics
        Q_at_dV_min = Q_ref_target + f_min  # f = Q_air - Q_target → Q_air = f + Q_target
        Q_at_dV_max = Q_ref_target + f_max

        return {
            'converged': False,
            'dV_fan': dV_fallback,
            'UA': UA_fb,
            'T_ou_a_mid': cu.K2C(T_mid_fb),
            'Q_ou_air': Q_fb,
            'epsilon': eps_fb,
            # Diagnostic fields for callers
            'Q_at_dV_min': Q_at_dV_min,
            'Q_at_dV_max': Q_at_dV_max,
            'Q_ref_target': Q_ref_target,
            'dV_min': dV_min,
            'dV_max': dV_max,
            'hint': hint,
        }

    sol = root_scalar(_error_function, bracket=[dV_min, dV_max], method='bisect')
    
    if sol.converged:
        # 수렴된 dV_fan 값을 사용하여 최종 값들 계산
        dV_fan_converged = sol.root
        UA = calc_UA_from_dV_fan(dV_fan_converged, dV_fan_design, A_cross, UA_design)
        epsilon = 1 - np.exp(-UA / (c_a * rho_a * dV_fan_converged))
        # 증발기 계산이므로 T_ref_evap_sat_K 사용 (포화 증발 온도)
        T_ou_a_mid_K = T_ref_evap_sat_K + epsilon * (T_ou_a_in_K - T_ref_evap_sat_K)  # Heating assumption (Q_ref_target > 0)
        
        Q_ou_air = c_a * rho_a * dV_fan_converged * (T_ou_a_in_K - T_ou_a_mid_K)
        
        return {
            'converged': True,  # 명시적으로 converged 플래그 추가
            'dV_fan': dV_fan_converged,
            'UA': UA,
            'T_ou_a_mid': cu.K2C(T_ou_a_mid_K),
            'Q_ou_air': Q_ou_air,
            'epsilon': epsilon,
            }
    else:
        return {
            'converged': False,
            'dV_fan': np.nan,
            'UA': np.nan,
            'T_ou_a_mid': np.nan,
            'Q_ou_air': np.nan,
            'epsilon': np.nan
        }

def calc_fan_power_from_dV_fan(dV_fan, fan_params, vsd_coeffs, is_active=True):
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
    is_active : bool, optional
        활성화 여부 (기본값: True)
        is_active=False일 때 np.nan 반환
    
    Returns:
    --------
    float
        Fan power [W]
        Returns np.nan if is_active=False
    """
    if not is_active:
        return np.nan
    
    # Extract design parameters
    fan_design_flow_rate = fan_params.get('fan_design_flow_rate', None)
    fan_design_power = fan_params.get('fan_design_power', None)
    
    # Error if design parameters are missing
    if fan_design_flow_rate is None or fan_design_power is None:
        raise ValueError("fan_design_flow_rate and fan_design_power must be provided in fan_params")
    
    # Flow rate validation
    if dV_fan < 0:
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

def check_hp_schedule_active(hour, hp_on_schedule):
    """
    주어진 시간이 히트펌프 운전 허용 구간에 포함되는지 확인합니다.
    
    Parameters
    ----------
    hour : float
        현재 시간 [시] (0.0 ~ 24.0)
    hp_on_schedule : list of tuple
        히트펌프 운전 허용 구간 리스트. 각 튜플은 (시작_시, 종료_시) 형식.
        예: [(0.0, 5.0), (8.0, 24.0)] → 0~5시, 8~24시 구간에만 운전 허용
    
    Returns
    -------
    bool
        hour이 hp_on_schedule의 어떤 구간에 포함되면 True, 아니면 False.
        구간은 [start, end) 형식 (start 포함, end 미포함).
    
    Examples
    --------
    >>> check_hp_schedule_active(3.0, [(0.0, 5.0), (8.0, 24.0)])
    True
    >>> check_hp_schedule_active(6.0, [(0.0, 5.0), (8.0, 24.0)])
    False
    >>> check_hp_schedule_active(10.0, [(0.0, 5.0), (8.0, 24.0)])
    True
    """
    for start_hour, end_hour in hp_on_schedule:
        if start_hour <= hour < end_hour:
            return True
    return False

def build_dhw_usage_ratio(entries, t_array):
    """
    Build schedule ratio array from schedule entries for each timestep (t_array).

    Parameters
    ----------
    entries : list of tuple
        Schedule entry list. Each item is (start_str, end_str, frac) format.
        - start_str, end_str: "H:M", "H:M:S", "H" format string (e.g., "6:00", "23:30:20", "24:00", etc.).
          "24:00" or "24:00:00" is specially handled as end of day (= 24*cu.h2s seconds).
        - frac: Usage ratio (float) for that interval. Clipped to 0.0 ~ 1.0 range.
        Intervals are treated as half-open [start, end).

    t_array : numpy.ndarray
        Timestep array in seconds (e.g., np.arange(0, sim_sec, dt)). Each element
        is mapped to time within the same day using modulo operation with 24*cu.h2s.

    Returns
    -------
    numpy.ndarray
        Array with same shape as t_array. Schedule ratio (0.0 ~ 1.0) for each timestep.
        If multiple entries overlap, the value at that position is the maximum of the frac values.

    Summary
    -------
    - Time strings are converted to seconds internally using _time_str_to_sec.
    - If end is 24*cu.h2s (e.g., "24:00", "24:00:00"), it is adjusted to last value before end of day.
    - If interval crosses midnight (e.g., start=23:00, end=02:00), OR mask is used to
      correctly cover intervals that cross midnight.
    - Ratio (frac) is clipped to 0~1 range using np.clip.
    - Return value is finally clipped to 0~1 range to guarantee bounds.

    Examples
    --------
    entries = [("6:00","7:00",0.5), ("6:30:15","8:00:30",0.8)]
    -> Interval 6:30:15~7:00:00 has max(0.5,0.8)=0.8 applied.

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
        Convert time string format (e.g., "H", "H:M", "H:M:S") to integer seconds within day (0 ~ 86400).

        Parameters
        ----------
        time_str : str
            Time string in "H", "H:M", or "H:M:S" format.
            - "H" represents hour, integer in 0~24 range.
            - "H:M" format, hour and minute, minute is 0~59 integer.
            - "H:M:S" format, hour, minute, and second, second is 0~59 integer.
            - "24:00" or "24:00:00" is specially handled as end of day (= 24*cu.h2s seconds).

        Behavior
        --------
        - Separate hour, minute, and second, convert to seconds: seconds = (h % 24) * 3600 + m * 60 + s
        - If input is "24:00" or "24:00:00" (string starts with '24' and h%24 == 0), return 24*cu.h2s.
        - Hour (h) is modulo operated with 24, so notation >= 24 is mapped to 0..23 except for "24:00"/"24:00:00".

        Returns
        -------
        int
            Integer seconds within day (0 ~ 86400).
        """
        fields = time_str.strip().split(':')
        # Pad missing values: ["H"], ["H","M"], ["H","M","S"]
        while len(fields) < 3:
            fields.append('0')
        h, m, s = (int(x) for x in fields[:3])
        # "24:00" or "24:00:00" => end of day
        if h == 0 and time_str.strip().startswith('24'):
            return 24*cu.h2s
        h = h % 24
        return h*cu.h2s + m*60 + s

    # Process schedule entries
    for start_str, end_str, frac in entries:
        s = _time_str_to_sec(start_str)
        e = _time_str_to_sec(end_str)
        if e == 24*cu.h2s:
            e = 24*cu.h2s - 1e-9
        ratio = np.clip(frac, 0.0, 1.0)
        if s == e:
            continue

        if s < e:
            mask = (secs_mod >= s) & (secs_mod < e)
        else:
            mask = (secs_mod >= s) | (secs_mod < e)
        sched[mask] = np.maximum(sched[mask], ratio)

    return np.clip(sched, 0.0, 1.0)

def get_uv_params_from_turbidity(turbidity_ntu):
    """
    Turbidity 값에 따라 UV 파라미터를 반환하는 함수
    
    테이블 데이터 기반 (Table 1. Effect of Turbidity on UVT, UV Absorbance, UV Intensity, and Exposure Time)
    공식: ae = 2.303 × A254 (ae는 자연대수 흡수 계수, A254는 UV Absorbance)
    
    Parameters:
    -----------
    turbidity_ntu : float
        탁도 값 [NTU]
    
    Returns:
    --------
    dict
        {
            'uv_absorbance': float,  # UV Absorbance (A254)
            'uv_transmittance_percent': float,  # % UVT
            'reference_intensity_mw_cm2': float,  # UV Intensity (mW/cm²)
            'reference_exposure_time_sec': float  # Exposure time for 5 mJ/cm² dose (s)
        }
    """
    # 테이블 데이터: [Turbidity (NTU), % UVT, UV Absorbance (A254), UV Intensity (mW/cm²), Exposure time (s)]
    turbidity_table = [
        [0.25, 86, 0.07, 0.40, 12.4],
        [5.0, 78, 0.11, 0.39, 12.8],
        [10.0, 71, 0.15, 0.36, 13.9],
        [20.1, 59, 0.23, 0.33, 15.0]
    ]
    
    # 테이블에서 가장 가까운 값 찾기 또는 보간
    turbidity_values = [row[0] for row in turbidity_table]
    
    # 입력값이 테이블 범위를 벗어나는 경우 처리
    if turbidity_ntu <= turbidity_values[0]:
        # 최소값 이하: 첫 번째 행 사용
        row = turbidity_table[0]
        return {
            'uv_absorbance': row[2],
            'uv_transmittance_percent': row[1],
            'reference_intensity_mw_cm2': row[3],
            'reference_exposure_time_sec': row[4]
        }
    elif turbidity_ntu >= turbidity_values[-1]:
        # 최대값 이상: 마지막 행 사용
        row = turbidity_table[-1]
        return {
            'uv_absorbance': row[2],
            'uv_transmittance_percent': row[1],
            'reference_intensity_mw_cm2': row[3],
            'reference_exposure_time_sec': row[4]
        }
    else:
        # 선형 보간
        for i in range(len(turbidity_values) - 1):
            if turbidity_values[i] <= turbidity_ntu < turbidity_values[i + 1]:
                # 두 점 사이에서 보간
                t1, t2 = turbidity_values[i], turbidity_values[i + 1]
                row1, row2 = turbidity_table[i], turbidity_table[i + 1]
                
                # 보간 비율
                ratio = (turbidity_ntu - t1) / (t2 - t1)
                
                return {
                    'uv_absorbance': row1[2] + ratio * (row2[2] - row1[2]),
                    'uv_transmittance_percent': row1[1] + ratio * (row2[1] - row1[1]),
                    'reference_intensity_mw_cm2': row1[3] + ratio * (row2[3] - row1[3]),
                    'reference_exposure_time_sec': row1[4] + ratio * (row2[4] - row1[4])
                }
        
        # 정확히 일치하는 경우
        for i, t_val in enumerate(turbidity_values):
            if abs(turbidity_ntu - t_val) < 1e-6:
                row = turbidity_table[i]
                return {
                    'uv_absorbance': row[2],
                    'uv_transmittance_percent': row[1],
                    'reference_intensity_mw_cm2': row[3],
                    'reference_exposure_time_sec': row[4]
                }
    
    # 기본값 반환 (발생하지 않아야 함)
    row = turbidity_table[0]
    return {
        'uv_absorbance': row[2],
        'uv_transmittance_percent': row[1],
        'reference_intensity_mw_cm2': row[3],
        'reference_exposure_time_sec': row[4]
    }

def calc_uv_exposure_time(radius_cm, uvc_output_W, lamp_arc_length_cm, 
                               target_dose_mj_cm2=186, turbidity_ntu=0.25):
    """
    ADA453967.pdf 문서의 Radial Model을 기반으로 UV 램프의 필요 가동 시간을 계산하는 함수
    https://apps.dtic.mil/sti/tr/pdf/ADA453967.pdf
    
    UV lamp catalog
    https://www.assets.signify.com/is/content/Signify/Assets/philips-lighting/global/catalogue-uv-c-disinfection-nov2025.pdf
    
    I(r) = (P_L / (2 * pi * r)) * exp(-ae * r)
    Time = Target Dose / I(r)
    ae = 2.303 × A254 (ae는 자연대수 흡수 계수, A254는 UV Absorbance)

    Parameters:
    -----------
    radius_cm : float
        저탕조의 반지름 (cm) - 램프에서 가장 먼 벽까지의 거리
    uvc_output_W : float
        램프의 순수 UV-C 출력 [W]
    lamp_arc_length_cm : float
        램프의 발광부 길이 [cm]
    target_dose_mj_cm2 : float, optional
        목표 살균 선량 [mJ/cm²]. 기본값 186은 EPA의 4-log 바이러스 살균 기준.
    turbidity_ntu : float, optional
        탁도 값 [NTU]. 제공되면 테이블 데이터를 기반으로 UV Absorbance를 조회하고
        absorption_coeff를 자동 계산합니다. 기본값 0.25 NTU 수준의 맑은 물 기준.
    absorption_coeff : float, optional
        물의 자연대수 흡수 계수 ae [1/cm]. turbidity_ntu가 제공되지 않을 때만 사용됩니다.
        기본값 0.16은 탁도 0.25 NTU 수준의 맑은 물 기준.

    Returns:
    --------
    required_time_min : float
        필요 1회 노출 시간 [분]
    """
    
    # absorption_coeff 결정: turbidity가 제공되면 자동 계산
    # Table 1의 데이터는 수질에 따른 빛 손실 원리를 보여주는 참고 자료이며,
    # 최종 목표는 target_dose_mj_cm2 (기본값 186 mJ/cm²)입니다.
    # 공식: ae = 2.303 × A254 (수질에 따른 빛 손실 원리 반영)
    uv_params = get_uv_params_from_turbidity(turbidity_ntu)
    uv_absorbance = uv_params['uv_absorbance']
    absorption_coeff = 2.303 * uv_absorbance
    
    # 1. 선형 출력 밀도 P_L (Power emitted per unit arc length) 계산 [단위: mW/cm]
    # 입력된 Watts를 mW로 변환 후 길이로 나눔
    p_l_mw_cm = (uvc_output_W * 1000) / lamp_arc_length_cm
    
    # 2. 탱크 벽면(거리 r)에서의 UV 강도 I(r) 계산 [단위: mW/cm²]
    # 공식: I(r) = (P_L / 2πr) * e^(-ae * r) [cite: 1479]
    intensity_mw_cm2 = (p_l_mw_cm / (2 * math.pi * radius_cm)) * math.exp(-absorption_coeff * radius_cm)
    
    required_time_sec = target_dose_mj_cm2 / intensity_mw_cm2 
    required_time_min = required_time_sec / 60

    return required_time_min

def make_dhw_schedule_from_Annex_42_profile(flow_rate_array, df_time_step, simulation_time_step):
    """
    Generate DHW schedule list from flow profile data.
    
    This function implements the logic to convert a flow profile (L/min) 
    into a schedule list with specified time step.
    
    Args:
        flow_rate_array (array-like): Flow rate data in L/min .
        simulation_time_step (float): Simulation time step in seconds.
    
    Returns:
        list: List of tuples (start_time, end_time, fraction).
    """
    df_time_step = 60
    
    # Peak 유량 산출 (Fraction = 1.0 기준값)
    if hasattr(flow_rate_array, 'max'):
         peak_flow_rate_array = flow_rate_array.max()
    else:
         peak_flow_rate_array = max(flow_rate_array)

    schedule_entries = []
    num_slots_per_min = int(df_time_step // simulation_time_step)
    
    for i, flow in enumerate(flow_rate_array):
        # i번째 1분 데이터의 시작 시간(초 단위, 0-based)
        minute_start_sec = i * df_time_step
        # 각 1분 데이터를 10초 단위 구간 6개로 쪼개 작성
        for slot in range(num_slots_per_min):
            slot_start_sec = minute_start_sec + slot * simulation_time_step
            slot_end_sec = slot_start_sec + simulation_time_step

            # 시작 시간표시(H:MM:SS)
            sh = slot_start_sec // cu.h2s
            sm = (slot_start_sec % cu.h2s) // cu.m2s
            ss = slot_start_sec % cu.m2s
            start_time = f"{int(sh)}:{int(sm):02d}:{int(ss):02d}"

            # 종료 시간 계산 (마지막 구간에서 24:00:00로 처리)
            if slot_end_sec >= 24 * cu.h2s:
                end_time = "24:00:00"
            else:
                eh = slot_end_sec // cu.h2s
                em = (slot_end_sec % cu.h2s) // cu.m2s
                es = slot_end_sec % cu.m2s
                end_time = f"{int(eh)}:{int(em):02d}:{int(es):02d}"

            fraction = (flow / peak_flow_rate_array) if peak_flow_rate_array > 0 else 0.0

            schedule_entries.append((start_time, end_time, fraction))
            
    return schedule_entries


def calc_total_water_use_from_schedule(schedule, peak_load_m3s, info = True, info_unit = 'L'):
    '''
    Calculate total water use from schedule.

    Parameters
    ----------
    schedule : list of tuple
        Schedule list. Each item is (start_str, end_str, ratio) format.
        - start_str, end_str: "H:M" or "H:M:S" format string (e.g., "6:00", "23:30:15", "24:00").
        - ratio: Usage ratio (float) for that interval. Clipped to 0.0 ~ 1.0 range.
    peak_load_m3s : float
        Peak load flow rate [m³/s].
    
    Returns
    -------
    float
        Total daily water use [L]

    Examples
    --------
    schedule = [("6:00","7:00",0.5), ("6:30","8:00",0.8)]
    -> Interval 6:30~7:00 has max(0.5,0.8)=0.8 applied.

    Notes
    -----
    - schedule must be list of tuple, each item is (start_str, end_str, ratio) format.
    '''
    peak_load_lpm = peak_load_m3s * cu.m32L / cu.s2m
    total_use = 0
    if info:
        print(f'Peak load: {peak_load_lpm:.2f} L/min')
        print(f"{'Start':>8} ~ {'End':>8} | {'Ratio':>5} | {'Liters':>8}")
        print("-" * 45)
    
    for start, end, ratio in schedule:
        # 시간 문자열 파싱 (H, H:M, H:M:S 지원)
        def parse_to_min(time_str):
            parts = list(map(float, time_str.split(':')))
            if len(parts) == 1: # H
                return parts[0] * 60
            elif len(parts) == 2: # H:M
                return parts[0] * 60 + parts[1]
            elif len(parts) == 3: # H:M:S
                return parts[0] * 60 + parts[1] + parts[2] / 60.0
            return 0.0

        t1_min = parse_to_min(start)
        t2_min = parse_to_min(end)
        
        duration_min = t2_min - t1_min
        
        # 24:00 처리 (다음날 0시)
        if duration_min < 0: duration_min += 24 * 60 

        liters = ratio * peak_load_lpm * duration_min
        total_use += liters
        
        if info:
            if info_unit == 'L':
                val_str = f"{liters:>8.1f} L"
            elif info_unit == 'mL':
                val_str = f"{liters*1000:>8.1f} mL"
            elif info_unit == 'm3':
                val_str = f"{liters*cu.L2m3:>8.4f} m3"
            else:
                raise ValueError(f"Invalid info_unit: {info_unit}")

            print(f"{start:>8} ~ {end:>8} | {ratio:>5.2f} | {val_str}")
            
    if info:
        print("-" * 45)
        print(f"Total daily water use: {total_use:.2f} Liters")
    return total_use

def calc_cold_water_temp(df, target_date_str: str) -> float:
    """
    기상자료개방포털(https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36)의
    월간 평균온도 데이터(DataFrame)와 날짜를 입력받아, EnergyPlus 알고리즘으로 상수도 온도를 계산합니다.

    [적용된 공식 및 계수] (모두 화씨 계산)
    1. Offset = 6 F  
    2. Ratio  = 0.4 + 0.01 * (연평균기온_F - 44)
    3. Lag    = 35 - 1.0 * (연평균기온_F - 44)
    4. 최종 결과를 cu.F2C로 변환 (°C 반환)

    Parameters:
    -----------
    df : pd.DataFrame
        '평균기온(°C)' 컬럼이 포함된 DataFrame.
    target_date_str : str
        계산할 날짜 (형식: 'YYYY-MM-DD').

    Returns:
    --------
    float
        계산된 상수도 온도 (°C)
    """

    # 1. DataFrame 전처리
    df.columns = df.columns.str.strip()
    target_col = '평균기온(°C)'
    if target_col not in df.columns:
        raise ValueError(f"DataFrame에 '{target_col}' 컬럼이 존재하지 않습니다.")

    # 2. 기상 통계 추출 (섭씨 기준)
    t_avg_annual_c = df[target_col].mean()          # 연평균 기온 (C)
    t_max_monthly_c = df[target_col].max()          # 월최대 기온 (C)
    t_min_monthly_c = df[target_col].min()          # 월최소 기온 (C)

    # 3. 모든 값 화씨 단위로 변환
    t_avg_annual_f = cu.C2F(t_avg_annual_c)         # 연평균 기온 (F)
    t_max_monthly_f = cu.C2F(t_max_monthly_c)       # 월최대 기온 (F)
    t_min_monthly_f = cu.C2F(t_min_monthly_c)       # 월최소 기온 (F)
    t_diff_max_f = t_max_monthly_f - t_min_monthly_f # 최대 온도차 (F)

    # 4. 계수 정의 및 공식 적용 (전부 화씨)
    offset_f = 6.0
    ratio = 0.4 + 0.01 * (t_avg_annual_f - 44)
    lag_days = 35 - 1.0 * (t_avg_annual_f - 44)

    # 5. 날짜 처리
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
    day_of_year = target_date.timetuple().tm_yday

    # 6. 최종 공식 적용 (화씨)
    # T_mains_f = (T_avg_F + Offset_F) + Ratio * (T_diff_F / 2) * sin(...)
    degrees = 0.986 * (day_of_year - 15 - lag_days) - 90
    radians = np.radians(degrees)
    t_mains_f = (t_avg_annual_f + offset_f) + ratio * (t_diff_max_f / 2) * np.sin(radians)

    # 7. 화씨 -> 섭씨 변환
    t_mains_c = cu.F2C(t_mains_f)
    return round(t_mains_c, 2)

def calc_ref_state(
    T_evap_K,  # 증발 온도 [K] (포화 온도로 해석)
    T_cond_K,  # 응축 온도 [K] (포화 온도로 해석)
    refrigerant,  # 냉매 이름
    eta_cmp_isen,  # 압축기 단열 효율
    T0_K=None,  # 기준 온도 [K] (엑서지 계산용, 선택적)
    P0=101325,  # 기준 압력 [Pa] (엑서지 계산용, 선택적)
    mode='heating',  # 작동 모드 ('heating' 또는 'cooling')
    dT_superheat=0.0,  # [K] 증발기 출구 과열도 (State 1* → 1)
    dT_subcool=0.0,  # [K] 응축기 출구 과냉각도 (State 3* → 3)
    is_active=True,  # 활성화 여부 (False일 때 nan 값 반환)
):
    """
    냉매 사이클의 State 1-4 열역학 물성치를 계산하는 공통 함수.
    (수정됨: 과열도(Superheating) 및 과냉각도(Subcooling) 고려 모델)
    
    이 함수는 히트펌프 사이클의 4개 주요 상태점을 계산합니다:
    
    난방 모드 (mode='heating'):
    - State 1*: 증발기 포화 증기 (x=1) - 포화 온도점
    - State 1: 압축기 입구 (증발기 출구, 저압 과열 증기) = State 1* + dT_superheat
    - State 2: 압축기 출구 (응축기 입구, 고압 과열 증기)
    - State 3*: 응축기 포화 액체 (x=0) - 포화 온도점
    - State 3: 응축기 출구 (팽창밸브 입구, 고압 과냉 액체) = State 3* - dT_subcool
    - State 4: 팽창밸브 출구 (증발기 입구, 저압 액체+기체 혼합물)
    
    냉방 모드 (mode='cooling', 4-way 밸브로 인한 역순환):
    - State 1: 압축기 출구 (응축기 입구, 고압 과열 증기)
    - State 2: 압축기 입구 (증발기 출구, 저압 포화 증기)
    - State 3: 팽창밸브 출구 (증발기 입구, 저압 액체+기체 혼합물)
    - State 4: 응축기 출구 (팽창밸브 입구, 고압 포화 액체)
    
    알고리즘:
    1. 증발기와 응축기 포화 압력 계산
    2. State 1*: 저압 포화 증기 상태 계산 (T1_star_K = T_evap_K)
    3. State 1: 과열 증기 상태 계산 (T1_K = T1_star_K + dT_superheat)
    4. State 2: 단열 압축 후 실제 압축(비단열) 계산
       - 등엔트로피 압축 후 엔탈피 계산 (이상적)
       - 압축기 효율을 고려한 실제 엔탈피 계산
    5. State 3*: 고압 포화 액체 상태 계산 (T3_star_K = T_cond_K)
    6. State 3: 과냉 액체 상태 계산 (T3_K = T3_star_K - dT_subcool)
    7. State 4: 등엔탈피 팽창 (h4 = h3) 후 상태 계산
    
    호출 관계:
    - 호출자: AirSourceHeatPumpBoiler._calc_state, GroundSourceHeatPumpBoiler._calc_on_state
    - 호출 함수: CoolProp.PropsSI (냉매 물성 계산)
    
    Args:
        - T_evap_K (float): 증발 포화 온도 [K]
        - T_cond_K (float): 응축 포화 온도 [K]
        - refrigerant (str): 냉매 이름 (CoolProp 형식, 예: 'R410A')
        - eta_cmp_isen (float): 압축기 단열 효율 [0-1]
            - 실제 압축 전력 = 이론 압축 전력 / eta_cmp_isen
        - T0_K (float, optional): 기준 온도 [K] (엑서지 계산용)
            - 제공되면 State 1-4의 엑서지 계산 수행
        - P0 (float, optional): 기준 압력 [Pa] (엑서지 계산용, 기본값: 101325)
        - mode (str, optional): 작동 모드 ('heating' 또는 'cooling', 기본값: 'heating')
            - 'heating': 난방 모드 (기본 계산, State 1=압축기 유입)
            - 'cooling': 냉방 모드 (4-way 밸브 역순환, State 2=압축기 유입으로 재매핑)
        - dT_superheat (float, optional): 증발기 출구 과열도 [K] (기본값: 0.0)
            - dT_superheat=0이면 포화 증기 (기존 동작 유지)
        - dT_subcool (float, optional): 응축기 출구 과냉각도 [K] (기본값: 0.0)
            - dT_subcool=0이면 포화 액체 (기존 동작 유지)
        - is_active (bool, optional): 활성화 여부 (기본값: True)
            - is_active=False일 때 모든 값이 nan인 딕셔너리 반환
    
    Returns:
        dict: State 1-4의 물성치를 포함한 딕셔너리
        - P1, P2, P3, P4: 압력 [Pa] (모드에 따라 물리적 위치에 맞게 재매핑됨)
        - T1_K, T2_K, T3_K, T4_K: 온도 [K] (실제 상태점, 모드에 따라 재매핑됨)
        - T1_star_K, T3_star_K: 포화 온도 [K] (포화 상태점)
        - h1, h2, h3, h4: 엔탈피 [J/kg] (모드에 따라 물리적 위치에 맞게 재매핑됨)
        - s1, s2, s3, s4: 엔트로피 [J/kgK] (모드에 따라 물리적 위치에 맞게 재매핑됨)
        - rho: 압축기 유입 밀도 [kg/m³] (냉매 유량 계산에 사용)
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
        - dT_superheat=0, dT_subcool=0이면 기존 동작과 동일 (하위 호환성 유지)
    """
    
    # is_active=False일 때 nan 값으로 채워진 딕셔너리 반환
    if not is_active:
        return {
            'P_ref_cmp_in': np.nan,
            'P_ref_cmp_out': np.nan,
            'P_ref_exp_in': np.nan,
            'P_ref_exp_out': np.nan,
            'T_ref_cmp_in_K': np.nan,
            'T_ref_cmp_out_K': np.nan,
            'T_ref_exp_in_K': np.nan,
            'T_ref_exp_out_K': np.nan,
            'T_ref_evap_sat_K': np.nan,
            'T_ref_cond_sat_v_K': np.nan,
            'T_ref_cond_sat_l_K': np.nan,
            'P_ref_cond_sat_v': np.nan,
            'h_ref_cmp_in': np.nan,
            'h_ref_cmp_out': np.nan,
            'h_ref_cond_sat_v': np.nan,
            'h_ref_exp_in': np.nan,
            'h_ref_exp_out': np.nan,
            's_ref_cmp_in': np.nan,
            's_ref_cmp_out': np.nan,
            's_ref_cond_sat_v': np.nan,
            's_ref_exp_in': np.nan,
            's_ref_exp_out': np.nan,
            'rho_ref_cmp_in': np.nan,
            'x_ref_cmp_in': np.nan,
            'x_ref_cmp_out': np.nan,
            'x_ref_cond_sat_v': np.nan,
            'x_ref_exp_in': np.nan,
            'x_ref_exp_out': np.nan,
            'mode': mode,
        }
    
    # 1단계: 포화 온도 및 압력 계산
    T_ref_evap_sat_K = T_evap_K  # 증발기 포화 증기 온도 (State 1*)
    T_ref_cond_sat_l_K = T_cond_K  # 응축기 포화 액체 온도 (State 3*)
    
    P_evap = CP.PropsSI('P', 'T', T_ref_evap_sat_K, 'Q', 1, refrigerant)  # 증발기 포화 압력
    P_cond = CP.PropsSI('P', 'T', T_ref_cond_sat_l_K, 'Q', 0, refrigerant)  # 응축기 포화 압력
    
    # 2단계: State 1* (포화 증기) 및 State 1 (실제 과열 증기) 계산
    # State 1*: 포화 증기 상태 (참조용)
    h1_star = CP.PropsSI('H', 'P', P_evap, 'Q', 1, refrigerant)
    s1_star = CP.PropsSI('S', 'P', P_evap, 'Q', 1, refrigerant)
    
    # State 1: 실제 압축기 입구 (과열 증기)
    T_ref_cmp_in_K = T_ref_evap_sat_K + dT_superheat  # 과열도 적용
    
    # dT_superheat = 0일 때는 포화 증기 상태로 처리 (CoolProp 에러 방지)
    if abs(dT_superheat) < 1e-6:  # 0에 가까우면 포화 상태
        h_ref_cmp_in = CP.PropsSI('H', 'P', P_evap, 'Q', 1, refrigerant)
        s_ref_cmp_in = CP.PropsSI('S', 'P', P_evap, 'Q', 1, refrigerant)
        rho_ref_cmp_in = CP.PropsSI('D', 'P', P_evap, 'Q', 1, refrigerant)  # 압축기 유입 밀도
    else:  # 과열 상태
        h_ref_cmp_in = CP.PropsSI('H', 'T', T_ref_cmp_in_K, 'P', P_evap, refrigerant)
        s_ref_cmp_in = CP.PropsSI('S', 'T', T_ref_cmp_in_K, 'P', P_evap, refrigerant)
        rho_ref_cmp_in = CP.PropsSI('D', 'T', T_ref_cmp_in_K, 'P', P_evap, refrigerant)  # 압축기 유입 밀도
    
    # 3단계: State 2 계산 - 압축기 출구 (고압 과열 증기)
    h2_isen = CP.PropsSI('H', 'P', P_cond, 'S', s_ref_cmp_in, refrigerant)  # 등엔트로피 압축 후 엔탈피
    
    h_ref_cmp_out = h_ref_cmp_in + (h2_isen - h_ref_cmp_in) / eta_cmp_isen
    T_ref_cmp_out_K = CP.PropsSI('T', 'P', P_cond, 'H', h_ref_cmp_out, refrigerant)  # 과열 온도
    P_ref_cmp_out = P_cond  # 압력은 응축기 압력과 동일
    s_ref_cmp_out = CP.PropsSI('S', 'P', P_cond, 'H', h_ref_cmp_out, refrigerant)  # 실제 엔트로피 (s1보다 큼)
    
    # 3.5단계: State 2* 계산 - 응축기 입구에서 포화 증기에 처음 도달하는 지점
    # T2_star: P_cond 압력에서 포화 증기(Q=1) 상태
    T_ref_cond_sat_v_K = T_ref_cond_sat_l_K  # 응축기 포화 온도와 동일
    P_ref_cond_sat_v = P_cond  # 응축기 포화 압력
    h_ref_cond_sat_v = CP.PropsSI('H', 'P', P_cond, 'Q', 1, refrigerant)  # 포화 증기 엔탈피
    s_ref_cond_sat_v = CP.PropsSI('S', 'P', P_cond, 'Q', 1, refrigerant)  # 포화 증기 엔트로피
    
    # 4단계: State 3* (포화 액체) 및 State 3 (실제 과냉 액체) 계산
    # State 3*: 포화 액체 상태 (참조용)
    h3_star = CP.PropsSI('H', 'P', P_cond, 'Q', 0, refrigerant)
    s3_star = CP.PropsSI('S', 'P', P_cond, 'Q', 0, refrigerant)
    
    # State 3: 실제 응축기 출구 (과냉 액체)
    T_ref_exp_in_K = T_ref_cond_sat_l_K - dT_subcool  # 과냉각도 적용
    
    # dT_subcool = 0일 때는 포화 액체 상태로 처리 (CoolProp 에러 방지)
    if abs(dT_subcool) < 1e-6:  # 0에 가까우면 포화 상태
        h_ref_exp_in = CP.PropsSI('H', 'P', P_cond, 'Q', 0, refrigerant)
        s_ref_exp_in = CP.PropsSI('S', 'P', P_cond, 'Q', 0, refrigerant)
    else:  # 과냉 상태
        h_ref_exp_in = CP.PropsSI('H', 'T', T_ref_exp_in_K, 'P', P_cond, refrigerant)
        s_ref_exp_in = CP.PropsSI('S', 'T', T_ref_exp_in_K, 'P', P_cond, refrigerant)
    
    # 5단계: State 4 계산 - 팽창밸브 출구 (저압 액체+기체 혼합물)
    h_ref_exp_out = h_ref_exp_in  # 등엔탈피 팽창
    P_ref_exp_out = P_evap  # 압력은 증발기 압력과 동일
    T_ref_exp_out_K = CP.PropsSI('T', 'P', P_evap, 'H', h_ref_exp_out, refrigerant)  # 저압에서 엔탈피 h4에 해당하는 온도
    s_ref_exp_out = CP.PropsSI('S', 'P', P_evap, 'H', h_ref_exp_out, refrigerant)  # 팽창 후 엔트로피
    
    # 엑서지 계산용 기준 상태
    h0 = CP.PropsSI('H', 'T', T0_K, 'P', P0, refrigerant)
    s0 = CP.PropsSI('S', 'T', T0_K, 'P', P0, refrigerant)
    
    if mode == 'cooling':
        result = {
            # 냉방 모드 기준 물성치 (물리적 위치에 따라 재매핑)
            'P_ref_cmp_in': P_ref_cmp_out,
            'P_ref_cmp_out': P_evap,
            'P_ref_exp_in': P_ref_exp_out,
            'P_ref_exp_out': P_cond,
            'T_ref_cmp_in_K': T_ref_cmp_out_K,
            'T_ref_cmp_out_K': T_ref_cmp_in_K,
            'T_ref_exp_in_K': T_ref_exp_out_K,
            'T_ref_exp_out_K': T_ref_exp_in_K,
            'T_ref_evap_sat_K': T_ref_cmp_out_K,  # 냉방 모드에서는 재매핑 필요 없음 (참조용)
            'T_ref_cond_sat_l_K': T_ref_exp_in_K,  # 냉방 모드에서는 재매핑 필요 없음 (참조용)
            'h_ref_cmp_in': h_ref_cmp_out,
            'h_ref_cmp_out': h_ref_cmp_in,
            'h_ref_exp_in': h_ref_exp_out,
            'h_ref_exp_out': h_ref_exp_in,
            's_ref_cmp_in': s_ref_cmp_out,
            's_ref_cmp_out': s_ref_cmp_in,
            's_ref_exp_in': s_ref_exp_out,
            's_ref_exp_out': s_ref_exp_in,
            'rho_ref_cmp_in': rho_ref_cmp_in,
            'mode': 'cooling',
        }
    else:
        # 난방 모드: 기본 계산값 그대로 사용
        result = {
            'P_ref_cmp_in': P_evap,
            'P_ref_cmp_out': P_cond,
            'P_ref_exp_in': P_cond,
            'P_ref_exp_out': P_evap,
            'T_ref_cmp_in_K': T_ref_cmp_in_K,
            'T_ref_cmp_out_K': T_ref_cmp_out_K,
            'T_ref_exp_in_K': T_ref_exp_in_K,
            'T_ref_exp_out_K': T_ref_exp_out_K,
            'T_ref_evap_sat_K': T_ref_evap_sat_K,  # 포화 증기 온도
            'T_ref_cond_sat_v_K': T_ref_cond_sat_v_K,  # 응축기 포화 증기 온도
            'T_ref_cond_sat_l_K': T_ref_cond_sat_l_K,  # 포화 액체 온도
            'P_ref_cond_sat_v': P_ref_cond_sat_v,  # 응축기 포화 압력
            'h_ref_cmp_in': h_ref_cmp_in,
            'h_ref_cmp_out': h_ref_cmp_out,
            'h_ref_cond_sat_v': h_ref_cond_sat_v,  # 포화 증기 엔탈피
            'h_ref_exp_in': h_ref_exp_in,
            'h_ref_exp_out': h_ref_exp_out,
            's_ref_cmp_in': s_ref_cmp_in,
            's_ref_cmp_out': s_ref_cmp_out,
            's_ref_cond_sat_v': s_ref_cond_sat_v,  # 포화 증기 엔트로피
            's_ref_exp_in': s_ref_exp_in,
            's_ref_exp_out': s_ref_exp_out,
            'rho_ref_cmp_in': rho_ref_cmp_in,  # 압축기 유입 밀도 (State 1)
            'x_ref_cmp_in': (h_ref_cmp_in-h0) - T0_K*(s_ref_cmp_in - s0),
            'x_ref_cmp_out': (h_ref_cmp_out-h0) - T0_K*(s_ref_cmp_out - s0),
            'x_ref_cond_sat_v': (h_ref_cond_sat_v-h0) - T0_K*(s_ref_cond_sat_v - s0),  # 포화 증기 엑서지
            'x_ref_exp_in': (h_ref_exp_in-h0) - T0_K*(s_ref_exp_in - s0),
            'x_ref_exp_out': (h_ref_exp_out-h0) - T0_K*(s_ref_exp_out - s0),
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
        │       └─ calc_ref_state 호출
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

def update_tank_temperature(
    T_tank_w_K,
    Q_gain,
    UA_tank,
    T0_K,
    C_tank,
    dt):
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



def load_kma_solar_csv(csv_path, encoding='euc-kr'):
    """Load KMA (기상청) 1-minute cumulative solar irradiance CSV.

    The KMA ASOS solar radiation data provides cumulative Global Horizontal
    Irradiance (GHI) at 1-minute resolution in MJ/m².  This function
    converts it to instantaneous GHI in W/m².

    Parameters
    ----------
    csv_path : str
        Path to the KMA solar CSV file.
    encoding : str, optional
        File encoding (default: ``'euc-kr'``).

    Returns
    -------
    pd.Series
        Instantaneous GHI [W/m²] with ``DatetimeIndex`` (timezone-naive,
        KST assumed).  Index name is ``'datetime'``.

    Notes
    -----
    - Cumulative MJ/m² is differenced then divided by 60 s to get W/m².
    - Conversion: 1 MJ = 1e6 J,  power = ΔE / Δt = ΔMJ×1e6 / 60  [W/m²].
    - Negative values (from noise) are clipped to 0.
    """
    df = pd.read_csv(csv_path, encoding=encoding)
    # Columns: 지점, 지점명, 일시, 일사(MJ/m^2)
    col_datetime = df.columns[2]   # '일시'
    col_ghi_cum  = df.columns[3]   # '일사(MJ/m^2)'

    df[col_datetime] = pd.to_datetime(df[col_datetime])
    df = df.set_index(col_datetime).sort_index()

    ghi_cum = pd.to_numeric(df[col_ghi_cum], errors='coerce').fillna(method='ffill').fillna(0)

    # Differentiate cumulative → incremental (MJ/m² per minute)
    ghi_delta_mj = ghi_cum.diff().fillna(0).clip(lower=0)

    # Convert MJ/m²/min → W/m²:  W = MJ * 1e6 / 60
    ghi_wm2 = ghi_delta_mj * 1e6 / 60.0

    ghi_wm2.index.name = 'datetime'
    ghi_wm2.name = 'ghi_wm2'
    return ghi_wm2


def load_kma_T0_sol_hourly_csv(csv_path, encoding='euc-kr'):
    """Load KMA hourly T0 + solar CSV (Seoul_25_T0_Sol format).

    컬럼명 패턴으로 일시·기온·일사 컬럼을 자동 식별하여,
    컬럼 순서나 추가 컬럼이 있어도 동작합니다.

    일사는 시간평균 MJ/m² → W/m² = MJ/m² × 1e6 / 3600 으로 변환합니다.

    Parameters
    ----------
    csv_path : str
        Path to the KMA T0+solar CSV file.
    encoding : str, optional
        File encoding (default: ``'euc-kr'``).

    Returns
    -------
    pd.DataFrame
        DatetimeIndex, columns: ``T0`` [°C], ``GHI`` [W/m²].
        365일(8760행)까지만 포함. 2026-01-01 행은 제외.

    Notes
    -----
    - 일시: '일시', 'datetime', '날짜', 'date' 등 포함
    - 기온: '기온', '온도', 'temp', 'T0', '°C', '℃' 등 포함
    - 일사: '일사', 'ghi', 'irradiance', 'MJ', 'solar' 등 포함
    """
    df = pd.read_csv(csv_path, encoding=encoding)

    def _find_col(patterns):
        for c in df.columns:
            c_lower = str(c).lower().strip()
            for p in patterns:
                if p in c_lower or p in str(c):
                    return c
        return None

    col_datetime = _find_col(['일시', 'datetime', '날짜', 'date'])
    col_t0 = _find_col(['기온', '온도', 'temp', 't0', '°c', '℃'])
    col_solar = _find_col(['일사', 'ghi', 'irradiance', 'mj', 'solar'])

    if col_datetime is None:
        raise ValueError(f"일시 컬럼을 찾을 수 없습니다. columns: {list(df.columns)}")
    if col_t0 is None:
        raise ValueError(f"기온 컬럼을 찾을 수 없습니다. columns: {list(df.columns)}")
    if col_solar is None:
        raise ValueError(f"일사 컬럼을 찾을 수 없습니다. columns: {list(df.columns)}")

    df[col_datetime] = pd.to_datetime(df[col_datetime])
    df = df.set_index(col_datetime).sort_index()

    T0 = pd.to_numeric(df[col_t0], errors='coerce').ffill().bfill()
    ghi_mj_h = pd.to_numeric(df[col_solar], errors='coerce').fillna(0).clip(lower=0)
    GHI = (ghi_mj_h * 1e6 / 3600.0).rename('GHI')

    result = pd.DataFrame({'T0': T0, 'GHI': GHI}, index=df.index)
    result.index.name = 'datetime'

    # 365일(8760행)까지만 사용
    if len(result) > 8760:
        result = result.iloc[:8760]
    return result


def decompose_ghi_to_poa(
    ghi,
    latitude,
    longitude,
    tilt,
    azimuth,
    altitude=0,
    tz='Asia/Seoul',
    decomposition='erbs',
    transposition='perez',
):
    """Decompose GHI into DNI+DHI and transpose to plane-of-array irradiance.

    Uses ``pvlib`` to convert Global Horizontal Irradiance to
    Plane-of-Array (POA) direct and diffuse components for a tilted
    solar collector.

    Parameters
    ----------
    ghi : pd.Series
        Instantaneous GHI [W/m²] with ``DatetimeIndex``.
        If timezone-naive, *tz* is applied (localized).
    latitude : float
        Site latitude [°N].
    longitude : float
        Site longitude [°E].
    tilt : float
        Collector tilt from horizontal [°].  0 = horizontal, 90 = vertical.
    azimuth : float
        Collector azimuth [°].  180 = south-facing (Northern Hemisphere).
    altitude : float, optional
        Site altitude above sea level [m] (default 0).
    tz : str, optional
        Timezone string (default ``'Asia/Seoul'``).
    decomposition : str, optional
        GHI decomposition model: ``'erbs'`` (default) or ``'erbs_driesse'``.
    transposition : str, optional
        POA transposition model: ``'perez'`` (default) or ``'isotropic'``.

    Returns
    -------
    pd.DataFrame
        Columns:

        - ``ghi``: original GHI [W/m²]
        - ``dni``: Direct Normal Irradiance [W/m²]
        - ``dhi``: Diffuse Horizontal Irradiance [W/m²]
        - ``poa_direct``:  POA beam irradiance [W/m²]  → use as ``I_DN_schedule``
        - ``poa_diffuse``: POA diffuse irradiance [W/m²] → use as ``I_dH_schedule``
        - ``poa_global``:  total POA irradiance [W/m²]

    Raises
    ------
    ImportError
        If ``pvlib`` is not installed.

    Examples
    --------
    >>> ghi = load_kma_solar_csv('Seoul_250101_Sol.csv')
    >>> poa = decompose_ghi_to_poa(ghi, latitude=37.57, longitude=126.97,
    ...                             tilt=35.0, azimuth=180.0)
    >>> I_DN_schedule = poa['poa_direct'].values
    >>> I_dH_schedule = poa['poa_diffuse'].values
    """
    try:
        import pvlib
        from pvlib.irradiance import get_total_irradiance, erbs, erbs_driesse
        from pvlib.solarposition import get_solarposition
    except ImportError:
        raise ImportError(
            "pvlib is required for GHI decomposition. "
            "Install it with: uv add pvlib"
        )

    # Ensure timezone-aware index
    if ghi.index.tz is None:
        ghi = ghi.copy()
        ghi.index = ghi.index.tz_localize(tz)

    # --- 1. Solar position ---
    solpos = get_solarposition(
        ghi.index, latitude, longitude, altitude=altitude
    )
    solar_zenith  = solpos['apparent_zenith']
    solar_azimuth = solpos['azimuth']

    # --- 2. GHI decomposition → DNI + DHI ---
    # Extra-terrestrial radiation for decomposition
    dni_extra = pvlib.irradiance.get_extra_radiation(ghi.index)

    if decomposition == 'erbs_driesse':
        decomp = erbs_driesse(ghi, solar_zenith, datetime_or_doy=ghi.index)
    else:  # default: 'erbs'
        decomp = erbs(ghi, solar_zenith, datetime_or_doy=ghi.index)

    dni = decomp['dni'].fillna(0).clip(lower=0)
    dhi = decomp['dhi'].fillna(0).clip(lower=0)

    # --- 3. Transposition to POA ---
    poa = get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solar_zenith,
        solar_azimuth=solar_azimuth,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        dni_extra=dni_extra,
        model=transposition,
    )

    result = pd.DataFrame({
        'ghi': ghi.values,
        'dni': dni.values,
        'dhi': dhi.values,
        'poa_direct':  poa['poa_direct'].fillna(0).clip(lower=0).values,
        'poa_diffuse': poa['poa_diffuse'].fillna(0).clip(lower=0).values,
        'poa_global':  poa['poa_global'].fillna(0).clip(lower=0).values,
    }, index=ghi.index)

    return result


def calc_stc_performance(
    I_DN_stc,              # 직달일사 [W/m²]
    I_dH_stc,              # 확산일사 [W/m²]
    T_stc_w_in_K,          # STC 입수 온도 (저탕조 온도) [K]
    T0_K,                  # 기준 온도 [K]
    A_stc_pipe,            # STC 파이프 면적 [m²]
    alpha_stc,             # 흡수율 [-]
    h_o_stc,               # 외부 대류 열전달계수 [W/m²K]
    h_r_stc,               # 공기층 복사 열전달계수 [W/m²K]
    k_ins_stc,             # 단열재 열전도도 [W/mK]
    x_air_stc,             # 공기층 두께 [m]
    x_ins_stc,             # 단열재 두께 [m]
    dV_stc,                # STC 유량 [m³/s]
    E_pump,                # 펌프 소모 전력 [W]
    is_active=True,        # 활성화 여부 (기본값: True)
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
        STC 유량 [m³/s]
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
    import math
    from .constants import c_w, rho_w, k_D, k_d, k_a
    
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
            'I_sol_stc': np.nan,
            'Q_sol_stc': np.nan,
            'Q_stc_w_in': np.nan,
            'Q_stc_w_out': np.nan,
            'ksi_stc': np.nan,
            'T_stc_w_final_K': T_stc_w_in_K,  # 입수 온도와 동일
            'T_stc_w_out_K': T_stc_w_in_K,  # 입수 온도와 동일
            'T_stc_w_in_K': T_stc_w_in_K,
            'T_stc_K': np.nan,
            'Q_l_stc': np.nan,
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
    T_stc_w_out_numerator = T0_K + (
        Q_sol_stc + Q_stc_w_in
        + A_stc_pipe * U_stc * (ksi_stc * T_stc_w_in_K / (1 - ksi_stc))
        + A_stc_pipe * U_stc * T0_K
    ) / G_stc
    
    T_stc_w_out_denominator = 1 + (A_stc_pipe * U_stc) / ((1 - ksi_stc) * G_stc)
    
    T_stc_w_out_K = T_stc_w_out_numerator / T_stc_w_out_denominator
    T_stc_w_final_K = T_stc_w_out_K + E_pump / G_stc
    T_stc_K = 1/(1-ksi_stc) * T_stc_w_out_K - ksi_stc/(1-ksi_stc) * T_stc_w_in_K
    
    # STC 출수 열량 - calc_energy_flow 사용
    Q_stc_w_out = calc_energy_flow(G_stc, T_stc_w_out_K, T0_K)
    
    # 집열판 열 손실
    Q_l_stc = A_stc_pipe * U_stc * (T_stc_K - T0_K)
    
    return {
        'I_sol_stc': I_sol_stc,
        'Q_sol_stc': Q_sol_stc,
        'Q_stc_w_in': Q_stc_w_in,
        'Q_stc_w_out': Q_stc_w_out,
        'ksi_stc': ksi_stc,
        'T_stc_w_final_K': T_stc_w_final_K,
        'T_stc_w_out_K': T_stc_w_out_K,
        'T_stc_w_in_K': T_stc_w_in_K,
        'T_stc_K': T_stc_K,
        'Q_l_stc': Q_l_stc,
    }

def print_simulation_summary(df, simulation_time_step, dV_ou_a_design):
    """
    Reads simulation result DataFrame and prints comprehensive statistics in English.
    """
    required_columns = ['converged', 'E_ou_fan [W]', 'E_tot [W]', 'dV_ou_a [m3/s]', 'cmp_rpm [rpm]']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Required columns not found in DataFrame: {missing_columns}")

    print("="*50)
    # 1. Convergence Status
    converged_all = df['converged'].all()
    print(f"[Convergence Status] All converged: {converged_all}")
    if not converged_all:
        nonconverged_count = (~df['converged']).sum()
        print(f"  - Non-converged steps: {nonconverged_count} / {len(df)}")
    print("-" * 50)

    # 2. Compressor Statistics
    cmp_rpm_nonzero = df.loc[df['cmp_rpm [rpm]'] > 0, 'cmp_rpm [rpm]']
    print("[Compressor Speed]")
    if not cmp_rpm_nonzero.empty:
        print(f"  - Min: {cmp_rpm_nonzero.min():.1f} rpm | Max: {cmp_rpm_nonzero.max():.1f} rpm")
        print(f"  - Avg (active): {cmp_rpm_nonzero.mean():.1f} rpm")
    else:
        print("  - No active data.")
    print("-" * 50)

    # 3. Fan Flow Rate Statistics
    fan_nonzero = df.loc[df['dV_ou_a [m3/s]'] > 0, 'dV_ou_a [m3/s]']
    print("[Fan Flow Rate]")
    if not fan_nonzero.empty:
        fan_avg = fan_nonzero.mean()
        fan_avg_pct = (fan_avg / dV_ou_a_design) * 100
        print(f"  - Min: {fan_nonzero.min():.3f} m³/s | Max: {fan_nonzero.max():.3f} m³/s")
        print(f"  - Avg: {fan_avg:.3f} m³/s ({fan_avg_pct:.1f}% of design)")
    else:
        print("  - No active data.")
    print("-" * 50)

    # 3-1. Fan Velocity & Pressure Statistics
    if 'v_ou_a [m/s]' in df.columns:
        v_fan_nonzero = df.loc[df['v_ou_a [m/s]'] > 0, 'v_ou_a [m/s]']
        print("[Fan Velocity]")
        if not v_fan_nonzero.empty:
            print(f"  - Min: {v_fan_nonzero.min():.2f} m/s | Max: {v_fan_nonzero.max():.2f} m/s")
            print(f"  - Avg: {v_fan_nonzero.mean():.2f} m/s")
        else:
            print("  - No active data.")
        print("-" * 50)

    if 'dP_ou_fan_static [Pa]' in df.columns and 'dP_ou_fan_dynamic [Pa]' in df.columns:
        # Filter based on active fan (using dV_ou_a > 0)
        active_idx = df['dV_ou_a [m3/s]'] > 0
        dP_static = df.loc[active_idx, 'dP_ou_fan_static [Pa]']
        dP_dynamic = df.loc[active_idx, 'dP_ou_fan_dynamic [Pa]']
        
        print("[Fan Pressure (Static / Dynamic)]")
        if not dP_static.empty:
            print(f"  - Static  : Avg {dP_static.mean():.1f} Pa | Min {dP_static.min():.1f} Pa | Max {dP_static.max():.1f} Pa")
            print(f"  - Dynamic : Avg {dP_dynamic.mean():.1f} Pa | Min {dP_dynamic.min():.1f} Pa | Max {dP_dynamic.max():.1f} Pa")
        else:
            print("  - No active data.")
        print("-" * 50)

    # 4. Fan Power Statistics
    fan_p_nonzero = df.loc[df['E_ou_fan [W]'] > 0, 'E_ou_fan [W]']
    print("[Fan Power Use]")
    if not fan_p_nonzero.empty:
        print(f"  - Min: {fan_p_nonzero.min():.1f} W | Max: {fan_p_nonzero.max():.1f} W")
        print(f"  - Avg: {fan_p_nonzero.mean():.1f} W")
    else:
        print("  - No active data.")
    print("-" * 50)

    # 5. System Efficiency Metrics
    total_fan_energy = df['E_ou_fan [W]'].sum() * simulation_time_step
    total_energy = df['E_tot [W]'].sum() * simulation_time_step
    fan_ratio = (total_fan_energy / total_energy * 100) if total_energy > 0 else 0
    print(f"[Fan Power Ratio] {fan_ratio:.1f}% (Typical: 5-10%)")
    print("-" * 50)

    # 6. Heat Exchange Performance: Outdoor Air
    if 'T_ou_a_in [°C]' in df.columns and 'T_ou_a_out [°C]' in df.columns:
        valid_idx = df['T_ou_a_out [°C]'].notna()
        print("[Outdoor Air Temperature Difference (In - Out)]")
        if valid_idx.any():
            delta_T = df.loc[valid_idx, 'T_ou_a_in [°C]'] - df.loc[valid_idx, 'T_ou_a_out [°C]']
            print(f"  - Avg Delta T: {delta_T.mean():.2f} K | Max Delta T: {delta_T.max():.2f} K")
        else:
            print("  - No active data.")
        print("-" * 50)

    # 7. Heat Exchange Performance: Temp Differences
    print("[Heat Exchanger Temperature Differences]")
    
    # Condenser (T_cond - T_tank_w)
    if 'T_ref_cond_sat_l [°C]' in df.columns and 'T_tank_w [°C]' in df.columns:
        T_cond = df.loc[:, 'T_ref_cond_sat_l [°C]']
        T_tank_w = df.loc[:, 'T_tank_w [°C]']
        
        if not T_cond.empty and not T_tank_w.empty:
            dT_cond = T_cond - T_tank_w
            # Filter valid (active) steps if possible, e.g. dT > 0 or Q > 0
            # For simplicity, calculate stats for all active timestamps
            print(f"  - Condenser (T_cond - T_tank) Avg: {dT_cond.mean():.2f} K | Min: {dT_cond.min():.2f} K | Max: {dT_cond.max():.2f} K")
        else:
            print("  - Condenser: No data")

    # Evaporator (T_air_in - T_evap) & (T_air_in - T_air_out)
    if 'T_ou_a_in [°C]' in df.columns and 'T_ref_evap_sat [°C]' in df.columns and 'T_ou_a_out [°C]' in df.columns:
        T_air_in = df.loc[:, 'T_ou_a_in [°C]']
        T_evap_sat = df.loc[:, 'T_ref_evap_sat [°C]']
        T_air_out = df.loc[:, 'T_ou_a_out [°C]']
        
        if not T_air_in.empty:
            dT_evap_drive = T_air_in - T_evap_sat
            dT_air_drop = T_air_in - T_air_out
            
            print(f"  - Evap Drive (T_air_in - T_evap) Avg: {dT_evap_drive.mean():.2f} K")
            print(f"  - Air Drop (T_air_in - T_air_out) Avg: {dT_air_drop.mean():.2f} K")
        else:
            print("  - Evaporator: No data")

    print("="*50)

#%%
def plot_th_diagram(ax, result, refrigerant, T_tank, T0, fs, pad):
    """Plot T-h diagram on given axis. 
    If csv_path and timestep_idx are provided, draw horizontal lines for tank water temp / outdoor temp for that timestep.
    """
    # 색상 정의
    color1 = 'oc.blue5'   
    color2 = 'oc.red5'    
    color3 = 'black'
    color4 = 'oc.gray6'  
    line_color = 'oc.gray5'

    # 축 범위 설정
    xmin, xmax = 0, 600  # 엔탈피 [kJ/kg]
    ymin, ymax = -40, 120  # 온도 [°C]

    # 포화선 계산
    T_critical = cu.K2C(CP.PropsSI('Tcrit', refrigerant))
    temps = np.linspace(cu.K2C(CP.PropsSI('Tmin', refrigerant)) + 1, T_critical, 600)
    h_liq = [CP.PropsSI('H', 'T', cu.C2K(T), 'Q', 0, refrigerant) / 1000 for T in temps]
    h_vap = [CP.PropsSI('H', 'T', cu.C2K(T), 'Q', 1, refrigerant) / 1000 for T in temps]

    # 7개 상태점 추출 (신규 T_ref_*/h_ref_* 또는 구형 T1/h1 키 지원)
    h1_star = result.get('h_ref_evap_sat [J/kg]', result.get('h1_star [J/kg]', np.nan)) * cu.J2kJ
    h1 = result.get('h_ref_cmp_in [J/kg]', result.get('h1 [J/kg]', np.nan)) * cu.J2kJ
    h2 = result.get('h_ref_cmp_out [J/kg]', result.get('h2 [J/kg]', np.nan)) * cu.J2kJ
    h2_star = result.get('h_ref_cond_sat_v [J/kg]', result.get('h2_star [J/kg]', np.nan)) * cu.J2kJ
    h3_star = result.get('h_ref_cond_sat_l [J/kg]', result.get('h3_star [J/kg]', np.nan)) * cu.J2kJ
    h3 = result.get('h_ref_exp_in [J/kg]', result.get('h3 [J/kg]', np.nan)) * cu.J2kJ
    h4 = result.get('h_ref_exp_out [J/kg]', result.get('h4 [J/kg]', np.nan)) * cu.J2kJ
    T1_star = result.get('T_ref_evap_sat [°C]', result.get('T1_star [°C]', np.nan))
    T1 = result.get('T_ref_cmp_in [°C]', result.get('T1 [°C]', np.nan))
    T2 = result.get('T_ref_cmp_out [°C]', result.get('T2 [°C]', np.nan))
    T2_star = result.get('T_ref_cond_sat_v [°C]', result.get('T2_star [°C]', np.nan))
    T3_star = result.get('T_ref_cond_sat_l [°C]', result.get('T3_star [°C]', np.nan))
    T3 = result.get('T_ref_exp_in [°C]', result.get('T3 [°C]', np.nan))
    T4 = result.get('T_ref_exp_out [°C]', result.get('T4 [°C]', np.nan))

    # superheating=0, subcooling=0 시 *_star 누락 fallback (1*=1, 3*=3이면 동일값 사용)
    if np.isnan(h1_star) and not np.isnan(h1):
        h1_star, T1_star = h1, T1
    if np.isnan(h3_star) and not np.isnan(h3):
        h3_star, T3_star = h3, T3

    # 포화선 그리기
    ax.plot(h_liq, temps, color=color1, label='Saturated liquid', linewidth=dm.lw(0))
    ax.plot(h_vap, temps, color=color2, label='Saturated vapor', linewidth=dm.lw(0))

    # 마커 색상 결정
    cycle_markerfacecolor = color3 if result.get('hp_is_on', result.get('is_on', False)) else color4
    cycle_markeredgecolor = color3 if result.get('hp_is_on', result.get('is_on', False)) else color4

    def points_are_close(x1, y1, x2, y2, tol_atol=0.1):
        """두 점이 시각적으로 동일한지 확인 (subcooling/superheating≈0 시 degenerate 세그먼트 감지용)"""
        if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
            return False
        return (np.isclose(x1, x2, atol=tol_atol) and np.isclose(y1, y2, atol=tol_atol))

    # 물리적 프로세스 경로 그리기
    # T4 → T1_star: 등압 증발 (포화 액체선→포화 증기선)
    if not (np.isnan(h4) or np.isnan(h1_star) or np.isnan(T4) or np.isnan(T1_star)):
        ax.plot([h4, h1_star], [T4, T1_star], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    
    # T1_star → T1: 등압 superheating (superheating=0이면 끝점 동일 → 점으로 표현)
    if not (np.isnan(h1_star) or np.isnan(h1) or np.isnan(T1_star) or np.isnan(T1)):
        if points_are_close(h1_star, T1_star, h1, T1):
            ax.plot(h1, T1, marker='o', linestyle='None', color=line_color, markersize=2.5, zorder=1)
        else:
            ax.plot([h1_star, h1], [T1_star, T1], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    
    # T1 → T2: 압축
    if not (np.isnan(h1) or np.isnan(h2) or np.isnan(T1) or np.isnan(T2)):
        ax.plot([h1, h2], [T1, T2], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    
    # T2 → T2_star: 등압 냉각 (2=2*이면 끝점 동일 → 점으로 표현)
    if not (np.isnan(h2) or np.isnan(h2_star) or np.isnan(T2) or np.isnan(T2_star)):
        if points_are_close(h2, T2, h2_star, T2_star):
            ax.plot(h2, T2, marker='o', linestyle='None', color=line_color, markersize=2.5, zorder=1)
        else:
            ax.plot([h2, h2_star], [T2, T2_star], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    
    # T2_star → T3_star: 등압 응축 (포화 증기선 → 포화 액체선)
    if not (np.isnan(h2_star) or np.isnan(h3_star) or np.isnan(T2_star) or np.isnan(T3_star)):
        ax.plot([h2_star, h3_star], [T2_star, T3_star], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    
    # T3_star → T3: 등압 subcooling (subcooling=0이면 끝점 동일 → 점으로 표현)
    if not (np.isnan(h3_star) or np.isnan(h3) or np.isnan(T3_star) or np.isnan(T3)):
        if points_are_close(h3_star, T3_star, h3, T3):
            ax.plot(h3, T3, marker='o', linestyle='None', color=line_color, markersize=2.5, zorder=1)
        else:
            ax.plot([h3_star, h3], [T3_star, T3], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    
    # T3 → T4: 등엔탈피 팽창 (수직선)
    if not (np.isnan(h3) or np.isnan(h4) or np.isnan(T3) or np.isnan(T4)):
        ax.plot([h3, h4], [T3, T4], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)

    # 모든 상태점 마커 표시 (동일한 star/일반 지점 통합)
    
    points = []
    
    # 1*와 1 비교
    if points_are_close(h1_star, T1_star, h1, T1):
        points.append((h1, T1, '1=1$^*$'))
    else:
        if not (np.isnan(h1_star) or np.isnan(T1_star)):
            points.append((h1_star, T1_star, '1$^*$'))
        if not (np.isnan(h1) or np.isnan(T1)):
            points.append((h1, T1, '1'))
    
    # 2와 2* 비교
    if points_are_close(h2, T2, h2_star, T2_star):
        points.append((h2, T2, '2=2$^*$'))
    else:
        if not (np.isnan(h2) or np.isnan(T2)):
            points.append((h2, T2, '2'))
        if not (np.isnan(h2_star) or np.isnan(T2_star)):
            points.append((h2_star, T2_star, '2$^*$'))
    
    # 3*와 3 비교
    if points_are_close(h3_star, T3_star, h3, T3):
        points.append((h3, T3, '3=3$^*$'))
    else:
        if not (np.isnan(h3_star) or np.isnan(T3_star)):
            points.append((h3_star, T3_star, '3$^*$'))
        if not (np.isnan(h3) or np.isnan(T3)):
            points.append((h3, T3, '3'))
    
    # 4는 항상 표시
    if not (np.isnan(h4) or np.isnan(T4)):
        points.append((h4, T4, '4'))
    
    for h_val, T_val, label in points:
        ax.plot(h_val, T_val, marker='o', markersize=2.5, 
               markerfacecolor=cycle_markerfacecolor, 
               markeredgecolor=cycle_markeredgecolor,
               markeredgewidth=0, zorder=2)
        ax.annotate(label, (h_val, T_val),
                   xytext=(0, 3), textcoords='offset points',
                   ha='center', va='bottom',
                   fontsize=fs['legend'])

    ax.axhline(y=T_tank, color='oc.red5', linestyle=':', linewidth=dm.lw(0))
    ax.text(xmin + 20, T_tank + 2, f'Tank: {T_tank:.1f}°C', color='oc.red5',
            fontsize=fs['legend'], ha='left', va='bottom')
    ax.axhline(y=T0, color='oc.orange5', linestyle=':', linewidth=dm.lw(0))
    ax.text(xmin + 20, T0 - 2, f'Outdoor: {T0:.1f}°C', color='oc.orange5',
            fontsize=fs['legend'], ha='left', va='top')

    # 축 설정
    ax.set_xlabel('Enthalpy [kJ/kg]', fontsize=fs['label'], labelpad=pad['label'])
    ax.set_ylabel('Temperature [°C]', fontsize=fs['label'], labelpad=pad['label'])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.tick_params(axis='both', which='major', labelsize=fs['tick'], pad=pad['tick'])
    
    # legend 만들기 - 항상 마커는 검은색, 사이클 선도 검은색 마커, 선은 gray4로 표기
    legend_handles = []
    handle1, = ax.plot([], [], color=color1, linewidth=dm.lw(0), label='Saturated liquid')
    handle2, = ax.plot([], [], color=color2, linewidth=dm.lw(0), label='Saturated vapor')
    handle3, = ax.plot([], [], color=line_color, linewidth=dm.lw(0), marker='o', linestyle=':', markersize=2.5, markerfacecolor=color3, markeredgecolor=color3, label='Refrigerant cycle', markeredgewidth=0,)
    legend_handles.append(handle1)
    legend_handles.append(handle2)
    legend_handles.append(handle3)
    ax.legend(
        handles=legend_handles,
        loc='upper left', bbox_to_anchor=(0.0, 0.99),
        handlelength=1.5, labelspacing=0.5, columnspacing=2,
        ncol=1, frameon=False, fontsize=fs['legend']
    )
    
def plot_ph_diagram(ax, result, refrigerant, fs, pad):
    """Plot P-h diagram on given axis. 
    If csv_path and timestep_idx are provided, draw horizontal lines for tank water temp / outdoor temp for that timestep.
    """
    # 색상 정의
    color1 = 'oc.blue5'   # 포화 액체선
    color2 = 'oc.red5'    # 포화 증기선
    color3 = 'black'
    color4 = 'oc.gray6'
    line_color = 'oc.gray4'

    # 축 범위 설정
    xmin, xmax = 0, 600  # 엔탈피 [kJ/kg]
    ymin, ymax = 100, 10**4  # 압력 [kPa]

    # 포화선 계산
    T_critical = cu.K2C(CP.PropsSI('Tcrit', refrigerant))
    temps = np.linspace(cu.K2C(CP.PropsSI('Tmin', refrigerant)) + 1, T_critical, 600)
    h_liq = [CP.PropsSI('H', 'T', cu.C2K(T), 'Q', 0, refrigerant) / 1000 for T in temps]
    h_vap = [CP.PropsSI('H', 'T', cu.C2K(T), 'Q', 1, refrigerant) / 1000 for T in temps]
    p_sat = [CP.PropsSI('P', 'T', cu.C2K(T), 'Q', 0, refrigerant) / 1000 for T in temps]

    # 7개 상태점 추출 (T1_star, T1, T2, T2_star, T3_star, T3, T4)
    P1_star = (result.get('P_ref_evap_sat [Pa]') or result.get('P1_star [Pa]', np.nan)) * cu.Pa2kPa
    P1 = (result.get('P_ref_cmp_in [Pa]') or result.get('P1 [Pa]', np.nan)) * cu.Pa2kPa
    P2 = (result.get('P_ref_cmp_out [Pa]') or result.get('P2 [Pa]', np.nan)) * cu.Pa2kPa
    P2_star = (result.get('P_ref_cond_sat_v [Pa]') or result.get('P2_star [Pa]', np.nan)) * cu.Pa2kPa
    P3_star = (result.get('P_ref_cond_sat_l [Pa]') or result.get('P3_star [Pa]', np.nan)) * cu.Pa2kPa
    P3 = (result.get('P_ref_exp_in [Pa]') or result.get('P3 [Pa]', np.nan)) * cu.Pa2kPa
    P4 = (result.get('P_ref_exp_out [Pa]') or result.get('P4 [Pa]', np.nan)) * cu.Pa2kPa
    
    h1_star = (result.get('h_ref_evap_sat [J/kg]') or result.get('h1_star [J/kg]', np.nan)) * cu.J2kJ
    h1 = (result.get('h_ref_cmp_in [J/kg]') or result.get('h1 [J/kg]', np.nan)) * cu.J2kJ
    h2 = (result.get('h_ref_cmp_out [J/kg]') or result.get('h2 [J/kg]', np.nan)) * cu.J2kJ
    h2_star = (result.get('h_ref_cond_sat_v [J/kg]') or result.get('h2_star [J/kg]', np.nan)) * cu.J2kJ
    h3_star = (result.get('h_ref_cond_sat_l [J/kg]') or result.get('h3_star [J/kg]', np.nan)) * cu.J2kJ
    h3 = (result.get('h_ref_exp_in [J/kg]') or result.get('h3 [J/kg]', np.nan)) * cu.J2kJ
    h4 = (result.get('h_ref_exp_out [J/kg]') or result.get('h4 [J/kg]', np.nan)) * cu.J2kJ

    # superheating=0, subcooling=0 시 *_star 누락 fallback (1*=1, 3*=3이면 동일값 사용)
    if np.isnan(h1_star) and not np.isnan(h1):
        h1_star, P1_star = h1, P1
    if np.isnan(h3_star) and not np.isnan(h3):
        h3_star, P3_star = h3, P3

    # 포화선 그리기
    ax.plot(h_liq, p_sat, color=color1, label='Saturated liquid', linewidth=dm.lw(0))
    ax.plot(h_vap, p_sat, color=color2, label='Saturated vapor', linewidth=dm.lw(0))
    
    # 마커 색상 결정
    cycle_markerfacecolor = color3 if result.get('hp_is_on', result.get('is_on', False)) else color4
    cycle_markeredgecolor = color3 if result.get('hp_is_on', result.get('is_on', False)) else color4

    def points_are_close(x1, y1, x2, y2, tol_atol=0.1):
        """두 점이 시각적으로 동일한지 확인 (subcooling/superheating≈0 시 degenerate 세그먼트 감지용)"""
        if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
            return False
        return (np.isclose(x1, x2, atol=tol_atol) and np.isclose(y1, y2, atol=tol_atol))

    # 물리적 프로세스 경로 그리기
    # T4 → T1_star: 등압 증발 (포화 액체선→포화 증기선, P4=P1_star)
    if not (np.isnan(h4) or np.isnan(h1_star) or np.isnan(P4) or np.isnan(P1_star)):
        ax.plot([h4, h1_star], [P4, P1_star], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    
    # T1_star → T1: 등압 superheating (superheating=0이면 끝점 동일 → 점으로 표현)
    if not (np.isnan(h1_star) or np.isnan(h1) or np.isnan(P1_star) or np.isnan(P1)):
        if points_are_close(h1_star, P1_star, h1, P1):
            ax.plot(h1, P1, marker='o', linestyle='None', color=line_color, markersize=2.5, zorder=1)
        else:
            ax.plot([h1_star, h1], [P1_star, P1], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    
    # T1 → T2: 압축
    if not (np.isnan(h1) or np.isnan(h2) or np.isnan(P1) or np.isnan(P2)):
        ax.plot([h1, h2], [P1, P2], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    
    # T2 → T2_star: 등압 냉각 (2=2*이면 끝점 동일 → 점으로 표현)
    if not (np.isnan(h2) or np.isnan(h2_star) or np.isnan(P2) or np.isnan(P2_star)):
        if points_are_close(h2, P2, h2_star, P2_star):
            ax.plot(h2, P2, marker='o', linestyle='None', color=line_color, markersize=2.5, zorder=1)
        else:
            ax.plot([h2, h2_star], [P2, P2_star], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    
    # T2_star → T3_star: 등압 응축 (포화 증기선 → 포화 액체선, P2_star=P3_star)
    if not (np.isnan(h2_star) or np.isnan(h3_star) or np.isnan(P2_star) or np.isnan(P3_star)):
        ax.plot([h2_star, h3_star], [P2_star, P3_star], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    
    # T3_star → T3: 등압 subcooling (subcooling=0이면 끝점 동일 → 점으로 표현)
    if not (np.isnan(h3_star) or np.isnan(h3) or np.isnan(P3_star) or np.isnan(P3)):
        if points_are_close(h3_star, P3_star, h3, P3):
            ax.plot(h3, P3, marker='o', linestyle='None', color=line_color, markersize=2.5, zorder=1)
        else:
            ax.plot([h3_star, h3], [P3_star, P3], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    
    # T3 → T4: 등엔탈피 팽창 (수직선, h3=h4)
    if not (np.isnan(h3) or np.isnan(h4) or np.isnan(P3) or np.isnan(P4)):
        ax.plot([h3, h4], [P3, P4], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)

    # 모든 상태점 마커 표시 (동일한 star/일반 지점 통합)
    points = []
    
    # 1*와 1 비교
    if points_are_close(h1_star, P1_star, h1, P1):
        points.append((h1, P1, '1=1$^*$'))
    else:
        if not (np.isnan(h1_star) or np.isnan(P1_star)):
            points.append((h1_star, P1_star, '1$^*$'))
        if not (np.isnan(h1) or np.isnan(P1)):
            points.append((h1, P1, '1'))
    
    # 2와 2* 비교
    if points_are_close(h2, P2, h2_star, P2_star):
        points.append((h2, P2, '2=2$^*$'))
    else:
        if not (np.isnan(h2) or np.isnan(P2)):
            points.append((h2, P2, '2'))
        if not (np.isnan(h2_star) or np.isnan(P2_star)):
            points.append((h2_star, P2_star, '2$^*$'))
    
    # 3*와 3 비교
    if points_are_close(h3_star, P3_star, h3, P3):
        points.append((h3, P3, '3=3$^*$'))
    else:
        if not (np.isnan(h3_star) or np.isnan(P3_star)):
            points.append((h3_star, P3_star, '3$^*$'))
        if not (np.isnan(h3) or np.isnan(P3)):
            points.append((h3, P3, '3'))
    
    # 4는 항상 표시
    if not (np.isnan(h4) or np.isnan(P4)):
        points.append((h4, P4, '4'))
    
    for h_val, p_val, label in points:
        ax.plot(h_val, p_val, marker='o', markersize=2.5, 
               markerfacecolor=cycle_markerfacecolor, 
               markeredgecolor=cycle_markeredgecolor,
               markeredgewidth=0, zorder=2)
        ax.annotate(label, (h_val, p_val),
                   xytext=(0, 3), textcoords='offset points',
                   ha='center', va='bottom',
                   fontsize=fs['legend'])

    # 축 설정
    ax.set_xlabel('Enthalpy [kJ/kg]', fontsize=fs['label'], labelpad=pad['label'])
    ax.set_ylabel('Pressure [kPa]', fontsize=fs['label'], labelpad=pad['label'])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=fs['tick'], pad=pad['tick'])
    
    legend_handles = []
    handle1, = ax.plot([], [], color=color1, linewidth=dm.lw(0), label='Saturated liquid')
    handle2, = ax.plot([], [], color=color2, linewidth=dm.lw(0), label='Saturated vapor')
    handle3, = ax.plot([], [], color=line_color, linewidth=dm.lw(0), marker='o', linestyle=':', markersize=2.5, markerfacecolor=color3, markeredgecolor=color3, label='Refrigerant cycle', markeredgewidth=0,)
    legend_handles.append(handle1)
    legend_handles.append(handle2)
    legend_handles.append(handle3)
    ax.legend(
        handles=legend_handles,
        loc='upper left', bbox_to_anchor=(0.0, 0.99),
        handlelength=1.5, labelspacing=0.5, columnspacing=2,
        ncol=1, frameon=False, fontsize=fs['legend']
    )
    
def plot_ts_diagram(ax, result, refrigerant, T_tank, T0, fs, pad):
    """Plot T-s diagram on given axis with super heating/cooling considered.
    Shows 6 points: T1_star, T1, T2, T3_star, T3, T4 with physical process paths.
    """
    # 색상 정의
    color1 = 'oc.blue5'   
    color2 = 'oc.red5'    
    color3 = 'black'
    color4 = 'oc.gray6'  
    line_color = 'oc.gray5'

    # 축 범위 설정
    xmin, xmax = 0, 2.0  # 엔트로피 [kJ/(kg·K)]
    ymin, ymax = -40, 120  # 온도 [°C]

    # 포화선 계산
    T_critical = cu.K2C(CP.PropsSI('Tcrit', refrigerant))
    temps = np.linspace(cu.K2C(CP.PropsSI('Tmin', refrigerant)) + 1, T_critical, 600)
    s_liq = [CP.PropsSI('S', 'T', cu.C2K(T), 'Q', 0, refrigerant) / 1000 for T in temps]
    s_vap = [CP.PropsSI('S', 'T', cu.C2K(T), 'Q', 1, refrigerant) / 1000 for T in temps]

    # 포화선 그리기
    ax.plot(s_liq, temps, color=color1, label='Saturated liquid', linewidth=dm.lw(0))
    ax.plot(s_vap, temps, color=color2, label='Saturated vapor', linewidth=dm.lw(0))

    # 7개 상태점 추출 (T1_star, T1, T2, T2_star, T3_star, T3, T4)
    s1_star = (result.get('s_ref_evap_sat [J/(kg·K)]') or result.get('s1_star [J/(kg·K)]', np.nan)) / 1000
    s1 = (result.get('s_ref_cmp_in [J/(kg·K)]') or result.get('s1 [J/(kg·K)]', np.nan)) / 1000
    s2 = (result.get('s_ref_cmp_out [J/(kg·K)]') or result.get('s2 [J/(kg·K)]', np.nan)) / 1000
    s2_star = (result.get('s_ref_cond_sat_v [J/(kg·K)]') or result.get('s2_star [J/(kg·K)]', np.nan)) / 1000
    s3_star = (result.get('s_ref_cond_sat_l [J/(kg·K)]') or result.get('s3_star [J/(kg·K)]', np.nan)) / 1000
    s3 = (result.get('s_ref_exp_in [J/(kg·K)]') or result.get('s3 [J/(kg·K)]', np.nan)) / 1000
    s4 = (result.get('s_ref_exp_out [J/(kg·K)]') or result.get('s4 [J/(kg·K)]', np.nan)) / 1000
    
    T1_star = result.get('T_ref_evap_sat [°C]') or result.get('T1_star [°C]', np.nan)
    T1 = result.get('T_ref_cmp_in [°C]') or result.get('T1 [°C]', np.nan)
    T2 = result.get('T_ref_cmp_out [°C]') or result.get('T2 [°C]', np.nan)
    T2_star = result.get('T_ref_cond_sat_v [°C]') or result.get('T2_star [°C]', np.nan)
    T3_star = result.get('T_ref_cond_sat_l [°C]') or result.get('T3_star [°C]', np.nan)
    T3 = result.get('T_ref_exp_in [°C]') or result.get('T3 [°C]', np.nan)
    T4 = result.get('T_ref_exp_out [°C]') or result.get('T4 [°C]', np.nan)

    # superheating=0, subcooling=0 시 *_star 누락 fallback (1*=1, 3*=3이면 동일값 사용)
    if np.isnan(s1_star) and not np.isnan(s1):
        s1_star, T1_star = s1, T1
    if np.isnan(s3_star) and not np.isnan(s3):
        s3_star, T3_star = s3, T3

    # 마커 색상 결정
    cycle_markerfacecolor = color3 if result.get('hp_is_on', result.get('is_on', False)) else color4
    cycle_markeredgecolor = color3 if result.get('hp_is_on', result.get('is_on', False)) else color4

    def points_are_close(x1, y1, x2, y2, tol_atol=0.1):
        """두 점이 시각적으로 동일한지 확인 (subcooling/superheating≈0 시 degenerate 세그먼트 감지용)"""
        if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
            return False
        return (np.isclose(x1, x2, atol=tol_atol) and np.isclose(y1, y2, atol=tol_atol))

    # 물리적 프로세스 경로 그리기
    # T4 → T1_star: 등압 증발 (포화 액체선→포화 증기선)
    if not (np.isnan(s4) or np.isnan(s1_star) or np.isnan(T4) or np.isnan(T1_star)):
        ax.plot([s4, s1_star], [T4, T1_star], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    
    # T1_star → T1: 등압 superheating (superheating=0이면 끝점 동일 → 점으로 표현)
    if not (np.isnan(s1_star) or np.isnan(s1) or np.isnan(T1_star) or np.isnan(T1)):
        if points_are_close(s1_star, T1_star, s1, T1):
            ax.plot(s1, T1, marker='o', linestyle='None', color=line_color, markersize=2.5, zorder=1)
        else:
            ax.plot([s1_star, s1], [T1_star, T1], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    
    # T1 → T2: 압축 (등엔트로피는 아니지만 실제 압축, s 증가)
    if not (np.isnan(s1) or np.isnan(s2) or np.isnan(T1) or np.isnan(T2)):
        ax.plot([s1, s2], [T1, T2], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    
    # T2 → T2_star: 등압 냉각 (2=2*이면 끝점 동일 → 점으로 표현)
    if not (np.isnan(s2) or np.isnan(s2_star) or np.isnan(T2) or np.isnan(T2_star)):
        if points_are_close(s2, T2, s2_star, T2_star):
            ax.plot(s2, T2, marker='o', linestyle='None', color=line_color, markersize=2.5, zorder=1)
        else:
            ax.plot([s2, s2_star], [T2, T2_star], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    
    # T2_star → T3_star: 등압 응축 (포화 증기선 → 포화 액체선)
    if not (np.isnan(s2_star) or np.isnan(s3_star) or np.isnan(T2_star) or np.isnan(T3_star)):
        ax.plot([s2_star, s3_star], [T2_star, T3_star], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    
    # T3_star → T3: 등압 subcooling (subcooling=0이면 끝점 동일 → 점으로 표현)
    if not (np.isnan(s3_star) or np.isnan(s3) or np.isnan(T3_star) or np.isnan(T3)):
        if points_are_close(s3_star, T3_star, s3, T3):
            ax.plot(s3, T3, marker='o', linestyle='None', color=line_color, markersize=2.5, zorder=1)
        else:
            ax.plot([s3_star, s3], [T3_star, T3], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    
    # T3 → T4: 등엔탈피 팽창 (엔트로피 증가)
    if not (np.isnan(s3) or np.isnan(s4) or np.isnan(T3) or np.isnan(T4)):
        ax.plot([s3, s4], [T3, T4], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)

    # 모든 상태점 마커 표시 (동일한 star/일반 지점 통합)
    points = []
    
    # 1*와 1 비교
    if points_are_close(s1_star, T1_star, s1, T1):
        points.append((s1, T1, '1=1$^*$'))
    else:
        if not (np.isnan(s1_star) or np.isnan(T1_star)):
            points.append((s1_star, T1_star, '1$^*$'))
        if not (np.isnan(s1) or np.isnan(T1)):
            points.append((s1, T1, '1'))
    
    # 2와 2* 비교
    if points_are_close(s2, T2, s2_star, T2_star):
        points.append((s2, T2, '2=2$^*$'))
    else:
        if not (np.isnan(s2) or np.isnan(T2)):
            points.append((s2, T2, '2'))
        if not (np.isnan(s2_star) or np.isnan(T2_star)):
            points.append((s2_star, T2_star, '2$^*$'))
    
    # 3*와 3 비교
    if points_are_close(s3_star, T3_star, s3, T3):
        points.append((s3, T3, '3=3$^*$'))
    else:
        if not (np.isnan(s3_star) or np.isnan(T3_star)):
            points.append((s3_star, T3_star, '3$^*$'))
        if not (np.isnan(s3) or np.isnan(T3)):
            points.append((s3, T3, '3'))
    
    # 4는 항상 표시
    if not (np.isnan(s4) or np.isnan(T4)):
        points.append((s4, T4, '4'))
    
    for s_val, T_val, label in points:
        ax.plot(s_val, T_val, marker='o', markersize=2.5, 
               markerfacecolor=cycle_markerfacecolor, 
               markeredgecolor=cycle_markeredgecolor,
               markeredgewidth=0, zorder=2)
        ax.annotate(label, (s_val, T_val),
                   xytext=(0, 3), textcoords='offset points',
                   ha='center', va='bottom',
                   fontsize=fs['legend'])

    ax.axhline(y=T_tank, color='oc.red5', linestyle=':', linewidth=dm.lw(0))
    ax.text(xmin + 0.05, T_tank + 2, f'Tank: {T_tank:.1f}°C', color='oc.red5',
            fontsize=fs['legend'], ha='left', va='bottom')
    ax.axhline(y=T0, color='oc.orange5', linestyle=':', linewidth=dm.lw(0))
    ax.text(xmin + 0.05, T0 - 2, f'Outdoor: {T0:.1f}°C', color='oc.orange5',
            fontsize=fs['legend'], ha='left', va='top')

    # 축 설정
    ax.set_xlabel('Entropy [kJ/(kg·K)]', fontsize=fs['label'], labelpad=pad['label'])
    ax.set_ylabel('Temperature [°C]', fontsize=fs['label'], labelpad=pad['label'])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.tick_params(axis='both', which='major', labelsize=fs['tick'], pad=pad['tick'])
    
    # legend 만들기 - 항상 마커는 검은색, 사이클 선도 검은색 마커, 선은 gray5로 표기
    legend_handles = []
    handle1, = ax.plot([], [], color=color1, linewidth=dm.lw(0), label='Saturated liquid')
    handle2, = ax.plot([], [], color=color2, linewidth=dm.lw(0), label='Saturated vapor')
    handle3, = ax.plot([], [], color=line_color, linewidth=dm.lw(0), marker='o', linestyle=':', markersize=2.5, markerfacecolor=color3, markeredgecolor=color3, label='Refrigerant cycle', markeredgewidth=0,)
    legend_handles.append(handle1)
    legend_handles.append(handle2)
    legend_handles.append(handle3)
    ax.legend(
        handles=legend_handles,
        loc='upper left', bbox_to_anchor=(0.0, 0.99),
        handlelength=1.5, labelspacing=0.5, columnspacing=2,
        ncol=1, frameon=False, fontsize=fs['legend']
    )
