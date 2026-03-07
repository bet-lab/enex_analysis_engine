from .dhw import (
    build_dhw_usage_ratio,
    calc_cold_water_temp,
    calc_total_water_use_from_schedule,
    make_dhw_schedule_from_Annex_42_profile,
)
from .heat_transfer import (
    calc_h_vertical_plate,
    calc_LMTD_counter_flow,
    calc_LMTD_parallel_flow,
    calc_simple_tank_UA,
    calc_UA_tank_arr,
    darcy_friction_factor,
)
from .refrigerant import (
    calc_ref_state,
    create_lmtd_constraints,
    find_ref_loop_optimal_operation,
)
from .thermodynamics import (
    calc_energy_flow,
    calc_exergy_flow,
    calc_refrigerant_exergy,
    convert_electricity_to_exergy,
    generate_entropy_exergy_term,
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
from scipy import integrate
from scipy.optimize import root_scalar
from scipy.special import erf

try:
    import dartwork_mpl as dm
except ImportError:
    # dartwork_mpl이 없는 경우를 대비한 fallback
    dm = None
from . import calc_util as cu
from .constants import SP, c_a, c_w, rho_a, rho_w

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




def calc_UA_from_dV_fan(dV_fan, dV_fan_design, A_cross, UA):
    """
    Calculate heat transfer coefficient based on Dittus-Boelter equation.
    
    This function calculates heat transfer coefficient based on air velocity.
    Dittus-Boelter equation: Nu = 0.023 * Re^0.8 * Pr^n
    Proportional to velocity^0.8.
    
    Parameters:
    -----------
    dV_fan : float
        Fan flow rate [m3/s]
    dV_fan_design : float
        Design fan flow rate [m3/s]
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

    dV_min = dV_fan_design * 0.1 # [m3/s]
    dV_max = dV_fan_design # [m3/s]

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
        Current flow rate [m3/s]
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
    dV_stc,                # STC 유량 [m3/s]
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


#%%


