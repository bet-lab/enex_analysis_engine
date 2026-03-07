"""
Heat transfer and fluid dynamics calculations.
"""
import math

import numpy as np


def darcy_friction_factor(
    Re: float, e: float, d: float, is_active: bool = True
) -> float:
    """Calculate the Darcy friction factor.
    
    Uses Haaland equation.

    Parameters
    ----------
    Re : float
        Reynolds number.
    e : float
        Surface roughness [m].
    d : float
        Diameter [m].
    is_active : bool, optional
        If False, returns np.nan.

    Returns
    -------
    float
        Friction factor.
    """
    if not is_active:
        return np.nan

    if Re < 2300:
        return 64.0 / max(Re, 1e-10)

    return 1.0 / (-1.8 * math.log10((e / d / 3.7)**1.11 + 6.9 / Re))**2

def calc_h_vertical_plate(
    T_s: float, T_inf: float, L: float, fluid: str = 'Air', is_active: bool = True
) -> float:
    """Calculate natural convection heat transfer coefficient for a vertical plate.

    Parameters
    ----------
    T_s : float
        Surface temperature [K].
    T_inf : float
        Fluid temperature [K].
    L : float
        Characteristic length [m].
    fluid : str, optional
        Fluid name. Default is 'Air'.
    is_active : bool, optional
        If False, returns np.nan.

    Returns
    -------
    float
        Heat transfer coefficient [W/m2K].
    """
    if not is_active:
        return np.nan

    import CoolProp.CoolProp as CP

    T_f = (T_s + T_inf) / 2
    P = 101325

    # Calculate properties at film temperature
    beta = CP.PropsSI('isobaric_expansion_coefficient', 'T', T_f, 'P', P, fluid)
    nu = CP.PropsSI('V', 'T', T_f, 'P', P, fluid) / CP.PropsSI('D', 'T', T_f, 'P', P, fluid)
    Pr = CP.PropsSI('PRANDTL', 'T', T_f, 'P', P, fluid)
    k = CP.PropsSI('L', 'T', T_f, 'P', P, fluid)

    g = 9.81
    Ra = g * beta * abs(T_s - T_inf) * L**3 / (nu * (k / (CP.PropsSI('D', 'T', T_f, 'P', P, fluid) * CP.PropsSI('C', 'T', T_f, 'P', P, fluid))))

    if Ra < 1e9:
        Nu = 0.59 * Ra**0.25
    else:
        Nu = 0.1 * Ra**(1/3)

    return Nu * k / L

def calc_UA_tank_arr(
    arr_D_in: float,
    arr_D_out: float,
    arr_L: float,
    arr_k: float,
    h_in: float,
    h_out: float
) -> float:
    """Calculate thermal conductance (UA) of a multi-layer cylindrical tank.

    Parameters
    ----------
    arr_D_in : array_like
        Inner diameters of layers [m].
    arr_D_out : array_like
        Outer diameters of layers [m].
    arr_L : array_like
        Lengths of layers [m].
    arr_k : array_like
        Thermal conductivities of layers [W/mK].
    h_in : float
        Inner convection coefficient [W/m2K].
    h_out : float
        Outer convection coefficient [W/m2K].

    Returns
    -------
    float
        Thermal conductance [W/K].
    """
    R_in = 1 / (max(h_in, 1e-10) * math.pi * arr_D_in[0] * arr_L[0])
    R_out = 1 / (max(h_out, 1e-10) * math.pi * arr_D_out[-1] * arr_L[-1])

    R_cond = 0.0
    for i in range(len(arr_D_in)):
        R_cond += math.log(arr_D_out[i] / arr_D_in[i]) / (2 * math.pi * max(arr_k[i], 1e-10) * arr_L[i])

    return 1 / (R_in + R_cond + R_out)

def calc_simple_tank_UA(
    V: float,
    t_ins: float,
    k_ins: float = 0.04,
    aspect_ratio: float = 2.0
) -> float:
    """Calculate simple thermal conductance of a cylindrical tank.

    Parameters
    ----------
    V : float
        Volume [m3].
    t_ins : float
        Insulation thickness [m].
    k_ins : float, optional
        Insulation thermal conductivity [W/mK]. Default is 0.04.
    aspect_ratio : float, optional
        Height / Diameter ratio. Default is 2.0.

    Returns
    -------
    float
        Thermal conductance [W/K].
    """
    R = (V / (math.pi * aspect_ratio))**(1/3)
    H = aspect_ratio * 2 * R
    A_surf = 2 * math.pi * R * H + 2 * math.pi * R**2
    return A_surf * k_ins / max(t_ins, 1e-10)

def calc_LMTD_counter_flow(
    Th_in: float, Th_out: float, Tc_in: float, Tc_out: float
) -> float:
    """Calculate Log-Mean Temperature Difference for counter-flow heat exchanger.

    Parameters
    ----------
    Th_in : float
        Hot stream inlet temp [K].
    Th_out : float
        Hot stream outlet temp [K].
    Tc_in : float
        Cold stream inlet temp [K].
    Tc_out : float
        Cold stream outlet temp [K].

    Returns
    -------
    float
        LMTD [K].
    """
    dT1 = Th_in - Tc_out
    dT2 = Th_out - Tc_in

    if dT1 <= 0 or dT2 <= 0:
        return np.nan

    if abs(dT1 - dT2) < 1e-5:
        return dT1

    return (dT1 - dT2) / math.log(dT1 / dT2)

def calc_LMTD_parallel_flow(
    Th_in: float, Th_out: float, Tc_in: float, Tc_out: float
) -> float:
    """Calculate Log-Mean Temperature Difference for parallel-flow heat exchanger.

    Parameters
    ----------
    Th_in : float
        Hot stream inlet temp [K].
    Th_out : float
        Hot stream outlet temp [K].
    Tc_in : float
        Cold stream inlet temp [K].
    Tc_out : float
        Cold stream outlet temp [K].

    Returns
    -------
    float
        LMTD [K].
    """
    dT1 = Th_in - Tc_in
    dT2 = Th_out - Tc_out

    if dT1 <= 0 or dT2 <= 0:
        return np.nan

    if abs(dT1 - dT2) < 1e-5:
        return dT1

    return (dT1 - dT2) / math.log(dT1 / dT2)

def TRIDIAG_MATRIX_ALGORITHM(
    a_M: list[float],
    a_P: list[float],
    a_E: list[float],
    a_W: list[float],
    b_P: list[float]
) -> list[float]:
    """Solve tridiagonal matrix system using Thomas algorithm.

    Parameters
    ----------
    a_M : list[float]
        Main diagonal (a_P in standard notation).
    a_P : list[float]
        Not used directly, maintained for signature compatibility.
    a_E : list[float]
        Upper diagonal.
    a_W : list[float]
        Lower diagonal.
    b_P : list[float]
        RHS vector.

    Returns
    -------
    list[float]
        Solution vector.
    """
    n = len(b_P)
    c_star = np.zeros(n)
    d_star = np.zeros(n)
    phi = np.zeros(n)

    c_star[0] = a_E[0] / a_M[0]
    d_star[0] = b_P[0] / a_M[0]

    for i in range(1, n-1):
        inv_denom = 1.0 / (a_M[i] - a_W[i] * c_star[i-1])
        c_star[i] = a_E[i] * inv_denom
        d_star[i] = (b_P[i] + a_W[i] * d_star[i-1]) * inv_denom

    d_star[n-1] = (b_P[n-1] + a_W[n-1] * d_star[n-2]) / (a_M[n-1] - a_W[n-1] * c_star[n-2])

    phi[n-1] = d_star[n-1]
    for i in range(n-2, -1, -1):
        phi[i] = c_star[i] * phi[i+1] + d_star[i]

    return phi.tolist()
