"""Borehole g-function and air property helpers.

Provides:
- Finite Line Source (FLS) g-function for borehole heat exchangers
- Air dynamic viscosity (Sutherland's formula) and Prandtl number
"""

from typing import Any

import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.special import erf

from . import calc_util as cu
from .constants import SP

try:
    import pygfunction as gt

    HAS_PYGFUNCTION = True
except ImportError:
    HAS_PYGFUNCTION = False


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
    return x * erf(x) - (1 - np.exp(-(x**2))) / SP


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

    temp = np.exp(-((rb * s) ** 2)) / (h * s)
    Is = 2 * f(h) + 2 * f(h + 2 * d) - f(2 * h + 2 * d) - f(2 * d)

    return temp * Is


_g_func_cache: dict[str, Any] = {}


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


def precompute_gfunction(
    N_1: int,
    N_2: int,
    B: float,
    H_b: float,
    D_b: float,
    r_b: float,
    alpha_s: float,
    k_s: float,
    t_max_s: float,
    dt_s: float,
) -> interp1d:
    """Precompute g-function using pygfunction and return an interpolator.

    Creates a rectangular borehole field and computes the g-function
    for log-spaced time steps up to t_max_s (plus an extra margin).
    Returns a callable `interp1d` object predicting the g-function [mK/W].

    Parameters
    ----------
    N_1 : int
        Number of boreholes in x-direction.
    N_2 : int
        Number of boreholes in y-direction.
    B : float
        Borehole spacing [m].
    H_b : float
        Borehole depth/length [m].
    D_b : float
        Buried depth [m].
    r_b : float
        Borehole radius [m].
    alpha_s : float
        Ground thermal diffusivity [m²/s].
    k_s : float
        Ground thermal conductivity [W/mK].
    t_max_s : float
        Maximum simulation time [s].
    dt_s : float
        Simulation timestep [s].

    Returns
    -------
    scipy.interpolate.interp1d
        Interpolator function mapping `time [s]` to `g-function [mK/W]`.
    """

    if not HAS_PYGFUNCTION:
        raise ImportError(
            "pygfunction is not installed. Run `uv pip install pygfunction` to use multi-borehole features."
        )

    # Evaluate from 1 hour to bypass the short-term numerical noise (Fo < 0.1)
    # of the finite line source BEM discretization.
    # The first point is safely evaluated at 3600s where the numerical noise floor is cleared.
    t_min = max(dt_s, 3600.0)
    times = np.geomspace(t_min, t_max_s * 1.5, num=100)

    boreField = gt.boreholes.rectangle_field(N_1=N_1, N_2=N_2, B_1=B, B_2=B, H=H_b, D=D_b, r_b=r_b)

    # Use uniform_heat_flux to ensure stability and compatibility with fundamental FLS assumptions
    options = {"method": "uniform_heat_flux"}
    gfunc_obj = gt.gfunction.gFunction(boreField, alpha_s, time=times, options=options)
    g_vals_dim = gfunc_obj.gFunc / (2 * np.pi * k_s)

    # Prepend 0.0 for t=0.
    # This automatically provides a noise-free linear interpolation for any dt < 3600s !
    times = np.concatenate(([0.0], times))
    g_vals_dim = np.concatenate(([0.0], g_vals_dim))

    # Create interpolator
    return interp1d(times, g_vals_dim, kind="linear", bounds_error=False, fill_value="extrapolate")


def chi_mfls(s, r, H, x_prime, U, alpha_s, z0=0):
    """
    Helper function for MFLS (Moving Finite Line Source) G-function calculation.
    
    Ref: Molina-Giraldo et al. (2011), "A moving finite line source model 
    to simulate borehole heat exchangers with groundwater advection"
    """
    if s == 0:
        return 0.0
    val = chi(s, r, H, z0)

    # Advective multiplier
    adv_mult = np.exp((U * x_prime) / (2 * alpha_s) - (U**2) / (16 * (alpha_s**2) * (s**2)))
    return val * adv_mult


def G_MFLS_Field(
    times: np.ndarray,
    boreholes: list,
    v_gw: float,
    theta_gw: float,
    rho_w: float,
    c_w: float,
    alpha_s: float,
    k_s: float,
    rho_s: float,
    c_s: float,
) -> np.ndarray:
    """Calculate the spatial superposition of the MFLS response for a bore field.

    Parameters
    ----------
    times : np.ndarray
        Array of time values [s]
    boreholes : list
        List of pygfunction Borehole objects
    v_gw : float
        Darcy velocity of groundwater [m/s]
    theta_gw : float
        Direction of groundwater flow [rad]
    rho_w : float
        Density of groundwater [kg/m³]
    c_w : float
        Specific heat capacity of groundwater [J/kgK]
    alpha_s : float
        Ground thermal diffusivity [m²/s]
    k_s : float
        Ground thermal conductivity [W/mK]
    rho_s : float
        Density of ground [kg/m³]
    c_s : float
        Specific heat capacity of ground [J/kgK]

    Returns
    -------
    np.ndarray
        Dimensional g-values for the entire field over time [mK/W]
    """
    U = v_gw * (rho_w * c_w) / (rho_s * c_s)
    N_bh = len(boreholes)

    field_g_vals = np.zeros(len(times))
    factor = 1 / (4 * np.pi * k_s)

    # Evaluate integrals for each pair
    # To optimize, we loop through times and pairs
    for i, b_i in enumerate(boreholes):
        for j, b_j in enumerate(boreholes):
            dx = b_i.x - b_j.x
            dy = b_i.y - b_j.y
            r = np.sqrt(dx**2 + dy**2)

            # Using r_b for self-response
            if i == j:
                r = b_i.r_b
                x_prime = 0.0
            else:
                x_prime = dx * np.cos(theta_gw) + dy * np.sin(theta_gw)

            H = b_j.H
            D = b_j.D

            for t_idx, t in enumerate(times):
                if t <= 0:
                    continue
                lbs = 1 / np.sqrt(4 * alpha_s * t)

                # Single integration from lbs to infinity
                # For high limits, quad works effectively
                integral_val = integrate.quad(chi_mfls, lbs, np.inf, args=(r, H, x_prime, U, alpha_s, D), limit=100)[0]

                # Each source influences target, so we accumulate the dimensional temp rise
                field_g_vals[t_idx] += factor * integral_val

    # Average temperature response of the field
    field_g_vals /= N_bh
    return field_g_vals


def precompute_gfunction_mls(
    N_1: int,
    N_2: int,
    B: float,
    H_b: float,
    D_b: float,
    r_b: float,
    alpha_s: float,
    k_s: float,
    rho_s: float,
    c_s: float,
    v_gw: float,
    theta_gw: float,
    rho_w: float,
    c_w: float,
    t_max_s: float,
    dt_s: float,
) -> interp1d:
    """Precompute the MFLS g-function and return an interpolator."""
    if not HAS_PYGFUNCTION:
        raise ImportError("pygfunction is not installed.")

    t_min = max(dt_s, 3600.0)
    times = np.geomspace(t_min, t_max_s * 1.5, num=50)

    boreField = gt.boreholes.rectangle_field(N_1=N_1, N_2=N_2, B_1=B, B_2=B, H=H_b, D=D_b, r_b=r_b)

    g_vals_dim = G_MFLS_Field(
        times=times,
        boreholes=boreField,
        v_gw=v_gw,
        theta_gw=theta_gw,
        rho_w=rho_w,
        c_w=c_w,
        alpha_s=alpha_s,
        k_s=k_s,
        rho_s=rho_s,
        c_s=c_s,
    )

    times = np.concatenate(([0.0], times))
    g_vals_dim = np.concatenate(([0.0], g_vals_dim))

    return interp1d(times, g_vals_dim, kind="linear", bounds_error=False, fill_value="extrapolate")


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
    T0 = cu.C2K(0)  # Reference temperature [K]
    mu0 = 1.716e-5  # Reference viscosity [Pa·s] at T0
    S = 110.4  # Sutherland constant [K] for air

    mu = mu0 * ((T_K / T0) ** 1.5) * ((T0 + S) / (T_K + S))
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
