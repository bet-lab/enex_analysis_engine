"""Heat pump Coefficient of Performance (COP) models.

Provides empirical COP correlations for:
- Air source heat pumps (cooling and heating modes)
- Ground source heat pumps (modified Carnot model)
"""

from . import calc_util as cu


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
    COP = (
        -7.46 * (PLR - 0.0047 * cu.K2C(T0) - 0.477) ** 2
        + 0.0941 * cu.K2C(T0)
        + 4.34
    )
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
        raise ValueError(
            "T_cond must be greater than T_evap for a valid COP calculation."
        )

    delta_T = Tg - T_evap

    # Compute COP using the modified Carnot expression
    denominator = 1 - (Tg / T_cond) + (delta_T / (T_cond * theta_hat))

    if denominator <= 0:
        return float("nan")  # Avoid division by zero or negative COP

    COP = 1 / denominator
    return COP
