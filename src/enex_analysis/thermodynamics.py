"""
Thermodynamic property and exergy calculations.
"""
from typing import Literal

import CoolProp.CoolProp as CP

from .constants import P0_PA, T0_K


def generate_entropy_exergy_term(
    fluid: str,
    T: float,
    P: float,
    Q: float,
    T0: float,
    P0: float,
    phase: Literal['gas', 'liquid', 'twophase'] = 'gas'
) -> tuple[float, float, float]:
    """Calculate entropy, enthalpy, and exergy.

    Parameters
    ----------
    fluid : str
        Fluid name.
    T : float
        Temperature [K].
    P : float
        Pressure [Pa].
    Q : float
        Quality (0 to 1).
    T0 : float
        Dead state temperature [K].
    P0 : float
        Dead state pressure [Pa].
    phase : Literal['gas', 'liquid', 'twophase'], optional
        Fluid phase. Default is 'gas'.

    Returns
    -------
    tuple[float, float, float]
        Entropy [J/kg-K], Enthalpy [J/kg], Exergy [J/kg].
    """
    if phase == 'twophase':
        s = CP.PropsSI('S', 'T', T, 'Q', Q, fluid)
        h = CP.PropsSI('H', 'T', T, 'Q', Q, fluid)
    else:
        s = CP.PropsSI('S', 'T', T, 'P', P, fluid)
        h = CP.PropsSI('H', 'T', T, 'P', P, fluid)

    s0 = CP.PropsSI('S', 'T', T0, 'P', P0, fluid)
    h0 = CP.PropsSI('H', 'T', T0, 'P', P0, fluid)
    exergy = (h - h0) - T0 * (s - s0)

    return s, h, exergy

def calc_energy_flow(mass_flow: float, h1: float, h2: float) -> float:
    """Calculate energy flow rate.

    Parameters
    ----------
    mass_flow : float
        Mass flow rate [kg/s].
    h1 : float
        Enthalpy state 1 [J/kg].
    h2 : float
        Enthalpy state 2 [J/kg].

    Returns
    -------
    float
        Energy flow rate [W].
    """
    return mass_flow * abs(h1 - h2)

def calc_exergy_flow(mass_flow: float, ex1: float, ex2: float) -> float:
    """Calculate exergy flow rate.

    Parameters
    ----------
    mass_flow : float
        Mass flow rate [kg/s].
    ex1 : float
        Exergy state 1 [J/kg].
    ex2 : float
        Exergy state 2 [J/kg].

    Returns
    -------
    float
        Exergy flow rate [W].
    """
    return mass_flow * abs(ex1 - ex2)

def calc_refrigerant_exergy(
    fluid: str,
    T: float,
    P: float,
    Q: float,
    T0: float = T0_K,
    P0: float = P0_PA,
    phase: Literal['gas', 'liquid', 'twophase'] = 'gas'
) -> float:
    """Calculate specific exergy of refrigerant.

    Parameters
    ----------
    fluid : str
        Refrigerant name.
    T : float
        Temperature [K].
    P : float
        Pressure [Pa].
    Q : float
        Quality.
    T0 : float, optional
        Dead state temperature [K]. Default is T0_K.
    P0 : float, optional
        Dead state pressure [Pa]. Default is P0_PA.
    phase : Literal['gas', 'liquid', 'twophase'], optional
        Fluid phase. Default is 'gas'.

    Returns
    -------
    float
        Specific exergy [J/kg].
    """
    _, _, ex = generate_entropy_exergy_term(fluid, T, P, Q, T0, P0, phase)
    return ex

def convert_electricity_to_exergy(P_elec: float) -> float:
    """Convert electrical power to exergy rate.
    
    Electricity is a pure form of work, so its exergy equals its energy.

    Parameters
    ----------
    P_elec : float
        Electrical power [W].

    Returns
    -------
    float
        Exergy rate [W].
    """
    return P_elec
