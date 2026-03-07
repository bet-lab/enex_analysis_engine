"""
Visualization and summary output functions.
"""
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def print_simulation_summary(
    df: pd.DataFrame, simulation_time_step: int, dV_ou_a_design: float
) -> None:
    """Print a comprehensive summary of simulation results.

    Parameters
    ----------
    df : pd.DataFrame
        Simulation result DataFrame.
    simulation_time_step : int
        Time step [s].
    dV_ou_a_design : float
        Design airflow rate of outdoor unit [m3/s].
    """
    if df.empty:
        print("Empty DataFrame provided.")
        return

    energy_kJ = simulation_time_step / 1000  # J to kJ conversion factor

    print("-" * 50)
    print("SIMULATION SUMMARY")
    print("-" * 50)

    # Energy metrics
    E_comp = df['W_comp'].sum() * energy_kJ if 'W_comp' in df else 0
    E_pump = df['W_pump'].sum() * energy_kJ if 'W_pump' in df else 0
    E_fan = df['W_fan'].sum() * energy_kJ if 'W_fan' in df else 0

    total_elec = E_comp + E_pump + E_fan

    print(f"Total Electricity Consumption: {total_elec / 3600:.2f} kWh")
    print(f"  - Compressor: {E_comp / 3600:.2f} kWh")
    print(f"  - Pump:       {E_pump / 3600:.2f} kWh")
    print(f"  - Fan:        {E_fan / 3600:.2f} kWh")
    print("-" * 50)

def _points_are_close(x1: float, y1: float, x2: float, y2: float, tol: float = 0.1) -> bool:
    """Check if two points are visually identical on a plot."""
    return abs(x1 - x2) < tol and abs(y1 - y2) < tol

def plot_th_diagram(
    ax: plt.Axes,
    result: dict[str, float],
    refrigerant: str,
    T_tank: float | None = None,
    T0: float | None = None,
    fs: int = 10,
    pad: float = 5.0
) -> None:
    """Plot Temperature-Enthalpy (T-h) diagram.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object.
    result : dict[str, float]
        Refrigerant cycle state points.
    refrigerant : str
        Refrigerant name.
    T_tank : float, optional
        Tank temperature [K].
    T0 : float, optional
        Outdoor temperature [K].
    fs : int, optional
        Font size. Default is 10.
    pad : float, optional
        Padding for axes limits. Default is 5.0.
    """
    if not isinstance(result, dict) or 'T1' not in result:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
        return

    try:
        T_crit = CP.PropsSI('TCRIT', refrigerant)
        T_min = CP.PropsSI('TMIN', refrigerant)

        # Draw saturation dome
        T_dome = np.linspace(T_min, T_crit - 0.1, 100)
        H_sat_liq = [CP.PropsSI('H', 'T', t, 'Q', 0, refrigerant) / 1000 for t in T_dome]
        H_sat_vap = [CP.PropsSI('H', 'T', t, 'Q', 1, refrigerant) / 1000 for t in T_dome]

        ax.plot(H_sat_liq, T_dome - 273.15, 'k-', linewidth=1.5, alpha=0.5)
        ax.plot(H_sat_vap, T_dome - 273.15, 'k-', linewidth=1.5, alpha=0.5)

        # Plot cycle points
        H = [result['H1']/1000, result['H2']/1000, result['H3']/1000, result['H4']/1000, result['H1']/1000]
        T_C = [result['T1']-273.15, result['T2']-273.15, result['T3']-273.15, result['T4']-273.15, result['T1']-273.15]

        ax.plot(H, T_C, 'b-o', linewidth=2, markersize=6)

        # Add labels
        for i, (h, t) in enumerate(zip(H[:-1], T_C[:-1])):
            ax.text(h, t + pad/2, f"{i+1}", fontsize=fs, ha='center', va='bottom')

        ax.set_xlabel('Enthalpy [kJ/kg]', fontsize=fs)
        ax.set_ylabel('Temperature [°C]', fontsize=fs)
        ax.set_title(f"T-h Diagram ({refrigerant})", fontsize=fs+2)
        ax.grid(True, linestyle='--', alpha=0.7)
    except Exception as e:
        ax.text(0.5, 0.5, f"Plot Error: {e}", ha='center', va='center', transform=ax.transAxes)

def plot_ph_diagram(
    ax: plt.Axes,
    result: dict[str, float],
    refrigerant: str,
    fs: int = 10,
    pad: float = 5.0
) -> None:
    """Plot Pressure-Enthalpy (P-h) diagram.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object.
    result : dict[str, float]
        Refrigerant cycle state points.
    refrigerant : str
        Refrigerant name.
    fs : int, optional
        Font size. Default is 10.
    pad : float, optional
        Padding for axes limits. Default is 5.0.
    """
    if not isinstance(result, dict) or 'P1' not in result:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
        return

    try:
        T_crit = CP.PropsSI('TCRIT', refrigerant)
        T_min = CP.PropsSI('TMIN', refrigerant)

        # Draw saturation dome
        T_dome = np.linspace(T_min, T_crit - 0.1, 100)
        H_sat_liq = [CP.PropsSI('H', 'T', t, 'Q', 0, refrigerant) / 1000 for t in T_dome]
        H_sat_vap = [CP.PropsSI('H', 'T', t, 'Q', 1, refrigerant) / 1000 for t in T_dome]
        P_sat = [CP.PropsSI('P', 'T', t, 'Q', 0, refrigerant) / 1000 for t in T_dome]

        ax.semilogy(H_sat_liq, P_sat, 'k-', linewidth=1.5, alpha=0.5)
        ax.semilogy(H_sat_vap, P_sat, 'k-', linewidth=1.5, alpha=0.5)

        # Plot cycle points
        H = [result['H1']/1000, result['H2']/1000, result['H3']/1000, result['H4']/1000, result['H1']/1000]
        P = [result['P1']/1000, result['P2']/1000, result['P3']/1000, result['P4']/1000, result['P1']/1000]

        ax.plot(H, P, 'r-s', linewidth=2, markersize=6)

        # Add labels
        for i, (h, p) in enumerate(zip(H[:-1], P[:-1])):
            ax.text(h, p * 1.1, f"{i+1}", fontsize=fs, ha='center', va='bottom')

        ax.set_xlabel('Enthalpy [kJ/kg]', fontsize=fs)
        ax.set_ylabel('Pressure [kPa] (log scale)', fontsize=fs)
        ax.set_title(f"P-h Diagram ({refrigerant})", fontsize=fs+2)
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
    except Exception as e:
        ax.text(0.5, 0.5, f"Plot Error: {e}", ha='center', va='center', transform=ax.transAxes)

def plot_ts_diagram(
    ax: plt.Axes,
    result: dict[str, float],
    refrigerant: str,
    T_tank: float | None = None,
    T0: float | None = None,
    fs: int = 10,
    pad: float = 5.0
) -> None:
    """Plot Temperature-Entropy (T-s) diagram.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object.
    result : dict[str, float]
        Refrigerant cycle state points.
    refrigerant : str
        Refrigerant name.
    T_tank : float, optional
        Tank temperature [K].
    T0 : float, optional
        Outdoor temperature [K].
    fs : int, optional
        Font size. Default is 10.
    pad : float, optional
        Padding for axes limits. Default is 5.0.
    """
    if not isinstance(result, dict) or 'S1' not in result:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
        return

    try:
        T_crit = CP.PropsSI('TCRIT', refrigerant)
        T_min = CP.PropsSI('TMIN', refrigerant)

        # Draw saturation dome
        T_dome = np.linspace(T_min, T_crit - 0.1, 100)
        S_sat_liq = [CP.PropsSI('S', 'T', t, 'Q', 0, refrigerant) / 1000 for t in T_dome]
        S_sat_vap = [CP.PropsSI('S', 'T', t, 'Q', 1, refrigerant) / 1000 for t in T_dome]

        ax.plot(S_sat_liq, T_dome - 273.15, 'k-', linewidth=1.5, alpha=0.5)
        ax.plot(S_sat_vap, T_dome - 273.15, 'k-', linewidth=1.5, alpha=0.5)

        # Plot cycle points
        S = [result['S1']/1000, result['S2']/1000, result['S3']/1000, result['S4']/1000, result['S1']/1000]
        T_C = [result['T1']-273.15, result['T2']-273.15, result['T3']-273.15, result['T4']-273.15, result['T1']-273.15]

        ax.plot(S, T_C, 'g-^', linewidth=2, markersize=6)

        # Add labels
        for i, (s, t) in enumerate(zip(S[:-1], T_C[:-1])):
            ax.text(s, t + pad/2, f"{i+1}", fontsize=fs, ha='center', va='bottom')

        ax.set_xlabel('Entropy [kJ/kg-K]', fontsize=fs)
        ax.set_ylabel('Temperature [°C]', fontsize=fs)
        ax.set_title(f"T-s Diagram ({refrigerant})", fontsize=fs+2)
        ax.grid(True, linestyle='--', alpha=0.7)
    except Exception as e:
        ax.text(0.5, 0.5, f"Plot Error: {e}", ha='center', va='center', transform=ax.transAxes)
