"""
Visualization and summary output functions.
"""

import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dartwork_mpl as dm

from . import calc_util as cu


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

    required_columns = ['converged', 'E_ou_fan [W]', 'E_tot [W]', 'dV_ou_a [m3/s]', 'cmp_rpm [rpm]']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Required columns not found in DataFrame: {missing_columns}")

    active_mask = df['cmp_rpm [rpm]'] > 0

    print("="*50)
    # 1. Convergence Status
    converged_all = df['converged'].all()
    print(f"[Convergence Status] All converged: {converged_all}")
    if not converged_all:
        nonconverged_count = (~df['converged']).sum()
        print(f"  - Non-converged steps: {nonconverged_count} / {len(df)}")
    print("-" * 50)

    # 2. Compressor Statistics
    cmp_rpm_active = df.loc[active_mask, 'cmp_rpm [rpm]']
    print("[Compressor Speed]")
    if not cmp_rpm_active.empty:
        print(f"  - Min: {cmp_rpm_active.min():.1f} rpm | Max: {cmp_rpm_active.max():.1f} rpm")
        print(f"  - Avg (active): {cmp_rpm_active.mean():.1f} rpm")
    else:
        print("  - No active data.")
    print("-" * 50)

    # 3. Fan Flow Rate Statistics
    fan_active = df.loc[active_mask, 'dV_ou_a [m3/s]']
    print("[Fan Flow Rate]")
    if not fan_active.empty:
        fan_avg = fan_active.mean()
        fan_avg_pct = (fan_avg / dV_ou_a_design) * 100
        print(f"  - Min: {fan_active.min():.3f} m³/s | Max: {fan_active.max():.3f} m³/s")
        print(f"  - Avg: {fan_avg:.3f} m³/s ({fan_avg_pct:.1f}% of design)")
    else:
        print("  - No active data.")
    print("-" * 50)

    # 3-1. Fan Velocity & Pressure Statistics
    if 'v_ou_a [m/s]' in df.columns:
        v_fan_active = df.loc[active_mask, 'v_ou_a [m/s]']
        print("[Fan Velocity]")
        if not v_fan_active.empty:
            print(f"  - Min: {v_fan_active.min():.2f} m/s | Max: {v_fan_active.max():.2f} m/s")
            print(f"  - Avg: {v_fan_active.mean():.2f} m/s")
        else:
            print("  - No active data.")
        print("-" * 50)

    if 'dP_ou_fan_static [Pa]' in df.columns and 'dP_ou_fan_dynamic [Pa]' in df.columns:
        dP_static = df.loc[active_mask, 'dP_ou_fan_static [Pa]']
        dP_dynamic = df.loc[active_mask, 'dP_ou_fan_dynamic [Pa]']
        
        print("[Fan Pressure (Static / Dynamic)]")
        if not dP_static.empty:
            print(f"  - Static  : Avg {dP_static.mean():.1f} Pa | Min {dP_static.min():.1f} Pa | Max {dP_static.max():.1f} Pa")
            print(f"  - Dynamic : Avg {dP_dynamic.mean():.1f} Pa | Min {dP_dynamic.min():.1f} Pa | Max {dP_dynamic.max():.1f} Pa")
        else:
            print("  - No active data.")
        print("-" * 50)

    # 4. Fan Power Statistics
    fan_p_active = df.loc[active_mask, 'E_ou_fan [W]']
    print("[Fan Power Use]")
    if not fan_p_active.empty:
        print(f"  - Min: {fan_p_active.min():.1f} W | Max: {fan_p_active.max():.1f} W")
        print(f"  - Avg: {fan_p_active.mean():.1f} W")
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
        print("[Outdoor Air Temperature Difference (In - Out)]")
        if active_mask.any():
            delta_T = df.loc[active_mask, 'T_ou_a_in [°C]'] - df.loc[active_mask, 'T_ou_a_out [°C]']
            print(f"  - Avg Delta T: {delta_T.mean():.2f} K | Max Delta T: {delta_T.max():.2f} K")
        else:
            print("  - No active data.")
        print("-" * 50)

    # 7. Heat Exchange Performance: Temp Differences
    print("[Heat Exchanger Temperature Differences]")
    
    # Condenser (T_cond - T_tank_w)
    if 'T_ref_cond_sat_l [°C]' in df.columns and 'T_tank_w [°C]' in df.columns:
        T_cond = df.loc[active_mask, 'T_ref_cond_sat_l [°C]']
        T_tank_w = df.loc[active_mask, 'T_tank_w [°C]']
        
        if not T_cond.empty and not T_tank_w.empty:
            dT_cond = T_cond - T_tank_w
            print(f"  - Condenser (T_cond - T_tank) Avg: {dT_cond.mean():.2f} K | Min: {dT_cond.min():.2f} K | Max: {dT_cond.max():.2f} K")
        else:
            print("  - Condenser: No data")

    # Evaporator (T_air_in - T_evap) & (T_air_in - T_air_out)
    if 'T_ou_a_in [°C]' in df.columns and 'T_ref_evap_sat [°C]' in df.columns and 'T_ou_a_out [°C]' in df.columns:
        T_air_in = df.loc[active_mask, 'T_ou_a_in [°C]']
        T_evap_sat = df.loc[active_mask, 'T_ref_evap_sat [°C]']
        T_air_out = df.loc[active_mask, 'T_ou_a_out [°C]']
        
        if not T_air_in.empty:
            dT_evap_drive = T_air_in - T_evap_sat
            dT_air_drop = T_air_in - T_air_out
            
            print(f"  - Evap Drive (T_air_in - T_evap) Avg: {dT_evap_drive.mean():.2f} K")
            print(f"  - Air Drop (T_air_in - T_air_out) Avg: {dT_air_drop.mean():.2f} K")
        else:
            print("  - Evaporator: No data")

    print("="*50)


def _points_are_close(
    x1: float, y1: float, x2: float, y2: float, tol: float = 0.1
) -> bool:
    """Check if two points are visually identical on a plot."""
    return abs(x1 - x2) < tol and abs(y1 - y2) < tol


def plot_th_diagram(
    ax: plt.Axes,
    result: dict[str, float],
    refrigerant: str,
    T_tank: float | None = None,
    T0: float | None = None,
) -> None:
    """Plot T-h diagram on given axis."""
    color1, color2, color3, color4, line_color = 'oc.blue5', 'oc.red5', 'black', 'oc.gray6', 'oc.gray5'
    xmin, xmax, ymin, ymax = 0, 600, -40, 120

    T_critical = cu.K2C(CP.PropsSI('Tcrit', refrigerant))
    temps = np.linspace(cu.K2C(CP.PropsSI('Tmin', refrigerant)) + 1, T_critical, 600)
    h_liq = [CP.PropsSI('H', 'T', cu.C2K(T), 'Q', 0, refrigerant) / 1000 for T in temps]
    h_vap = [CP.PropsSI('H', 'T', cu.C2K(T), 'Q', 1, refrigerant) / 1000 for T in temps]

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

    if np.isnan(h1_star) and not np.isnan(h1): h1_star, T1_star = h1, T1
    if np.isnan(h3_star) and not np.isnan(h3): h3_star, T3_star = h3, T3

    ax.plot(h_liq, temps, color=color1, label='Saturated liquid', linewidth=dm.lw(0))
    ax.plot(h_vap, temps, color=color2, label='Saturated vapor', linewidth=dm.lw(0))

    cycle_markerfacecolor = color3 if result.get('hp_is_on', result.get('is_on', False)) else color4
    cycle_markeredgecolor = color3 if result.get('hp_is_on', result.get('is_on', False)) else color4

    def points_are_close(x1, y1, x2, y2, tol_atol=0.1):
        if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2): return False
        return (np.isclose(x1, x2, atol=tol_atol) and np.isclose(y1, y2, atol=tol_atol))

    if not (np.isnan(h4) or np.isnan(h1_star) or np.isnan(T4) or np.isnan(T1_star)):
        ax.plot([h4, h1_star], [T4, T1_star], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    if not (np.isnan(h1_star) or np.isnan(h1) or np.isnan(T1_star) or np.isnan(T1)):
        if points_are_close(h1_star, T1_star, h1, T1): ax.plot(h1, T1, marker='o', linestyle='None', color=line_color, markersize=2.5, zorder=1)
        else: ax.plot([h1_star, h1], [T1_star, T1], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    if not (np.isnan(h1) or np.isnan(h2) or np.isnan(T1) or np.isnan(T2)):
        ax.plot([h1, h2], [T1, T2], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    if not (np.isnan(h2) or np.isnan(h2_star) or np.isnan(T2) or np.isnan(T2_star)):
        if points_are_close(h2, T2, h2_star, T2_star): ax.plot(h2, T2, marker='o', linestyle='None', color=line_color, markersize=2.5, zorder=1)
        else: ax.plot([h2, h2_star], [T2, T2_star], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    if not (np.isnan(h2_star) or np.isnan(h3_star) or np.isnan(T2_star) or np.isnan(T3_star)):
        ax.plot([h2_star, h3_star], [T2_star, T3_star], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    if not (np.isnan(h3_star) or np.isnan(h3) or np.isnan(T3_star) or np.isnan(T3)):
        if points_are_close(h3_star, T3_star, h3, T3): ax.plot(h3, T3, marker='o', linestyle='None', color=line_color, markersize=2.5, zorder=1)
        else: ax.plot([h3_star, h3], [T3_star, T3], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    if not (np.isnan(h3) or np.isnan(h4) or np.isnan(T3) or np.isnan(T4)):
        ax.plot([h3, h4], [T3, T4], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)

    points = []
    if points_are_close(h1_star, T1_star, h1, T1): points.append((h1, T1, '1=1\''))
    else:
        if not (np.isnan(h1_star) or np.isnan(T1_star)): points.append((h1_star, T1_star, '1\''))
        if not (np.isnan(h1) or np.isnan(T1)): points.append((h1, T1, '1'))
    
    if points_are_close(h2, T2, h2_star, T2_star): points.append((h2, T2, '2=2\''))
    else:
        if not (np.isnan(h2) or np.isnan(T2)): points.append((h2, T2, '2'))
        if not (np.isnan(h2_star) or np.isnan(T2_star)): points.append((h2_star, T2_star, '2\''))
    
    if points_are_close(h3_star, T3_star, h3, T3): points.append((h3, T3, '3=3\''))
    else:
        if not (np.isnan(h3_star) or np.isnan(T3_star)): points.append((h3_star, T3_star, '3\''))
        if not (np.isnan(h3) or np.isnan(T3)): points.append((h3, T3, '3'))
    
    if not (np.isnan(h4) or np.isnan(T4)): points.append((h4, T4, '4'))
    
    for h_val, T_val, label in points:
        ax.plot(h_val, T_val, marker='o', markersize=2.5, markerfacecolor=cycle_markerfacecolor, markeredgecolor=cycle_markeredgecolor, markeredgewidth=0, zorder=2)
        ax.annotate(label, (h_val, T_val), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

    if T_tank is not None and not np.isnan(T_tank):
        ax.axhline(y=T_tank, color='oc.red5', linestyle=':', linewidth=dm.lw(0))
        ax.text(xmin + 20, T_tank + 2, f'Tank: {T_tank:.1f}°C', color='oc.red5', ha='left', va='bottom')
    if T0 is not None and not np.isnan(T0):
        ax.axhline(y=T0, color='oc.orange5', linestyle=':', linewidth=dm.lw(0))
        ax.text(xmin + 20, T0 - 2, f'Outdoor: {T0:.1f}°C', color='oc.orange5', ha='left', va='top')

    ax.set_xlabel('Enthalpy [kJ/kg]')
    ax.set_ylabel('Temperature [°C]')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    legend_handles = [
        ax.plot([], [], color=color1, linewidth=dm.lw(0), label='Saturated liquid')[0],
        ax.plot([], [], color=color2, linewidth=dm.lw(0), label='Saturated vapor')[0],
        ax.plot([], [], color=line_color, linewidth=dm.lw(0), marker='o', linestyle=':', markersize=2.5, markerfacecolor=color3, markeredgecolor=color3, label='Refrigerant cycle', markeredgewidth=0)[0]
    ]
    ax.legend(handles=legend_handles, loc='upper right', handlelength=1.5, labelspacing=0.5, columnspacing=2, ncol=1, frameon=False)


def plot_ph_diagram(
    ax: plt.Axes,
    result: dict[str, float],
    refrigerant: str,
) -> None:
    """Plot P-h diagram on given axis."""
    color1, color2, color3, color4, line_color = 'oc.blue5', 'oc.red5', 'black', 'oc.gray6', 'oc.gray4'
    xmin, xmax, ymin, ymax = 0, 600, 100, 10**4

    T_critical = cu.K2C(CP.PropsSI('Tcrit', refrigerant))
    temps = np.linspace(cu.K2C(CP.PropsSI('Tmin', refrigerant)) + 1, T_critical, 600)
    h_liq = [CP.PropsSI('H', 'T', cu.C2K(T), 'Q', 0, refrigerant) / 1000 for T in temps]
    h_vap = [CP.PropsSI('H', 'T', cu.C2K(T), 'Q', 1, refrigerant) / 1000 for T in temps]
    p_sat = [CP.PropsSI('P', 'T', cu.C2K(T), 'Q', 0, refrigerant) / 1000 for T in temps]

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

    if np.isnan(h1_star) and not np.isnan(h1): h1_star, P1_star = h1, P1
    if np.isnan(h3_star) and not np.isnan(h3): h3_star, P3_star = h3, P3

    ax.plot(h_liq, p_sat, color=color1, label='Saturated liquid', linewidth=dm.lw(0))
    ax.plot(h_vap, p_sat, color=color2, label='Saturated vapor', linewidth=dm.lw(0))
    
    cycle_markerfacecolor = color3 if result.get('hp_is_on', result.get('is_on', False)) else color4
    cycle_markeredgecolor = color3 if result.get('hp_is_on', result.get('is_on', False)) else color4

    def points_are_close(x1, y1, x2, y2, tol_atol=0.1):
        if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2): return False
        return (np.isclose(x1, x2, atol=tol_atol) and np.isclose(y1, y2, atol=tol_atol))

    if not (np.isnan(h4) or np.isnan(h1_star) or np.isnan(P4) or np.isnan(P1_star)):
        ax.plot([h4, h1_star], [P4, P1_star], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    if not (np.isnan(h1_star) or np.isnan(h1) or np.isnan(P1_star) or np.isnan(P1)):
        if points_are_close(h1_star, P1_star, h1, P1): ax.plot(h1, P1, marker='o', linestyle='None', color=line_color, markersize=2.5, zorder=1)
        else: ax.plot([h1_star, h1], [P1_star, P1], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    if not (np.isnan(h1) or np.isnan(h2) or np.isnan(P1) or np.isnan(P2)):
        ax.plot([h1, h2], [P1, P2], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    if not (np.isnan(h2) or np.isnan(h2_star) or np.isnan(P2) or np.isnan(P2_star)):
        if points_are_close(h2, P2, h2_star, P2_star): ax.plot(h2, P2, marker='o', linestyle='None', color=line_color, markersize=2.5, zorder=1)
        else: ax.plot([h2, h2_star], [P2, P2_star], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    if not (np.isnan(h2_star) or np.isnan(h3_star) or np.isnan(P2_star) or np.isnan(P3_star)):
        ax.plot([h2_star, h3_star], [P2_star, P3_star], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    if not (np.isnan(h3_star) or np.isnan(h3) or np.isnan(P3_star) or np.isnan(P3)):
        if points_are_close(h3_star, P3_star, h3, P3): ax.plot(h3, P3, marker='o', linestyle='None', color=line_color, markersize=2.5, zorder=1)
        else: ax.plot([h3_star, h3], [P3_star, P3], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    if not (np.isnan(h3) or np.isnan(h4) or np.isnan(P3) or np.isnan(P4)):
        ax.plot([h3, h4], [P3, P4], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)

    points = []
    if points_are_close(h1_star, P1_star, h1, P1): points.append((h1, P1, '1=1\''))
    else:
        if not (np.isnan(h1_star) or np.isnan(P1_star)): points.append((h1_star, P1_star, '1\''))
        if not (np.isnan(h1) or np.isnan(P1)): points.append((h1, P1, '1'))
    
    if points_are_close(h2, P2, h2_star, P2_star): points.append((h2, P2, '2=2\''))
    else:
        if not (np.isnan(h2) or np.isnan(P2)): points.append((h2, P2, '2'))
        if not (np.isnan(h2_star) or np.isnan(P2_star)): points.append((h2_star, P2_star, '2\''))
    
    if points_are_close(h3_star, P3_star, h3, P3): points.append((h3, P3, '3=3\''))
    else:
        if not (np.isnan(h3_star) or np.isnan(P3_star)): points.append((h3_star, P3_star, '3\''))
        if not (np.isnan(h3) or np.isnan(P3)): points.append((h3, P3, '3'))
    
    if not (np.isnan(h4) or np.isnan(P4)): points.append((h4, P4, '4'))
    
    for h_val, p_val, label in points:
        ax.plot(h_val, p_val, marker='o', markersize=2.5, markerfacecolor=cycle_markerfacecolor, markeredgecolor=cycle_markeredgecolor, markeredgewidth=0, zorder=2)
        ax.annotate(label, (h_val, p_val), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

    ax.set_xlabel('Enthalpy [kJ/kg]')
    ax.set_ylabel('Pressure [kPa]')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_yscale('log')
    
    legend_handles = [
        ax.plot([], [], color=color1, linewidth=dm.lw(0), label='Saturated liquid')[0],
        ax.plot([], [], color=color2, linewidth=dm.lw(0), label='Saturated vapor')[0],
        ax.plot([], [], color=line_color, linewidth=dm.lw(0), marker='o', linestyle=':', markersize=2.5, markerfacecolor=color3, markeredgecolor=color3, label='Refrigerant cycle', markeredgewidth=0)[0]
    ]
    ax.legend(handles=legend_handles, loc='upper right', handlelength=1.5, labelspacing=0.5, columnspacing=2, ncol=1, frameon=False)


def plot_ts_diagram(
    ax: plt.Axes,
    result: dict[str, float],
    refrigerant: str,
    T_tank: float | None = None,
    T0: float | None = None,
) -> None:
    """Plot T-s diagram on given axis with super heating/cooling considered."""
    color1, color2, color3, color4, line_color = 'oc.blue5', 'oc.red5', 'black', 'oc.gray6', 'oc.gray5'
    xmin, xmax, ymin, ymax = 0, 2.0, -40, 120

    T_critical = cu.K2C(CP.PropsSI('Tcrit', refrigerant))
    temps = np.linspace(cu.K2C(CP.PropsSI('Tmin', refrigerant)) + 1, T_critical, 600)
    s_liq = [CP.PropsSI('S', 'T', cu.C2K(T), 'Q', 0, refrigerant) / 1000 for T in temps]
    s_vap = [CP.PropsSI('S', 'T', cu.C2K(T), 'Q', 1, refrigerant) / 1000 for T in temps]

    ax.plot(s_liq, temps, color=color1, label='Saturated liquid', linewidth=dm.lw(0))
    ax.plot(s_vap, temps, color=color2, label='Saturated vapor', linewidth=dm.lw(0))

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

    if np.isnan(s1_star) and not np.isnan(s1): s1_star, T1_star = s1, T1
    if np.isnan(s3_star) and not np.isnan(s3): s3_star, T3_star = s3, T3

    cycle_markerfacecolor = color3 if result.get('hp_is_on', result.get('is_on', False)) else color4
    cycle_markeredgecolor = color3 if result.get('hp_is_on', result.get('is_on', False)) else color4

    def points_are_close(x1, y1, x2, y2, tol_atol=0.1):
        if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2): return False
        return (np.isclose(x1, x2, atol=tol_atol) and np.isclose(y1, y2, atol=tol_atol))

    if not (np.isnan(s4) or np.isnan(s1_star) or np.isnan(T4) or np.isnan(T1_star)):
        ax.plot([s4, s1_star], [T4, T1_star], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    if not (np.isnan(s1_star) or np.isnan(s1) or np.isnan(T1_star) or np.isnan(T1)):
        if points_are_close(s1_star, T1_star, s1, T1): ax.plot(s1, T1, marker='o', linestyle='None', color=line_color, markersize=2.5, zorder=1)
        else: ax.plot([s1_star, s1], [T1_star, T1], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    if not (np.isnan(s1) or np.isnan(s2) or np.isnan(T1) or np.isnan(T2)):
        ax.plot([s1, s2], [T1, T2], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    if not (np.isnan(s2) or np.isnan(s2_star) or np.isnan(T2) or np.isnan(T2_star)):
        if points_are_close(s2, T2, s2_star, T2_star): ax.plot(s2, T2, marker='o', linestyle='None', color=line_color, markersize=2.5, zorder=1)
        else: ax.plot([s2, s2_star], [T2, T2_star], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    if not (np.isnan(s2_star) or np.isnan(s3_star) or np.isnan(T2_star) or np.isnan(T3_star)):
        ax.plot([s2_star, s3_star], [T2_star, T3_star], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    if not (np.isnan(s3_star) or np.isnan(s3) or np.isnan(T3_star) or np.isnan(T3)):
        if points_are_close(s3_star, T3_star, s3, T3): ax.plot(s3, T3, marker='o', linestyle='None', color=line_color, markersize=2.5, zorder=1)
        else: ax.plot([s3_star, s3], [T3_star, T3], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)
    if not (np.isnan(s3) or np.isnan(s4) or np.isnan(T3) or np.isnan(T4)):
        ax.plot([s3, s4], [T3, T4], color=line_color, linewidth=dm.lw(0), linestyle=':', zorder=1)

    points = []
    if points_are_close(s1_star, T1_star, s1, T1): points.append((s1, T1, '1=1\''))
    else:
        if not (np.isnan(s1_star) or np.isnan(T1_star)): points.append((s1_star, T1_star, '1\''))
        if not (np.isnan(s1) or np.isnan(T1)): points.append((s1, T1, '1'))
    
    if points_are_close(s2, T2, s2_star, T2_star): points.append((s2, T2, '2=2\''))
    else:
        if not (np.isnan(s2) or np.isnan(T2)): points.append((s2, T2, '2'))
        if not (np.isnan(s2_star) or np.isnan(T2_star)): points.append((s2_star, T2_star, '2\''))
    
    if points_are_close(s3_star, T3_star, s3, T3): points.append((s3, T3, '3=3\''))
    else:
        if not (np.isnan(s3_star) or np.isnan(T3_star)): points.append((s3_star, T3_star, '3\''))
        if not (np.isnan(s3) or np.isnan(T3)): points.append((s3, T3, '3'))
    
    if not (np.isnan(s4) or np.isnan(T4)): points.append((s4, T4, '4'))
    
    for s_val, T_val, label in points:
        ax.plot(s_val, T_val, marker='o', markersize=2.5, markerfacecolor=cycle_markerfacecolor, markeredgecolor=cycle_markeredgecolor, markeredgewidth=0, zorder=2)
        ax.annotate(label, (s_val, T_val), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

    if T_tank is not None and not np.isnan(T_tank):
        ax.axhline(y=T_tank, color='oc.red5', linestyle=':', linewidth=dm.lw(0))
        ax.text(xmin + 0.05, T_tank + 2, f'Tank: {T_tank:.1f}°C', color='oc.red5', ha='left', va='bottom')
    if T0 is not None and not np.isnan(T0):
        ax.axhline(y=T0, color='oc.orange5', linestyle=':', linewidth=dm.lw(0))
        ax.text(xmin + 0.05, T0 - 2, f'Outdoor: {T0:.1f}°C', color='oc.orange5', ha='left', va='top')

    ax.set_xlabel('Entropy [kJ/(kg·K)]')
    ax.set_ylabel('Temperature [°C]')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    legend_handles = [
        ax.plot([], [], color=color1, linewidth=dm.lw(0), label='Saturated liquid')[0],
        ax.plot([], [], color=color2, linewidth=dm.lw(0), label='Saturated vapor')[0],
        ax.plot([], [], color=line_color, linewidth=dm.lw(0), marker='o', linestyle=':', markersize=2.5, markerfacecolor=color3, markeredgecolor=color3, label='Refrigerant cycle', markeredgewidth=0)[0]
    ]
    ax.legend(handles=legend_handles, loc='upper right', handlelength=1.5, labelspacing=0.5, columnspacing=2, ncol=1, frameon=False)
