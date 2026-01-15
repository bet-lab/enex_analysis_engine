"""
Visualization and LaTeX export utilities for DHW system analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional

# Try to use seaborn style, fallback to default if not available
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')


def export_latex_table(
    df: pd.DataFrame,
    filename: str,
    caption: str = "",
    label: str = "",
    float_format: str = "{:.3f}"
) -> None:
    """
    Export DataFrame to LaTeX table format.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to export
    filename : str
        Output filename (should end with .tex)
    caption : str
        Table caption
    label : str
        LaTeX label for referencing
    float_format : str
        Format string for floating point numbers
    """
    latex_str = df.to_latex(
        index=False,
        float_format=lambda x: float_format.format(x) if pd.notna(x) else "",
        caption=caption,
        label=label,
        escape=False
    )
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(latex_str)
    
    print(f"LaTeX table exported to {filename}")


def plot_grid_mix_comparison(
    df: pd.DataFrame,
    hpb_primary_energy: float,
    gb_primary_energy: float,
    hpb_co2: float,
    gb_co2: float,
    save_path: Optional[str] = None
) -> None:
    """
    Plot comparison of HPB vs GB for different grid mixes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame from sweep_renewable_fraction()
    hpb_primary_energy : float
        HPB primary energy consumption (without grid factor) [kWh]
    gb_primary_energy : float
        GB primary energy consumption [kWh]
    hpb_co2 : float
        HPB CO2 emissions (without grid factor) [kg CO2]
    gb_co2 : float
        GB CO2 emissions [kg CO2]
    save_path : str, optional
        Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Calculate total values
    hpb_pe_total = hpb_primary_energy * df['pef'].values
    hpb_co2_total = hpb_co2 * df['co2_factor'].values
    
    # Plot primary energy comparison
    ax1.plot(df['renewable_fraction'] * 100, hpb_pe_total, 
             'b-', label='HPB', linewidth=2)
    ax1.axhline(y=gb_primary_energy, color='r', linestyle='--', 
                label='GB', linewidth=2)
    ax1.set_xlabel('Renewable Energy Fraction [%]', fontsize=12)
    ax1.set_ylabel('Primary Energy Consumption [kWh]', fontsize=12)
    ax1.set_title('Primary Energy Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot CO2 comparison
    ax2.plot(df['renewable_fraction'] * 100, hpb_co2_total,
             'b-', label='HPB', linewidth=2)
    ax2.axhline(y=gb_co2, color='r', linestyle='--',
                label='GB', linewidth=2)
    ax2.set_xlabel('Renewable Energy Fraction [%]', fontsize=12)
    ax2.set_ylabel('CO2 Emissions [kg CO2]', fontsize=12)
    ax2.set_title('CO2 Emissions Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_well_to_tap_sankey(
    wtt_results: Dict[str, Dict[str, float]],
    system_name: str,
    save_path: Optional[str] = None
) -> None:
    """
    Create Sankey-like diagram for well-to-tap analysis.
    
    Parameters:
    -----------
    wtt_results : dict
        Well-to-tap results from WellToTapAnalyzer
    system_name : str
        System name ('EB', 'GB', or 'HPB')
    save_path : str, optional
        Path to save figure
    """
    results = wtt_results[system_name]
    breakdown = results['stage_breakdown']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for Sankey diagram
    stages = []
    values = []
    colors = []
    
    if system_name in ['EB', 'HPB']:
        # Electric system stages
        stages = ['Primary\nEnergy', 'Generation\nOutput', 'Grid\nOutput', 'Building\nDemand']
        gen_output = breakdown['generation']['output']
        grid_output = breakdown['grid']['output']
        building_demand = breakdown['grid']['output']
        
        values = [
            results['primary_energy_input'],
            gen_output,
            grid_output,
            building_demand
        ]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    else:
        # Gas system stages
        stages = ['Extraction', 'Transport', 'Distribution', 'Building\nDemand']
        extraction_output = breakdown['extraction']['output']
        transport_output = breakdown['transport']['output']
        distribution_output = breakdown['distribution']['output']
        
        values = [
            results['primary_energy_input'],
            extraction_output,
            transport_output,
            distribution_output
        ]
        colors = ['#FF6B6B', '#FFA07A', '#FFD700', '#98D8C8']
    
    # Create horizontal bar chart
    y_pos = np.arange(len(stages))
    width = 0.6
    
    bars = ax.barh(y_pos, values, width, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + max(values) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.2f} kWh', va='center', fontsize=10, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stages, fontsize=11)
    ax.set_xlabel('Energy Flow [kWh]', fontsize=12)
    ax.set_title(f'Well-to-Tap Energy Flow: {system_name}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_exergy_destruction_comparison(
    wtt_results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> None:
    """
    Plot exergy destruction comparison across systems and stages.
    
    Parameters:
    -----------
    wtt_results : dict
        Well-to-tap results for all systems
    save_path : str, optional
        Path to save figure
    """
    systems = ['EB', 'GB', 'HPB']
    stages = ['Generation', 'Grid', 'Total']
    
    # Prepare data
    data = []
    for system in systems:
        if system in wtt_results:
            results = wtt_results[system]
            if system in ['EB', 'HPB']:
                data.append([
                    results.get('exergy_destroyed_generation', 0),
                    results.get('exergy_destroyed_grid', 0),
                    results.get('exergy_destroyed_total', 0)
                ])
            else:
                data.append([
                    0,
                    results.get('exergy_destroyed_supply', 0),
                    results.get('exergy_destroyed_total', 0)
                ])
        else:
            data.append([0, 0, 0])
    
    data = np.array(data)
    
    # Create grouped bar chart
    x = np.arange(len(systems))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, stage in enumerate(stages):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, data[:, i], width, label=stage)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('System Type', fontsize=12)
    ax.set_ylabel('Exergy Destroyed [kWh]', fontsize=12)
    ax.set_title('Exergy Destruction Comparison by Stage', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()

