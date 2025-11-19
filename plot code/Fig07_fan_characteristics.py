#%%
# import libraries
import enex_analysis as enex
import matplotlib.pyplot as plt
import dartwork_mpl as dm
import numpy as np
from enex_analysis.plot_style import fs, pad
dm.use_style()

fan_model = enex.Fan()

# 유량 범위
int_flow_range = np.linspace(1.0, 3.0, 100)
ext_flow_range = np.linspace(1.5, 3.5, 100)

# Common settings
line_width = 1.0

# Color
colors = ['dm.soft blue', 'dm.cool green', 'dm.red5']

fig, axes = plt.subplots(2, 1, figsize=(dm.cm2in(9), dm.cm2in(12)))
plt.subplots_adjust(left=0.12, right=0.7, top=0.94, bottom=0.1, wspace=0.9, hspace=0.5)

fan_labels = ['Indoor unit', 'Outdoor unit']
flow_ranges = [int_flow_range, ext_flow_range]
ylim_pressure = [(130, 210), (250, 370)]
yticks_pressure = [np.linspace(130, 210, 5), np.linspace(250, 370, 5)]
ylim_efficiency = [(50, 70), (50, 70)]
yticks_efficiency = [np.linspace(50, 70, 5), np.linspace(50, 70, 5)]
ylim_power = [(0.2, 1.0), (0.6, 2.2)]
yticks_power = [np.linspace(0.2, 1.0, 5), np.linspace(0.6, 2.2, 5)]
xlims = [(1.0, 3.0), (1.5, 3.5)]
xticks = [np.arange(1.0, 3.1, 0.5), np.arange(1.5, 3.6, 0.5)]

for ax, fan, label, flow_range, ylim_p, yticks_p, ylim_e, yticks_e, ylim_pow, yticks_pow, xlim, xtick in zip(
    axes, fan_model.fan_list, fan_labels, flow_ranges, ylim_pressure, yticks_pressure, ylim_efficiency, yticks_efficiency, ylim_power, yticks_power, xlims, xticks
):
    pressure = fan_model.get_pressure(fan, flow_range)
    efficiency = fan_model.get_efficiency(fan, flow_range)
    power = fan_model.get_power(fan, flow_range)

    # Efficiency - Left y-axis
    ax.set_xlabel('Air flow rate [m$^3$/s]', fontsize=fs['label'], labelpad=pad['label'])
    ax.set_ylabel('Efficiency [%]', color=colors[1], fontsize=fs['label'], labelpad=pad['label'])
    eff_line, = ax.plot(flow_range, efficiency * 100, color=colors[1], linestyle='-', label='Efficiency', linewidth=line_width)
    ax.tick_params(axis='x', labelsize=fs['tick'], pad=pad['tick'])
    ax.tick_params(axis='y', labelsize=fs['tick'], colors=colors[1], pad=pad['tick'])
    ax.set_xlim(xlim)
    ax.set_xticks(xtick)
    ax.set_ylim(ylim_e)
    ax.set_yticks(yticks_e)
    ax.spines['left'].set_color(colors[1])

    # Power - First right y-axis
    ax2 = ax.twinx()
    ax2.set_ylabel('Power [kW]', color=colors[2], fontsize=fs['label'], labelpad=pad['label'])
    power_line, = ax2.plot(flow_range, power*enex.W2kW, color=colors[2], linestyle=':', label='Power', linewidth=line_width)
    ax2.tick_params(axis='y', labelsize=fs['tick'], colors=colors[2], pad=pad['tick'])
    ax2.set_ylim(ylim_pow)
    ax2.set_yticks(yticks_pow)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_color(colors[2])

    # Pressure drop - Second right y-axis
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('axes', 1.25))
    ax3.set_ylabel('Pressure drop [Pa]', color=colors[0], fontsize=fs['label'], labelpad=pad['label'])
    pressure_line, = ax3.plot(flow_range, pressure, color=colors[0], linestyle='--', label='Pressure drop', linewidth=line_width)
    ax3.tick_params(axis='y', labelsize=fs['tick'], colors=colors[0], pad=pad['tick'])
    ax3.set_ylim(ylim_p)
    ax3.set_yticks(yticks_p)
    ax3.spines['top'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.spines['right'].set_color(colors[0])

    ax.grid(True)

    # remove minor ticks
    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.tick_params(axis='y', which='minor', left=False)
    ax.tick_params(axis='y', which='minor', right=False)
    ax2.tick_params(axis='y', which='minor', right=False)
    ax3.tick_params(axis='y', which='minor', right=False)

    # Add titles to each subplot
    ax.text(0.01, 1.13, f'({chr(97 + axes.tolist().index(ax))}) {label} fan',
            transform=ax.transAxes, fontsize=fs['subtitle'], va='top', ha='left')

    # Add legend
    lines = [eff_line, power_line, pressure_line]
    labels_ = [line.get_label() for line in lines]
    ax.legend([pressure_line, eff_line, power_line], 
              ['Pressure drop', 'Efficiency', 'Power'], 
              loc='upper left', fontsize=fs['legend'])

# Save and show
plt.savefig('../figure/Fig. 7.png', dpi=600)
plt.savefig('../figure/Fig. 7.pdf', dpi=600)
dm.util.save_and_show(fig)
# %%
