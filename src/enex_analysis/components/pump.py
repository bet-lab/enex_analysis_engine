from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

try:
    import dartwork_mpl as dm
except ImportError:
    dm = None

from .. import calc_util as cu
from ..enex_functions import cubic_function


@dataclass
class Pump:
    """Pump performance dataclass with curve-fit interpolation.

    Stores flow-rate vs efficiency data for two reference pumps.
    Provides cubic curve-fit prediction of efficiency and power draw.
    """

    def __post_init__(self):
        """Store flow-rate and efficiency data for two reference pumps."""
        self.pump1 = {
            "flow rate": np.array([2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6])
            / cu.h2s,  # m3/s
            "efficiency": [
                0.255,
                0.27,
                0.3,
                0.33,
                0.34,
                0.33,
                0.32,
                0.3,
                0.26,
            ],  # [-]
        }
        self.pump2 = {
            "flow rate": np.array(
                [1.8, 2.2, 2.8, 3.3, 3.8, 4.3, 4.8, 5.3, 5.8]
            )
            / cu.h2s,  # m3/s
            "efficiency": [
                0.23,
                0.26,
                0.29,
                0.32,
                0.35,
                0.34,
                0.33,
                0.31,
                0.28,
            ],  # [-]
        }
        self.pump_list = [self.pump1, self.pump2]

    def get_efficiency(self, pump, dV_pmp):
        """Return pump efficiency via cubic curve fit.

        Parameters
        ----------
        pump : dict
            Reference pump data (``self.pump1`` or ``self.pump2``).
        dV_pmp : float
            Volume flow rate [m³/s].
        """
        self.efficiency_coeffs, _ = curve_fit(
            cubic_function, pump["flow rate"], pump["efficiency"]
        )
        eff = cubic_function(dV_pmp, *self.efficiency_coeffs)
        return eff

    def get_power(self, pump, V_pmp, dP_pmp):
        """Compute pump power consumption.

        Parameters
        ----------
        pump : dict
            Reference pump data.
        V_pmp : float
            Volume flow rate [m³/s].
        dP_pmp : float
            Pressure rise [Pa].

        Returns
        -------
        float
            Electrical power draw [W].
        """
        efficiency = self.get_efficiency(pump, V_pmp)
        power = (V_pmp * dP_pmp) / efficiency
        return power

    def show_graph(self):
        """Plot flow-rate vs efficiency curves for all pumps.

        Raw datapoints are shown as dots; cubic curve fits as lines.
        """
        fig, ax = plt.subplots(figsize=(dm.cm2in(10), dm.cm2in(5)))

        # 그래프 색상 설정
        scatter_colors = ["dm.red3", "dm.blue3", "dm.green3", "dm.orange3"]
        plot_colors = ["dm.red6", "dm.blue6", "dm.green6", "dm.orange6"]

        for i, pump in enumerate(self.pump_list):
            # 원본 데이터 (dot 형태)
            ax.scatter(
                pump["flow rate"] * cu.h2s,
                pump["efficiency"],
                label=f"Pump {i + 1} Data",
                color=scatter_colors[i],
                s=2,
            )

            # 곡선 피팅 수행
            coeffs, _ = curve_fit(
                cubic_function, pump["flow rate"] * cu.h2s, pump["efficiency"]
            )
            flow_range = (
                np.linspace(
                    min(pump["flow rate"]), max(pump["flow rate"]), 100
                )
                * cu.h2s
            )
            fitted_values = cubic_function(flow_range, *coeffs)

            # 피팅된 곡선 (line 형태)
            a, b, c, d = coeffs
            ax.plot(
                flow_range,
                fitted_values,
                label=f"Pump {i + 1} Fit",
                color=plot_colors[i],
                linestyle="-",
            )
            print(f"fan {i + 1}: {a:.4f}x³ + {b:.4f}x² + {c:.4f}x + {d:.4f}")

        ax.set_xlabel("Flow Rate [m$^3$/h]", fontsize=dm.fs(0.5))
        ax.set_ylabel("Efficiency [-]", fontsize=dm.fs(0.5))
        ax.legend()

        dm.simple_layout(
            fig,
            margins=(0.05, 0.05, 0.05, 0.05),
            bbox=(0, 1, 0, 1),
            verbose=False,
        )
        dm.save_and_show(fig)
