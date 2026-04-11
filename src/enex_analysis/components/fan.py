from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

try:
    import dartwork_mpl as dm
except ImportError:
    dm = None

from .. import calc_util as cu
from ..enex_functions import cubic_function, quartic_function


@dataclass
class Fan:
    def __post_init__(self):
        # Fan reference: https://www.krugerfan.com/public/uploads/KATCAT006.pdf
        self.fan1 = {
            "flow rate": [
                0.6,
                0.8,
                1.0,
                1.2,
                1.4,
                1.6,
                1.8,
                2.0,
                2.5,
                3.0,
            ],  # [m3/s]
            "pressure": [
                140,
                136,
                137,
                147,
                163,
                178,
                182,
                190,
                198,
                181,
            ],  # [Pa]
            "efficiency": [
                0.43,
                0.48,
                0.52,
                0.55,
                0.60,
                0.65,
                0.68,
                0.66,
                0.63,
                0.52,
            ],  # [-]
            "fan type": "centrifugal",
        }
        # self.fan2 = {
        #     'flow rate'  : [0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0], # [m3/s]
        #     'pressure'   : [137, 138, 143, 168, 182, 191, 198, 200, 201, 170, 160], # [Pa]
        #     'efficiency' : [0.45, 0.49, 0.57, 0.62, 0.67, 0.69, 0.68, 0.67, 0.63, 0.40, 0.48], # [-]
        #     'fan type' : 'centrifugal',
        # }
        self.fan2 = {
            "flow rate": [
                0.8,
                1.0,
                1.2,
                1.4,
                1.6,
                1.8,
                2.0,
                2.5,
                3.0,
                3.5,
                4.0,
                5.0,
            ],  # [m3/s]
            "pressure": [
                244,
                241,
                239,
                242,
                260,
                290,
                305,
                340,
                345,
                350,
                320,
                230,
            ],  # [Pa]
            "efficiency": [
                0.44,
                0.47,
                0.50,
                0.52,
                0.56,
                0.58,
                0.63,
                0.67,
                0.65,
                0.60,
                0.55,
                0.31,
            ],  # [-]
            "fan type": "centrifugal",
        }

        self.fan3 = {  # https://ventilatorry.ru/downloads/ebmpapst/datasheet/w3g710-go81-01-en-datasheet-ebmpapst.pdf
            "flow rate": [
                0 / cu.h2s,
                6245 / cu.h2s,
                8330 / cu.h2s,
                10410 / cu.h2s,
                12610 / cu.h2s,
            ],  # [m3/s]
            "power": [0, 100, 238, 465, 827],  # [-]
            "fan type": "axial",
        }
        self.fan_list = [self.fan1, self.fan2, self.fan3]

    def get_efficiency(self, fan, dV_fan):
        if "efficiency" not in fan:
            raise ValueError("Selected fan does not have efficiency data.")
        self.efficiency_coeffs, _ = curve_fit(cubic_function, fan["flow rate"], fan["efficiency"])
        eff = cubic_function(dV_fan, *self.efficiency_coeffs)
        return eff

    def get_pressure(self, fan, dV_fan):
        if "pressure" not in fan:
            raise ValueError("Selected fan does not have pressure data.")
        self.pressure_coeffs, _ = curve_fit(cubic_function, fan["flow rate"], fan["pressure"])
        pressure = cubic_function(dV_fan, *self.pressure_coeffs)
        return pressure

    def get_power(self, fan, dV_fan):
        if "efficiency" in fan and "pressure" in fan:
            eff = self.get_efficiency(fan, dV_fan)
            pressure = self.get_pressure(fan, dV_fan)
            power = pressure * dV_fan / eff
        elif "power" in fan:
            self.power_coeffs, _ = curve_fit(quartic_function, fan["flow rate"], fan["power"])
            power = quartic_function(dV_fan, *self.power_coeffs)
        else:
            raise ValueError("Fan must have either ('efficiency' + 'pressure') or 'power' data to compute power.")
        return power

    def show_graph(self):
        """Plot flow-rate vs pressure and efficiency curves for all fans.

        Raw datapoints are shown as dots; cubic curve fits as lines.

        Raises
        ------
        ImportError
            If ``dartwork_mpl`` is not installed.
        """
        if dm is None:
            raise ImportError("dartwork_mpl is required for show_graph(). Install it with: pip install dartwork-mpl")
        fig, axes = plt.subplots(1, 2, figsize=(dm.cm2in(15), dm.cm2in(5)))

        # 그래프 색상 설정
        scatter_colors = ["dm.red3", "dm.blue3", "dm.green3", "dm.orange3"]
        plot_colors = ["dm.red6", "dm.blue6", "dm.green6", "dm.orange6"]

        data_pairs = [
            ("pressure", "Pressure [Pa]", "Flow Rate vs Pressure"),
            ("efficiency", "Efficiency [-]", "Flow Rate vs Efficiency"),
        ]

        for ax, (key, ylabel, title) in zip(axes, data_pairs, strict=False):
            print(f"\n{'=' * 10} {title} {'=' * 10}")
            for i, fan in enumerate(self.fan_list):
                # 원본 데이터 (dot 형태)
                ax.scatter(
                    fan["flow rate"],
                    fan[key],
                    label=f"Fan {i + 1} Data",
                    color=scatter_colors[i],
                    s=2,
                )

                # 곡선 피팅 수행
                coeffs, _ = curve_fit(cubic_function, fan["flow rate"], fan[key])
                flow_range = np.linspace(min(fan["flow rate"]), max(fan["flow rate"]), 100)
                fitted_values = cubic_function(flow_range, *coeffs)

                # 피팅된 곡선 (line 형태)
                ax.plot(
                    flow_range,
                    fitted_values,
                    label=f"Fan {i + 1} Fit",
                    color=plot_colors[i],
                    linestyle="-",
                )
                a, b, c, d = coeffs
                print(f"fan {i + 1}: {a:.4f}x³ + {b:.4f}x² + {c:.4f}x + {d:.4f}")

            ax.set_xlabel("Flow Rate [m$^3$/s]", fontsize=dm.fs(0.5))
            ax.set_ylabel(ylabel, fontsize=dm.fs(0.5))
            ax.set_title(title, fontsize=dm.fs(0.5))
            ax.legend()

        plt.subplots_adjust(wspace=0.3)
        dm.simple_layout(
            fig,
            margins=(0.05, 0.05, 0.05, 0.05),
            bbox=(0, 1, 0, 1),
            verbose=False,
        )
        dm.save_and_show(fig)
