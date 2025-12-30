"""
Energy, Entropy, and Exergy Analysis Engine.

This module contains classes for modeling various energy systems including:
- Domestic hot water systems (electric boiler, gas boiler, heat pump boiler)
- Air source heat pumps (cooling and heating modes)
- Ground source heat pumps (cooling and heating modes)
- Solar-assisted systems
- Electric heaters
"""

import numpy as np
import math
from dataclasses import dataclass
import dartwork_mpl as dm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from . import calc_util as cu
from .constants import (
    c_a, rho_a, k_a, c_w, rho_w, mu_w, k_w, sigma, k_D, k_d, ex_eff_NG, SP
)

from .enex_functions import (
    darcy_friction_factor,
    calc_h_vertical_plate,
    linear_function,
    quadratic_function,
    cubic_function,
    quartic_function,
    print_balance,
    calculate_ASHP_cooling_COP,
    calculate_ASHP_heating_COP,
    calculate_GSHP_COP,
    G_FLS
)


#%%
# class - Fan & Pump
@dataclass
class Fan:
    def __post_init__(self): 
        # Fan reference: https://www.krugerfan.com/public/uploads/KATCAT006.pdf
        self.fan1 = {
            'flow rate'  : [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0], # [m3/s]
            'pressure'   : [140, 136, 137, 147, 163, 178, 182, 190, 198, 181], # [Pa]
            'efficiency' : [0.43, 0.48, 0.52, 0.55, 0.60, 0.65, 0.68, 0.66, 0.63, 0.52], # [-]
            'fan type' : 'centrifugal',
        }
        # self.fan2 = {
        #     'flow rate'  : [0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0], # [m3/s]
        #     'pressure'   : [137, 138, 143, 168, 182, 191, 198, 200, 201, 170, 160], # [Pa]
        #     'efficiency' : [0.45, 0.49, 0.57, 0.62, 0.67, 0.69, 0.68, 0.67, 0.63, 0.40, 0.48], # [-]
        #     'fan type' : 'centrifugal',
        # }
        self.fan2 = {
            'flow rate'  : [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0], # [m3/s]
            'pressure'   : [244, 241, 239, 242, 260, 290, 305, 340, 345, 350, 320, 230], # [Pa]
            'efficiency' : [0.44, 0.47, 0.50, 0.52, 0.56, 0.58, 0.63, 0.67, 0.65, 0.60, 0.55, 0.31], # [-]
            'fan type' : 'centrifugal',
        }

        self.fan3 = { # https://ventilatorry.ru/downloads/ebmpapst/datasheet/w3g710-go81-01-en-datasheet-ebmpapst.pdf
            'flow rate' : [0/cu.h2s, 6245/cu.h2s, 8330/cu.h2s, 10410/cu.h2s, 12610/cu.h2s], # [m3/s]
            'power' : [0, 100, 238, 465, 827], # [-]
            'fan type' : 'axial',
        }
        self.fan_list = [self.fan1, self.fan2, self.fan3]

    def get_efficiency(self, fan, dV_fan):
        if 'efficiency' not in fan:
            raise ValueError("Selected fan does not have efficiency data.")
        self.efficiency_coeffs, _ = curve_fit(cubic_function, fan['flow rate'], fan['efficiency'])
        eff = cubic_function(dV_fan, *self.efficiency_coeffs)
        return eff
    
    def get_pressure(self, fan, dV_fan):
        if 'pressure' not in fan:
            raise ValueError("Selected fan does not have pressure data.")
        self.pressure_coeffs, _ = curve_fit(cubic_function, fan['flow rate'], fan['pressure'])
        pressure = cubic_function(dV_fan, *self.pressure_coeffs)
        return pressure
    
    def get_power(self, fan, dV_fan):
        if 'efficiency' in fan and 'pressure' in fan:
            eff = self.get_efficiency(fan, dV_fan)
            pressure = self.get_pressure(fan, dV_fan)
            power = pressure * dV_fan / eff
        elif 'power' in fan:
            self.power_coeffs, _ = curve_fit(quartic_function, fan['flow rate'], fan['power'])
            power = quartic_function(dV_fan, *self.power_coeffs)
        return power

    def show_graph(self):
        """
        유량(flow rate) 대비 압력(pressure) 및 효율(efficiency) 그래프를 출력.
        - 원본 데이터는 점(dot)으로 표시.
        - 커브 피팅된 곡선은 선(line)으로 표시.
        """
        fig, axes = plt.subplots(1, 2, figsize=(dm.cm2in(15), dm.cm2in(5)))

        # 그래프 색상 설정
        scatter_colors = ['dm.red3', 'dm.blue3', 'dm.green3', 'dm.orange3']
        plot_colors = ['dm.red6', 'dm.blue6', 'dm.green6', 'dm.orange6']

        data_pairs = [
            ("pressure", "Pressure [Pa]", "Flow Rate vs Pressure"),
            ("efficiency", "Efficiency [-]", "Flow Rate vs Efficiency"),
        ]

        for ax, (key, ylabel, title) in zip(axes, data_pairs):
            print(f"\n{'='*10} {title} {'='*10}")
            for i, fan in enumerate(self.fan_list):
                # 원본 데이터 (dot 형태)
                ax.scatter(fan['flow rate'], fan[key], label=f'Fan {i+1} Data', color=scatter_colors[i], s=2)

                # 곡선 피팅 수행
                coeffs, _ = curve_fit(cubic_function, fan['flow rate'], fan[key])
                flow_range = np.linspace(min(fan['flow rate']), max(fan['flow rate']), 100)
                fitted_values = cubic_function(flow_range, *coeffs)

                # 피팅된 곡선 (line 형태)
                ax.plot(flow_range, fitted_values, label=f'Fan {i+1} Fit', color=plot_colors[i], linestyle='-')
                a,b,c,d = coeffs
                print(f"fan {i+1}: {a:.4f}x³ + {b:.4f}x² + {c:.4f}x + {d:.4f}")

            ax.set_xlabel('Flow Rate [m$^3$/s]', fontsize=dm.fs(0.5))
            ax.set_ylabel(ylabel, fontsize=dm.fs(0.5))
            ax.set_title(title, fontsize=dm.fs(0.5))
            ax.legend()

        plt.subplots_adjust(wspace=0.3)
        dm.simple_layout(fig, margins=(0.05, 0.05, 0.05, 0.05), bbox=(0, 1, 0, 1), verbose=False)
        dm.save_and_show(fig)

@dataclass
class Pump:
    """
    Pump 클래스: 펌프의 성능 데이터를 저장하고 분석하는 클래스.
    
    - 유량(flow rate)과 효율(efficiency) 데이터를 보유.
    - 효율 데이터를 기반으로 곡선 피팅(curve fitting)을 수행하여 예측 값 계산.
    - 주어진 압력 차이(dP_pmp)와 유량(V_pmp)을 이용하여 펌프의 전력 소비량 계산.
    """

    def __post_init__(self):
        """
        클래스 초기화 후 자동 실행되는 메서드.
        두 개의 펌프의 유량 및 효율 데이터를 저장.
        """
        self.pump1 = {
            'flow rate'  : np.array([2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6])/cu.h2s, # m3/s
            'efficiency' : [0.255, 0.27, 0.3, 0.33, 0.34, 0.33, 0.32, 0.3, 0.26], # [-]
        }
        self.pump2 = {
            'flow rate'  : np.array([1.8, 2.2, 2.8, 3.3, 3.8, 4.3, 4.8, 5.3, 5.8])/cu.h2s, # m3/s
            'efficiency' : [0.23, 0.26, 0.29, 0.32, 0.35, 0.34, 0.33, 0.31, 0.28], # [-]
        }
        self.pump_list = [self.pump1, self.pump2]
        
    def get_efficiency(self, pump, dV_pmp):
        """
        주어진 유량(V_pmp)에 대해 3차 곡선 피팅을 통해 펌프 효율을 예측.
        
        :param pump: 선택한 펌프 (self.pump1 또는 self.pump2)
        :param V_pmp: 유량 (m3/h)
        :return: 예측된 펌프 효율
        """
        self.efficiency_coeffs, _ = curve_fit(cubic_function, pump['flow rate'], pump['efficiency'])
        eff = cubic_function(dV_pmp, *self.efficiency_coeffs)
        return eff

    def get_power(self, pump, V_pmp, dP_pmp):
        """
        주어진 유량(V_pmp)과 압력 차이(dP_pmp)를 이용하여 펌프의 전력 소비량을 계산.
        
        :param pump: 선택한 펌프 (self.pump1 또는 self.pump2)
        :param V_pmp: 유량 (m3/h)
        :param dP_pmp: 펌프 압력 차이 (Pa)
        :return: 펌프의 소비 전력 (W)
        """
        efficiency = self.get_efficiency(pump, V_pmp)
        power = (V_pmp * dP_pmp) / efficiency
        return power

    def show_graph(self):
        """
        유량(flow rate) 대비 효율(efficiency) 그래프를 출력.
        - 원본 데이터는 점(dot)으로 표시.
        - 커브 피팅된 곡선은 선(line)으로 표시.
        """
        fig, ax = plt.subplots(figsize=(dm.cm2in(10), dm.cm2in(5)))

        # 그래프 색상 설정
        scatter_colors = ['dm.red3', 'dm.blue3', 'dm.green3', 'dm.orange3']
        plot_colors = ['dm.red6', 'dm.blue6', 'dm.green6', 'dm.orange6']

        for i, pump in enumerate(self.pump_list):
            # 원본 데이터 (dot 형태)
            ax.scatter(pump['flow rate']*cu.h2s, pump['efficiency'], label=f'Pump {i+1} Data', color=scatter_colors[i], s=2)

            # 곡선 피팅 수행
            coeffs, _ = curve_fit(cubic_function, pump['flow rate']*cu.h2s, pump['efficiency'])
            flow_range = np.linspace(min(pump['flow rate']), max(pump['flow rate']), 100)*cu.h2s
            fitted_values = cubic_function(flow_range, *coeffs)

            # 피팅된 곡선 (line 형태)
            a,b,c,d = coeffs
            ax.plot(flow_range, fitted_values, label=f'Pump {i+1} Fit', color=plot_colors[i], linestyle='-')
            print(f"fan {i+1}: {a:.4f}x³ + {b:.4f}x² + {c:.4f}x + {d:.4f}")

        ax.set_xlabel('Flow Rate [m$^3$/h]', fontsize=dm.fs(0.5))
        ax.set_ylabel('Efficiency [-]', fontsize=dm.fs(0.5))
        ax.legend()

        dm.simple_layout(fig, margins=(0.05, 0.05, 0.05, 0.05), bbox=(0, 1, 0, 1), verbose=False)
        dm.save_and_show(fig)

#%%
# class - Domestic Hot Water System
@dataclass
class ElectricBoiler:

    def __post_init__(self):
        
        # Temperature [K]
        self.T_w_tank = 60
        self.T_w_sup  = 10
        self.T_w_serv = 45
        self.T0       = 0

        # Tank water use [L/min]
        self.dV_w_serv  = 1.2

        # Tank size [m]
        self.r0 = 0.2
        self.H = 0.8
        
        # Tank layer thickness [m]
        self.x_shell = 0.01 
        self.x_ins   = 0.10 
        
        # Tank thermal conductivity [W/mK]
        self.k_shell = 25   
        self.k_ins   = 0.03 

        # Overall heat transfer coefficient [W/m²K]
        self.h_o = 15 
        
    def system_update(self):
        # Temperature conversion: Celsius to Kelvin (create new variables with _K suffix)
        T_w_tank_K = cu.C2K(self.T_w_tank)  # Tank water temperature [K]
        T_w_sup_K = cu.C2K(self.T_w_sup)  # Supply water temperature [K]
        T_w_serv_K = cu.C2K(self.T_w_serv)  # Tap water temperature [K]
        T0_K = cu.C2K(self.T0)  # Reference temperature [K]
        
        # Unit conversion: L/min to m³/s
        dV_w_serv = self.dV_w_serv / 60 / 1000
        
        # Temperature [K]
        T_tank_is_K = T_w_tank_K  # Inner surface temperature of tank [K]

        # Geometric calculations
        r1 = self.r0 + self.x_shell
        r2 = r1 + self.x_ins
        
        # Tank surface areas [m²]
        A_side = 2 * math.pi * r2 * self.H
        A_base = math.pi * self.r0**2
        
        # Total tank volume [m³]
        V_tank = A_base * self.H

        # Volumetric flow rate ratio [-]
        alp = (T_w_serv_K - T_w_sup_K) / (T_w_tank_K - T_w_sup_K)
        if alp < 0:
            print("alp is negative")
        
        # Volumetric flow rates [m³/s]
        dV_w_sup_tank = alp * dV_w_serv
        dV_w_sup_mix = (1 - alp) * dV_w_serv

        # Thermal resistances per unit area/length
        R_base_unit = self.x_shell / self.k_shell + self.x_ins / self.k_ins  # [m²K/W]
        R_side_unit = math.log(r1 / self.r0) / (2 * math.pi * self.k_shell) + math.log(r2 / r1) / (2 * math.pi * self.k_ins)  # [mK/W]
        
        # Thermal resistances [K/W]
        R_base = R_base_unit / A_base  # [K/W]
        R_side = R_side_unit / self.H  # [K/W]
        
        # External thermal resistances [K/W]
        R_base_ext = 1 / (self.h_o * A_base)
        R_side_ext = 1 / (self.h_o * A_side)

        # Total thermal resistances [K/W]
        R_base_tot = R_base + R_base_ext
        R_side_tot = R_side + R_side_ext

        # U-value [W/K]
        U_tank = 2 / R_base_tot + 1 / R_side_tot

        # Heat transfer rates
        Q_w_tank = c_w * rho_w * dV_w_sup_tank * (T_w_tank_K - T0_K)
        Q_w_sup = c_w * rho_w * dV_w_sup_tank * (T_w_sup_K - T0_K)
        Q_l_tank = U_tank * (T_tank_is_K - T0_K)
        E_heater = Q_w_tank + Q_l_tank - Q_w_sup  # Electric power input [W]

        # Energy values
        Q_w_sup_tank = c_w * rho_w * dV_w_sup_tank * (T_w_sup_K - T0_K)
        Q_w_tank = c_w * rho_w * dV_w_sup_tank * (T_w_tank_K - T0_K)
        Q_w_sup_mix = c_w * rho_w * dV_w_sup_mix * (T_w_sup_K - T0_K)
        Q_w_serv = c_w * rho_w * dV_w_serv * (T_w_serv_K - T0_K)

        # Entropy values
        S_heater = (1 / float('inf')) * E_heater
        S_w_sup_tank = c_w * rho_w * dV_w_sup_tank * math.log(T_w_sup_K / T0_K)
        S_w_tank = c_w * rho_w * dV_w_sup_tank * math.log(T_w_tank_K / T0_K)
        S_l_tank = (1 / T_tank_is_K) * Q_l_tank
        S_w_sup_mix = c_w * rho_w * dV_w_sup_mix * math.log(T_w_sup_K / T0_K)
        S_w_serv = c_w * rho_w * dV_w_serv * math.log(T_w_serv_K / T0_K)
        S_g_tank = (S_w_tank + S_l_tank) - (S_heater + S_w_sup_tank)
        S_g_mix = S_w_serv - (S_w_tank + S_w_sup_mix)

        # Exergy values for hot water tank
        X_heater = E_heater - S_heater * T0_K
        X_w_sup_tank = c_w * rho_w * dV_w_sup_tank * ((T_w_sup_K - T0_K) - T0_K * math.log(T_w_sup_K / T0_K))
        X_w_tank = c_w * rho_w * dV_w_sup_tank * ((T_w_tank_K - T0_K) - T0_K * math.log(T_w_tank_K / T0_K))
        X_l_tank = (1 - T0_K / T_tank_is_K) * Q_l_tank
        X_c_tank = S_g_tank * T0_K

        # Exergy values for mixing valve
        X_w_sup_mix = c_w * rho_w * dV_w_sup_mix * ((T_w_sup_K - T0_K) - T0_K * math.log(T_w_sup_K / T0_K))
        X_w_serv = c_w * rho_w * dV_w_serv * ((T_w_serv_K - T0_K) - T0_K * math.log(T_w_serv_K / T0_K))
        X_c_mix = S_g_mix * T0_K
        
        # Total values
        X_c_tot = X_c_tank + X_c_mix
        X_eff = X_w_serv / X_heater

        # Assign all variables to self at the end
        self.T_w_tank_K = T_w_tank_K
        self.T_w_sup_K = T_w_sup_K
        self.T_w_serv_K = T_w_serv_K
        self.T0_K = T0_K
        self.dV_w_serv = dV_w_serv
        self.T_tank_is_K = T_tank_is_K
        self.r1 = r1
        self.r2 = r2
        self.A_side = A_side
        self.A_base = A_base
        self.V_tank = V_tank
        self.alp = alp
        self.dV_w_sup_tank = dV_w_sup_tank
        self.dV_w_sup_mix = dV_w_sup_mix
        self.R_base_unit = R_base_unit
        self.R_side_unit = R_side_unit
        self.R_base = R_base
        self.R_side = R_side
        self.R_base_ext = R_base_ext
        self.R_side_ext = R_side_ext
        self.R_base_tot = R_base_tot
        self.R_side_tot = R_side_tot
        self.U_tank = U_tank
        self.Q_w_tank = Q_w_tank
        self.Q_w_sup = Q_w_sup
        self.Q_l_tank = Q_l_tank
        self.E_heater = E_heater
        self.Q_w_sup_tank = Q_w_sup_tank
        self.Q_w_sup_mix = Q_w_sup_mix
        self.Q_w_serv = Q_w_serv
        self.S_heater = S_heater
        self.S_w_sup_tank = S_w_sup_tank
        self.S_w_tank = S_w_tank
        self.S_l_tank = S_l_tank
        self.S_w_sup_mix = S_w_sup_mix
        self.S_w_serv = S_w_serv
        self.S_g_tank = S_g_tank
        self.S_g_mix = S_g_mix
        self.X_heater = X_heater
        self.X_w_sup_tank = X_w_sup_tank
        self.X_w_tank = X_w_tank
        self.X_l_tank = X_l_tank
        self.X_c_tank = X_c_tank
        self.X_w_sup_mix = X_w_sup_mix
        self.X_w_serv = X_w_serv
        self.X_c_mix = X_c_mix
        self.X_c_tot = X_c_tot
        self.X_eff = X_eff

        # Create balance dictionaries
        self.energy_balance = {
            "hot water tank": {
                "in": {
                    "E_heater": self.E_heater,
                    "Q_w_sup_tank": self.Q_w_sup_tank
                },
                "out": {
                    "Q_w_tank": self.Q_w_tank,
                    "Q_l_tank": self.Q_l_tank
                }
            },
            "mixing valve": {
                "in": {
                    "Q_w_tank": self.Q_w_tank,
                    "Q_w_sup_mix": self.Q_w_sup_mix
                },
                "out": {
                    "Q_w_serv": self.Q_w_serv
                }
            }
        }

        self.entropy_balance = {
            "hot water tank": {
                "in": {
                    "S_heater": self.S_heater,
                    "S_w_sup_tank": self.S_w_sup_tank
                },
                "out": {
                    "S_w_tank": self.S_w_tank,
                    "S_l_tank": self.S_l_tank
                },
                "gen": {
                    "S_g_tank": self.S_g_tank
                }
            },
            "mixing valve": {
                "in": {
                    "S_w_tank": self.S_w_tank,
                    "S_w_sup_mix": self.S_w_sup_mix
                },
                "out": {
                    "S_w_serv": self.S_w_serv
                },
                "gen": {
                    "S_g_mix": self.S_g_mix
                }
            }
        }

        self.exergy_balance = {
            "hot water tank": {
                "in": {
                    "E_heater": self.E_heater,
                    "X_w_sup_tank": self.X_w_sup_tank
                },
                "out": {
                    "X_w_tank": self.X_w_tank,
                    "X_l_tank": self.X_l_tank
                },
                "con": {
                    "X_c_tank": self.X_c_tank
                }
            },
            "mixing valve": {
                "in": {
                    "X_w_tank": self.X_w_tank,
                    "X_w_sup_mix": self.X_w_sup_mix
                },
                "out": {
                    "X_w_serv": self.X_w_serv
                },
                "con": {
                    "X_c_mix": self.X_c_mix
                }
            }
        }

@dataclass
class GasBoiler:

    def __post_init__(self):
        
        # Efficiency [-]
        self.eta_comb = 0.9

        # Temperature [°C]
        self.T_w_tank = 60 
        self.T_w_sup  = 10
        self.T_w_serv  = 45 
        self.T0       = 0
        self.T_exh    = 70 

        # Tank water use [L/min]
        self.dV_w_serv  = 1.2

        # Tank size [m]
        self.r0 = 0.2
        self.H = 0.8
        
        # Tank layer thickness [m]
        self.x_shell = 0.01 
        self.x_ins   = 0.10 
        
        # Tank thermal conductivity [W/mK]
        self.k_shell = 25   
        self.k_ins   = 0.03 

        # Overall heat transfer coefficient [W/m²K]
        self.h_o = 15 
        
    def system_update(self):
        
        # Celcius to Kelvin
        self.T_w_tank = cu.C2K(self.T_w_tank) # tank water temperature [K]
        self.T_w_sup  = cu.C2K(self.T_w_sup)  # supply water temperature [K]
        self.T_w_serv  = cu.C2K(self.T_w_serv)  # tap water temperature [K]
        self.T0       = cu.C2K(self.T0)       # reference temperature [K]
        self.T_exh    = cu.C2K(self.T_exh)    # exhaust gas temperature [K]

        # L/min to m³/s
        self.dV_w_serv = self.dV_w_serv / 60 / 1000 # L/min to m³/s
        
        # Temperature [K]
        self.T_tank_is = self.T_w_tank # inner surface temperature of tank [K]

        # Surface areas
        self.r1 = self.r0 + self.x_shell
        self.r2 = self.r1 + self.x_ins
        
        # Tank surface areas [m²]
        self.A_side = 2 * math.pi * self.r2 * self.H
        self.A_base = math.pi * self.r0**2
        
        # Total tank volume [m³]
        self.V_tank = self.A_base * self.H

        # Volumetric flow rate ratio [-]
        self.alp = (self.T_w_serv - self.T_w_sup)/(self.T_w_tank - self.T_w_sup)
        self.alp = print("alp is negative") if self.alp < 0 else self.alp
        
        # Volumetric flow rates [m³/s]
        self.dV_w_sup_comb = self.alp * self.dV_w_serv
        self.dV_w_sup_mix  = (1-self.alp)*self.dV_w_serv

        # Thermal resistances per unit area/legnth
        self.R_base_unit = self.x_shell / self.k_shell + self.x_ins / self.k_ins # [m2K/W]
        self.R_side_unit = math.log(self.r1 / self.r0) / (2 * math.pi * self.k_shell) + math.log(self.r2 / self.r1) / (2 * math.pi * self.k_ins) # [mK/W]
        
        # Thermal resistances [K/W]
        self.R_base = self.R_base_unit / self.A_base # [K/W]
        self.R_side = self.R_side_unit / self.H # [K/W]
        
        # Thermal resistances [K/W]
        self.R_base_ext = 1 / (self.h_o * self.A_base)
        self.R_side_ext = 1 / (self.h_o * self.A_side)

        # Total thermal resistances [K/W]
        self.R_base_tot = self.R_base + self.R_base_ext
        self.R_side_tot = self.R_side + self.R_side_ext

        # U-value [W/K]
        self.U_tank = 2/self.R_base_tot + 1/self.R_side_tot
        self.Q_l_tank = self.U_tank * (self.T_tank_is - self.T0)  # Heat loss from tank

        # Temperature [K]
        self.T_w_comb = self.T_w_tank + self.Q_l_tank / (c_w * rho_w * self.dV_w_sup_comb)
        self.T_NG = self.T0 / (1 - ex_eff_NG) # eta_NG = 1 - T0/T_NG => T_NG = T0/(1-eta_NG) [K]
        
        # Pre-define variables for balance dictionaries
        self.E_NG     = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_comb - self.T_w_sup) / self.eta_comb
        self.Q_w_sup      = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_sup - self.T0)
        self.Q_exh        = (1 - self.eta_comb) * self.E_NG  # Heat loss from exhaust gases
        self.Q_w_comb_out = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_comb - self.T0)
        self.Q_w_tank     = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_tank - self.T0)
        self.Q_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * (self.T_w_sup - self.T0)
        self.Q_w_serv     = c_w * rho_w * self.dV_w_serv * (self.T_w_serv - self.T0)

        # Pre-calculate Entropy values for boiler
        self.S_NG         = (1 / self.T_NG) * self.E_NG
        self.S_w_sup      = c_w * rho_w * self.dV_w_sup_comb * math.log(self.T_w_sup / self.T0)
        self.S_w_comb_out = c_w * rho_w * self.dV_w_sup_comb * math.log(self.T_w_comb / self.T0)
        self.S_exh        = (1 / self.T_exh) * self.Q_exh
        self.S_g_comb     = (self.S_w_comb_out + self.S_exh) - (self.S_NG + self.S_w_sup)

        self.S_w_tank = c_w * rho_w * self.dV_w_sup_comb * math.log(self.T_w_tank / self.T0)
        self.S_l_tank = (1 / self.T_tank_is) * self.Q_l_tank
        self.S_g_tank = (self.S_w_tank + self.S_l_tank) - self.S_w_comb_out

        self.S_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * math.log(self.T_w_sup / self.T0)
        self.S_w_serv = c_w * rho_w * self.dV_w_serv * math.log(self.T_w_serv / self.T0)
        self.S_g_mix = self.S_w_serv - (self.S_w_tank + self.S_w_sup_mix)

        # Pre-calculate Exergy values for boiler
        self.X_NG = ex_eff_NG * self.E_NG
        self.X_w_sup = c_w * rho_w * self.dV_w_sup_comb * ((self.T_w_sup - self.T0) - self.T0 * math.log(self.T_w_sup / self.T0))
        self.X_w_comb_out = c_w * rho_w * self.dV_w_sup_comb * ((self.T_w_comb - self.T0) - self.T0 * math.log(self.T_w_comb / self.T0))
        self.X_exh = (1 - self.T0 / self.T_exh) * self.Q_exh
        self.X_c_comb = self.S_g_comb * self.T0

        self.X_w_tank = c_w * rho_w * self.dV_w_sup_comb * ((self.T_w_tank - self.T0) - self.T0 * math.log(self.T_w_tank / self.T0))
        self.X_l_tank = (1 - self.T0 / self.T_tank_is) * self.Q_l_tank
        self.X_c_tank = self.S_g_tank * self.T0

        self.X_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * ((self.T_w_sup - self.T0) - self.T0 * math.log(self.T_w_sup / self.T0))
        self.X_w_serv = c_w * rho_w * self.dV_w_serv * ((self.T_w_serv - self.T0) - self.T0 * math.log(self.T_w_serv / self.T0))
        self.X_c_mix = self.S_g_mix * self.T0
        
        # total
        self.X_c_tot = self.X_c_comb + self.X_c_tank + self.X_c_mix
        self.X_eff = self.X_w_serv / self.X_NG

        self.energy_balance = {}
        self.energy_balance["combustion chamber"] = {
            "in": {
            "E_NG": self.E_NG,
            "Q_w_sup": self.Q_w_sup
            },
            "out": {
            "Q_w_comb_out": self.Q_w_comb_out,
            "Q_exh": self.Q_exh
            }
        }

        self.energy_balance["hot water tank"] = {
            "in": {
            "Q_w_comb_out": self.Q_w_comb_out
            },
            "out": {
            "Q_w_tank": self.Q_w_tank,
            "Q_l_tank": self.Q_l_tank
            }
        }

        self.energy_balance["mixing valve"] = {
            "in": {
            "Q_w_tank": self.Q_w_tank,
            "Q_w_sup_mix": self.Q_w_sup_mix
            },
            "out": {
            "Q_w_serv": self.Q_w_serv
            }
        }

        self.entropy_balance = {}
        self.entropy_balance["combustion chamber"] = {
            "in": {
            "S_NG": self.S_NG,
            "S_w_sup": self.S_w_sup
            },
            "out": {
            "S_w_comb_out": self.S_w_comb_out,
            "S_exh": self.S_exh
            },
            "gen": {
            "S_g_comb": self.S_g_comb
            }
        }

        self.entropy_balance["hot water tank"] = {
            "in": {
            "S_w_comb_out": self.S_w_comb_out
            },
            "out": {
            "S_w_tank": self.S_w_tank,
            "S_l_tank": self.S_l_tank
            },
            "gen": {
            "S_g_tank": self.S_g_tank
            }
        }
        
        self.entropy_balance["mixing valve"] = {
            "in": {
            "S_w_tank": self.S_w_tank,
            "S_w_sup_mix": self.S_w_sup_mix
            },
            "out": {
            "S_w_serv": self.S_w_serv
            },
            "gen": {
            "S_g_mix": self.S_g_mix
            }
        }

        self.exergy_balance = {}
        self.exergy_balance["combustion chamber"] = {
            "in": {
            "X_NG": self.X_NG,
            "X_w_sup": self.X_w_sup
            },
            "out": {
            "X_w_comb_out": self.X_w_comb_out,
            "X_exh": self.X_exh
            },
            "con": {
            "X_c_comb": self.X_c_comb
            }
        }

        self.exergy_balance["hot water tank"] = {
            "in": {
            "X_w_comb_out": self.X_w_comb_out
            },
            "out": {
            "X_w_tank": self.X_w_tank,
            "X_l_tank": self.X_l_tank
            },
            "con": {
            "X_c_tank": self.X_c_tank
            }
        }

        self.exergy_balance["mixing valve"] = {
            "in": {
            "X_w_tank": self.X_w_tank,
            "X_w_sup_mix": self.X_w_sup_mix
            },
            "out": {
            "X_w_serv": self.X_w_serv
            },
            "con": {
            "X_c_mix": self.X_c_mix
            }
        }

@dataclass
class HeatPumpBoiler:

    def __post_init__(self): 
        
        # Efficiency [-]
        self.eta_fan = 0.6
        self.COP   = 2.5
                
        # Pressure [Pa]
        self.dP = 200 

        # Temperature [K]
        self.T0          = 0
        self.T_a_ext_out = self.T0 - 5
        self.T_r_ext     = self.T0 - 10
        
        self.T_w_tank    = 60
        self.T_r_tank    = self.T_w_tank + 5
        
        self.T_w_serv    = 45
        self.T_w_sup     = 10

        # Tank water use [L/min]
        self.dV_w_serv  = 1.2

        # Tank size [m]
        self.r0 = 0.2
        self.H = 0.8
        
        # Tank layer thickness [m]
        self.x_shell = 0.01 
        self.x_ins   = 0.10 
        
        # Tank thermal conductivity [W/mK]
        self.k_shell = 25   
        self.k_ins   = 0.03 

        # Overall heat transfer coefficient [W/m²K]
        self.h_o = 15 
        
    def system_update(self):
        
        # Celcius to Kelvin
        self.T0          = cu.C2K(self.T0)
        self.T_a_ext_out = cu.C2K(self.T_a_ext_out)
        self.T_r_ext     = cu.C2K(self.T_r_ext)
        self.T_r_tank    = cu.C2K(self.T_r_tank)
        self.T_w_tank    = cu.C2K(self.T_w_tank)
        self.T_w_serv     = cu.C2K(self.T_w_serv)
        self.T_w_sup     = cu.C2K(self.T_w_sup)
        
        # L/min to m³/s
        self.dV_w_serv = self.dV_w_serv / 60 / 1000 # L/min to m³/s
        
        # Temperature [K]
        self.T_tank_is = self.T_w_tank 
        self.T_a_ext_in = self.T0  # External unit inlet air temperature [K]

        # Surface areas
        self.r1 = self.r0 + self.x_shell
        self.r2 = self.r1 + self.x_ins
        
        # Tank surface areas [m²]
        self.A_side = 2 * math.pi * self.r2 * self.H
        self.A_base = math.pi * self.r0**2
        
        # Total tank volume [m³]
        self.V_tank = self.A_base * self.H

        # Volumetric flow rate ratio [-]
        self.alp = (self.T_w_serv - self.T_w_sup)/(self.T_w_tank - self.T_w_sup)
        self.alp = print("alp is negative") if self.alp < 0 else self.alp
        
        # Volumetric flow rates [m³/s]
        self.dV_w_sup_tank = self.alp * self.dV_w_serv
        self.dV_w_sup_mix  = (1-self.alp)*self.dV_w_serv

        # Thermal resistances per unit area/legnth
        self.R_base_unit = self.x_shell / self.k_shell + self.x_ins / self.k_ins # [m2K/W]
        self.R_side_unit = math.log(self.r1 / self.r0) / (2 * math.pi * self.k_shell) + math.log(self.r2 / self.r1) / (2 * math.pi * self.k_ins) # [mK/W]
        
        # Thermal resistances [K/W]
        self.R_base = self.R_base_unit / self.A_base
        self.R_side = self.R_side_unit / self.H 
        
        # Thermal resistances [K/W]
        self.R_base_ext = 1 / (self.h_o * self.A_base)
        self.R_side_ext = 1 / (self.h_o * self.A_side)

        # Total thermal resistances [K/W]
        self.R_base_tot = self.R_base + self.R_base_ext
        self.R_side_tot = self.R_side + self.R_side_ext

        # U-value [W/K]
        self.U_tank = 2/self.R_base_tot + 1/self.R_side_tot

        # Heat transfer
        self.Q_l_tank = self.U_tank * (self.T_tank_is - self.T0) # Tank heat losses
        self.Q_w_tank      = c_w * rho_w * self.dV_w_sup_tank * (self.T_w_tank - self.T0) # Heat transfer from tank water to mixing valve
        self.Q_w_sup_tank  = c_w * rho_w * self.dV_w_sup_tank * (self.T_w_sup - self.T0) # Heat transfer from supply water to tank water

        self.Q_r_tank = self.Q_l_tank + (self.Q_w_tank - self.Q_w_sup_tank) # Heat transfer from refrigerant to tank water
        self.E_cmp    = self.Q_r_tank/self.COP  # E_cmp [W]
        self.Q_r_ext  = self.Q_r_tank - self.E_cmp # Heat transfer from external unit to refrigerant

        def fan_equation(V_a_ext): 
            term1 = self.dP * V_a_ext / self.eta_fan # E_fan [W]
            term2 = c_a * rho_a * V_a_ext * (self.T_a_ext_in - self.T_a_ext_out) 
            return term1 + term2 - self.Q_r_ext
        
        # External fan air flow rate
        V_a_ext_initial_guess = 1.0

        from scipy.optimize import fsolve
        self.dV_a_ext = fsolve(fan_equation, V_a_ext_initial_guess)[0]
        if self.dV_a_ext < 0: 
            print("Negative air flow rate, check the input temperatures and heat transfer values.")
        self.E_fan   = self.dP * self.dV_a_ext/self.eta_fan  # Power input to external fan [W] (\Delta P = 0.5 * rho * V^2)

        self.Q_w_sup_tank = c_w * rho_w * self.dV_w_sup_tank * (self.T_w_sup - self.T0)
        self.Q_w_tank     = c_w * rho_w * self.dV_w_sup_tank * (self.T_w_tank - self.T0)
        self.Q_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * (self.T_w_sup - self.T0)
        self.Q_w_serv     = c_w * rho_w * self.dV_w_serv * (self.T_w_serv - self.T0)
        self.Q_a_ext_in   = c_a * rho_a * self.dV_a_ext * (self.T_a_ext_in - self.T0)
        self.Q_a_ext_out  = c_a * rho_a * self.dV_a_ext * (self.T_a_ext_out - self.T0)

        self.S_fan       = (1 / float('inf')) * self.E_fan
        self.S_a_ext_in  = c_a * rho_a * self.dV_a_ext * math.log(self.T_a_ext_in / self.T0)
        self.S_a_ext_out = c_a * rho_a * self.dV_a_ext * math.log(self.T_a_ext_out / self.T0)
        self.S_r_ext     = (1 / self.T_r_ext) * self.Q_r_ext
        self.S_cmp       = (1 / float('inf')) * self.E_cmp
        
        self.S_r_tank    = (1 / self.T_r_tank) * self.Q_r_tank
        self.S_w_sup_tank = c_w * rho_w * self.dV_w_sup_tank * math.log(self.T_w_sup / self.T0)
        self.S_w_tank     = c_w * rho_w * self.dV_w_sup_tank * math.log(self.T_w_tank / self.T0)
        self.S_l_tank     = (1 / self.T_tank_is) * self.Q_l_tank
        self.S_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * math.log(self.T_w_sup / self.T0)
        self.S_w_serv      = c_w * rho_w * self.dV_w_serv * math.log(self.T_w_serv / self.T0)

        self.S_g_ext = self.S_a_ext_out + self.S_r_ext - (self.S_fan + self.S_a_ext_in)
        self.S_g_r = self.S_r_tank - (self.S_cmp + self.S_r_ext)
        self.S_g_tank = (self.S_w_tank + self.S_l_tank) - (self.S_r_tank + self.S_w_sup_tank)
        self.S_g_mix = self.S_w_serv - (self.S_w_tank + self.S_w_sup_mix)

        self.X_fan = self.E_fan - self.S_fan * self.T0
        self.X_cmp = self.E_cmp - self.S_cmp * self.T0
        self.X_r_ext = -(1 - self.T0 / self.T_r_ext) * self.Q_r_ext
        self.X_r_tank = (1 - self.T0 / self.T_r_tank) * self.Q_r_tank
        self.X_w_sup_tank = c_w * rho_w * self.dV_w_sup_tank * ((self.T_w_sup - self.T0) - self.T0 * math.log(self.T_w_sup / self.T0))
        self.X_w_tank = c_w * rho_w * self.dV_w_sup_tank * ((self.T_w_tank - self.T0) - self.T0 * math.log(self.T_w_tank / self.T0))
        self.X_l_tank = (1 - self.T0 / self.T_tank_is) * self.Q_l_tank
        self.X_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * ((self.T_w_sup - self.T0) - self.T0 * math.log(self.T_w_sup / self.T0))
        self.X_w_serv = c_w * rho_w * self.dV_w_serv * ((self.T_w_serv - self.T0) - self.T0 * math.log(self.T_w_serv / self.T0))
        self.X_a_ext_in = c_a * rho_a * self.dV_a_ext * ((self.T_a_ext_in - self.T0) - self.T0 * math.log(self.T_a_ext_in / self.T0))
        self.X_a_ext_out = c_a * rho_a * self.dV_a_ext * ((self.T_a_ext_out - self.T0) - self.T0 * math.log(self.T_a_ext_out / self.T0))
        
        self.X_c_ext = self.S_g_ext * self.T0
        self.X_c_r = self.S_g_r * self.T0
        self.X_c_tank = self.S_g_tank * self.T0
        self.X_c_mix = self.S_g_mix * self.T0
        
        # total
        self.X_c_tot = self.X_c_ext + self.X_c_r + self.X_c_tank + self.X_c_mix
        self.X_eff = self.X_w_serv / (self.X_fan + self.X_cmp)

        self.energy_balance = {}
        self.energy_balance["external unit"] = {
            "in": {
            "E_fan": self.E_fan,
            "Q_a_ext_in": self.Q_a_ext_in,
            },
            "out": {
            "Q_a_ext_out": self.Q_a_ext_out,
            "Q_r_ext": self.Q_r_ext,
            }
        }

        self.energy_balance["refrigerant loop"] = {
            "in": {
            "E_cmp": self.E_cmp,
            "Q_r_ext": self.Q_r_ext
            },
            "out": {
            "Q_r_tank": self.Q_r_tank
            }
        }

        self.energy_balance["hot water tank"] = {
            "in": {
            "Q_r_tank": self.Q_r_tank,
            "Q_w_sup_tank": self.Q_w_sup_tank
            },
            "out": {
            "Q_w_tank": self.Q_w_tank,
            "Q_l_tank": self.Q_l_tank
            }
        }

        self.energy_balance["mixing valve"] = {
            "in": {
            "Q_w_tank": self.Q_w_tank,
            "Q_w_sup_mix": self.Q_w_sup_mix
            },
            "out": {
            "Q_w_serv": self.Q_w_serv
            }
        }

        ## Entropy Balance ========================================
        self.entropy_balance = {}

        self.entropy_balance["external unit"] = {
            "in": {
            "S_fan": self.S_fan,
            "S_a_ext_in": self.S_a_ext_in
            },
            "out": {
            "S_a_ext_out": self.S_a_ext_out,
            "S_r_ext": self.S_r_ext
            },
            "gen": {
            "S_g_ext": self.S_g_ext
            }
        }

        self.entropy_balance["refrigerant loop"] = {
            "in": {
            "S_cmp": self.S_cmp,
            "S_r_ext": self.S_r_ext
            },
            "out": {
            "S_r_tank": self.S_r_tank
            },
            "gen": {
            "S_g_r": self.S_g_r
            }
        }

        self.entropy_balance["hot water tank"] = {
            "in": {
            "S_r_tank": self.S_r_tank,
            "S_w_sup_tank": self.S_w_sup_tank
            },
            "out": {
            "S_w_tank": self.S_w_tank,
            "S_l_tank": self.S_l_tank
            },
            "gen": {
            "S_g_tank": self.S_g_tank
            }
        }

        self.entropy_balance["mixing valve"] = {
            "in": {
            "S_w_tank": self.S_w_tank,
            "S_w_sup_mix": self.S_w_sup_mix
            },
            "out": {
            "S_w_serv": self.S_w_serv
            },
            "gen": {
            "S_g_mix": self.S_g_mix
            }
        }

        ## Exergy Balance ========================================
        self.exergy_balance = {}

        self.exergy_balance["external unit"] = {
            "in": {
            "E_fan": self.E_fan,
            "X_r_ext": self.X_r_ext,
            "X_a_ext_in": self.X_a_ext_in
            },
            "con": {
            "X_c_ext": self.X_c_ext
            },
            "out": {
            "X_a_ext_out": self.X_a_ext_out
            }
        }

        self.exergy_balance["refrigerant loop"] = {
            "in": {
            "E_cmp": self.E_cmp
            },
            "con": {
            "X_c_r": self.X_c_r
            },
            "out": {
            "X_r_tank": self.X_r_tank,
            "X_r_ext": self.X_r_ext
            }
        }

        self.exergy_balance["hot water tank"] = {
            "in": {
            "X_r_tank": self.X_r_tank,
            "X_w_sup_tank": self.X_w_sup_tank
            },
            "con": {
            "X_c_tank": self.X_c_tank
            },
            "out": {
            "X_w_tank": self.X_w_tank,
            "X_l_tank": self.X_l_tank
            }
        }

        self.exergy_balance["mixing valve"] = {
            "in": {
            "X_w_tank": self.X_w_tank,
            "X_w_sup_mix": self.X_w_sup_mix
            },
            "con": {
            "X_c_mix": self.X_c_mix
            },
            "out": {
            "X_w_serv": self.X_w_serv
            }
        }

@dataclass
class SolarAssistedGasBoiler:

    def __post_init__(self):
        # Constants [-]
        self.alpha    = 0.95 # Absorptivity of collector
        self.eta_comb = 0.9 # Efficiency of combustion chamber

        # Solar radiation [W/m²]  
        self.I_DN = 500
        self.I_dH = 200

        # solar thermal collector
        self.A_stc = 2 # Solar thermal collector area [m²]

        # Temperature [°C]
        self.T0       = 0
        self.T_w_comb = 60
        self.T_w_serv  = 45
        self.T_w_sup  = 10
        self.T_exh    = 70

        # Tank water use [L/min]
        self.dV_w_serv = 1.2
        
        # Overall heat transfer coefficient [W/m²K]
        self.h_o = 15
        self.h_r = 2 # radiative heat transfer coefficient in air layer [W/m²K]
        
        # Thermal conductivity [W/mK]
        self.k_ins = 0.03 # Insulation thermal conductivity [W/mK]
        
        # Thickness [m]
        self.x_air = 0.01 # Air layer thickness [m]
        self.x_ins = 0.05 # insulation layer thickness [m]
        
    def system_update(self): 
    
        # L/min to m³/s
        self.dV_w_serv = self.dV_w_serv / 60 / 1000 # L/min to m³/s
        
        # Iradiance [W/m²]
        self.I_sol = self.I_DN + self.I_dH
        
        # Resistance [m2K/W] (conduction)
        self.R_air = self.x_air / k_a # [m2K/W]
        self.R_ins = self.x_ins / self.k_ins # [m2K/W]
        self.R_o   = 1/self.h_o
        self.R_r   = 1/self.h_r
        
        self.R1 = (self.R_r * self.R_air)/(self.R_r + self.R_air) + self.R_o
        self.R2 = self.R_ins + self.R_o
        
        # U-value [W/m²K]
        self.U1 = 1 / self.R1
        self.U2 = 1 / self.R2
        self.U  = self.U1 + self.U2 # 병렬
        
        
        # Celcius to Kelvin
        self.T0       = cu.C2K(self.T0)
        self.T_w_comb = cu.C2K(self.T_w_comb)
        self.T_w_serv  = cu.C2K(self.T_w_serv)
        self.T_w_sup  = cu.C2K(self.T_w_sup)
        self.T_exh    = cu.C2K(self.T_exh)
        self.T_NG     = self.T0 / (1 - ex_eff_NG)
        
        # Volumetric flow rate ratio [-]
        self.alp = (self.T_w_serv - self.T_w_sup)/(self.T_w_comb - self.T_w_sup)
        self.alp = print("alp is negative") if self.alp < 0 else self.alp
        
        # Volumetric flow rates [m³/s]
        self.dV_w_sup     = self.alp * self.dV_w_serv
        self.dV_w_sup_mix = (1-self.alp)*self.dV_w_serv
        
        # Demensionless numbers
        self.ksi_stc = np.exp(-self.A_stc * self.U/(c_w * rho_w * self.dV_w_sup))
        
        # Energy balance
        self.Q_w_sup     = c_w * rho_w * self.dV_w_sup * (self.T_w_sup - self.T0)
        self.Q_sol       = self.I_sol * self.A_stc * self.alpha
        
        T_w_stc_out_numerator = self.T0 + (
        self.Q_sol + self.Q_w_sup
        + self.A_stc * self.U * (self.ksi_stc * self.T_w_sup / (1 - self.ksi_stc))
        + self.A_stc * self.U * self.T0
        ) / (c_w * rho_w * self.dV_w_sup)

        T_w_stc_out_denominator = 1 + (self.A_stc * self.U) / ((1 - self.ksi_stc) * c_w * rho_w * self.dV_w_sup)

        self.T_w_stc_out = T_w_stc_out_numerator / T_w_stc_out_denominator
        self.T_stc = 1/(1-self.ksi_stc)*self.T_w_stc_out - self.ksi_stc/(1-self.ksi_stc)*self.T_w_sup

        self.Q_w_stc_out = c_w * rho_w * self.dV_w_sup * (self.T_w_stc_out - self.T0)
        self.Q_l         = self.A_stc * self.U * (self.T_stc - self.T0)
        
        self.E_NG     = c_w * rho_w * self.dV_w_sup * (self.T_w_comb - self.T_w_stc_out) / self.eta_comb
        self.Q_exh    = (1 - self.eta_comb) * self.E_NG  # Heat loss from exhaust gases
        self.Q_w_comb = c_w * rho_w * self.dV_w_sup * (self.T_w_comb - self.T0)
        
        self.Q_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * (self.T_w_sup - self.T0)
        self.Q_w_serv     = c_w * rho_w * self.dV_w_serv * (self.T_w_serv - self.T0)
        
        # Entropy balance
        self.S_w_sup = c_w * rho_w * self.dV_w_sup * math.log(self.T_w_sup / self.T0)
        self.S_DN = k_D * self.I_DN**(0.9)
        self.S_dH = k_d * self.I_dH**(0.9)
        self.S_sol = self.S_DN + self.S_dH
        self.S_w_stc_out = c_w * rho_w * self.dV_w_sup * math.log(self.T_w_stc_out / self.T0)       
        self.S_l = (1 / self.T_stc) * self.A_stc * self.U * (self.T_stc - self.T0)
        self.S_g_stc = self.S_w_stc_out + self.S_l - (self.S_sol + self.S_w_sup)
        
        self.S_NG = (1 / self.T_NG) * self.E_NG
        self.S_exh = (1 / self.T_exh) * self.Q_exh
        self.S_w_comb = c_w * rho_w * self.dV_w_sup * math.log(self.T_w_comb / self.T0)
        self.S_g_comb = (self.S_w_comb + self.S_exh) - (self.S_NG + self.S_w_stc_out)
        
        self.S_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * math.log(self.T_w_sup / self.T0)
        self.S_w_serv = c_w * rho_w * self.dV_w_serv * math.log(self.T_w_serv / self.T0)
        self.S_g_mix = self.S_w_serv - (self.S_w_comb + self.S_w_sup_mix)
        
        # Exergy balance
        self.X_w_sup = self.Q_w_sup - self.S_w_sup * self.T0
        self.X_sol = self.Q_sol - self.S_sol * self.T0
        self.X_w_stc_out = self.Q_w_stc_out - self.S_w_stc_out * self.T0
        self.X_l = self.Q_l - self.S_l * self.T0
        self.X_c_stc = self.S_g_stc * self.T0

        self.X_NG = ex_eff_NG * self.E_NG
        self.X_exh = (1 - self.T0 / self.T_exh) * self.Q_exh
        self.X_w_comb = self.Q_w_comb - self.S_w_comb * self.T0
        self.X_c_comb = self.S_g_comb * self.T0

        self.X_w_sup_mix = self.Q_w_sup_mix - self.S_w_sup_mix * self.T0
        self.X_w_serv = self.Q_w_serv - self.S_w_serv * self.T0 
        self.X_c_mix = self.S_g_mix * self.T0

        self.X_eff = self.X_w_serv / (self.X_NG)

        self.energy_balance = {}
        self.energy_balance["solar thermal panel"] = {
            "in": {
            "Q_sol": self.Q_sol,
            "Q_w_sup": self.Q_w_sup
            },
            "out": {
            "Q_w_stc_out": self.Q_w_stc_out,
            "Q_l": self.Q_l
            }
        }

        self.energy_balance["combustion chamber"] = {
            "in": {
            "Q_w_stc_out": self.Q_w_stc_out,
            "E_NG": self.E_NG,
            },
            "out": {
            "Q_exh": self.Q_exh,
            "Q_w_comb": self.Q_w_comb
            }
        }

        self.energy_balance["mixing valve"] = {
            "in": {
            "Q_w_comb": self.Q_w_comb,
            "Q_w_sup_mix": self.Q_w_sup_mix
            },
            "out": {
            "Q_w_serv": self.Q_w_serv
            }
        }

        ## Entropy Balance ========================================
        self.entropy_balance = {}

        self.entropy_balance["solar thermal panel"] = {
            "in": {
            "S_sol": self.S_sol,
            "S_w_sup": self.S_w_sup
            },
            "gen": {
            "S_g_stc": self.S_g_stc
            },
            "out": {
            "S_w_stc_out": self.S_w_stc_out,
            "S_l": self.S_l
            }
        }

        self.entropy_balance["combustion chamber"] = {
            "in": {
            "S_w_stc_out": self.S_w_stc_out,
            "S_NG": self.S_NG,
            },
            "gen": {
            "S_g_comb": self.S_g_comb
            },
            "out": {
            "S_exh": self.S_exh,
            "S_w_comb": self.S_w_comb
            }
        }

        self.entropy_balance["mixing valve"] = {
            "in": {
            "S_w_comb": self.S_w_comb,
            "S_w_sup_mix": self.S_w_sup_mix
            },
            "gen": {
            "S_g_mix": self.S_g_mix
            },
            "out": {
            "S_w_serv": self.S_w_serv
            }
        }


        ## Exergy Balance ========================================
        self.exergy_balance = {}

        self.exergy_balance["solar thermal panel"] = {
            "in": {
            "X_sol": self.X_sol,
            "X_w_sup": self.X_w_sup
            },
            "con": {
            "X_c_stc": self.X_c_stc
            },
            "out": {
            "X_w_stc_out": self.X_w_stc_out,
            "X_l": self.X_l
            }
        }

        self.exergy_balance["combustion chamber"] = {
            "in": {
            "X_w_stc_out": self.X_w_stc_out,
            "X_NG": self.X_NG,
            },
            "con": {
            "X_c_comb": self.X_c_comb
            },
            "out": {
            "X_exh": self.X_exh,
            "X_w_comb": self.X_w_comb
            }
        }

        self.exergy_balance["mixing valve"] = {
            "in": {
            "X_w_comb": self.X_w_comb,
            "X_w_sup_mix": self.X_w_sup_mix
            },
            "con": {
            "X_c_mix": self.X_c_mix
            },
            "out": {
            "X_w_serv": self.X_w_serv
            }
        }
              
@dataclass
class GroundSourceHeatPumpBoiler:

    def __post_init__(self): 
        self.time = 10 # [h]
        
        # Temperature [C]
        self.T0 = 0
        
        self.T_w_tank = 60
        self.T_w_serv = 45
        self.T_w_sup  = 10
        
        self.T_g      = 11
        self.T_r_tank = self.T_w_tank + 5

        self.dT_r_exch = -5  # 예시: 열교환기의 온도 - 열교환후 지중순환수 온도 [K]
        
        # Tank water use [L/min]
        self.dV_w_serv  = 1.2

        # Tank size [m]
        self.r0 = 0.2
        self.H = 0.8
        
        # Tank layer thickness [m]
        self.x_shell = 0.01
        self.x_ins   = 0.10
        
        # Tank thermal conductivity [W/mK]
        self.k_shell = 25   
        self.k_ins   = 0.03 

        # Overall heat transfer coefficient [W/m²K]
        self.h_o = 15 
        
        # Borehole parameters
        self.D_b = 0 # Borehole depth [m]
        self.H_b = 200 # Borehole height [m]
        self.r_b = 0.08 # Borehole radius [m]
        self.R_b = 0.108 # Effective borehole thermal resistance [mK/W]

        # Fluid parameters
        self.dV_f = 24 # Volumetric flow rate of fluid [L/min]

        # Ground parameters
        self.k_g = 2.0
        self.c_g = 800
        self.rho_g = 2000 

        # Pump of ground heat exchanger
        self.E_pmp = 200

    def system_update(self):
        
        if self.T_r_tank < self.T_w_tank:
            raise ValueError("T_r_tank cannot be smaller than T_w_tank")
        
        # L/min to m³/s
        self.dV_w_serv = self.dV_w_serv / 60 / 1000 # L/min to m³/s
        self.dV_f = self.dV_f / 60 / 1000 # L/min to m³/s

        # time
        self.time = self.time * cu.h2s  # Convert hours to seconds

        # Celcius to Kelvin
        self.T0 = cu.C2K(self.T0)
        self.T_w_tank = cu.C2K(self.T_w_tank)
        self.T_w_serv = cu.C2K(self.T_w_serv)
        self.T_w_sup = cu.C2K(self.T_w_sup)
        self.T_g = cu.C2K(self.T_g)
        self.T_r_tank = cu.C2K(self.T_r_tank)
                
        # Others
        self.alpha = self.k_g / (self.c_g * self.rho_g) # thermal diffusivity of ground [m²/s]
        
        # Temperature
        self.T_tank_is = self.T_w_tank # inner surface temperature of tank [K]
        
        # Surface areas
        self.r1 = self.r0 + self.x_shell
        self.r2 = self.r1 + self.x_ins
        
        # Tank surface areas [m²]
        self.A_side = 2 * math.pi * self.r2 * self.H
        self.A_base = math.pi * self.r0**2
        
        # Total tank volume [m³]
        self.V_tank = self.A_base * self.H

        # Volumetric flow rate ratio [-]
        self.alp = (self.T_w_serv - self.T_w_sup)/(self.T_w_tank - self.T_w_sup)
        self.alp = print("alp is negative") if self.alp < 0 else self.alp
        
        # Volumetric flow rates [m³/s]
        self.dV_w_sup_tank = self.alp * self.dV_w_serv
        self.dV_w_sup_mix  = (1-self.alp)*self.dV_w_serv

        # Thermal resistances per unit area/legnth
        self.R_base_unit = self.x_shell / self.k_shell + self.x_ins / self.k_ins # [m2K/W]
        self.R_side_unit = math.log(self.r1 / self.r0) / (2 * math.pi * self.k_shell) + math.log(self.r2 / self.r1) / (2 * math.pi * self.k_ins) # [mK/W]
        
        # Thermal resistances [K/W]
        self.R_base = self.R_base_unit / self.A_base # [K/W]
        self.R_side = self.R_side_unit / self.H # [K/W]
        
        # Thermal resistances [K/W]
        self.R_base_ext = 1 / (self.h_o * self.A_base)
        self.R_side_ext = 1 / (self.h_o * self.A_side)

        # Total thermal resistances [K/W]
        self.R_base_tot = self.R_base + self.R_base_ext
        self.R_side_tot = self.R_side + self.R_side_ext

        # U-value [W/K]
        self.U_tank = 2/self.R_base_tot + 1/self.R_side_tot
        
        # Load [W]
        self.Q_l_tank = self.U_tank * (self.T_tank_is - self.T0)
        self.Q_r_tank = c_w * rho_w * self.dV_w_sup_tank * (self.T_w_tank - self.T_w_sup)
        
        # 반복 수치해법 적용
        '''
        반복 수치해법을 사용하는 이유:
        1. 냉매 온도(T_r_exch)와 유체 입구 온도(T_f_in)가 서로 연동되어 직접 계산이 불가능함.
        2. 보어홀 열저항, 유량, 토양물성 등 시스템 파라미터가 COP, 온도, 효율에 반영되도록 하기 위함.
        3. 두 온도가 수렴할 때까지 반복 계산하여 물리적으로 일관된 해를 얻기 위함.
        '''
        max_iter = 20
        tol = 1e-3
        self.T_f = self.T_g  # 초기값
        self.T_f_in = self.T_f + self.dT_r_exch  # 초기값, 열교환기에서의 순환수 유입 온도

        for _ in range(max_iter):
            self.T_r_exch = self.T_f_in + self.dT_r_exch  # 5 K 높게 설정
            self.COP = calculate_GSHP_COP(Tg = self.T_g,
                                         T_cond = self.T_r_tank,
                                         T_evap = self.T_r_exch,
                                         theta_hat = 0.3)
            # Others
            self.E_cmp = self.Q_r_tank / self.COP # compressor power input [W]
            self.Q_r_exch = self.Q_r_tank - self.E_cmp  # changed from Q_r_ext to Q_r_exch
            # Borehole 
            self.Q_bh = (self.Q_r_exch - self.E_pmp) / self.H_b # heat flow rate from borehole to ground per unit length [W/m]
            self.g_i = G_FLS(t = self.time, ks = self.k_g, as_ = self.alpha, rb = self.r_b, H = self.H_b) # g-function [mK/W]
            # fluid temperature & borehole wall temperature [K]
            T_f_in_old = self.T_f_in  # 이전 유체 입구 온도 저장
            self.T_b = self.T_g - self.Q_bh * self.g_i # borehole wall temperature [K]
            self.T_f = self.T_b - self.Q_bh * self.R_b # fluid temperature in borehole [K]
            self.T_f_in = self.T_f - self.Q_bh * self.H_b / (2 * c_w * rho_w * self.dV_f) # fluid inlet temperature [K]
            self.T_f_out = self.T_f + self.Q_bh * self.H_b / (2 * c_w * rho_w * self.dV_f) # fluid outlet temperature [K]
            if abs(self.T_f_in - T_f_in_old) < tol:
                break

        # Exergy result
        self.X_w_sup_tank = c_w * rho_w * self.dV_w_sup_tank * ((self.T_w_sup - self.T0) - self.T0 * math.log(self.T_w_sup / self.T0))
        self.X_w_tank = c_w * rho_w * self.dV_w_sup_tank * ((self.T_w_tank - self.T0) - self.T0 * math.log(self.T_w_tank / self.T0))
        self.X_l_tank = (1 - self.T0 / self.T_tank_is) * self.Q_l_tank
        self.X_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * ((self.T_w_sup - self.T0) - self.T0 * math.log(self.T_w_sup / self.T0))
        self.X_w_serv = c_w * rho_w * self.dV_w_serv * ((self.T_w_serv - self.T0) - self.T0 * math.log(self.T_w_serv / self.T0))

        self.X_r_int = self.Q_r_tank * (1 - self.T0 / self.T_r_tank)
        self.X_r_exch = self.Q_r_exch * (1 - self.T0 / self.T_r_exch)  # changed from X_r_ext to X_r_exch

        self.X_pmp = self.E_pmp - (1 / float('inf')) * self.T0  
        self.X_cmp = self.E_cmp - (1 / float('inf')) * self.T0  
        
        self.X_f_in  = c_w * rho_w * self.dV_f * ((self.T_f_in - self.T0) - self.T0 * math.log(self.T_f_in / self.T0))
        self.X_f_out = c_w * rho_w * self.dV_f * ((self.T_f_out - self.T0) - self.T0 * math.log(self.T_f_out / self.T0))

        self.X_g = (1 - self.T0 / self.T_g) * (self.Q_bh * self.H_b)
        self.X_b = (1 - self.T0 / self.T_b) * (self.Q_bh * self.H_b)

        # Ground
        self.X_in_g = self.X_g
        self.X_out_g = self.X_b
        self.X_c_g = self.X_in_g - self.X_out_g

        # Ground heat exchanger
        self.X_in_GHE = self.E_pmp + self.X_out_g + self.X_f_in  # self.X_b 대신 self.X_out_g 사용
        self.X_out_GHE = self.X_f_out 
        self.X_c_GHE = self.X_in_GHE - self.X_out_GHE

        # Heat exchanger 
        self.X_in_exch = self.X_out_GHE 
        self.X_out_exch = self.X_r_exch + self.X_f_in
        self.X_c_exch = self.X_in_exch - self.X_out_exch

        # Closed refrigerant loop system
        self.X_in_r  = self.E_cmp + self.X_r_exch
        self.X_out_r = self.X_r_int
        self.X_c_r   = self.X_in_r - self.X_out_r

        # Tank
        self.X_in_tank  = self.X_r_int + self.X_w_sup_tank
        self.X_out_tank = self.X_w_tank + self.X_l_tank
        self.X_c_tank   = self.X_in_tank - self.X_out_tank

        # Mixing valve
        self.X_in_mix = self.X_w_tank + self.X_w_sup_mix
        self.X_out_mix = self.X_w_serv
        self.X_c_mix = self.X_in_mix - self.X_out_mix
        
        self.X_eff = self.X_w_serv / (self.X_pmp + self.X_cmp)

        ## Exergy Balance ========================================
        self.exergy_balance = {}

        # Mixing valve
        self.exergy_balance["mixing valve"] = {
            "in": {
            "$X_{w,tank}$": self.X_w_tank,
            "$X_{w,sup,mix}$": self.X_w_sup_mix,
            },
            "con": {
            "$X_{c,mix}$": self.X_c_mix,
            },
            "out": {
            "$X_{w,serv}$": self.X_w_serv,
            }
        }

        # Hot water tank
        self.exergy_balance["hot water tank"] = {
            "in": {
            "$X_{r,int}$": self.X_r_int,
            "$X_{w,sup}$": self.X_w_sup_tank,
            },
            "con": {
            "$X_{c,tank}$": self.X_c_tank,
            },
            "out": {
            "$X_{w,tank}$": self.X_w_tank,
            "$X_{l,tank}$": self.X_l_tank,
            }
        }

        # Refrigerant loop
        self.exergy_balance["refrigerant loop"] = {
            "in": {
            "$X_{cmp}$": self.X_cmp,
            "$X_{r,exch}$": self.X_r_exch,
            },
            "con": {
            "$X_{c,r}$": self.X_c_r,
            },
            "out": {
            "$X_{r,int}$": self.X_r_int,
            }
        }

        # Heat exchanger 
        self.exergy_balance["heat exchanger"] = {
            "in": {
            "$X_{f,out}$": self.X_f_out,
            },
            "con": {
            "$X_{c,exch}$": self.X_c_exch,
            },
            "out": {
            "$X_{r,exch}$": self.X_r_exch,
            "$X_{f,in}$": self.X_f_in,
            }
        }

        # Ground Heat Exchanger
        self.exergy_balance["ground heat exchanger"] = {
            "in": {
            "$X_{pmp}$": self.X_pmp,
            "$X_{b}$": self.X_b,
            "$X_{f,in}$": self.X_f_in,
            },
            "con": {
            "$X_{c,GHE}$": self.X_c_GHE,
            },
            "out": {
            "$X_{f,out}$": self.X_f_out,
            }
        }

        # Ground
        self.exergy_balance["ground"] = {
            "in": {
            "$X_{g}$": self.X_g,
            },
            "con": {
            "$X_{c,g}$": self.X_c_g,
            },
            "out": {
            "$X_{b}$": self.X_b,
            }
        }

#%%
# class - AirSourceHeatPump
@dataclass
class AirSourceHeatPump_cooling:

    def __post_init__(self):
        # fan
        self.fan_int = Fan().fan1
        self.fan_ext = Fan().fan2

        # COP
        self.Q_r_max = 9000 # [W]

        # temperature
        self.T0      = 32 # environmental temperature [°C]
        self.T_a_room = 20 # room air temperature [°C]
        
        self.T_r_int     = self.T_a_room - 15 # internal unit refrigerant temperature [°C]
        self.T_a_int_out = self.T_a_room - 10 # internal unit air outlet temperature [°C]
        
        self.T_a_ext_out = self.T0 + 10 # external unit air outlet temperature [°C]
        self.T_r_ext     = self.T0 + 15 # external unit refrigerant temperature [°C]
        
        # load
        self.Q_r_int = 6000 # [W]
        
        # COP의 reference로 삼을 수 있는 값
        self.COP_ref = 4

    def system_update(self):
        # Celcius to Kelvin
        self.T0 = cu.C2K(self.T0)
        self.T_a_room = cu.C2K(self.T_a_room)
        self.T_a_int_out = cu.C2K(self.T_a_int_out)
        self.T_a_ext_out = cu.C2K(self.T_a_ext_out)
        self.T_r_int = cu.C2K(self.T_r_int)
        self.T_r_ext = cu.C2K(self.T_r_ext)

        # temperature
        self.T_a_int_in  = self.T_a_room # internal unit air inlet temperature [K]
        self.T_a_ext_in  = self.T0 # external unit air inlet temperature [K]

        # others
        self.COP     = calculate_ASHP_cooling_COP(self.T_a_int_out, self.T_a_ext_in, self.Q_r_int, self.Q_r_max, self.COP_ref) # COP [-]
        self.E_cmp   = self.Q_r_int / self.COP # compressor power input [W]
        self.Q_r_ext = self.Q_r_int + self.E_cmp # heat transfer from external unit to refrigerant [W]

        # internal, external unit
        self.dV_int = self.Q_r_int / (c_a * rho_a * (abs(self.T_a_int_out - self.T_a_int_in))) # volumetric flow rate of internal unit [m3/s]
        self.dV_ext = self.Q_r_ext / (c_a * rho_a * (abs(self.T_a_ext_out - self.T_a_ext_in))) # volumetric flow rate of external unit [m3/s]

        # fan power
        self.E_fan_int = Fan().get_power(self.fan_int, self.dV_int) # power input of internal unit fan [W]
        self.E_fan_ext = Fan().get_power(self.fan_ext, self.dV_ext) # power input of external unit fan [W]

        # System COP
        self.COP_sys = self.Q_r_int / (self.E_fan_int + self.E_fan_ext + self.E_cmp)

        # exergy result 
        self.X_a_int_in  = c_a * rho_a * self.dV_int * ((self.T_a_int_in - self.T0) - self.T0 * math.log(self.T_a_int_in / self.T0))
        self.X_a_int_out = c_a * rho_a * self.dV_int * ((self.T_a_int_out - self.T0) - self.T0 * math.log(self.T_a_int_out / self.T0))
        self.X_a_ext_in  = c_a * rho_a * self.dV_ext * ((self.T_a_ext_in - self.T0) - self.T0 * math.log(self.T_a_ext_in / self.T0))
        self.X_a_ext_out = c_a * rho_a * self.dV_ext * ((self.T_a_ext_out - self.T0) - self.T0 * math.log(self.T_a_ext_out / self.T0))

        self.X_r_int   = - self.Q_r_int * (1 - self.T0 / self.T_r_int)
        self.X_r_ext   = self.Q_r_ext * (1 - self.T0 / self.T_r_ext)

        # Internal unit of ASHP
        self.X_in_int  = self.E_fan_int + self.X_r_int
        self.X_out_int = self.X_a_int_out - self.X_a_int_in
        self.X_c_int   = self.X_in_int - self.X_out_int

        # Closed refrigerant loop system of ASHP
        self.X_in_r  = self.E_cmp
        self.X_out_r = self.X_r_int + self.X_r_ext
        self.X_c_r   = self.X_in_r - self.X_out_r

        # External unit of ASHP
        self.X_in_ext  = self.E_fan_ext + self.X_r_ext
        self.X_out_ext = self.X_a_ext_out - self.X_a_ext_in
        self.X_c_ext   = self.X_in_ext - self.X_out_ext

        # Total exergy of ASHP
        self.X_in  = self.E_fan_int + self.E_cmp + self.E_fan_ext
        self.X_out = self.X_a_int_out - self.X_a_int_in
        self.X_c   = self.X_in - self.X_out
        
        self.X_eff = self.X_out/self.X_in
        
        ## Exergy Balance ========================================
        self.exergy_balance = {}
        # Internal Unit
        self.exergy_balance["internal unit"] = {
            "in": {
            "$E_{f,int}$": self.E_fan_int,
            "$X_{r,int}$": self.X_r_int,
            },
            "con": {
            "$X_{c,int}$": self.X_c_int,
            },
            "out": {
            "$X_{a,int,out}$": self.X_a_int_out,
            "$X_{a,int,in}$": self.X_a_int_in,
            }
        }
        
        # Refrigerant
        self.exergy_balance["refrigerant loop"] = {
            "in": {
            "$E_{cmp}$": self.E_cmp,
            },
            "con": {
            "$X_{c,r}$": self.X_c_r,
            },
            "out": {
            "$X_{r,int}$": self.X_r_int,
            "$X_{r,ext}$": self.X_r_ext,
            }
        }

        # External Unit
        self.exergy_balance["external unit"] = {
            "in": {
            "$E_{f,ext}$": self.E_fan_ext,
            "$X_{r,ext}$": self.X_r_ext,
            },
            "con": {
            "$X_{c,ext}$": self.X_c_ext,
            },
            "out": {
            "$X_{a,ext,out}$": self.X_a_ext_out,
            "$X_{a,ext,in}$": self.X_a_ext_in,
            }
        }

@dataclass
class AirSourceHeatPump_heating:

    def __post_init__(self):

        # fan
        self.fan_int = Fan().fan1
        self.fan_ext = Fan().fan2

        # COP
        self.Q_r_max = 9000 # maximum heating capacity [W]

        # temperature
        self.T0      = 0 # environmental temperature [°C]
        self.T_a_room = 20 # room air temperature [°C]
        
        self.T_r_int = self.T_a_room + 15 # internal unit refrigerant temperature [°C]
        self.T_a_int_out = self.T_a_room + 10 # internal unit air outlet temperature [°C]
        
        self.T_a_ext_out = self.T0 - 10 # external unit air outlet temperature [°C]
        self.T_r_ext = self.T0 - 15 # external unit refrigerant temperature [°C]

        # load
        self.Q_r_int = 6000 # [W]

    def system_update(self):
        
        # Celcius to Kelvin
        self.T0 = cu.C2K(self.T0)
        self.T_a_room = cu.C2K(self.T_a_room)
        self.T_a_int_out = cu.C2K(self.T_a_int_out)
        self.T_a_ext_out = cu.C2K(self.T_a_ext_out)
        self.T_r_int = cu.C2K(self.T_r_int)
        self.T_r_ext = cu.C2K(self.T_r_ext)
        
        # temperature
        self.T_a_int_in  = self.T_a_room
        self.T_a_ext_in  = self.T0 # external unit air inlet temperature [K]

        # others
        self.COP     = calculate_ASHP_heating_COP(T0 = self.T0, Q_r_int = self.Q_r_int, Q_r_max = self.Q_r_max) # COP [-]
        self.E_cmp   = self.Q_r_int / self.COP # compressor power input [W]
        self.Q_r_ext = self.Q_r_int - self.E_cmp # heat transfer from external unit to refrigerant [W]

        # internal, external unit
        self.dV_int = self.Q_r_int / (c_a * rho_a * abs(self.T_a_int_out - self.T_a_int_in)) # volumetric flow rate of internal unit [m3/s]
        self.dV_ext = self.Q_r_ext / (c_a * rho_a * abs(self.T_a_ext_out - self.T_a_ext_in)) # volumetric flow rate of external unit [m3/s]

        # fan power
        self.E_fan_int = Fan().get_power(self.fan_int, self.dV_int) # power input of internal unit fan [W]
        self.E_fan_ext = Fan().get_power(self.fan_ext, self.dV_ext) # power input of external unit fan [W]

        # System COP
        self.COP_sys = self.Q_r_int / (self.E_fan_int + self.E_fan_ext + self.E_cmp)

        # exergy result 
        self.X_a_int_in  = c_a * rho_a * self.dV_int * ((self.T_a_int_in - self.T0) - self.T0 * math.log(self.T_a_int_in / self.T0))
        self.X_a_int_out = c_a * rho_a * self.dV_int * ((self.T_a_int_out - self.T0) - self.T0 * math.log(self.T_a_int_out / self.T0))
        self.X_a_ext_in  = c_a * rho_a * self.dV_ext * ((self.T_a_ext_in - self.T0) - self.T0 * math.log(self.T_a_ext_in / self.T0))
        self.X_a_ext_out = c_a * rho_a * self.dV_ext * ((self.T_a_ext_out - self.T0) - self.T0 * math.log(self.T_a_ext_out / self.T0))

        self.X_r_int   = self.Q_r_int * (1 - self.T0 / self.T_r_int)
        self.X_r_ext   = - self.Q_r_ext * (1 - self.T0 / self.T_r_ext)

        # Internal unit of ASHP
        self.X_in_int = self.E_fan_int + self.X_r_int
        self.X_out_int = self.X_a_int_out - self.X_a_int_in
        self.X_c_int = self.E_fan_int + self.X_r_int - (self.X_a_int_out - self.X_a_int_in)

        # Refrigerant loop of ASHP
        self.X_in_r = self.E_cmp
        self.X_out_r = self.X_r_int + self.X_r_ext
        self.X_c_r = self.E_cmp - (self.X_r_int + self.X_r_ext)

        # External unit of ASHP
        self.X_in_ext = self.E_fan_ext + self.X_r_ext
        self.X_out_ext = self.X_a_ext_out - self.X_a_ext_in
        self.X_c_ext = self.E_fan_ext + self.X_r_ext - (self.X_a_ext_out - self.X_a_ext_in)
        
        # Total exergy of ASHP
        self.X_in  = self.E_fan_int + self.E_cmp + self.E_fan_ext
        self.X_out = self.X_a_int_out - self.X_a_int_in
        self.X_c   = self.X_in - self.X_out
        
        self.X_eff = self.X_out/self.X_in
        
        ## Exergy Balance ========================================
        self.exergy_balance = {}

        # Internal Unit of ASHP
        self.exergy_balance["internal unit"] = {
            "in": {
            "$E_{f,int}$": self.E_fan_int,
            "$X_{r,int}$": self.X_r_int,
            },
            "con": {
            "$X_{c,int}$": self.X_c_int,
            },
            "out": {
            "$X_{a,int,out}$": self.X_a_int_out,
            "$X_{a,int,in}$": self.X_a_int_in,
            }
        }
        
        # Refrigerant loop of ASHP
        self.exergy_balance["refrigerant loop"] = {
            "in": {
            "$E_{cmp}$": self.E_cmp,
            },
            "con": {
            "$X_{c,r}$": self.X_c_r,
            },
            "out": {
            "$X_{r,int}$": self.X_r_int,
            "$X_{r,ext}$": self.X_r_ext,
            }
        }

        # External Unit of ASHP
        self.exergy_balance["external unit"] = {
            "in": {
            "$E_{f,ext}$": self.E_fan_ext,
            "$X_{r,ext}$": self.X_r_ext,
            },
            "con": {
            "$X_{c,ext}$": self.X_c_ext,
            },
            "out": {
            "$X_{a,ext,out}$": self.X_a_ext_out,
            "$X_{a,ext,in}$": self.X_a_ext_in,
            }
        }

#%%
# class - GroundSourceHeatPump
@dataclass
class GroundSourceHeatPump_cooling:

    def __post_init__(self):
        # Time
        self.time = 10 # [h]
        
        # Borehole parameters
        self.D_b = 0 # Borehole depth [m]
        self.H_b = 200 # Borehole height [m]
        self.r_b = 0.08 # Borehole radius [m]
        self.R_b = 0.108 # Effective borehole thermal resistance [mK/W]

        # Fluid parameters
        self.dV_f = 24 # Volumetric flow rate of fluid [L/min]

        # Ground parameters
        self.k_g = 2.0 # Ground thermal conductivity [W/mK]
        self.c_g = 800 # Ground specific heat capacity [J/(kgK)]
        self.rho_g = 2000 # Ground density [kg/m³]

        # Pump power of ground heat exchanger
        self.E_pmp = 200 # Pump power input [W]

        # Fan
        self.fan_int = Fan().fan1     

        # Temperature
        self.dT_r_exch = 5  # 예시: 열교환기의 온도 - 열교환후 지중순환수 온도 [K]
        self.T0 = 32 # environmental temperature [°C]
        self.T_g = 15 # initial ground temperature [°C]
        self.T_a_room = 20 # room air temperature [°C]
        self.T_r_exch = 25 # heat exchanger side refrigerant temperature [°C]
        
        self.T_r_int     = self.T_a_room - 10 # internal unit refrigerant temperature [°C]
        self.T_a_int_out = self.T_a_room - 5 # internal unit air outlet temperature [°C]
        # Load
        self.Q_r_int = 6000 # W
    
    def system_update(self):
        
        # Unit conversion
        self.dV_f = self.dV_f / 60 / 1000 # L/min to m³/s
        
        self.time = self.time * cu.h2s  # Convert hours to seconds
        
        self.T0 = cu.C2K(self.T0)
        self.T_a_room = cu.C2K(self.T_a_room)
        self.T_a_int_out = cu.C2K(self.T_a_int_out)
        self.T_r_int = cu.C2K(self.T_r_int)
        self.T_g = cu.C2K(self.T_g)
        
        # Others
        self.alpha = self.k_g / (self.c_g * self.rho_g) # thermal diffusivity of ground [m²/s]
        self.Lx = 2*self.dV_f/(math.pi*self.alpha)
        self.x0 = self.H_b / self.Lx # dimensionless borehole depth
        self.k_sb = self.k_g/k_w # ratio of ground thermal conductivity
        
        # 반복 수치해법 적용
        '''
        반복 수치해법을 사용하는 이유:
        1. 냉매 온도(T_r_exch)와 유체 입구 온도(T_f_in)가 서로 연동되어 직접 계산이 불가능함.
        2. 보어홀 열저항, 유량, 토양물성 등 시스템 파라미터가 COP, 온도, 효율에 반영되도록 하기 위함.
        3. 두 온도가 수렴할 때까지 반복 계산하여 물리적으로 일관된 해를 얻기 위함.
        '''
        max_iter = 20
        tol = 1e-3
        self.T_f = self.T_g  # 초기값
        self.T_f_in = self.T_f + self.dT_r_exch  # 초기값, 열교환기에서의 순환수 유입 온도

        for _ in range(max_iter):
            self.T_r_exch = self.T_f_in + self.dT_r_exch  # 5 K 높게 설정
            self.COP = calculate_GSHP_COP(Tg = self.T_g,
                                         T_cond = self.T_r_exch,
                                         T_evap = self.T_r_int,
                                         theta_hat = 0.3)
            self.E_cmp = self.Q_r_int / self.COP # compressor power input [W]
            self.Q_r_exch = self.Q_r_int + self.E_cmp
            self.Q_bh = (self.Q_r_exch + self.E_pmp) / self.H_b 
            T_f_in_old = self.T_f_in
            self.g_i = G_FLS(t = self.time, ks = self.k_g, as_ = self.alpha, rb = self.r_b, H = self.H_b) # g-function [mK/W]
            self.T_b = self.T_g + self.Q_bh * self.g_i # borehole wall temperature [K]
            self.T_f = self.T_b + self.Q_bh * self.R_b
            self.T_f_in = self.T_f + self.Q_bh * self.H_b / (2 * c_w * rho_w * self.dV_f) # fluid inlet temperature [K]
            self.T_f_out = self.T_f - self.Q_bh * self.H_b / (2 * c_w * rho_w * self.dV_f) # fluid outlet temperature [K]
            if abs(self.T_f_in - T_f_in_old) < tol:
                break
        
        # Temperature
        self.T_a_int_in = self.T_a_room # internal unit air inlet temperature [K]

        # Internal unit
        self.dV_int = self.Q_r_int / (c_a * rho_a * (abs(self.T_a_int_out - self.T_a_int_in))) # volumetric flow rate of internal unit [m3/s]
            
        # Fan power
        self.E_fan_int = Fan().get_power(self.fan_int, self.dV_int) # power input of internal unit fan [W]

        # Exergy result
        self.X_a_int_in  = c_a * rho_a * self.dV_int * ((self.T_a_int_in - self.T0) - self.T0 * math.log(self.T_a_int_in / self.T0))
        self.X_a_int_out = c_a * rho_a * self.dV_int * ((self.T_a_int_out - self.T0) - self.T0 * math.log(self.T_a_int_out / self.T0))

        self.X_r_int  = - self.Q_r_int * (1 - self.T0 / self.T_r_int)
        self.X_r_exch = - self.Q_r_exch * (1 - self.T0 / self.T_r_exch)

        self.X_f_in = c_w * rho_w * self.dV_f * ((self.T_f_in - self.T0) - self.T0 * math.log(self.T_f_in / self.T0))
        self.X_f_out = c_w * rho_w * self.dV_f * ((self.T_f_out - self.T0) - self.T0 * math.log(self.T_f_out / self.T0))

        self.X_g = (1 - self.T0 / self.T_g) * (- self.Q_bh * self.H_b)
        self.X_b = (1 - self.T0 / self.T_b) * (- self.Q_bh * self.H_b)

        # Ground
        self.X_in_g = self.X_g
        self.X_out_g = self.X_b
        self.X_c_g = self.X_in_g - self.X_out_g

        # Ground heat exchanger
        self.X_in_GHE = self.E_pmp + self.X_out_g + self.X_f_in
        self.X_out_GHE = self.X_f_out 
        self.X_c_GHE = self.X_in_GHE - self.X_out_GHE

        # Heat exchanger
        self.X_in_exch = self.X_out_GHE 
        self.X_out_exch = self.X_r_exch + self.X_f_in
        self.X_c_exch = self.X_in_exch - self.X_out_exch

        # Closed refrigerant loop system
        self.X_in_r = self.E_cmp + self.X_r_exch
        self.X_out_r = self.X_r_int
        self.X_c_r = self.X_in_r - self.X_out_r

        # Internal unit
        self.X_in_int = self.E_fan_int + self.X_r_int + self.X_a_int_in
        self.X_out_int = self.X_a_int_out
        self.X_c_int = self.X_in_int - self.X_out_int

        # Exergy efficiency
        self.X_eff = (self.X_a_int_out - self.X_a_int_in) / (self.E_fan_int + self.E_cmp + self.E_pmp)

        ## Exergy Balance ========================================
        self.exergy_balance = {}

        # Internal Unit
        self.exergy_balance["internal unit"] = {
            "in": {
                "$X_{f,int}$": self.E_fan_int,
                "$X_{r,int}$": self.X_r_int,
                "$X_{a,int,in}$": self.X_a_int_in,
            },
            "con": {
                "$X_{c,int}$": self.X_c_int,
            },
            "out": {
                "$X_{a,int,out}$": self.X_a_int_out,
            }
        }

        # Refrigerant loop
        self.exergy_balance["refrigerant loop"] = {
            "in": {
                "$X_{cmp}$": self.E_cmp,
                "$X_{r,exch}$": self.X_r_exch,
            },
            "con": {
                "$X_{c,r}$": self.X_c_r,
            },
            "out": {
                "$X_{r,int}$": self.X_r_int,
            }
        }

        # Heat Exchanger
        self.exergy_balance["heat exchanger"] = {
            "in": {
                "$X_{f,out}$": self.X_f_out,
            },
            "con": {
                "$X_{c,exch}$": self.X_c_exch,
            },
            "out": {
                "$X_{r,exch}$": self.X_r_exch,
                "$X_{f,in}$": self.X_f_in,
            }
        }

        # Ground Heat Exchanger
        self.exergy_balance["ground heat exchanger"] = {
            "in": {
                "$E_{pmp}$": self.E_pmp,
                "$X_{b}$": self.X_b,
                "$X_{f,in}$": self.X_f_in,
            },
            "con": {
                "$X_{c,GHE}$": self.X_c_GHE,
            },
            "out": {
                "$X_{f,out}$": self.X_f_out,
            }
        }

        # Ground
        self.exergy_balance["ground"] = {
            "in": {
                "$X_{g}$": self.X_g,
            },
            "con": {
                "$X_{c,g}$": self.X_c_g,
            },
            "out": {
                "$X_{b}$": self.X_b,
            }
        }

@dataclass
class GroundSourceHeatPump_heating:
    def __post_init__(self):
        # Time
        self.time = 10 # [h]
        
        # Borehole parameters
        self.D_b = 0 # Borehole depth [m]
        self.H_b = 200 # Borehole height [m]
        self.r_b = 0.08 # Borehole radius [m]
        self.R_b = 0.108 # Effective borehole thermal resistance [mK/W]

        # Fluid parameters
        self.dV_f = 24 # Volumetric flow rate of fluid [L/min]

        # Ground parameters
        self.k_g = 2.0 # Ground thermal conductivity [W/mK]
        self.c_g = 800 # Ground specific heat capacity [J/(kgK)]
        self.rho_g = 2000 # Ground density [kg/m³]

        # Pump power of ground heat exchanger
        self.E_pmp = 200 # Pump power input [W]

        # Fan
        self.fan_int = Fan().fan1

        # Temperature
        self.dT_r_exch = -5  # 예시: 열교환기 측 냉매 온도 - 열교환후 지중순환수 온도 [K]
        self.T0 = 0 # environmental temperature [°C]
        self.T_g = 15 # initial ground temperature [°C]
        self.T_a_room = 20 # room air temperature [°C]
        self.T_r_exch = 5 # heat exchanger side refrigerant temperature [°C]
        
        self.T_r_int = self.T_a_room + 15 # internal unit refrigerant temperature [°C]
        self.T_a_int_out = self.T_a_room + 10 # internal unit air outlet temperature [°C]

        # Load
        self.Q_r_int = 6000 # W
        
    def system_update(self):
        # Unit conversion
        self.time = self.time * cu.h2s  # Convert hours to seconds
        self.dV_f = self.dV_f / 60 / 1000 # L/min to m³/s

        # Celcius to Kelvin
        self.T0 = cu.C2K(self.T0)
        self.T_a_room = cu.C2K(self.T_a_room)
        self.T_a_int_out = cu.C2K(self.T_a_int_out)
        self.T_r_int = cu.C2K(self.T_r_int)
        self.T_g = cu.C2K(self.T_g)
        
        # Others
        self.alpha = self.k_g / (self.c_g * self.rho_g) # thermal diffusivity of ground [m²/s]
        
        # 반복 수치해법 적용
        '''
        반복 수치해법을 사용하는 이유:
        1. 냉매 온도(T_r_exch)와 유체 입구 온도(T_f_in)가 서로 연동되어 직접 계산이 불가능함.
        2. 보어홀 열저항, 유량, 토양물성 등 시스템 파라미터가 COP, 온도, 효율에 반영되도록 하기 위함.
        3. 두 온도가 수렴할 때까지 반복 계산하여 물리적으로 일관된 해를 얻기 위함.
        '''
        max_iter = 20
        tol = 1e-3
        self.T_f = self.T_g  # 초기값
        self.T_f_in = self.T_f + self.dT_r_exch  # 초기값, 열교환기에서의 순환수 유입 온도

        for _ in range(max_iter):
            self.T_r_exch = self.T_f_in + self.dT_r_exch  # 5 K 높게 설정
            self.COP = calculate_GSHP_COP(Tg = self.T_g,
                                         T_cond = self.T_r_int,
                                         T_evap = self.T_r_exch,
                                         theta_hat = 0.3)
            # Others
            self.E_cmp = self.Q_r_int / self.COP # compressor power input [W]
            self.Q_r_exch = self.Q_r_int - self.E_cmp  # changed from Q_r_ext to Q_r_exch
            # Borehole 
            self.Q_bh = (self.Q_r_exch - self.E_pmp) / self.H_b # heat flow rate from borehole to ground per unit length [W/m]
            self.g_i = G_FLS(t = self.time, ks = self.k_g, as_ = self.alpha, rb = self.r_b, H = self.H_b) # g-function [mK/W]
            # fluid temperature & borehole wall temperature [K]
            T_f_in_old = self.T_f_in  # 이전 유체 입구 온도 저장
            self.T_b = self.T_g - self.Q_bh * self.g_i # borehole wall temperature [K]
            self.T_f = self.T_b - self.Q_bh * self.R_b # fluid temperature in borehole [K]
            self.T_f_in = self.T_f - self.Q_bh * self.H_b / (2 * c_w * rho_w * self.dV_f) # fluid inlet temperature [K]
            self.T_f_out = self.T_f + self.Q_bh * self.H_b / (2 * c_w * rho_w * self.dV_f) # fluid outlet temperature [K]
            if abs(self.T_f_in - T_f_in_old) < tol:
                break
        
        # Temperature
        self.T_a_int_in = self.T_a_room # internal unit air inlet temperature [K]

        # Internal unit
        self.dV_int = self.Q_r_int / (c_a * rho_a * (abs(self.T_a_int_out - self.T_a_int_in))) # volumetric flow rate of internal unit [m3/s]
            
        # Fan power
        self.E_fan_int = Fan().get_power(self.fan_int, self.dV_int) # power input of internal unit fan [W]

        # Exergy result
        self.X_a_int_in  = c_a * rho_a * self.dV_int * ((self.T_a_int_in - self.T0) - self.T0 * math.log(self.T_a_int_in / self.T0))
        self.X_a_int_out = c_a * rho_a * self.dV_int * ((self.T_a_int_out - self.T0) - self.T0 * math.log(self.T_a_int_out / self.T0))

        self.X_r_int   = self.Q_r_int * (1 - self.T0 / self.T_r_int)
        self.X_r_exch   = self.Q_r_exch * (1 - self.T0 / self.T_r_exch)

        self.X_f_in = c_w * rho_w * self.dV_f * ((self.T_f_in - self.T0) - self.T0 * math.log(self.T_f_in / self.T0))
        self.X_f_out = c_w * rho_w * self.dV_f * ((self.T_f_out - self.T0) - self.T0 * math.log(self.T_f_out / self.T0))

        self.X_g = (1 - self.T0 / self.T_g) * (self.Q_bh * self.H_b)
        self.X_b = (1 - self.T0 / self.T_b) * (self.Q_bh * self.H_b)

        # Internal unit
        self.X_in_int = self.E_fan_int + self.X_r_int + self.X_a_int_in
        self.X_out_int = self.X_a_int_out
        self.X_c_int = self.X_in_int - self.X_out_int

        # Closed refrigerant loop system
        self.X_in_r = self.E_cmp + self.X_r_exch
        self.X_out_r = self.X_r_int
        self.X_c_r = self.X_in_r - self.X_out_r

        # Heat exchanger
        self.X_in_exch = self.X_f_out
        self.X_out_exch = self.X_r_exch + self.X_f_in
        self.X_c_exch = self.X_in_exch - self.X_out_exch

        # Ground heat exchanger
        self.X_in_GHE = self.E_pmp + self.X_b + self.X_f_in
        self.X_out_GHE = self.X_f_out 
        self.X_c_GHE = self.X_in_GHE - self.X_out_GHE

        # Ground
        self.X_in_g = self.X_g
        self.X_out_g = self.X_b
        self.X_c_g = self.X_in_g - self.X_out_g
        
        # Exergy efficiency
        self.X_eff = (self.X_a_int_out - self.X_a_int_in) / (self.E_fan_int + self.E_cmp + self.E_pmp)

        ## Exergy Balance ========================================
        self.exergy_balance = {}

        # Internal Unit
        self.exergy_balance["internal unit"] = {
            "in": {
                "$X_{f,int}$": self.E_fan_int,
                "$X_{r,int}$": self.X_r_int,
                "$X_{a,int,in}$": self.X_a_int_in,
            },
            "con": {
                "$X_{c,int}$": self.X_c_int,
            },
            "out": {
                "$X_{a,int,out}$": self.X_a_int_out,
            }
        }

        # Refrigerant loop
        self.exergy_balance["refrigerant loop"] = {
            "in": {
                "$X_{cmp}$": self.E_cmp,
                "$X_{r,exch}$": self.X_r_exch,
            },
            "con": {
                "$X_{c,r}$": self.X_c_r,
            },
            "out": {
                "$X_{r,int}$": self.X_r_int,
            }
        }

        # Heat Exchanger
        self.exergy_balance["heat exchanger"] = {
            "in": {
                "$X_{f,out}$": self.X_f_out,
            },
            "con": {
                "$X_{c,exch}$": self.X_c_exch,
            },
            "out": {
                "$X_{r,exch}$": self.X_r_exch,
                "$X_{f,in}$": self.X_f_in,
            }
        }

        # Ground Heat Exchanger
        self.exergy_balance["ground heat exchanger"] = {
            "in": {
                "$E_{pmp}$": self.E_pmp,
                "$X_{b}$": self.X_b,
                "$X_{f,in}$": self.X_f_in,
            },
            "con": {
                "$X_{c,GHE}$": self.X_c_GHE,
            },
            "out": {
                "$X_{f,out}$": self.X_f_out,
            }
        }

        # Ground
        self.exergy_balance["ground"] = {
            "in": {
                "$X_{g}$": self.X_g,
            },
            "con": {
                "$X_{c,g}$": self.X_c_g,
            },
            "out": {
                "$X_{b}$": self.X_b,
            }
        }
 #%% 
# class - Electric heater
@dataclass
class ElectricHeater:

    def __post_init__(self): 
        
        # hb: heater body
        # hs: heater surface
        # ms: room surface
        
        # Heater material properties (냉간압연 탄소강판 SPCC)
        self.c   = 500 # [J/kgK]
        self.rho = 7800 # [kg/m3]
        self.k   = 50 # [W/mK]
    
        # Heater geometry [m]
        self.D = 0.005 
        self.H = 0.8 
        self.W = 1.0
        
        # Electricity input to the heater [W]
        self.E_heater = 1000
        
        # Temperature [°C]
        self.T0   = 0
        self.T_mr = 15
        self.T_init = 20 # Initial temperature of the heater [°C]
        self.T_a_room = 20 # Indoor air temperature [°C]
        
        # Emissivity [-]
        self.epsilon_hs = 1 # hs: heater surface
        self.epsilon_rs = 1 # rs: room surface
        
        # Time step [s]
        self.dt = 10
    
    def system_update(self):
        
        # Temperature [K]
        self.T0     = cu.C2K(self.T0) # 두번 system update를 할 경우 절대온도 변환 중첩됨
        self.T_mr   = cu.C2K(self.T_mr)
        self.T_a_room   = cu.C2K(self.T_a_room)
        self.T_init = cu.C2K(self.T_init)
        self.T_hb   = self.T_init # hb: heater body
        self.T_hs   = self.T_init # hs: heater surface
        
        # Heater material properties
        self.C = self.c * self.rho
        self.A = self.H * self.W * 2 # double side 
        self.V = self.H * self.W * self.D
        
        # Conductance [W/m²K]
        self.K_cond = self.k / (self.D / 2)
        
        # Iterative calculation
        self.time = []
        self.T_hb_list = []
        self.T_hs_list = []
        
        self.E_heater_list = []
        self.Q_st_list = []
        self.Q_cond_list = []
        self.Q_conv_list = []
        self.Q_rad_hs_list = []
        self.Q_rad_rs_list = []
        
        self.S_st_list = []
        self.S_heater_list = []
        self.S_cond_list = []
        self.S_conv_list = []
        self.S_rad_rs_list = []
        self.S_rad_hs_list = []
        self.S_g_hb_list = []
        self.S_g_hs_list = []
        
        self.X_st_list = [] 
        self.X_heater_list = []
        self.X_cond_list = []
        self.X_conv_list = []
        self.X_rad_rs_list = []
        self.X_rad_hs_list = []
        self.X_c_hb_list = []
        self.X_c_hs_list = []
        
        index = 0
        tolerance = 1e-8
        while True:
            self.time.append(index * self.dt)
            
            # Heat transfer coefficient [W/m²K]
            self.h_cp = calc_h_vertical_plate(self.T_hs, self.T0, self.H) 
            
            def residual_Tp(Tp_new):
                # 축열 항
                Q_st = self.rho * self.c * self.V * (Tp_new - self.T_hb) / self.dt

                # Tps 계산 (표면에너지 평형으로부터)
                Tps = (
                    self.K_cond * Tp_new
                    + self.h_cp * self.T_a_room
                    + self.epsilon_hs * self.epsilon_rs * sigma * (self.T_mr**4 - self.T0**4)
                    - self.epsilon_hs * self.epsilon_rs * sigma * (Tp_new**4 - self.T0**4)
                ) / (self.K_cond + self.h_cp)

                # 전도열
                Q_cond = self.A * self.K_cond * (Tp_new - Tps)

                return Q_st + Q_cond - self.E_heater
            
            self.T_hb_guess = self.T_hb # 초기 추정값
            
            from scipy.optimize import fsolve
            self.T_hb_next = fsolve(residual_Tp, self.T_hb_guess)[0]
            self.T_hb_old = self.T_hb
            
            # Temperature update
            self.T_hb = self.T_hb_next
            
            # T_hs update (Energy balance surface: Q_cond + Q_rad_rs = Q_conv + Q_rad_hs)
            self.T_hs = (
                self.K_cond * self.T_hb
                + self.h_cp * self.T_a_room
                + self.epsilon_hs * self.epsilon_rs * sigma * (self.T_mr ** 4 - self.T0 ** 4)
                - self.epsilon_hs * self.epsilon_rs * sigma * (self.T_hb ** 4 - self.T0 ** 4)
            ) / (self.K_cond + self.h_cp)
            
            # Temperature [K]
            self.T_hb_list.append(self.T_hb)
            self.T_hs_list.append(self.T_hs)
            
            # Conduction [W]
            self.Q_st = self.C * self.V * (self.T_hb_next - self.T_hb_old) / self.dt
            self.Q_cond = self.A * self.K_cond * (self.T_hb - self.T_hs)
            self.Q_conv = self.A * self.h_cp * (self.T_hs - self.T_a_room) # h_cp 추후 변하게
            self.Q_rad_rs = self.A * self.epsilon_hs * self.epsilon_rs * sigma * (self.T_mr ** 4 - self.T0 ** 4)
            self.Q_rad_hs = self.A * self.epsilon_hs * self.epsilon_rs * sigma * (self.T_hb ** 4 - self.T0 ** 4)
            
            self.E_heater_list.append(self.E_heater)
            self.Q_st_list.append(self.Q_st)
            self.Q_cond_list.append(self.Q_cond)
            self.Q_conv_list.append(self.Q_conv)
            self.Q_rad_hs_list.append(self.Q_rad_hs)
            self.Q_rad_rs_list.append(self.Q_rad_rs)
            
            # Entropy balance
            self.S_st = (1/self.T_hb) * (self.Q_st)
            self.S_heater = (1/float('inf')) * (self.E_heater)
            self.S_cond = (1/self.T_hb) * (self.Q_cond)
            self.S_conv = (1/self.T_hs) * (self.Q_conv)
            self.S_rad_rs  = 4/3 * self.A * self.epsilon_hs * self.epsilon_rs * sigma * (self.T_mr ** 3 - self.T0 ** 3)
            self.S_rad_hs  = 4/3 * self.A * self.epsilon_hs * self.epsilon_rs * sigma * (self.T_hb ** 3 - self.T0 ** 3)
            self.S_g_hb = self.S_st + self.S_conv - self.S_heater     
            self.S_g_hs = self.S_rad_hs + self.S_conv - self.S_cond - self.S_rad_rs

            self.S_st_list.append(self.S_st)
            self.S_heater_list.append(self.S_heater)
            self.S_cond_list.append(self.S_cond)
            self.S_conv_list.append(self.S_conv)
            self.S_rad_rs_list.append(self.S_rad_rs)
            self.S_rad_hs_list.append(self.S_rad_hs)
            self.S_g_hb_list.append(self.S_g_hb)
            self.S_g_hs_list.append(self.S_g_hs)
            
            # Exergy balance
            self.X_st = (1 - self.T0 / self.T_hb) * (self.Q_st)
            self.X_heater = (1 - self.T0 / float('inf')) * (self.E_heater)
            self.X_cond = (1 - self.T0 / self.T_hb) * (self.Q_cond)
            
            ###########################
            # self.X_conv = (1 - self.T0 / self.T_hs) * (self.Q_conv) # h_cp 추후 변하게
            self.X_conv = (1 - self.T0 / ((self.T_hs+self.T0)/2)) * (self.Q_conv) # 임시 변경 사항있으니 주의 필요 -----------------------------
            ############################
            
            self.X_rad_rs = self.Q_rad_rs - self.T0 * self.S_rad_rs
            self.X_rad_hs = self.Q_rad_hs - self.T0 * self.S_rad_hs
            self.X_c_hb = -(self.X_st + self.X_cond - self.X_heater)
            self.X_c_hs = -(self.X_rad_hs + self.X_conv - self.X_cond - self.X_rad_rs)
            
            self.X_st_list.append(self.X_st)
            self.X_heater_list.append(self.X_heater)
            self.X_cond_list.append(self.X_cond)
            self.X_conv_list.append(self.X_conv)
            self.X_rad_rs_list.append(self.X_rad_rs)
            self.X_rad_hs_list.append(self.X_rad_hs)
            self.X_c_hb_list.append(self.X_c_hb)
            self.X_c_hs_list.append(self.X_c_hs)
            
            index += 1
            T_hb_rel_change = abs(self.T_hb_next - self.T_hb_old) / max(abs(self.T_hb_next), 1e-8)
            if T_hb_rel_change < tolerance:
                break
            
            if index > 10000:
                print("time step is too short")
                break
        
        self.X_eff = (self.X_rad_hs + self.X_conv)/ self.X_heater 
        self.energy_balance = {}
        self.energy_balance["heater body"] = {
            "in": {
                "E_heater": self.E_heater,
            },
            "out": {
                "Q_st": self.Q_st,
                "Q_cond": self.Q_cond
            }
        }

        self.energy_balance["heater surface"] = {
            "in": {
                "Q_cond": self.Q_cond,
                "Q_rad_rs": self.Q_rad_rs,
            },
            "out": {
                "Q_conv": self.Q_conv,
                "Q_rad_hs": self.Q_rad_hs
            }
        }
        
        self.entropy_balance = {}
        self.entropy_balance["heater body"] = {
            "in": {
                "S_heater": self.S_heater,
            },
            "gen": {
                "S_g_hb": self.S_g_hb,
            },
            "out": {
                "S_st": self.S_st,
                "S_cond": self.S_cond
            }
        }

        self.entropy_balance["heater surface"] = {
            "in": {
                "S_cond":   self.S_cond,
                "S_rad_rs": self.S_rad_rs,
            },
            "gen": {
                "S_g_hs": self.S_g_hs,
            },
            "out": {
                "S_conv":   self.S_conv,
                "S_rad_hs": self.S_rad_hs
            }
        }
        
        self.exergy_balance = {}
        self.exergy_balance["heater body"] = {
            "in": {
                "X_heater": self.X_heater,
            },
            "con": {
                "X_c_hb": self.X_c_hb,
            },
            "out": {
                "X_st": self.X_st,
                "X_cond": self.X_cond
            }
        }

        self.exergy_balance["heater surface"] = {
            "in": {
                "X_cond":   self.X_cond,
                "X_rad_rs": self.X_rad_rs,
            },
            "con": {
                "X_c_hs": self.X_c_hs,
            },
            "out": {
                "X_conv":   self.X_conv,
                "X_rad_hs": self.X_rad_hs
            }
        }
        