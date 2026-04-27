"""
Energy, Entropy, and Exergy Analysis Engine.

This module contains classes for modeling various energy systems including:
- Domestic hot water systems (electric boiler, gas boiler, heat pump boiler)
- Air source heat pumps (cooling and heating modes)
- Ground source heat pumps (cooling and heating modes)
- Solar-assisted systems
- Electric heaters
"""

import math
from dataclasses import dataclass

import numpy as np

from . import calc_util as cu
from . import g_function as gf
from .air_source_heat_pump_boiler import AirSourceHeatPumpBoiler
from .components.fan import Fan
from .constants import c_a, c_w as c_f, k_w, rho_a, rho_w as rho_f
from .enex_functions import (
    calc_GSHP_COP,
    calc_fan_power_from_dV_fan,
)

# Phase 3 Refactoring: Re-exporting standalone boiler modules for backward compatibility

# Alias for backward compatibility since HeatPumpBoiler was the original name
HeatPumpBoiler = AirSourceHeatPumpBoiler

# class - Fan & Pump


# class - Domestic Hot Water System
# class - AirSourceHeatPump
@dataclass
class AirSourceHeatPump_cooling:
    """Air source heat pump model for cooling mode.

    Simulates a single-step refrigerant cycle with indoor/outdoor
    heat exchangers and fans.  Call ``system_update()`` after
    setting operating conditions to compute COP, capacities, and
    component powers.
    """

    def __post_init__(self):
        # fan
        self.fan_int = Fan().fan1
        self.fan_ext = Fan().fan2

        # COP
        self.Q_r_max = 20000  # [W]

        # temperature
        self.T0 = 32  # environmental temperature [°C]
        self.T_a_room = 20  # room air temperature [°C]

        self.T_r_int = self.T_a_room - 15  # internal unit refrigerant temperature [°C]
        self.T_a_int_out = self.T_a_room - 10  # internal unit air outlet temperature [°C]

        self.T_a_ext_out = self.T0 + 10  # external unit air outlet temperature [°C]
        self.T_r_ext = self.T0 + 15  # external unit refrigerant temperature [°C]

        # load
        self.Q_r_int = 6000  # [W]

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
        self.T_a_int_in = self.T_a_room  # internal unit air inlet temperature [K]
        self.T_a_ext_in = self.T0  # external unit air inlet temperature [K]

        # others
        self.COP = calc_ASHP_cooling_COP(
            self.T_a_int_out,
            self.T_a_ext_in,
            self.Q_r_int,
            self.Q_r_max,
            self.COP_ref,
        )  # COP [-]
        self.E_cmp = self.Q_r_int / self.COP  # compressor power input [W]
        self.Q_r_ext = self.Q_r_int + self.E_cmp  # heat transfer from external unit to refrigerant [W]

        # internal, external unit
        self.dV_int = self.Q_r_int / (
            c_a * rho_a * (abs(self.T_a_int_out - self.T_a_int_in))
        )  # volumetric flow rate of internal unit [m3/s]
        self.dV_ext = self.Q_r_ext / (
            c_a * rho_a * (abs(self.T_a_ext_out - self.T_a_ext_in))
        )  # volumetric flow rate of external unit [m3/s]

        # fan power
        self.E_fan_int = Fan().get_power(self.fan_int, self.dV_int)  # power input of internal unit fan [W]
        self.E_fan_ext = Fan().get_power(self.fan_ext, self.dV_ext)  # power input of external unit fan [W]

        # System COP
        self.COP_sys = self.Q_r_int / (self.E_fan_int + self.E_fan_ext + self.E_cmp)

        # exergy result
        self.X_a_int_in = (
            c_a * rho_a * self.dV_int * ((self.T_a_int_in - self.T0) - self.T0 * math.log(self.T_a_int_in / self.T0))
        )
        self.X_a_int_out = (
            c_a * rho_a * self.dV_int * ((self.T_a_int_out - self.T0) - self.T0 * math.log(self.T_a_int_out / self.T0))
        )
        self.X_a_ext_in = (
            c_a * rho_a * self.dV_ext * ((self.T_a_ext_in - self.T0) - self.T0 * math.log(self.T_a_ext_in / self.T0))
        )
        self.X_a_ext_out = (
            c_a * rho_a * self.dV_ext * ((self.T_a_ext_out - self.T0) - self.T0 * math.log(self.T_a_ext_out / self.T0))
        )

        self.X_r_int = -self.Q_r_int * (1 - self.T0 / self.T_r_int)
        self.X_r_ext = self.Q_r_ext * (1 - self.T0 / self.T_r_ext)

        # Internal unit of ASHP
        self.X_in_int = self.E_fan_int + self.X_r_int
        self.X_out_int = self.X_a_int_out - self.X_a_int_in
        self.X_c_int = self.X_in_int - self.X_out_int

        # Closed refrigerant loop system of ASHP
        self.X_in_r = self.E_cmp
        self.X_out_r = self.X_r_int + self.X_r_ext
        self.X_c_r = self.X_in_r - self.X_out_r

        # External unit of ASHP
        self.X_in_ext = self.E_fan_ext + self.X_r_ext
        self.X_out_ext = self.X_a_ext_out - self.X_a_ext_in
        self.X_c_ext = self.X_in_ext - self.X_out_ext

        # Total exergy of ASHP
        self.X_in = self.E_fan_int + self.E_cmp + self.E_fan_ext
        self.X_out = self.X_a_int_out - self.X_a_int_in
        self.X_c = self.X_in - self.X_out

        self.X_eff = self.X_out / self.X_in

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
            },
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
            },
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
            },
        }


@dataclass
class AirSourceHeatPump_heating:
    """Air source heat pump model for heating mode.

    Mirror of ``AirSourceHeatPump_cooling`` configured for space
    heating.  The condenser rejects heat to the indoor side while
    the evaporator absorbs from outdoor air.
    """

    def __post_init__(self):

        # fan
        self.fan_int = Fan().fan1
        self.fan_ext = Fan().fan2

        # COP
        self.Q_r_max = 20000  # maximum heating capacity [W]

        # temperature
        self.T0 = 0  # environmental temperature [°C]
        self.T_a_room = 20  # room air temperature [°C]

        self.T_r_int = self.T_a_room + 15  # internal unit refrigerant temperature [°C]
        self.T_a_int_out = self.T_a_room + 10  # internal unit air outlet temperature [°C]

        self.T_a_ext_out = self.T0 - 10  # external unit air outlet temperature [°C]
        self.T_r_ext = self.T0 - 15  # external unit refrigerant temperature [°C]

        # load
        self.Q_r_int = 6000  # [W]

    def system_update(self):

        # Celcius to Kelvin
        self.T0 = cu.C2K(self.T0)
        self.T_a_room = cu.C2K(self.T_a_room)
        self.T_a_int_out = cu.C2K(self.T_a_int_out)
        self.T_a_ext_out = cu.C2K(self.T_a_ext_out)
        self.T_r_int = cu.C2K(self.T_r_int)
        self.T_r_ext = cu.C2K(self.T_r_ext)

        # temperature
        self.T_a_int_in = self.T_a_room
        self.T_a_ext_in = self.T0  # external unit air inlet temperature [K]

        # others
        self.COP = calc_ASHP_heating_COP(T0=self.T0, Q_r_int=self.Q_r_int, Q_r_max=self.Q_r_max)  # COP [-]
        self.E_cmp = self.Q_r_int / self.COP  # compressor power input [W]
        self.Q_r_ext = self.Q_r_int - self.E_cmp  # heat transfer from external unit to refrigerant [W]

        # internal, external unit
        self.dV_int = self.Q_r_int / (
            c_a * rho_a * abs(self.T_a_int_out - self.T_a_int_in)
        )  # volumetric flow rate of internal unit [m3/s]
        self.dV_ext = self.Q_r_ext / (
            c_a * rho_a * abs(self.T_a_ext_out - self.T_a_ext_in)
        )  # volumetric flow rate of external unit [m3/s]

        # fan power
        self.E_fan_int = Fan().get_power(self.fan_int, self.dV_int)  # power input of internal unit fan [W]
        self.E_fan_ext = Fan().get_power(self.fan_ext, self.dV_ext)  # power input of external unit fan [W]

        # System COP
        self.COP_sys = self.Q_r_int / (self.E_fan_int + self.E_fan_ext + self.E_cmp)

        # exergy result
        self.X_a_int_in = (
            c_a * rho_a * self.dV_int * ((self.T_a_int_in - self.T0) - self.T0 * math.log(self.T_a_int_in / self.T0))
        )
        self.X_a_int_out = (
            c_a * rho_a * self.dV_int * ((self.T_a_int_out - self.T0) - self.T0 * math.log(self.T_a_int_out / self.T0))
        )
        self.X_a_ext_in = (
            c_a * rho_a * self.dV_ext * ((self.T_a_ext_in - self.T0) - self.T0 * math.log(self.T_a_ext_in / self.T0))
        )
        self.X_a_ext_out = (
            c_a * rho_a * self.dV_ext * ((self.T_a_ext_out - self.T0) - self.T0 * math.log(self.T_a_ext_out / self.T0))
        )

        self.X_r_int = self.Q_r_int * (1 - self.T0 / self.T_r_int)
        self.X_r_ext = -self.Q_r_ext * (1 - self.T0 / self.T_r_ext)

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
        self.X_in = self.E_fan_int + self.E_cmp + self.E_fan_ext
        self.X_out = self.X_a_int_out - self.X_a_int_in
        self.X_c = self.X_in - self.X_out

        self.X_eff = self.X_out / self.X_in

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
            },
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
            },
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
            },
        }


# class - GroundSourceHeatPump
@dataclass
class GroundSourceHeatPump_cooling:
    """Ground source heat pump model for cooling mode.

    Uses borehole heat exchangers with finite-line-source g-functions
    for soil thermal response.  Call ``system_update()`` each time
    step to advance the ground temperature history.
    """

    def __post_init__(self):
        # Time
        self.time = 10  # [h]

        # Borehole parameters
        self.D_b = 0  # Borehole depth [m]
        self.H_b = 200  # Borehole height [m]
        self.r_b = 0.08  # Borehole radius [m]
        self.R_b = 0.108  # Effective borehole thermal resistance [mK/W]

        # Fluid parameters
        self.dV_f = 24  # Volumetric flow rate of fluid [L/min]

        # Ground parameters
        self.k_g = 2.0  # Ground thermal conductivity [W/mK]
        self.c_g = 800  # Ground specific heat capacity [J/(kgK)]
        self.rho_g = 2000  # Ground density [kg/m³]

        # Pump power of ground heat exchanger
        self.E_pmp = 200  # Pump power input [W]

        # Fan
        self.fan_int = Fan().fan1

        # Temperature
        self.dT_r_exch = 5  # 예시: 열교환기의 온도 - 열교환후 지중순환수 온도 [K]
        self.T0 = 32  # environmental temperature [°C]
        self.T_g = 15  # initial ground temperature [°C]
        self.T_a_room = 20  # room air temperature [°C]
        self.T_r_exch = 25  # heat exchanger side refrigerant temperature [°C]

        self.T_r_int = self.T_a_room - 10  # internal unit refrigerant temperature [°C]
        self.T_a_int_out = self.T_a_room - 5  # internal unit air outlet temperature [°C]
        # Load
        self.Q_r_int = 6000  # W

    def system_update(self):

        # Unit conversion
        self.dV_f = self.dV_f * cu.s2m * cu.L2m3  # L/min to m³/s

        self.time = self.time * cu.h2s  # Convert hours to seconds

        self.T0 = cu.C2K(self.T0)
        self.T_a_room = cu.C2K(self.T_a_room)
        self.T_a_int_out = cu.C2K(self.T_a_int_out)
        self.T_r_int = cu.C2K(self.T_r_int)
        self.T_g = cu.C2K(self.T_g)

        # Others
        self.alpha = self.k_g / (self.c_g * self.rho_g)  # thermal diffusivity of ground [m²/s]
        self.Lx = 2 * self.dV_f / (math.pi * self.alpha)
        self.x0 = self.H_b / self.Lx  # dimensionless borehole depth
        self.k_sb = self.k_g / k_w  # ratio of ground thermal conductivity

        # 반복 수치해법 적용
        """
        반복 수치해법을 사용하는 이유:
        1. 냉매 온도(T_r_exch)와 유체 입구 온도(T_f_in)가 서로 연동되어 직접 계산이 불가능함.
        2. 보어홀 열저항, 유량, 토양물성 등 시스템 파라미터가 COP, 온도, 효율에 반영되도록 하기 위함.
        3. 두 온도가 수렴할 때까지 반복 계산하여 물리적으로 일관된 해를 얻기 위함.
        """
        max_iter = 20
        tol = 1e-3
        self.T_f = self.T_g  # 초기값
        self.T_f_in = self.T_f + self.dT_r_exch  # 초기값, 열교환기에서의 순환수 유입 온도

        for _ in range(max_iter):
            self.T_r_exch = self.T_f_in + self.dT_r_exch  # 5 K 높게 설정
            self.COP = calc_GSHP_COP(
                Tg=self.T_g,
                T_cond=self.T_r_exch,
                T_evap=self.T_r_int,
                theta_hat=0.3,
            )
            self.E_cmp = self.Q_r_int / self.COP  # compressor power input [W]
            self.Q_r_exch = self.Q_r_int + self.E_cmp
            self.Q_bh = (self.Q_r_exch + self.E_pmp) / self.H_b
            T_f_in_old = self.T_f_in
            self.g_i = G_FLS(
                t=self.time,
                ks=self.k_g,
                as_=self.alpha,
                rb=self.r_b,
                H=self.H_b,
            )  # g-function [mK/W]
            self.T_b = self.T_g + self.Q_bh * self.g_i  # borehole wall temperature [K]
            self.T_f = self.T_b + self.Q_bh * self.R_b
            self.T_f_in = self.T_f + self.Q_bh * self.H_b / (2 * c_w * rho_w * self.dV_f)  # fluid inlet temperature [K]
            self.T_f_out = self.T_f - self.Q_bh * self.H_b / (
                2 * c_w * rho_w * self.dV_f
            )  # fluid outlet temperature [K]
            if abs(self.T_f_in - T_f_in_old) < tol:
                break

        # Temperature
        self.T_a_int_in = self.T_a_room  # internal unit air inlet temperature [K]

        # Internal unit
        self.dV_int = self.Q_r_int / (
            c_a * rho_a * (abs(self.T_a_int_out - self.T_a_int_in))
        )  # volumetric flow rate of internal unit [m3/s]

        # Fan power
        self.E_fan_int = Fan().get_power(self.fan_int, self.dV_int)  # power input of internal unit fan [W]

        # Exergy result
        self.X_a_int_in = (
            c_a * rho_a * self.dV_int * ((self.T_a_int_in - self.T0) - self.T0 * math.log(self.T_a_int_in / self.T0))
        )
        self.X_a_int_out = (
            c_a * rho_a * self.dV_int * ((self.T_a_int_out - self.T0) - self.T0 * math.log(self.T_a_int_out / self.T0))
        )

        self.X_r_int = -self.Q_r_int * (1 - self.T0 / self.T_r_int)
        self.X_r_exch = -self.Q_r_exch * (1 - self.T0 / self.T_r_exch)

        self.X_f_in = c_w * rho_w * self.dV_f * ((self.T_f_in - self.T0) - self.T0 * math.log(self.T_f_in / self.T0))
        self.X_f_out = c_w * rho_w * self.dV_f * ((self.T_f_out - self.T0) - self.T0 * math.log(self.T_f_out / self.T0))

        self.X_g = (1 - self.T0 / self.T_g) * (-self.Q_bh * self.H_b)
        self.X_b = (1 - self.T0 / self.T_b) * (-self.Q_bh * self.H_b)

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
            },
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
            },
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
            },
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
            },
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
            },
        }


@dataclass
class GroundSourceHeatPump_heating:
    """Ground source heat pump model for heating mode.

    Mirror of ``GroundSourceHeatPump_cooling`` configured for space
    heating.  The evaporator absorbs heat from the ground loop
    while the condenser supplies heat indoors.
    """

    def __post_init__(self):
        # Time
        self.time = 10  # [h]

        # Borehole parameters
        self.D_b = 0  # Borehole depth [m]
        self.H_b = 200  # Borehole height [m]
        self.r_b = 0.08  # Borehole radius [m]
        self.R_b = 0.108  # Effective borehole thermal resistance [mK/W]

        # Fluid parameters
        self.dV_f = 24  # Volumetric flow rate of fluid [L/min]

        # Ground parameters
        self.k_g = 2.0  # Ground thermal conductivity [W/mK]
        self.c_g = 800  # Ground specific heat capacity [J/(kgK)]
        self.rho_g = 2000  # Ground density [kg/m³]

        # Pump power of ground heat exchanger
        self.E_pmp = 200  # Pump power input [W]

        # Fan
        self.fan_int = Fan().fan1

        # Temperature
        self.dT_r_exch = -5  # 예시: 열교환기 측 냉매 온도 - 열교환후 지중순환수 온도 [K]
        self.T0 = 0  # environmental temperature [°C]
        self.T_g = 15  # initial ground temperature [°C]
        self.T_a_room = 20  # room air temperature [°C]
        self.T_r_exch = 5  # heat exchanger side refrigerant temperature [°C]

        self.T_r_int = self.T_a_room + 15  # internal unit refrigerant temperature [°C]
        self.T_a_int_out = self.T_a_room + 10  # internal unit air outlet temperature [°C]

        # Load
        self.Q_r_int = 6000  # W

    def system_update(self):
        # Unit conversion
        self.time = self.time * cu.h2s  # Convert hours to seconds
        self.dV_f = self.dV_f * cu.s2m * cu.L2m3  # L/min to m³/s

        # Celcius to Kelvin
        self.T0 = cu.C2K(self.T0)
        self.T_a_room = cu.C2K(self.T_a_room)
        self.T_a_int_out = cu.C2K(self.T_a_int_out)
        self.T_r_int = cu.C2K(self.T_r_int)
        self.T_g = cu.C2K(self.T_g)

        # Others
        self.alpha = self.k_g / (self.c_g * self.rho_g)  # thermal diffusivity of ground [m²/s]

        # 반복 수치해법 적용
        """
        반복 수치해법을 사용하는 이유:
        1. 냉매 온도(T_r_exch)와 유체 입구 온도(T_f_in)가 서로 연동되어 직접 계산이 불가능함.
        2. 보어홀 열저항, 유량, 토양물성 등 시스템 파라미터가 COP, 온도, 효율에 반영되도록 하기 위함.
        3. 두 온도가 수렴할 때까지 반복 계산하여 물리적으로 일관된 해를 얻기 위함.
        """
        max_iter = 20
        tol = 1e-3
        self.T_f = self.T_g  # 초기값
        self.T_f_in = self.T_f + self.dT_r_exch  # 초기값, 열교환기에서의 순환수 유입 온도

        for _ in range(max_iter):
            self.T_r_exch = self.T_f_in + self.dT_r_exch  # 5 K 높게 설정
            self.COP = calc_GSHP_COP(
                Tg=self.T_g,
                T_cond=self.T_r_int,
                T_evap=self.T_r_exch,
                theta_hat=0.3,
            )
            # Others
            self.E_cmp = self.Q_r_int / self.COP  # compressor power input [W]
            self.Q_r_exch = self.Q_r_int - self.E_cmp  # changed from Q_r_ext to Q_r_exch
            # Borehole
            self.Q_bh = (
                self.Q_r_exch - self.E_pmp
            ) / self.H_b  # heat flow rate from borehole to ground per unit length [W/m]
            self.g_i = G_FLS(
                t=self.time,
                ks=self.k_g,
                as_=self.alpha,
                rb=self.r_b,
                H=self.H_b,
            )  # g-function [mK/W]
            # fluid temperature & borehole wall temperature [K]
            T_f_in_old = self.T_f_in  # 이전 유체 입구 온도 저장
            self.T_b = self.T_g - self.Q_bh * self.g_i  # borehole wall temperature [K]
            self.T_f = self.T_b - self.Q_bh * self.R_b  # fluid temperature in borehole [K]
            self.T_f_in = self.T_f - self.Q_bh * self.H_b / (2 * c_w * rho_w * self.dV_f)  # fluid inlet temperature [K]
            self.T_f_out = self.T_f + self.Q_bh * self.H_b / (
                2 * c_w * rho_w * self.dV_f
            )  # fluid outlet temperature [K]
            if abs(self.T_f_in - T_f_in_old) < tol:
                break

        # Temperature
        self.T_a_int_in = self.T_a_room  # internal unit air inlet temperature [K]

        # Internal unit
        self.dV_int = self.Q_r_int / (
            c_a * rho_a * (abs(self.T_a_int_out - self.T_a_int_in))
        )  # volumetric flow rate of internal unit [m3/s]

        # Fan power
        self.E_fan_int = Fan().get_power(self.fan_int, self.dV_int)  # power input of internal unit fan [W]

        # Exergy result
        self.X_a_int_in = (
            c_a * rho_a * self.dV_int * ((self.T_a_int_in - self.T0) - self.T0 * math.log(self.T_a_int_in / self.T0))
        )
        self.X_a_int_out = (
            c_a * rho_a * self.dV_int * ((self.T_a_int_out - self.T0) - self.T0 * math.log(self.T_a_int_out / self.T0))
        )

        self.X_r_int = self.Q_r_int * (1 - self.T0 / self.T_r_int)
        self.X_r_exch = self.Q_r_exch * (1 - self.T0 / self.T_r_exch)

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
            },
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
            },
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
            },
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
            },
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
            },
        }


@dataclass
class GroundSourceHeatPump:
    """Ground source heat pump model for both cooling and heating mode.

    Uses borehole heat exchangers with pygfunction step-response
    factor array for precise soil thermal response with temporal
    superposition of dynamic building loads.
    """
    # 1. Borehole parameters
    H_b: float = 150.0 # Borehole height [m]
    D_b: float = 2.0   # Borehole burial depth [m]
    r_b: float = 0.08  # Borehole radius [m]
    
    # 2. Pipe & Grout parameters
    k_p: float = 0.4      # Pipe thermal conductivity [W/mK] (HDPE)
    k_grout: float = 1.5  # Grout thermal conductivity [W/mK]
    r_out: float = 0.016  # Pipe outer radius [m] (32mm OD / 2)
    r_in: float = 0.013   # Pipe inner radius [m] (26mm ID / 2)
    D_s: float = 0.05     # Shank spacing [m]
    
    # 3. Ground parameters
    k_g: float = 2.0     # Ground thermal conductivity [W/mK]
    c_g: float = 800.0   # Ground specific heat capacity [J/(kgK)]
    rho_g: float = 2000.0 # Ground density [kg/m³]
    T_g: float = 15.0    # Initial ground temperature [°C]
    
    # 4. Fluid parameters
    dV_f: float = 16.0   # Volumetric flow rate of fluid [L/min]
    
    # 5. Rated Performance & Design
    Q_rated_cooling: float = 20590.0 # [W]
    Q_rated_heating: float = 16450.0 # [W]
    E_pmp: float = 250.0            # Pump power input [W]
    dP_iu_fan_design: float = 60.0   # Design pressure drop [Pa]
    eta_iu_fan_design: float = 0.6   # Design fan efficiency
    
    # 6. Simulation Control
    dt_hours: int = 1
    sim_hours: int = 8760
    
    # 7. Runtime Inputs (per-timestep)
    Q_r_iu: float = 0.0
    T0: float = 20.0

    def __post_init__(self):
        # Initialize historical and temporal states
        self.time = 0.0
        self.dt_sec = self.dt_hours * 3600.0
        self.q_b_history = [0.0]

        # Calculate Effective Borehole Thermal Resistance R_b
        # Using water properties at approx 15-20 degC
        m_flow_pipe = (self.dV_f * cu.L2m3 * cu.s2m * rho_f) / 2.0 # Mass flow per pipe [kg/s]
        self.R_b = gf.calc_borehole_thermal_resistance(
            k_s=self.k_g, k_g=self.k_grout, k_p=self.k_p,
            r_b=self.r_b, r_out=self.r_out, r_in=self.r_in, D_s=self.D_s,
            m_flow_pipe=m_flow_pipe, rho_f=rho_f, mu_f=0.00114, cp_f=c_f, k_f=k_w
        )

        # Fan parameters (VSD model)
        _hp_capacity = max(self.Q_rated_cooling, self.Q_rated_heating)
        self.dV_iu_fan_design = _hp_capacity / (rho_a * c_a * 10.0) 
        self.E_iu_fan_design = (
            self.dV_iu_fan_design * self.dP_iu_fan_design / self.eta_iu_fan_design
        )
        self.vsd_coeffs_iu = {
            "c1": 0.0013, "c2": 0.1470, "c3": 0.9506, "c4": -0.0998, "c5": 0.0,
        }
        self.fan_params_iu = {
            "fan_design_flow_rate": self.dV_iu_fan_design,
            "fan_design_power": self.E_iu_fan_design,
        }

        # Precompute dimensional g-function interpolator [mK/W]
        alpha = self.k_g / (self.rho_g * self.c_g)
        self.g_func_interp = gf.precompute_gfunction(
            N_1=1, N_2=1, B=6.0, 
            H_b=self.H_b, D_b=self.D_b, r_b=self.r_b,
            alpha_s=alpha, k_s=self.k_g,
            t_max_s=self.sim_hours * 3600.0,
            dt_s=self.dt_sec
        )
        # ---------------------------------------------------------------------

    def system_update(self):
        # Unit conversion
        dV_f_m3s = self.dV_f * cu.s2m * cu.L2m3  # Nominal flow rate [m³/s]

        if not hasattr(self, "T0"):
            raise AttributeError("T0 must be provided before system_update().")

        # Determine mode based on load sign
        if self.Q_r_iu > 0:
            mode = "cooling"
            self.T_a_room = 27  # Room air temperature [°C]
            self.dT_r_ghx = 3 # GHX refrigerant - GHX outlet water [K]
            self.dT_r_iu = -15 # Indoor unit refrigerant - Indoor unit inlet air [K]
            self.T_r_iu = self.T_a_room + self.dT_r_iu # Indoor unit refrigerant [°C]
            dT_a_iu = -10 # Indoor unit outlet air - Room air [K]
            dV_f_m3s_active = dV_f_m3s
            E_pmp_active = self.E_pmp  # Pump power input [W]
        elif self.Q_r_iu < 0:
            mode = "heating"
            self.T_a_room = 21  # Room air temperature [°C]
            self.dT_r_ghx = -3 # GHX refrigerant - GHX outlet water [K] 
            self.dT_r_iu = 15 # Indoor unit refrigerant - Indoor unit inlet air [K]
            self.T_r_iu = self.T_a_room + self.dT_r_iu # Indoor unit refrigerant [°C]
            dT_a_iu = 10 # Indoor unit outlet air - Room air [K]
            dV_f_m3s_active = dV_f_m3s
            E_pmp_active = self.E_pmp  # Pump power input [W]
        else:
            mode = "off"
            self.T_a_room = 22  # Room air temperature [°C]
            self.dT_r_ghx = 0
            self.T_r_ghx = self.T0
            self.T_r_iu = self.T0
            dT_a_iu = 0
            dV_f_m3s_active = 0.0
            dV_f_m3s_active = 0.0
            E_pmp_active = 0.0

        # Temperatures in Kelvin
        self.T0_K = cu.C2K(self.T0)
        self.T_a_room_K = cu.C2K(self.T_a_room)
        
        self.T_a_iu_out_K = self.T_a_room_K + dT_a_iu
             
        self.T_r_iu_K = cu.C2K(self.T_r_iu)
        self.T_g_K = cu.C2K(self.T_g)

        # ---------------------------------------------------------------------
        # A. Pre-calculate the Historical Temperature Effect (Superposition)
        # ---------------------------------------------------------------------
        T_b_history_effect = 0.0

        for i in range(1, len(self.q_b_history)):
            delta_Q = self.q_b_history[i] - self.q_b_history[i - 1]
            elapsed_time = (len(self.q_b_history) - i + 1) * self.dt_sec
            
            # Use dimensional g-function from interpolator [mK/W]
            g_val_dim = float(self.g_func_interp(elapsed_time))
            T_b_history_effect += delta_Q * g_val_dim
        # ---------------------------------------------------------------------

        max_iter = 20
        tol = 1e-2
        # ------------------------------------------------------------------
        # Airflow calculation (indoor unit air volume flow rate).
        # Must be computed BEFORE the COP iteration loop.
        # ------------------------------------------------------------------

        if self.Q_r_iu == 0:
            self.dV_a = 0.0
        else:
            self.dV_a = abs(self.Q_r_iu) / (
                c_a * rho_a * abs(self.T_a_iu_out_K - self.T_a_room_K)
            )

        # Synchronize dV_a_ratio with fan design flow rate
        dV_a_ratio = self.dV_a / self.dV_iu_fan_design if self.dV_iu_fan_design > 0 else 1.0
        # --------------------------------------------------------------------------

        self.T_f = self.T_g_K  # 초기값
        self.T_f_in = self.T_f
        self.T_f_out = self.T_f

        for _ in range(max_iter):
            T_f_in_old = self.T_f_in

            if mode == "cooling":
                self.COP = calc_GSHP_COP(
                    T_a_iu_in_K=self.T_a_room_K,    # T_db of indoor air entering coil
                    T_f_out_K=self.T_f_out,     # water entering HP from borehole [K]
                    dV_a_ratio=dV_a_ratio,
                    mode="cooling",
                )
                self.E_cmp = self.Q_r_iu / self.COP
            elif mode == "heating":
                self.COP = calc_GSHP_COP(
                    T_a_iu_in_K=self.T_a_room_K,    # T_db of indoor air entering coil
                    T_f_out_K=self.T_f_out,     # water entering HP from borehole [K]
                    dV_a_ratio=dV_a_ratio,
                    mode="heating",
                )
                self.E_cmp = -self.Q_r_iu / self.COP
            else:
                self.COP = 0.0
                self.E_cmp = 0.0

            self.Q_r_ghx = self.Q_r_iu + self.E_cmp
            self.q_b = (self.Q_r_ghx + E_pmp_active) / self.H_b

            # -----------------------------------------------------------------
            # B. Core Calculation: Borehole Wall Temp with Superposition
            # -----------------------------------------------------------------
            # Dimensional g-value for the current step (dt_sec) [mK/W]
            self.g_i_dim = float(self.g_func_interp(self.dt_sec))
            self.T_b_history_effect = T_b_history_effect  
            self.T_b = (
                self.T_g_K 
                + T_b_history_effect 
                + (self.q_b - self.q_b_history[-1]) * self.g_i_dim
            )
            # -----------------------------------------------------------------

            self.T_f = self.T_b + self.q_b * self.R_b
            if dV_f_m3s_active > 0:
                delta_T_fluid = self.q_b * self.H_b / (2 * c_f * rho_f * dV_f_m3s_active)
            else:
                delta_T_fluid = 0.0

            self.T_f_in = self.T_f + delta_T_fluid
            self.T_f_out = self.T_f - delta_T_fluid

            if abs(self.T_f_in - T_f_in_old) < tol or mode == "off":
                break

        # Finalize refrigerant temperature based on converged fluid temperature
        self.T_r_ghx_K = self.T_f_out + self.dT_r_ghx

        # ---------------------------------------------------------------------
        # C. Store the finalized load to history for the next timestep
        # ---------------------------------------------------------------------
        self.q_b_history.append(self.q_b)
        self.time += self.dt_hours
        # ---------------------------------------------------------------------

        # Temperature
        self.T_a_iu_in_K = self.T_a_room_K
        self.T_a_iu_in = self.T_a_room
        self.T_a_iu_out = cu.K2C(self.T_a_iu_out_K)

        # Fan power - Updated to hx_fan VSD model
        self.E_fan_iu = calc_fan_power_from_dV_fan(
            dV_fan=self.dV_a,
            fan_params=self.fan_params_iu,
            vsd_coeffs=self.vsd_coeffs_iu,
            is_active=(self.Q_r_iu != 0.0)
        )

        # System COP calculation
        total_pwr = self.E_cmp + self.E_fan_iu + E_pmp_active
        if total_pwr > 0:
            self.COP_sys = abs(self.Q_r_iu) / total_pwr
        else:
            self.COP_sys = 0.0

        # Helper for thermal exergy
        def get_thermal_exergy(c, rho, dV, T_stream, T_env):
            if T_stream <= 0 or dV <= 0:
                return 0.0
            return c * rho * dV * ((T_stream - T_env) - T_env * math.log(T_stream / T_env))

        # -------------------------------------------------------------
        # Exergy of air streams
        # -------------------------------------------------------------
        self.X_a_iu_in = get_thermal_exergy(c_a, rho_a, self.dV_a, self.T_a_iu_in_K, self.T0_K)
        self.X_a_iu_out = get_thermal_exergy(c_a, rho_a, self.dV_a, self.T_a_iu_out_K, self.T0_K)

        # -------------------------------------------------------------
        # Exergy of refrigerant streams
        # -------------------------------------------------------------
        if self.Q_r_iu == 0:
            self.X_g = 0.0
            self.X_b = 0.0
            self.X_r_iu = 0.0
            self.X_r_ghx = 0.0
        else:
            self.X_g = (1 - self.T0_K / self.T_g_K) * (-self.q_b * self.H_b)
            self.X_b = (1 - self.T0_K / self.T_b) * (-self.q_b * self.H_b)
            self.X_r_iu = -self.Q_r_iu * (1 - self.T0_K / self.T_r_iu_K)
            self.X_r_ghx = -self.Q_r_ghx * (1 - self.T0_K / self.T_r_ghx_K)

        self.T_r_ghx = cu.K2C(self.T_r_ghx_K)

        # -------------------------------------------------------------
        # Exergy of water streams
        # -------------------------------------------------------------
        self.X_f_in = get_thermal_exergy(c_f, rho_f, dV_f_m3s_active, self.T_f_in, self.T0_K)
        self.X_f_out = get_thermal_exergy(c_f, rho_f, dV_f_m3s_active, self.T_f_out, self.T0_K)

        # -------------------------------------------------------------
        # Component exergy balance
        # -------------------------------------------------------------
        if mode == "off":
            self.X_in_g = self.X_out_g = self.X_c_g = 0.0
            self.X_in_ghx = self.X_out_ghx = self.X_c_ghx = 0.0
            self.X_in_r = self.X_out_r = self.X_c_r = 0.0
            self.X_in_iu = self.X_out_iu = self.X_c_iu = 0.0
        else:
            # Ground
            self.X_in_g = self.X_g
            self.X_out_g = self.X_b
            self.X_c_g = self.X_in_g - self.X_out_g

            # Ground heat exchanger
            self.X_in_ghx = self.X_b + E_pmp_active
            self.X_out_ghx = self.X_r_ghx
            self.X_c_ghx = self.X_in_ghx - self.X_out_ghx

            # Refrigerant loop
            self.X_in_r = self.X_r_ghx + self.E_cmp
            self.X_out_r = self.X_r_iu
            self.X_c_r = self.X_in_r - self.X_out_r

            # Indoor unit
            self.X_in_iu = self.E_fan_iu + self.X_r_iu
            self.X_out_iu = self.X_a_iu_out - self.X_a_iu_in
            self.X_c_iu = self.X_in_iu - self.X_out_iu

        # -------------------------------------------------------------
        # Exergy efficiency
        # -------------------------------------------------------------
        if self.Q_r_iu == 0:
            self.X_eff = 0.0
        else:
            self.X_eff = (self.X_a_iu_out - self.X_a_iu_in) / (self.E_fan_iu + self.E_cmp + E_pmp_active)

        # -------------------------------------------------------------
        # Structured exergy balance
        # -------------------------------------------------------------
        self.exergy_bal = {
            "indoor unit": {
                "in": {
                    "X_r_iu": self.X_r_iu,
                    "E_fan_iu": self.E_fan_iu,
                },
                "out": {
                    "X_a_iu_out": self.X_a_iu_out,
                    "X_a_iu_in": self.X_a_iu_in,
                },
                "con": {"X_c_iu": self.X_c_iu},
            },
            "refrigerant loop": {
                "in": {
                    "X_r_ghx": self.X_r_ghx,
                    "E_cmp": self.E_cmp,
                },
                "out": {"X_r_iu": self.X_r_iu},
                "con": {"X_c_r": self.X_c_r},
            },
            "ground heat exchanger": {
                "in": {
                    "X_b": self.X_b,
                    "E_pmp": E_pmp_active,
                },
                "out": {"X_r_ghx": self.X_r_ghx},
                "con": {"X_c_ghx": self.X_c_ghx},
            },
            "ground": {
                "in": {"X_g": self.X_g},
                "out": {"X_b": self.X_b},
                "con": {"X_c_g": self.X_c_g},
            },
        }
