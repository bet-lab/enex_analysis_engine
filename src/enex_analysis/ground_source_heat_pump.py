"""
Ground Source Heat Pump module for Exergy Analysis.
Contains unified cooling, heating, and off mode simulation class with Temporal Superposition.
"""

import math
from dataclasses import dataclass

import numpy as np
import pygfunction as gt

from . import calc_util as cu
from .components.fan import Fan
from .constants import c_a, rho_a
from .constants import c_w as c_f
from .constants import rho_w as rho_f
from .enex_functions import calc_GSHP_COP


@dataclass
class GroundSourceHeatPump:
    """Ground source heat pump model for both cooling and heating mode.

    Uses borehole heat exchangers with pygfunction step-response
    factor array for precise soil thermal response with temporal
    superposition of dynamic building loads. Call ``system_update()``
    each time step to advance the ground temperature history.
    """

    def __post_init__(self):
        # Time and Simulation Control
        self.time = 0.0  # [h] Will be updated continuously

        # Borehole parameters
        self.D_b = 2.0  # Borehole burial depth [m]
        self.H_b = 200  # Borehole height [m]
        self.r_b = 0.07  # Borehole radius [m]
        self.R_b = 0.108  # Effective borehole thermal resistance [mK/W]

        # Fluid parameters
        self.dV_f = 20.04  # Volumetric flow rate of fluid [L/min]

        # Ground parameters
        self.k_g = 2.0  # Ground thermal conductivity [W/mK]
        self.c_g = 800  # Ground specific heat capacity [J/(kgK)]
        self.rho_g = 2000  # Ground density [kg/m³]

        # Pump power of ground heat exchanger
        self.E_pmp = 100  # Pump power input [W]

        # Fan
        self.fan_iu = Fan().fan1

        # Universal Initial Temperatures
        self.T_g = 16  # Initial ground temperature [°C]

        # Initial Load
        self.Q_r_iu = 0  # Indoor thermal load [W]

        # ---------------------------------------------------------------------
        # Temporal Superposition & pygfunction Initialization
        # ---------------------------------------------------------------------
        self.q_b_history = [0.0]  # Store historical loads

        # Determine simulation length and timestep (Default: 8760 hours, 1 hour step)
        self.sim_hours = 8760
        self.dt_hours = 1
        self.dt_sec = self.dt_hours * 3600.0

        borehole = gt.boreholes.Borehole(H=self.H_b, D=self.D_b, r_b=self.r_b, x=0.0, y=0.0)

        n_steps = int(self.sim_hours / self.dt_hours)
        time_array = np.arange(1, n_steps + 1) * self.dt_sec
        alpha = self.k_g / (self.rho_g * self.c_g) # Soil thermal diffusivity [m²/s]

        self.g_func_list = gt.gfunction.gFunction(
            [borehole], alpha, time=time_array,
        ).gFunc
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
            elapsed_steps = len(self.q_b_history) - i

            idx = elapsed_steps
            if idx < len(self.g_func_list):
                g_val = self.g_func_list[idx]
            else:
                g_val = self.g_func_list[-1]  # fallback

            T_b_history_effect += (delta_Q / (2 * math.pi * self.k_g)) * g_val
        # ---------------------------------------------------------------------

        max_iter = 20
        tol = 1e-2
        # ------------------------------------------------------------------
        # Pre-calculate dV_a (indoor unit air volume flow rate) and air flow ratio
        # Must be computed BEFORE the COP iteration loop.
        # ------------------------------------------------------------------
        # Rated airflow: V_a_ref = Q_rated / (rho_a * c_a * dT_rated=10K)
        if mode == "cooling":
            _Q_rated = 20590.0  # [W] TCH072_GLHP cooling rated capacity
        elif mode == "heating":
            _Q_rated = 16450.0  # [W] TCH072_GLHP heating rated capacity
        else:
            _Q_rated = 20590.0

        _dT_rated = 10.0  # [K] rated temperature difference across indoor coil
        V_a_ref = _Q_rated / (rho_a * c_a * _dT_rated)  # [m³/s]

        if self.Q_r_iu == 0:
            self.dV_a = 0.0
        else:
            self.dV_a = abs(self.Q_r_iu) / (
                c_a * rho_a * abs(self.T_a_iu_out_K - self.T_a_room_K)
            )

        dV_a_ratio = self.dV_a / V_a_ref if V_a_ref > 0 else 1.0
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
            self.g_i = self.g_func_list[0]
            self.T_b_history_effect = T_b_history_effect  # Expose history penalty
            self.T_b = (
                self.T_g_K
                + T_b_history_effect
                + ((self.q_b - self.q_b_history[-1]) / (2 * math.pi * self.k_g)) * self.g_i
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

        # Fan power
        if self.dV_a > 0:
            self.E_fan_iu = Fan().get_power(self.fan_iu, self.dV_a)
        else:
            self.E_fan_iu = 0.0

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
                "consumption": {"X_c_iu": self.X_c_iu},
            },
            "refrigerant loop": {
                "in": {
                    "X_r_ghx": self.X_r_ghx,
                    "E_cmp": self.E_cmp,
                },
                "out": {"X_r_iu": self.X_r_iu},
                "consumption": {"X_c_r": self.X_c_r},
            },
            "ground heat exchanger": {
                "in": {
                    "X_b": self.X_b,
                    "E_pmp": E_pmp_active,
                },
                "out": {"X_r_ghx": self.X_r_ghx},
                "consumption": {"X_c_ghx": self.X_c_ghx},
            },
            "ground": {
                "in": {"X_g": self.X_g},
                "out": {"X_b": self.X_b},
                "consumption": {"X_c_g": self.X_c_g},
            },
        }
