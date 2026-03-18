"""
Ground Source Heat Pump module for Exergy Analysis.
Contains cooling and heating mode simulation classes with Temporal Superposition.
"""

import math
from dataclasses import dataclass
import numpy as np
import pygfunction as gt

from . import calc_util as cu
from .components.fan import Fan
from .constants import c_a, c_w, k_w, rho_a, rho_w
from .enex_functions import calc_GSHP_COP

@dataclass
class GroundSourceHeatPump_cooling:
    """Ground source heat pump model for cooling mode.

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
        self.fan_iu = Fan().fan1

        # Temperature Setup
        self.dT_r_rwhx = 5
        self.T0 = 32  # environmental temperature [°C]
        self.T_g = 15  # initial ground temperature [°C]
        self.T_a_room = 20  # room air temperature [°C]
        self.T_r_rwhx = 25  # refrigerant-to-water heat exchanger side refrigerant temperature [°C]
        self.T_r_iu = 15  # indoor unit refrigerant temperature [°C]

        self.Q_r_iu = 20000  # [W] (Initial load)

        # ---------------------------------------------------------------------
        # Temporal Superposition & pygfunction Initialization
        # ---------------------------------------------------------------------
        self.Q_bh_history = [0.0]  # Store historical loads
        
        # Determine simulation length and timestep (Default: 8760 hours, 1 hour step)
        self.sim_hours = 8760
        self.dt_hours = 1
        self.dt_sec = self.dt_hours * 3600.0
        
        # 1. Define borehole using pygfunction
        borehole = gt.boreholes.Borehole(
            H=self.H_b, D=self.D_b, r_b=self.r_b, x=0.0, y=0.0
        )
        
        # 2. Extract step-response g-function list correctly matching our timesteps
        n_steps = int(self.sim_hours / self.dt_hours)
        time_array = np.arange(1, n_steps + 1) * self.dt_sec
        alpha = self.k_g / (self.rho_g * self.c_g)
        
        # g_func_list[0] is response after 1 dt, g_func_list[1] after 2 dt, etc.
        self.g_func_list = gt.gfunction.gFunction(
            [borehole], alpha, time=time_array
        ).gFunc
        # ---------------------------------------------------------------------

    def system_update(self):
        # Unit conversion
        self.dV_f = self.dV_f * cu.s2m * cu.L2m3  # L/min to m³/s

        # self.time handling: assuming the outer loop updates self.time in hours
        # BUT since we use the step array logic, we don't directly feed time to g_func.
        # We process current step properties based on the history array.
        
        self.T0 = cu.C2K(self.T0)
        self.T_a_room = cu.C2K(self.T_a_room)
        # Assuming iu_out is room temp minus something or handled externally,
        # but in original code T_a_iu_out is used later. Wait, T_a_iu_out is missing in __post_init__?
        # Let's define default T_a_iu_out to prevent errors if not set externally.
        if not hasattr(self, 'T_a_iu_out'):
             self.T_a_iu_out = self.T_a_room - 10 # Default cooling delta
        else:
             self.T_a_iu_out = cu.C2K(self.T_a_iu_out)
             
        self.T_r_iu = cu.C2K(self.T_r_iu)
        self.T_g = cu.C2K(self.T_g)

        # ---------------------------------------------------------------------
        # A. Pre-calculate the Historical Temperature Effect (Superposition)
        # ---------------------------------------------------------------------
        T_b_history_effect = 0.0
        
        # Summation over all past steps
        # i represents the index of historical load states
        for i in range(1, len(self.Q_bh_history)):
            # Load step change at that historical timestep
            delta_Q = self.Q_bh_history[i] - self.Q_bh_history[i-1]
            
            # How many timesteps has this delta_Q been actively affecting the ground?
            # E.g., at end of step 2, Q_bh_1 has been active for 2 steps.
            elapsed_steps = len(self.Q_bh_history) - i
            
            # g_func_list index 0 represents 1 timestep elapsed.
            idx = elapsed_steps
            if idx < len(self.g_func_list):
                g_val = self.g_func_list[idx]
            else:
                g_val = self.g_func_list[-1]  # fallback if array exhausted
                
            T_b_history_effect += delta_Q * g_val
        # ---------------------------------------------------------------------

        max_iter = 20
        tol = 1e-2
        self.T_f = self.T_g  # 초기값
        self.T_f_in = self.T_f + self.dT_r_rwhx

        for _ in range(max_iter):
            self.T_r_rwhx = self.T_f_in + self.dT_r_rwhx  # 5 K 높게 설정
            self.COP = calc_GSHP_COP(
                Tg=self.T_g,
                T_cond=self.T_r_rwhx,
                T_evap=self.T_r_iu,
                theta_hat=0.3,
            )
            self.E_cmp = self.Q_r_iu / self.COP  # compressor power input [W]
            self.Q_r_rwhx = self.Q_r_iu + self.E_cmp
            self.Q_bh = (self.Q_r_rwhx + self.E_pmp) / self.H_b
            T_f_in_old = self.T_f_in
            
            # -----------------------------------------------------------------
            # B. Core Calculation: Borehole Wall Temp with Superposition 
            # -----------------------------------------------------------------
            # Current step's response factor (1 timestep elapsed)
            self.g_i = self.g_func_list[0]
            
            # Final Superposed Wall Temperature Calculation
            self.T_b = (
                self.T_g 
                + T_b_history_effect 
                + (self.Q_bh - self.Q_bh_history[-1]) * self.g_i
            )
            # -----------------------------------------------------------------
            
            self.T_f = self.T_b + self.Q_bh * self.R_b
            self.T_f_in = self.T_f + self.Q_bh * self.H_b / (
                2 * c_w * rho_w * self.dV_f
            )  # fluid inlet temperature [K]
            self.T_f_out = self.T_f - self.Q_bh * self.H_b / (
                2 * c_w * rho_w * self.dV_f
            )  # fluid outlet temperature [K]
            if abs(self.T_f_in - T_f_in_old) < tol:
                break
                
        # ---------------------------------------------------------------------
        # C. Store the finalized load to history for the next timestep
        # ---------------------------------------------------------------------
        self.Q_bh_history.append(self.Q_bh)
        # Update internal timer tracking for consistency
        self.time += self.dt_hours
        # ---------------------------------------------------------------------

        # Temperature
        self.T_a_iu_in = self.T_a_room  # indoor unit air inlet temperature [K]

        # Indoor unit
        self.dV_iu = self.Q_r_iu / (
            c_a * rho_a * (abs(self.T_a_iu_out - self.T_a_iu_in))
        )

        # Fan power
        self.E_fan_iu = Fan().get_power(self.fan_iu, self.dV_iu)

        # Exergy result
        self.X_a_iu_in = (
            c_a * rho_a * self.dV_iu * (
                (self.T_a_iu_in - self.T0) - self.T0 * math.log(self.T_a_iu_in / self.T0)
            )
        )
        self.X_a_iu_out = (
            c_a * rho_a * self.dV_iu * (
                (self.T_a_iu_out - self.T0) - self.T0 * math.log(self.T_a_iu_out / self.T0)
            )
        )

        self.X_r_iu = -self.Q_r_iu * (1 - self.T0 / self.T_r_iu)
        self.X_r_rwhx = -self.Q_r_rwhx * (1 - self.T0 / self.T_r_rwhx)

        self.X_f_in = (
            c_w * rho_w * self.dV_f * (
                (self.T_f_in - self.T0) - self.T0 * math.log(self.T_f_in / self.T0)
            )
        )
        self.X_f_out = (
            c_w * rho_w * self.dV_f * (
                (self.T_f_out - self.T0) - self.T0 * math.log(self.T_f_out / self.T0)
            )
        )

        self.X_g = (1 - self.T0 / self.T_g) * (-self.Q_bh * self.H_b)
        self.X_b = (1 - self.T0 / self.T_b) * (-self.Q_bh * self.H_b)

        # Ground
        self.X_in_g = self.X_g
        self.X_out_g = self.X_b
        self.X_c_g = self.X_in_g - self.X_out_g

        # Ground heat exchanger
        self.X_in_ghx = self.E_pmp + self.X_out_g + self.X_f_in
        self.X_out_ghx = self.X_f_out
        self.X_c_ghx = self.X_in_ghx - self.X_out_ghx

        # Refrigerant-to-water heat exchanger
        self.X_in_rwhx = self.X_out_ghx
        self.X_out_rwhx = self.X_r_rwhx + self.X_f_in
        self.X_c_rwhx = self.X_in_rwhx - self.X_out_rwhx

        # Closed refrigerant loop system
        self.X_in_r = self.E_cmp + self.X_r_rwhx
        self.X_out_r = self.X_r_iu
        self.X_c_r = self.X_in_r - self.X_out_r

        # Indoor unit
        self.X_in_iu = self.E_fan_iu + self.X_r_iu + self.X_a_iu_in
        self.X_out_iu = self.X_a_iu_out
        self.X_c_iu = self.X_in_iu - self.X_out_iu

        # Exergy efficiency
        self.X_eff = (self.X_a_iu_out - self.X_a_iu_in) / (
            self.E_fan_iu + self.E_cmp + self.E_pmp
        )

        ## Exergy Balance
        self.exergy_bal = {
            "indoor unit": {
                "in": {
                    "X_a_iu_in": self.X_a_iu_in,
                    "X_r_iu": self.X_r_iu,
                    "E_fan_iu": self.E_fan_iu,
                },
                "out": {"X_a_iu_out": self.X_a_iu_out},
                "con": {"X_c_iu": self.X_c_iu},
            },
            "refrigerant loop": {
                "in": {"X_r_rwhx": self.X_r_rwhx, "E_cmp": self.E_cmp},
                "out": {"X_r_iu": self.X_r_iu},
                "con": {"X_c_r": self.X_c_r},
            },
            "refrigerant-to-water heat exchanger": {
                "in": {"X_out_ghx": self.X_out_ghx},
                "out": {"X_r_rwhx": self.X_r_rwhx, "X_f_in": self.X_f_in},
                "con": {"X_c_rwhx": self.X_c_rwhx},
            },
            "ground heat exchanger": {
                "in": {
                    "X_out_g": self.X_out_g,
                    "X_f_in": self.X_f_in,
                    "E_pmp": self.E_pmp,
                },
                "out": {"X_out_ghx": self.X_out_ghx},
                "con": {"X_c_ghx": self.X_c_ghx},
            },
            "ground": {
                "in": {"X_in_g": self.X_in_g},
                "out": {"X_out_g": self.X_out_g},
                "con": {"X_c_g": self.X_c_g},
            },
        }

@dataclass
class GroundSourceHeatPump_heating:
    """Ground source heat pump model for heating mode.

    Mirror of ``GroundSourceHeatPump_cooling`` configured for space 
    heating using temporal superposition on ground temperature responses 
    via pygfunction.
    """

    def __post_init__(self):
        # Time
        self.time = 0.0  # [h]

        # Borehole parameters
        self.D_b = 2.0  # Borehole burial depth [m]
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
        self.fan_iu = Fan().fan1

        # Temperature
        self.dT_r_rwhx = -5
        self.T0 = 0  # environmental temperature [°C]
        self.T_g = 15  # initial ground temperature [°C]
        self.T_a_room = 20  # room air temperature [°C]
        self.T_r_rwhx = 5  # refrigerant-to-water heat exchanger side refrigerant temperature [°C]
        self.T_r_iu = 35  # indoor unit refrigerant temperature [°C]

        self.Q_r_iu = -20000  # [W] (Heating load is nominally negative to represent extracting heat from ground)

        # ---------------------------------------------------------------------
        # Temporal Superposition & pygfunction Initialization
        # ---------------------------------------------------------------------
        self.Q_bh_history = [0.0]  # Store historical loads
        
        self.sim_hours = 8760
        self.dt_hours = 1
        self.dt_sec = self.dt_hours * 3600.0
        
        borehole = gt.boreholes.Borehole(
            H=self.H_b, D=self.D_b, r_b=self.r_b, x=0.0, y=0.0
        )
        
        n_steps = int(self.sim_hours / self.dt_hours)
        time_array = np.arange(1, n_steps + 1) * self.dt_sec
        alpha = self.k_g / (self.rho_g * self.c_g)
        
        self.g_func_list = gt.gfunction.gFunction(
            [borehole], alpha, time=time_array
        ).gFunc
        # ---------------------------------------------------------------------

    def system_update(self):
        # Unit conversion
        self.dV_f = self.dV_f * cu.s2m * cu.L2m3

        self.T0 = cu.C2K(self.T0)
        self.T_a_room = cu.C2K(self.T_a_room)
        
        if not hasattr(self, 'T_a_iu_out'):
             self.T_a_iu_out = self.T_a_room + 15 # Default heating delta
        else:
             self.T_a_iu_out = cu.C2K(self.T_a_iu_out)
             
        self.T_r_iu = cu.C2K(self.T_r_iu)
        self.T_g = cu.C2K(self.T_g)

        # ---------------------------------------------------------------------
        # A. Pre-calculate the Historical Temperature Effect (Superposition)
        # ---------------------------------------------------------------------
        T_b_history_effect = 0.0
        
        for i in range(1, len(self.Q_bh_history)):
            delta_Q = self.Q_bh_history[i] - self.Q_bh_history[i-1]
            elapsed_steps = len(self.Q_bh_history) - i
            idx = elapsed_steps
            if idx < len(self.g_func_list):
                g_val = self.g_func_list[idx]
            else:
                g_val = self.g_func_list[-1]
                
            T_b_history_effect += delta_Q * g_val
        # ---------------------------------------------------------------------

        max_iter = 20
        tol = 1e-2
        self.T_f = self.T_g  # 초기값
        self.T_f_in = self.T_f + self.dT_r_rwhx

        for _ in range(max_iter):
            self.T_r_rwhx = self.T_f_in + self.dT_r_rwhx  # -5 K 낮게 설정
            self.COP = calc_GSHP_COP(
                Tg=self.T_g,
                T_cond=self.T_r_iu,
                T_evap=self.T_r_rwhx,
                theta_hat=0.3,
            )
            self.E_cmp = -self.Q_r_iu / self.COP  # compressor power input [W]
            self.Q_r_rwhx = self.Q_r_iu + self.E_cmp
            self.Q_bh = (self.Q_r_rwhx + self.E_pmp) / self.H_b
            T_f_in_old = self.T_f_in
            
            # -----------------------------------------------------------------
            # B. Core Calculation: Borehole Wall Temp with Superposition 
            # -----------------------------------------------------------------
            self.g_i = self.g_func_list[0]
            
            self.T_b = (
                self.T_g 
                + T_b_history_effect 
                + (self.Q_bh - self.Q_bh_history[-1]) * self.g_i
            )
            # -----------------------------------------------------------------
            
            self.T_f = self.T_b + self.Q_bh * self.R_b
            self.T_f_in = self.T_f + self.Q_bh * self.H_b / (
                2 * c_w * rho_w * self.dV_f
            )  # fluid inlet temperature [K]
            self.T_f_out = self.T_f - self.Q_bh * self.H_b / (
                2 * c_w * rho_w * self.dV_f
            )  # fluid outlet temperature [K]
            if abs(self.T_f_in - T_f_in_old) < tol:
                break
                
        # ---------------------------------------------------------------------
        # C. Store the finalized load to history for the next timestep
        # ---------------------------------------------------------------------
        self.Q_bh_history.append(self.Q_bh)
        self.time += self.dt_hours
        # ---------------------------------------------------------------------

        # Temperature
        self.T_a_iu_in = self.T_a_room  # indoor unit air inlet temperature [K]

        # Indoor unit
        self.dV_iu = -self.Q_r_iu / (
            c_a * rho_a * (abs(self.T_a_iu_out - self.T_a_iu_in))
        )  # Volumetric flow rate of indoor unit [m³/s]

        # Fan power
        self.E_fan_iu = Fan().get_power(self.fan_iu, self.dV_iu)

        # Exergy result
        self.X_a_iu_in = (
            c_a * rho_a * self.dV_iu * (
                (self.T_a_iu_in - self.T0) - self.T0 * math.log(self.T_a_iu_in / self.T0)
            )
        )
        self.X_a_iu_out = (
            c_a * rho_a * self.dV_iu * (
                (self.T_a_iu_out - self.T0) - self.T0 * math.log(self.T_a_iu_out / self.T0)
            )
        )

        self.X_r_iu = -self.Q_r_iu * (1 - self.T0 / self.T_r_iu)
        self.X_r_rwhx = -self.Q_r_rwhx * (1 - self.T0 / self.T_r_rwhx)

        self.X_f_in = (
            c_w * rho_w * self.dV_f * (
                (self.T_f_in - self.T0) - self.T0 * math.log(self.T_f_in / self.T0)
            )
        )
        self.X_f_out = (
            c_w * rho_w * self.dV_f * (
                (self.T_f_out - self.T0) - self.T0 * math.log(self.T_f_out / self.T0)
            )
        )

        self.X_g = (1 - self.T0 / self.T_g) * (-self.Q_bh * self.H_b)
        self.X_b = (1 - self.T0 / self.T_b) * (-self.Q_bh * self.H_b)

        # Ground
        self.X_in_g = self.X_g
        self.X_out_g = self.X_b
        self.X_c_g = self.X_in_g - self.X_out_g

        # Ground heat exchanger
        self.X_in_ghx = self.E_pmp + self.X_out_g + self.X_f_in
        self.X_out_ghx = self.X_f_out
        self.X_c_ghx = self.X_in_ghx - self.X_out_ghx

        # Refrigerant-to-water heat exchanger
        self.X_in_rwhx = self.X_out_ghx
        self.X_out_rwhx = self.X_r_rwhx + self.X_f_in
        self.X_c_rwhx = self.X_in_rwhx - self.X_out_rwhx

        # Closed refrigerant loop system
        self.X_in_r = self.E_cmp + self.X_r_rwhx
        self.X_out_r = self.X_r_iu
        self.X_c_r = self.X_in_r - self.X_out_r

        # Indoor unit
        self.X_in_iu = self.E_fan_iu + self.X_r_iu + self.X_a_iu_in
        self.X_out_iu = self.X_a_iu_out
        self.X_c_iu = self.X_in_iu - self.X_out_iu

        # Exergy efficiency
        self.X_eff = (self.X_a_iu_out - self.X_a_iu_in) / (
            self.E_fan_iu + self.E_cmp + self.E_pmp
        )

        ## Exergy Balance
        self.exergy_bal = {
            "indoor unit": {
                "in": {
                    "X_a_iu_in": self.X_a_iu_in,
                    "X_r_iu": self.X_r_iu,
                    "E_fan_iu": self.E_fan_iu,
                },
                "out": {"X_a_iu_out": self.X_a_iu_out},
                "con": {"X_c_iu": self.X_c_iu},
            },
            "refrigerant loop": {
                "in": {"X_r_iu": self.X_r_iu, "E_cmp": self.E_cmp},
                "out": {"X_r_rwhx": self.X_r_rwhx},
                "con": {"X_c_r": self.X_c_r},
            },
            "refrigerant-to-water heat exchanger": {
                "in": {"X_r_rwhx": self.X_r_rwhx, "X_out_ghx": self.X_out_ghx},
                "out": {"X_f_in": self.X_f_in},
                "con": {"X_c_rwhx": self.X_c_rwhx},
            },
            "ground heat exchanger": {
                "in": {
                    "X_f_in": self.X_f_in,
                    "X_out_g": self.X_out_g,
                    "E_pmp": self.E_pmp,
                },
                "out": {"X_out_ghx": self.X_out_ghx},
                "con": {"X_c_ghx": self.X_c_ghx},
            },
            "ground": {
                "in": {"X_in_g": self.X_in_g},
                "out": {"X_out_g": self.X_out_g},
                "con": {"X_c_g": self.X_c_g},
            },
        }
