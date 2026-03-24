"""
Ground Source Heat Pump module for Exergy Analysis.
Contains unified cooling and heating mode simulation class with Temporal Superposition.
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

        # Universal Initial Temperatures
        self.T_g = 15  # initial ground temperature [°C]
        self.T_a_room = 20  # room air temperature [°C]

        # Initial Load
        self.Q_r_iu = 0  # [W] Initialize with zero or be overridden by user

        # ---------------------------------------------------------------------
        # Temporal Superposition & pygfunction Initialization
        # ---------------------------------------------------------------------
        self.Q_bh_history = [0.0]  # Store historical loads
        
        # Determine simulation length and timestep (Default: 8760 hours, 1 hour step)
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
        dV_f_m3s = self.dV_f * cu.s2m * cu.L2m3  # L/min to m³/s

        # Determine mode based on load sign
        if self.Q_r_iu >= 0:
            mode = "cooling"
            self.dT_r_rwhx = 5
            self.T0 = 32
            self.T_r_rwhx = 25
            self.T_r_iu = 15
            T_a_iu_out_delta = -10
        else:
            mode = "heating"
            self.dT_r_rwhx = -5
            self.T0 = 0
            self.T_r_rwhx = 5
            self.T_r_iu = 35
            T_a_iu_out_delta = 15

        # Temperatures in Kelvin
        T0_K = cu.C2K(self.T0)
        T_a_room_K = cu.C2K(self.T_a_room)

        if not hasattr(self, 'T_a_iu_out'):
             T_a_iu_out_K = T_a_room_K + T_a_iu_out_delta
        else:
             T_a_iu_out_K = cu.C2K(self.T_a_iu_out)
             
        T_r_iu_K = cu.C2K(self.T_r_iu)
        T_g_K = cu.C2K(self.T_g)

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
                g_val = self.g_func_list[-1]  # fallback
                
            T_b_history_effect += delta_Q * g_val
        # ---------------------------------------------------------------------

        max_iter = 20
        tol = 1e-2
        self.T_f = T_g_K  # 초기값
        self.T_f_in = self.T_f + self.dT_r_rwhx

        for _ in range(max_iter):
            T_r_rwhx_K = self.T_f_in + self.dT_r_rwhx
            
            if mode == "cooling":
                self.COP = calc_GSHP_COP(
                    Tg=T_g_K, T_cond=T_r_rwhx_K, T_evap=T_r_iu_K, theta_hat=0.3
                )
                self.E_cmp = self.Q_r_iu / self.COP
            else:
                self.COP = calc_GSHP_COP(
                    Tg=T_g_K, T_cond=T_r_iu_K, T_evap=T_r_rwhx_K, theta_hat=0.3
                )
                self.E_cmp = -self.Q_r_iu / self.COP

            self.Q_r_rwhx = self.Q_r_iu + self.E_cmp
            self.Q_bh = (self.Q_r_rwhx + self.E_pmp) / self.H_b
            T_f_in_old = self.T_f_in
            
            # -----------------------------------------------------------------
            # B. Core Calculation: Borehole Wall Temp with Superposition 
            # -----------------------------------------------------------------
            self.g_i = self.g_func_list[0]
            self.T_b = (
                T_g_K 
                + T_b_history_effect 
                + (self.Q_bh - self.Q_bh_history[-1]) * self.g_i
            )
            # -----------------------------------------------------------------
            
            self.T_f = self.T_b + self.Q_bh * self.R_b
            delta_T_fluid = self.Q_bh * self.H_b / (2 * c_w * rho_w * dV_f_m3s)
            self.T_f_in = self.T_f + delta_T_fluid
            self.T_f_out = self.T_f - delta_T_fluid
            
            if abs(self.T_f_in - T_f_in_old) < tol:
                break
                
        # ---------------------------------------------------------------------
        # C. Store the finalized load to history for the next timestep
        # ---------------------------------------------------------------------
        self.Q_bh_history.append(self.Q_bh)
        self.time += self.dt_hours
        # ---------------------------------------------------------------------

        # Temperature
        T_a_iu_in_K = T_a_room_K

        # Indoor unit volume flow
        dV_iu = abs(self.Q_r_iu) / (
            c_a * rho_a * abs(T_a_iu_out_K - T_a_iu_in_K)
        )

        # Fan power
        self.E_fan_iu = Fan().get_power(self.fan_iu, dV_iu)

        # Helper for thermal exergy
        def get_thermal_exergy(c, rho, dV, T_stream, T_env):
            if T_stream <= 0: return 0
            return c * rho * dV * ((T_stream - T_env) - T_env * math.log(T_stream / T_env))

        # Exergy result calculations
        self.X_a_iu_in = get_thermal_exergy(c_a, rho_a, dV_iu, T_a_iu_in_K, T0_K)
        self.X_a_iu_out = get_thermal_exergy(c_a, rho_a, dV_iu, T_a_iu_out_K, T0_K)

        self.X_r_iu = -self.Q_r_iu * (1 - T0_K / T_r_iu_K)
        self.X_r_rwhx = -self.Q_r_rwhx * (1 - T0_K / T_r_rwhx_K)

        self.X_f_in = get_thermal_exergy(c_w, rho_w, dV_f_m3s, self.T_f_in, T0_K)
        self.X_f_out = get_thermal_exergy(c_w, rho_w, dV_f_m3s, self.T_f_out, T0_K)

        self.X_g = (1 - T0_K / T_g_K) * (-self.Q_bh * self.H_b)
        self.X_b = (1 - T0_K / self.T_b) * (-self.Q_bh * self.H_b)

        # Setting Input and Output dynamically based on mode
        if mode == "cooling":
            # Ground (Heat flows to Ground)
            X_input_g = self.X_g
            X_output_g = self.X_b
            self.X_c_g = X_input_g - X_output_g

            # Ground heat exchanger (Heat flows from fluid to ground)
            X_input_ghx = self.E_pmp + X_output_g + self.X_f_in
            X_output_ghx = self.X_f_out
            self.X_c_ghx = X_input_ghx - X_output_ghx

            # Refrigerant-to-water heat exchanger
            X_input_rwhx = X_output_ghx
            X_output_rwhx = self.X_r_rwhx + self.X_f_in
            self.X_c_rwhx = X_input_rwhx - X_output_rwhx

            # Closed refrigerant loop system
            X_input_r = self.E_cmp + self.X_r_rwhx
            X_output_r = self.X_r_iu
            self.X_c_r = X_input_r - X_output_r

        else: # heating mode
            X_input_g = self.X_g
            X_output_g = self.X_b
            self.X_c_g = X_input_g - X_output_g

            # Ground heat exchanger (Heat flows from ground wall to fluid)
            X_input_ghx = self.E_pmp + X_output_g + self.X_f_in
            X_output_ghx = self.X_f_out
            self.X_c_ghx = X_input_ghx - X_output_ghx

            # Refrigerant-to-water heat exchanger
            X_input_rwhx = self.X_r_rwhx + X_output_ghx
            X_output_rwhx = self.X_f_in
            self.X_c_rwhx = X_input_rwhx - X_output_rwhx

            # Closed refrigerant loop system
            X_input_r = self.E_cmp + self.X_r_iu
            X_output_r = self.X_r_rwhx
            self.X_c_r = X_input_r - X_output_r

        # Indoor unit (Same logical flow for both modes if Q_r_iu appropriately signed)
        X_input_iu = self.E_fan_iu + self.X_r_iu + self.X_a_iu_in
        X_output_iu = self.X_a_iu_out
        self.X_c_iu = X_input_iu - X_output_iu

        # Exergy efficiency
        self.X_eff = (self.X_a_iu_out - self.X_a_iu_in) / (
            self.E_fan_iu + self.E_cmp + self.E_pmp
        )

        ## Exergy Balance mapping with Input / Output / Consumption
        self.exergy_bal = {
            "indoor unit": {
                "input": {
                    "X_a_iu_in": self.X_a_iu_in,
                    "X_r_iu": self.X_r_iu,
                    "E_fan_iu": self.E_fan_iu,
                },
                "output": {"X_a_iu_out": self.X_a_iu_out},
                "consumption": {"X_c_iu": self.X_c_iu},
            },
            "refrigerant loop": {
                "input": {"X_r_rwhx": self.X_r_rwhx, "E_cmp": self.E_cmp} if mode == "cooling" else {"X_r_iu": self.X_r_iu, "E_cmp": self.E_cmp},
                "output": {"X_r_iu": self.X_r_iu} if mode == "cooling" else {"X_r_rwhx": self.X_r_rwhx},
                "consumption": {"X_c_r": self.X_c_r},
            },
            "refrigerant-to-water heat exchanger": {
                "input": {"X_output_ghx": X_output_ghx} if mode == "cooling" else {"X_r_rwhx": self.X_r_rwhx, "X_output_ghx": X_output_ghx},
                "output": {"X_r_rwhx": self.X_r_rwhx, "X_f_in": self.X_f_in} if mode == "cooling" else {"X_f_in": self.X_f_in},
                "consumption": {"X_c_rwhx": self.X_c_rwhx},
            },
            "ground heat exchanger": {
                "input": {
                    "X_output_g": X_output_g,
                    "X_f_in": self.X_f_in,
                    "E_pmp": self.E_pmp,
                },
                "output": {"X_output_ghx": X_output_ghx},
                "consumption": {"X_c_ghx": self.X_c_ghx},
            },
            "ground": {
                "input": {"X_input_g": X_input_g},
                "output": {"X_output_g": X_output_g},
                "consumption": {"X_c_g": self.X_c_g},
            },
        }
