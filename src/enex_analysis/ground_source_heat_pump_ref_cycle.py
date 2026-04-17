"""
Ground Source Heat Pump module with detailed refrigerant cycle for Exergy Analysis.

Unified single class handling cooling, heating, and off modes.
"""

import math
from dataclasses import dataclass

import numpy as np
import pygfunction as gt
from CoolProp.CoolProp import PropsSI
from scipy.optimize import brentq, minimize_scalar

from . import calc_util as cu
from .components.fan import Fan
from .constants import c_a, c_w as c_f, rho_a, rho_w as rho_f


def _ref_flow_exergy(m_r: float, h: float, s: float, h0: float, s0: float, T0_K: float) -> float:
    return m_r * ((h - h0) - T0_K * (s - s0))

def _solve_Ca_from_NTU(UA: float, Q: float, T_air_in_K: float, T_ref_K: float) -> float:
    dT = abs(T_air_in_K - T_ref_K)
    if dT < 1e-6:
        raise ValueError("T_air_in == T_ref: zero driving force.")
    capacity_limit = UA * dT
    if Q >= capacity_limit:
        raise ValueError(f"Q >= UA*dT")

    rhs = Q / (UA * dT)
    def _f(ntu):
        if ntu < 1e-10:
            return 1.0 - rhs
        return (1.0 - math.exp(-ntu)) / ntu - rhs
    
    ntu_sol = brentq(_f, 1e-8, 500.0, xtol=1e-8, maxiter=200)
    return UA / ntu_sol

@dataclass
class GroundSourceHeatPump_RefCycle:
    """Unified GSHP model with detailed ref cycle."""
    def __post_init__(self):
        self.time = 0.0

        self.ref = "R32"
        self.SH = 5.0
        self.SC = 5.0
        self.eta_is = 0.70
        self.eta_el = 0.95

        self.T_g = 16.0
        self.Q_r_iu = 0.0

        self.UA_iu = 2000.0
        self.fan_iu = Fan().fan1

        self.D_b = 2.0
        self.H_b = 100.0
        self.r_b = 0.07 
        self.R_b = 0.108

        self.dV_f = 20.04 

        self.k_g = 2.0
        self.c_g = 800.0
        self.rho_g = 2000.0

        self.E_pmp_nom = 100.0 
        self.UA_rwhx = 3000.0

        self.q_b_history = [0.0]
        self.sim_hours = 8760
        self.dt_hours = 1
        self.dt_sec = self.dt_hours * 3600.0

        borehole = gt.boreholes.Borehole(H=self.H_b, D=self.D_b, r_b=self.r_b, x=0.0, y=0.0)
        n_steps = int(self.sim_hours / self.dt_hours)
        time_array = np.arange(1, n_steps + 1) * self.dt_sec
        alpha = self.k_g / (self.rho_g * self.c_g)
        self.g_func_list = gt.gfunction.gFunction([borehole], alpha, time=time_array).gFunc

    def system_update(self):
        dV_f_m3s = self.dV_f * cu.s2m * cu.L2m3

        if not hasattr(self, 'T0'):
            raise AttributeError("T0 required")

        if self.Q_r_iu > 0:
            mode = "cooling"
            self.T_a_room = 27
            dV_f_m3s_active = dV_f_m3s
            self.E_pmp = self.E_pmp_nom
        elif self.Q_r_iu < 0:
            mode = "heating"
            self.T_a_room = 21
            dV_f_m3s_active = dV_f_m3s
            self.E_pmp = self.E_pmp_nom
        else:
            mode = "off"
            self.T_a_room = 22
            dV_f_m3s_active = 0.0
            self.E_pmp = 0.0

        T0_K = cu.C2K(self.T0)
        T_g_K = cu.C2K(self.T_g)
        T_room_K = cu.C2K(self.T_a_room)
        T_a_iu_in_K = T_room_K

        p0 = 101325.0
        h0 = PropsSI("H", "T", T0_K, "P", p0, self.ref)
        s0 = PropsSI("S", "T", T0_K, "P", p0, self.ref)

        T_b_history_effect = 0.0
        for i in range(1, len(self.q_b_history)):
            delta_q = self.q_b_history[i] - self.q_b_history[i - 1]
            idx = len(self.q_b_history) - i
            g_val = self.g_func_list[idx] if idx < len(self.g_func_list) else self.g_func_list[-1]
            T_b_history_effect += (delta_q / (2 * math.pi * self.k_g)) * g_val

        def _cooling_cycle(T_r_iu_C: float, T_f_out_K: float):
            T_r_iu_K = cu.C2K(T_r_iu_C)
            eps_rwhx = 1.0 - math.exp(-self.UA_rwhx / (c_f * rho_f * dV_f_m3s_active))
            
            def _rwhx_resid(T_rwhx: float) -> float:
                if T_rwhx <= T_r_iu_K + 0.5:
                    return 1e6
                try:
                    p1_ = PropsSI("P", "T", T_r_iu_K, "Q", 1, self.ref)
                    h1_ = PropsSI("H", "T", T_r_iu_K + self.SH, "P", p1_, self.ref)
                    s1_ = PropsSI("S", "T", T_r_iu_K + self.SH, "P", p1_, self.ref)
                    p2_ = PropsSI("P", "T", T_rwhx, "Q", 1, self.ref)
                    h2s_ = PropsSI("H", "P", p2_, "S", s1_, self.ref)
                    h2_  = h1_ + (h2s_ - h1_) / self.eta_is
                    h3_  = PropsSI("H", "T", T_rwhx - self.SC, "P", p2_, self.ref)
                    dh   = h1_ - h3_
                    if dh < 1.0: return 1e6
                    Q_rwhx = (abs(self.Q_r_iu) / dh) * (h2_ - h3_)
                    return T_rwhx - (T_f_out_K + Q_rwhx / (eps_rwhx * c_f * rho_f * dV_f_m3s_active))
                except Exception:
                    return 1e6

            T_c_lb = max(T_f_out_K + 0.1, T_r_iu_K + 1.0)
            T_c_ub = 353.15  
            T_r_rwhx_K = brentq(_rwhx_resid, T_c_lb, T_c_ub, xtol=0.01, maxiter=60)

            p1 = PropsSI("P", "T", T_r_iu_K, "Q", 1, self.ref)
            T1_K = T_r_iu_K + self.SH
            h1 = PropsSI("H", "T", T1_K, "P", p1, self.ref)
            s1 = PropsSI("S", "T", T1_K, "P", p1, self.ref)

            p2 = PropsSI("P", "T", T_r_rwhx_K, "Q", 1, self.ref)
            h2s = PropsSI("H", "P", p2, "S", s1, self.ref)
            h2 = h1 + (h2s - h1) / self.eta_is
            T2_K = PropsSI("T", "P", p2, "H", h2, self.ref)
            s2 = PropsSI("S", "P", p2, "H", h2, self.ref)

            p3 = p2
            T3_K = T_r_rwhx_K - self.SC
            h3 = PropsSI("H", "T", T3_K, "P", p3, self.ref)
            s3 = PropsSI("S", "T", T3_K, "P", p3, self.ref)

            p4 = p1
            h4 = h3
            T4_K = PropsSI("T", "P", p4, "H", h4, self.ref)
            s4 = PropsSI("S", "P", p4, "H", h4, self.ref)

            dh_eff = h1 - h4
            m_r = abs(self.Q_r_iu) / dh_eff
            E_cmp = m_r * (h2 - h1) / self.eta_el

            C_a_iu = _solve_Ca_from_NTU(self.UA_iu, abs(self.Q_r_iu), T_a_iu_in_K, T_r_iu_K)
            m_a_iu = C_a_iu / c_a
            V_a_iu = m_a_iu / rho_a
            E_fan_iu = Fan().get_power(self.fan_iu, V_a_iu)

            return (E_cmp, E_fan_iu, m_r, V_a_iu, m_a_iu, C_a_iu, p1, h1, s1, T1_K, p2, h2, s2, T2_K, p3, h3, s3, T3_K, p4, h4, s4, T4_K, T_r_rwhx_K)

        def _heating_cycle(T_r_iu_C: float, T_f_out_K: float):
            T_r_iu_K = cu.C2K(T_r_iu_C)
            eps_rwhx = 1.0 - math.exp(-self.UA_rwhx / (c_f * rho_f * dV_f_m3s_active))

            p2   = PropsSI("P", "T", T_r_iu_K, "Q", 1, self.ref)
            T3_K = T_r_iu_K - self.SC
            h3   = PropsSI("H", "T", T3_K, "P", p2, self.ref)
            s3   = PropsSI("S", "T", T3_K, "P", p2, self.ref)

            def _rwhx_resid(T_r_rwhx: float) -> float:
                if T_r_rwhx >= T_r_iu_K - 0.5 or T_r_rwhx >= T_f_out_K - 0.05:
                    return 1e6
                try:
                    p1_  = PropsSI("P", "T", T_r_rwhx, "Q", 1, self.ref)
                    T1_  = T_r_rwhx + self.SH
                    h1_  = PropsSI("H", "T", T1_, "P", p1_, self.ref)
                    s1_  = PropsSI("S", "T", T1_, "P", p1_, self.ref)
                    h2s_ = PropsSI("H", "P", p2, "S", s1_, self.ref)
                    h2_ = h1_ + (h2s_ - h1_) / self.eta_is
                    dh_c = h2_ - h3
                    if dh_c < 1.0: return 1e6
                    m_r_  = abs(self.Q_r_iu) / dh_c
                    Q_ref = m_r_ * (h1_ - h3)   
                    Q_wat = eps_rwhx * c_f * rho_f * dV_f_m3s_active * (T_f_out_K - T_r_rwhx)
                    return Q_ref - Q_wat
                except Exception:
                    return 1e6

            T_e_ub = min(T_r_iu_K - 1.0, T_f_out_K - 0.1)
            T_e_lb = 233.15  
            T_r_rwhx_K = brentq(_rwhx_resid, T_e_lb, T_e_ub, xtol=0.01, maxiter=60)

            p1   = PropsSI("P", "T", T_r_rwhx_K, "Q", 1, self.ref)
            T1_K = T_r_rwhx_K + self.SH
            h1   = PropsSI("H", "T", T1_K, "P", p1, self.ref)
            s1   = PropsSI("S", "T", T1_K, "P", p1, self.ref)

            h2s = PropsSI("H", "P", p2, "S", s1, self.ref)
            h2 = h1 + (h2s - h1) / self.eta_is
            T2_K = PropsSI("T", "P", p2, "H", h2, self.ref)
            s2 = PropsSI("S", "P", p2, "H", h2, self.ref)

            p3 = p2
            p4 = p1
            h4 = h3
            T4_K = PropsSI("T", "P", p4, "H", h4, self.ref)
            s4 = PropsSI("S", "P", p4, "H", h4, self.ref)

            dh_eff = h2 - h3
            if dh_eff < 1.0: raise ValueError("dh_eff small")
            m_r = abs(self.Q_r_iu) / dh_eff
            E_cmp = m_r * (h2 - h1) / self.eta_el

            C_a_iu   = _solve_Ca_from_NTU(self.UA_iu, abs(self.Q_r_iu), T_a_iu_in_K, T_r_iu_K)
            m_a_iu   = C_a_iu / c_a
            V_a_iu = m_a_iu / rho_a
            E_fan_iu = Fan().get_power(self.fan_iu, V_a_iu)

            return (E_cmp, E_fan_iu, m_r, V_a_iu, m_a_iu, C_a_iu, p1, h1, s1, T1_K, p2, h2, s2, T2_K, p3, h3, s3, T3_K, p4, h4, s4, T4_K, T_r_rwhx_K)

        T_f_in_K = T_g_K
        T_f_out_K = T_g_K

        self.E_cmp, self.E_fan_iu, self.m_r, self.V_a_iu = 0.0, 0.0, 0.0, 0.0
        self.Q_r_rwhx, self.Q_r_ghx, self.q_b = 0.0, 0.0, 0.0
        self.COP = 0.0

        if mode != "off":
            max_outer = 30
            for _outer in range(max_outer):
                if mode == "cooling":
                    T_r_iu_ub_C = (T_a_iu_in_K - abs(self.Q_r_iu)/self.UA_iu - 0.5) - 273.15
                    T_r_iu_lb_C = 0.0
                    def obj(T_C):
                        try:
                            E_c, E_f, *_ = _cooling_cycle(T_C, T_f_out_K)
                            return E_c + E_f
                        except: return 1e12
                    
                    try:
                        rst = minimize_scalar(obj, bounds=(T_r_iu_lb_C, T_r_iu_ub_C), method="bounded", options={"xatol":0.5})
                        _res = _cooling_cycle(rst.x, T_f_out_K)
                        Q_r_rwhx_val = _res[2] * (_res[11] - _res[15]) 
                    except Exception:
                        # Pinch-Point Crossover Safety Fallback for Cooling
                        fallback_E_cmp = abs(self.Q_r_iu) * 2.0  # COP = 0.5 Extreme Penalty
                        _res = (fallback_E_cmp, Fan().get_power(self.fan_iu, 0.5), 0.0, 0.5, 0.0, 0.0,
                                p0, h0, s0, T0_K, p0, h0, s0, T0_K, p0, h0, s0, T0_K, p0, h0, s0, T0_K, T0_K)
                        Q_r_rwhx_val = abs(self.Q_r_iu) + fallback_E_cmp
                        class _Obj: pass
                        rst = _Obj()
                        rst.x = cu.K2C(T0_K) 
                else:
                    T_r_iu_lb_C = (T_a_iu_in_K + abs(self.Q_r_iu)/self.UA_iu + 0.5) - 273.15
                    T_r_iu_ub_C = 65.0
                    def obj(T_C):
                        try:
                            E_c, E_f, *_ = _heating_cycle(T_C, T_f_out_K)
                            return E_c + E_f
                        except: return 1e12
                    
                    try:
                        rst = minimize_scalar(obj, bounds=(T_r_iu_lb_C, T_r_iu_ub_C), method="bounded", options={"xatol":0.5})
                        _res = _heating_cycle(rst.x, T_f_out_K)
                        Q_r_rwhx_val = - _res[2] * (_res[7] - _res[15])
                    except Exception:
                        # Pinch-Point Crossover Safety Fallback for Heating
                        fallback_E_cmp = abs(self.Q_r_iu) * 2.0  # COP = 0.5 Extreme Penalty
                        _res = (fallback_E_cmp, Fan().get_power(self.fan_iu, 0.5), 0.0, 0.5, 0.0, 0.0,
                                p0, h0, s0, T0_K, p0, h0, s0, T0_K, p0, h0, s0, T0_K, p0, h0, s0, T0_K, T0_K)
                        Q_r_rwhx_val = -abs(self.Q_r_iu) + fallback_E_cmp
                        class _Obj: pass
                        rst = _Obj()
                        rst.x = cu.K2C(T0_K)
                (self.E_cmp, self.E_fan_iu, self.m_r, self.V_a_iu, self.m_a_iu, self.C_a_iu,
                 p1, h1, s1, T1_K, p2, h2, s2, T2_K, p3, h3, s3, T3_K, p4, h4, s4, T4_K, T_r_rwhx_K) = _res
                T_r_iu_K = cu.C2K(rst.x)
                
                self.p1, self.h1, self.s1, self.T1_K = p1, h1, s1, T1_K
                self.p2, self.h2, self.s2, self.T2_K = p2, h2, s2, T2_K
                self.p3, self.h3, self.s3, self.T3_K = p3, h3, s3, T3_K
                self.p4, self.h4, self.s4, self.T4_K = p4, h4, s4, T4_K
                
                self.Q_r_rwhx = Q_r_rwhx_val
                self.Q_r_ghx = self.Q_r_rwhx + self.E_pmp
                self.q_b = self.Q_r_ghx / self.H_b

                self.g_i = self.g_func_list[0]
                self.T_b_K = T_g_K + T_b_history_effect + ((self.q_b - self.q_b_history[-1]) / (2*math.pi*self.k_g)) * self.g_i
                
                T_f_K = self.T_b_K + self.q_b * self.R_b
                dT_f = self.q_b * self.H_b / (2.0 * c_f * rho_f * dV_f_m3s_active)
                T_f_in_new_K = T_f_K + dT_f
                T_f_out_new_K = T_f_K - dT_f

                if abs(T_f_in_new_K - T_f_in_K) < 1e-3:
                    T_f_in_K, T_f_out_K = T_f_in_new_K, T_f_out_new_K
                    break
                T_f_in_K, T_f_out_K = T_f_in_new_K, T_f_out_new_K
            
            self.COP = abs(self.Q_r_iu) / self.E_cmp
            self.COP_sys = abs(self.Q_r_iu) / (self.E_cmp + self.E_fan_iu + self.E_pmp) if (self.E_cmp + self.E_fan_iu + self.E_pmp) > 0 else 0.0
            self.T_r_iu_K = T_r_iu_K
            self.T_r_rwhx_K = T_r_rwhx_K
            
            # COP and temperatures assigned above
            
        else:
            self.E_cmp, self.E_fan_iu, self.m_r, self.V_a_iu = 0.0, 0.0, 0.0, 0.0
            self.Q_r_rwhx, self.Q_r_ghx, self.q_b = 0.0, 0.0, 0.0
            self.COP = 0.0
            self.COP_sys = 0.0
            self.m_a_iu = 0.0
            self.C_a_iu = 0.0
            
            self.g_i = self.g_func_list[0]
            self.T_b_K = T_g_K + T_b_history_effect + ((self.q_b - self.q_b_history[-1]) / (2*math.pi*self.k_g)) * self.g_i
            T_f_K = self.T_b_K
            T_f_in_K, T_f_out_K = T_f_K, T_f_K

            self.T_r_iu_K = T0_K
            self.T_r_rwhx_K = T0_K
            self.p1, self.h1, self.s1, self.T1_K = p0, h0, s0, T0_K
            self.p2, self.h2, self.s2, self.T2_K = p0, h0, s0, T0_K
            self.p3, self.h3, self.s3, self.T3_K = p0, h0, s0, T0_K
            self.p4, self.h4, self.s4, self.T4_K = p0, h0, s0, T0_K

        self.q_b_history.append(self.q_b)
        self.time += self.dt_hours

        self.T_f_K = (T_f_in_K + T_f_out_K) / 2.0
        self.T_f_in_K = T_f_in_K
        self.T_f_out_K = T_f_out_K

        if mode == "off":
            X1, X2, X3, X4 = 0.0, 0.0, 0.0, 0.0
            X_a_iu_in, X_a_iu_out = 0.0, 0.0
            X_f_in, X_f_out = 0.0, 0.0
            X_g, X_b = 0.0, 0.0
            self.X_eff = 0.0
            X_c_iu, X_c_cmp, X_c_exp, X_c_rwhx, X_c_ghx, X_c_g = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            X_in_iu, X_out_iu = 0.0, 0.0
            X_in_cmp, X_out_cmp = 0.0, 0.0
            X_in_exp, X_out_exp = 0.0, 0.0
            X_in_rwhx, X_out_rwhx = 0.0, 0.0
            X_in_ghx, X_out_ghx = 0.0, 0.0
            X_in_g, X_out_g = 0.0, 0.0
            
            # Reset more variables for off mode
            self.eps_iu = 0.0
            self.NTU_iu = 0.0
            self.E_tot = 0.0
            
        else:
            X1 = _ref_flow_exergy(self.m_r, self.h1, self.s1, h0, s0, T0_K)
            X2 = _ref_flow_exergy(self.m_r, self.h2, self.s2, h0, s0, T0_K)
            X3 = _ref_flow_exergy(self.m_r, self.h3, self.s3, h0, s0, T0_K)
            X4 = _ref_flow_exergy(self.m_r, self.h4, self.s4, h0, s0, T0_K)

            T_a_iu_out_K = T_a_iu_in_K - self.Q_r_iu / (c_a * rho_a * self.V_a_iu)

            def _air_exergy(T_K):
                return c_a * rho_a * self.V_a_iu * ((T_K - T0_K) - T0_K * math.log(T_K / T0_K))
            
            X_a_iu_in = _air_exergy(T_a_iu_in_K)
            X_a_iu_out = _air_exergy(T_a_iu_out_K)

            # Calculation of heat exchanger indicators
            self.NTU_iu = self.UA_iu / (c_a * rho_a * self.V_a_iu)
            self.eps_iu = 1.0 - math.exp(-self.NTU_iu)

            def _fluid_exergy(T_K):
                return c_f * rho_f * dV_f_m3s_active * ((T_K - T0_K) - T0_K * math.log(T_K / T0_K))
            
            X_f_in = _fluid_exergy(T_f_in_K)
            X_f_out = _fluid_exergy(T_f_out_K)

            X_g = (1.0 - T0_K / T_g_K) * (-self.Q_r_ghx)
            X_b = (1.0 - T0_K / self.T_b_K) * (-self.Q_r_ghx)

            if mode == "cooling":
                X_in_iu = self.E_fan_iu + X4 + X_a_iu_in
                X_out_iu = X1 + X_a_iu_out
                X_in_cmp = self.E_cmp + X1
                X_out_cmp = X2
                X_in_exp = X3
                X_out_exp = X4
                X_in_rwhx = X2 + X_f_out
                X_out_rwhx = X3 + X_f_in
            elif mode == "heating":
                X_in_iu = self.E_fan_iu + X2 + X_a_iu_in
                X_out_iu = X3 + X_a_iu_out
                X_in_cmp = self.E_cmp + X1
                X_out_cmp = X2
                X_in_exp = X3
                X_out_exp = X4
                X_in_rwhx = X4 + X_f_out
                X_out_rwhx = X1 + X_f_in

            X_c_iu = X_in_iu - X_out_iu
            X_c_cmp = X_in_cmp - X_out_cmp
            X_c_exp = X_in_exp - X_out_exp
            X_c_rwhx = X_in_rwhx - X_out_rwhx

            X_in_ghx = self.E_pmp + X_f_in + X_b
            X_out_ghx = X_f_out
            X_c_ghx = X_in_ghx - X_out_ghx

            X_in_g = X_g
            X_out_g = X_b
            X_c_g = X_in_g - X_out_g

            self.E_tot = self.E_cmp + self.E_fan_iu + self.E_pmp
            self.X_out_tot = X_a_iu_out - X_a_iu_in
            self.X_eff = (X_a_iu_out - X_a_iu_in) / self.E_tot if self.E_tot > 0 else 0.0

        # Bind variables to instance for Jupyter extraction
        self.X1, self.X2, self.X3, self.X4 = X1, X2, X3, X4
        self.X_a_iu_in, self.X_a_iu_out = X_a_iu_in, X_a_iu_out
        self.X_f_in, self.X_f_out = X_f_in, X_f_out
        self.X_g, self.X_b = X_g, X_b
        self.X_in_iu, self.X_out_iu, self.X_c_iu = X_in_iu, X_out_iu, X_c_iu
        self.X_in_cmp, self.X_out_cmp, self.X_c_cmp = X_in_cmp, X_out_cmp, X_c_cmp
        self.X_in_exp, self.X_out_exp, self.X_c_exp = X_in_exp, X_out_exp, X_c_exp
        self.X_in_rwhx, self.X_out_rwhx, self.X_c_rwhx = X_in_rwhx, X_out_rwhx, X_c_rwhx
        self.X_in_ghx, self.X_out_ghx, self.X_c_ghx = X_in_ghx, X_out_ghx, X_c_ghx
        self.X_in_g, self.X_out_g, self.X_c_g = X_in_g, X_out_g, X_c_g

        # Temperature unit conversions for visualization (Celsius)
        self.T1 = cu.K2C(self.T1_K)
        self.T2 = cu.K2C(self.T2_K)
        self.T3 = cu.K2C(self.T3_K)
        self.T4 = cu.K2C(self.T4_K)
        self.T_r_iu = cu.K2C(self.T_r_iu_K)
        self.T_r_rwhx = cu.K2C(self.T_r_rwhx_K)
        self.T_b = cu.K2C(self.T_b_K)
        self.T_f_in = cu.K2C(self.T_f_in_K)
        self.T_f_out = cu.K2C(self.T_f_out_K)
        if mode != "off":
            self.T_a_iu_out_K = T_a_iu_out_K
            self.T_a_iu_out = cu.K2C(T_a_iu_out_K)
        else:
            self.T_a_iu_out_K = T0_K
            self.T_a_iu_out = self.T0

        self.exergy_bal = {
            "indoor unit": {
                "in":  {"E_fan_iu": self.E_fan_iu, "X_a_iu_in": X_a_iu_in, "X2_or_X4": X_in_iu - self.E_fan_iu - X_a_iu_in},
                "out": {"X_a_iu_out": X_a_iu_out, "X1_or_X3": X_out_iu - X_a_iu_out},
                "con": {"X_c_iu": X_c_iu},
            },
            "compressor": {
                "in":  {"E_cmp": self.E_cmp, "X1": X_in_cmp - self.E_cmp},
                "out": {"X2": X_out_cmp},
                "con": {"X_c_cmp": X_c_cmp},
            },
            "expansion valve": {
                "in":  {"X3": X_in_exp},
                "out": {"X4": X_out_exp},
                "con": {"X_c_exp": X_c_exp},
            },
            "refrigerant-to-water heat exchanger": {
                "in":  {"X_f_out": X_f_out, "X_ref_in": X_in_rwhx - X_f_out},
                "out": {"X_f_in": X_f_in, "X_ref_out": X_out_rwhx - X_f_in},
                "con": {"X_c_rwhx": X_c_rwhx},
            },
            "ground heat exchanger": {
                "in": {"E_pmp": self.E_pmp, "X_f_in": X_f_in, "X_b": X_b},
                "out": {"X_f_out": X_f_out},
                "con": {"X_c_ghx": X_c_ghx},
            },
            "ground": {
                "in": {"X_in_g": X_in_g},
                "out": {"X_out_g": X_out_g},
                "con": {"X_c_g": X_c_g},
            },
        }
