import math
from dataclasses import dataclass

import numpy as np
from CoolProp.CoolProp import PropsSI

from . import calc_util as cu
from .components.fan import Fan
from .constants import c_a, rho_a


def epsilon_from_NTU(NTU):
    # 2상(liquid, vapor) 조건에서의 열교환률 계산
    return 1 - np.exp(-NTU)

def epsilon_from_temperature(T_f_in, T_f_out, T_s):
    # 열교환기 냉매의 열교환률 계산
    return (T_f_in - T_f_out) / (T_f_in - T_s)

def NTU_from_epsilon(epsilon):
    # 열교환기 냉매의 NTU 계산
    return -np.log(1 - epsilon)

def Ts_from_otherfluid(T_f_in, T_f_out, epsilon):
    # 열교환기 냉매의 포화 온도 계산
    return T_f_in - (T_f_in - T_f_out) / epsilon

def _calc_X_flow(m, h_list, s_list, h0, s0, T0_K):
    # 냉매 상태별 엑서지 흐름 계산 헬퍼 함수
    return [m * ((h - h0) - T0_K * (s - s0)) for h, s in zip(h_list, s_list)]

@dataclass
class AirSourceHeatPump_cooling:
    """Air source heat pump model for cooling mode (4 sub-systems)."""

    def __post_init__(self):
        # Refrigerant state
        self.ref = 'R32' # refrigerant
        self.SH = 6 # superheat [K]
        self.SC = 6 # subcooling [K]

        # Temperature
        self.T0 = 35 # environment temperature [°C]
        self.T_a_room = 27 # room air temperature [°C]

        # Load
        self.Q_r_iu = 7000 # thermal load [W]

        # Heat exchanger UA
        self.UA_iu = 800 # indoor unit heat exchanger UA [W/K]
        self.UA_ou = 800 # outdoor unit heat exchanger UA [W/K]

        # Compressor parameter
        self.eta_is = 0.7 # isentropic efficiency [-]
        self.eta_el = 0.95 # electrical efficiency [-]

        # Fan parameter
        self.fan_iu = Fan().fan1 # indoor unit fan
        self.fan_ou = Fan().fan2 # outdoor unit fan

    def system_update(self):
        if self.Q_r_iu == 0:
            # System is Off (Reset logic)
            self.T_a_iu_in = self.T_a_room
            self.T_a_iu_out = self.T_a_iu_in
            self.T_a_ou_in = self.T0
            self.T_a_ou_out = self.T_a_ou_in
            self.T1 = self.T2 = self.T3 = self.T4 = self.T0
            self.p1 = self.p2 = self.p3 = self.p4 = 101325.0
            self.h1 = self.h2 = self.h3 = self.h4 = 0.0
            self.s1 = self.s2 = self.s3 = self.s4 = 0.0
            self.m_r = 0.0
            self.E_cmp = self.E_fan_iu = self.E_fan_ou = 0.0
            self.E_shaft = 0.0
            self.COP = self.COP_sys = 0.0
            self.Q_ou = 0.0
            self.E1 = self.E2 = self.E3 = self.E4 = 0.0
            self.S1 = self.S2 = self.S3 = self.S4 = 0.0
            self.X1 = self.X2 = self.X3 = self.X4 = 0.0
            self.X_cmp = self.X_fan_iu = self.X_fan_ou = 0.0
            self.X_in_tot = 1e-6
            self.X_out_tot = 0.0
            self.X_eff = 0.0
            self.X_in_cmp = self.X_out_cmp = self.X_c_cmp = 0.0
            self.X_in_ou = self.X_out_ou = self.X_c_ou = 0.0
            self.X_in_exp = self.X_out_exp = self.X_c_exp = 0.0
            self.X_in_iu = self.X_out_iu = self.X_c_iu = 0.0
            self.X_a_iu_in = self.X_a_iu_out = 0.0
            self.X_a_ou_in = self.X_a_ou_out = 0.0
            return

        def solve_with_dT(dT_iu, dT_ou):
            res = {}
            res['T_a_iu_in'] = self.T_a_room
            res['T_a_iu_out'] = self.T_a_room - dT_iu
            res['T_a_ou_in'] = self.T0
        p0 = 101325
        h0 = PropsSI('H', 'P', p0, 'T', cu.C2K(self.T0), self.ref)
        s0 = PropsSI('S', 'P', p0, 'T', cu.C2K(self.T0), self.ref)

        if self.Q_r_iu == 0:
            self.E_cmp, self.E_fan_iu, self.E_fan_ou, self.E_tot = 0.0, 0.0, 0.0, 0.0
            self.m_r, self.COP, self.COP_sys, self.X_eff = 0.0, 0.0, 0.0, 0.0
            self.T1, self.T2, self.T3, self.T4 = self.T0, self.T0, self.T0, self.T0
            self.p1, self.p2, self.p3, self.p4 = 101325, 101325, 101325, 101325
            self.h1, self.h2, self.h3, self.h4 = 0.0, 0.0, 0.0, 0.0
            self.s1, self.s2, self.s3, self.s4 = 0.0, 0.0, 0.0, 0.0
            self.T_s_iu, self.T_s_ou = self.T_a_room, self.T0
            self.V_a_iu, self.m_a_iu, self.C_a_iu, self.eps_iu, self.NTU_iu = 0.0, 0.0, 0.0, 0.0, 0.0
            self.V_a_ou, self.m_a_ou, self.C_a_ou, self.eps_ou, self.NTU_ou = 0.0, 0.0, 0.0, 0.0, 0.0
            return

        def solve_with_dT(dt_iu, dt_ou):
            res = {'T_a_iu_in': self.T_a_room, 'T_a_iu_out': self.T_a_room - dt_iu,
                   'T_a_ou_in': self.T0, 'T_a_ou_out': self.T0 + dt_ou}

            # Indoor unit air flows and properties
            C_a_iu = self.Q_r_iu / dt_iu
            m_a_iu = C_a_iu / c_a
            V_a_iu = m_a_iu / rho_a
            NTU_iu = self.UA_iu / C_a_iu
            eps_iu = epsilon_from_NTU(NTU_iu)
            T_s_iu = Ts_from_otherfluid(res['T_a_iu_in'], res['T_a_iu_out'], eps_iu)

            p1 = PropsSI('P', 'T', cu.C2K(T_s_iu), 'Q', 1, self.ref)
            T1_K = cu.C2K(T_s_iu) + self.SH
            h1 = PropsSI('H', 'T', T1_K, 'P', p1, self.ref)
            s1 = PropsSI('S', 'T', T1_K, 'P', p1, self.ref)

            def residual(T_s_ou):
                try:
                    eps_ou = epsilon_from_temperature(res['T_a_ou_in'], res['T_a_ou_out'], T_s_ou)
                    if eps_ou <= 0 or eps_ou >= 1: return 1e9
                    NTU_ou = NTU_from_epsilon(eps_ou)
                    C_a_ou = self.UA_ou / NTU_ou
                    Q_air = C_a_ou * (res['T_a_ou_out'] - res['T_a_ou_in'])
                    p2 = PropsSI('P', 'T', cu.C2K(T_s_ou), 'Q', 1, self.ref)
                    T3 = T_s_ou - self.SC
                    h3 = PropsSI('H', 'T', cu.C2K(T3), 'P', p2, self.ref)
                    m_r = self.Q_r_iu / (h1 - h3) # h4 = h3
                    h2s = PropsSI('H', 'P', p2, 'S', s1, self.ref)
                    h2 = h1 + (h2s - h1) / self.eta_is
                    Q_ref = m_r * (h2 - h3)
                    return Q_ref - Q_air
                except: return 1e9

            # Efficient bisection for inner T_s_ou
            T_crit = PropsSI('Tcrit', self.ref) - 273.15
            lo, hi = res['T_a_ou_out'] + 0.1, T_crit - 1.0
            T_s_ou = (lo + hi) / 2
            res_lo = residual(lo)
            for _ in range(15):
                mid = (lo + hi) / 2
                res_mid = residual(mid)
                if abs(res_mid) < 1.0: T_s_ou = mid; break
                if res_lo * res_mid < 0: hi = mid
                else: lo = mid; res_lo = res_mid
                T_s_ou = mid

            eps_ou = epsilon_from_temperature(res['T_a_ou_in'], res['T_a_ou_out'], T_s_ou)
            NTU_ou = NTU_from_epsilon(eps_ou)
            C_a_ou = self.UA_ou / NTU_ou
            m_a_ou = C_a_ou / c_a
            V_a_ou = m_a_ou / rho_a

            p2 = PropsSI('P', 'T', cu.C2K(T_s_ou), 'Q', 1, self.ref)
            h2s = PropsSI('H', 'P', p2, 'S', s1, self.ref)
            h2 = h1 + (h2s - h1) / self.eta_is
            T3 = T_s_ou - self.SC
            h3 = PropsSI('H', 'T', cu.C2K(T3), 'P', p2, self.ref)
            s3 = PropsSI('S', 'T', cu.C2K(T3), 'P', p2, self.ref)
            m_r = self.Q_r_iu / (h1 - h3)

            E_cmp = (m_r * (h2 - h1)) / self.eta_el
            E_fan_iu = Fan().get_power(self.fan_iu, V_a_iu)
            E_fan_ou = Fan().get_power(self.fan_ou, V_a_ou)

            if E_cmp <= 0 or E_fan_iu <= 0 or E_fan_ou <= 0:
                return {'E_tot': 1e12}

            res.update({
                'E_tot': E_cmp + E_fan_iu + E_fan_ou, 'E_cmp': E_cmp, 'E_fan_iu': E_fan_iu, 'E_fan_ou': E_fan_ou,
                'm_r': m_r, 'h1': h1, 'h2': h2, 'h3': h3, 'h4': h3,
                's1': s1, 's2': PropsSI('S', 'H', h2, 'P', p2, self.ref), 's3': s3, 's4': PropsSI('S', 'H', h3, 'P', p1, self.ref),
                'p1': p1, 'p2': p2, 'p3': p2, 'p4': p1, 'T_s_iu': T_s_iu, 'T_s_ou': T_s_ou,
                'T1': T1_K-273.15, 'T2': PropsSI('T', 'H', h2, 'P', p2, self.ref)-273.15, 'T3': T3, 'T4': PropsSI('T', 'H', h3, 'P', p1, self.ref)-273.15,
                'V_a_iu': V_a_iu, 'm_a_iu': m_a_iu, 'C_a_iu': C_a_iu, 'eps_iu': eps_iu, 'NTU_iu': NTU_iu,
                'V_a_ou': V_a_ou, 'm_a_ou': m_a_ou, 'C_a_ou': C_a_ou, 'eps_ou': eps_ou, 'NTU_ou': NTU_ou
            })
            return res

        # 2D Grid Search for minimizing E_tot
        best_E = float('inf')
        best_res = None

        # Coarse Grid (Reduced to 3x3 for speed)
        dtiu_coarse = [7, 10, 13]
        dtou_coarse = [6, 15, 23]
        for dt_iu in dtiu_coarse:
            for dt_ou in dtou_coarse:
                try:
                    r = solve_with_dT(dt_iu, dt_ou)
                    if r['E_tot'] < best_E:
                        best_E = r['E_tot']
                        best_res = r
                        best_dt_iu, best_dt_ou = dt_iu, dt_ou
                except: continue

        # Fine Grid (Reduced to 3x3 around best)
        if best_res:
            dtiu_fine = [best_dt_iu-1.5, best_dt_iu, best_dt_iu+1.5]
            dtou_fine = [best_dt_ou-4.0, best_dt_ou, best_dt_ou+4.0]
            for dt_iu in dtiu_fine:
                for dt_ou in dtou_fine:
                    if dt_iu in dtiu_coarse and dt_ou in dtou_coarse: continue
                    try:
                        r = solve_with_dT(dt_iu, dt_ou)
                        if r['E_tot'] < best_E:
                            best_E = r['E_tot']
                            best_res = r
                    except: continue

        if not best_res:
            # Fallback to defaults if optimization fails
            best_res = solve_with_dT(8, 8)

        # Apply results to instance
        for k, v in best_res.items(): setattr(self, k, v)

        # Calculate derived exergy/energy metrics for best point
        p0 = 101325
        h0 = PropsSI('H', 'P', p0, 'T', cu.C2K(self.T0), self.ref)
        s0 = PropsSI('S', 'P', p0, 'T', cu.C2K(self.T0), self.ref)
        T0_K = cu.C2K(self.T0)

        self.E_tot = self.E_cmp + self.E_fan_iu + self.E_fan_ou
        self.COP_sys = self.Q_r_iu / self.E_tot
        self.COP = self.Q_r_iu / self.E_cmp
        self.Q_ou = self.m_r * (self.h2 - self.h3)
        self.X_in_tot = self.E_tot

        # Energy and Entropy flows
        self.E1, self.E2, self.E3, self.E4 = self.m_r*self.h1, self.m_r*self.h2, self.m_r*self.h3, self.m_r*self.h4
        self.S1, self.S2, self.S3, self.S4 = self.m_r*self.s1, self.m_r*self.s2, self.m_r*self.s3, self.m_r*self.s4
        self.S_cmp, self.S_fan_iu, self.S_fan_ou = 0.0, 0.0, 0.0

        # Air exergy and energy/entropy flows (Use standardized attributes)
        self.Q_a_iu_in = c_a * rho_a * self.V_a_iu * cu.C2K(self.T_a_iu_in)
        self.Q_a_iu_out = c_a * rho_a * self.V_a_iu * cu.C2K(self.T_a_iu_out)
        self.Q_a_ou_in = c_a * rho_a * self.V_a_ou * cu.C2K(self.T_a_ou_in)
        self.Q_a_ou_out = c_a * rho_a * self.V_a_ou * cu.C2K(self.T_a_ou_out)

        self.S_a_iu_in = c_a * rho_a * self.V_a_iu * math.log(cu.C2K(self.T_a_iu_in))
        self.S_a_iu_out = c_a * rho_a * self.V_a_iu * math.log(cu.C2K(self.T_a_iu_out))
        self.S_a_ou_in = c_a * rho_a * self.V_a_ou * math.log(cu.C2K(self.T_a_ou_in))
        self.S_a_ou_out = c_a * rho_a * self.V_a_ou * math.log(cu.C2K(self.T_a_ou_out))

        # Entropy generation
        self.S_g_cmp = self.S2 - self.S1
        self.S_g_ou  = (self.S3 + self.S_a_ou_out) - (self.S2 + self.S_a_ou_in)
        self.S_g_exp = self.S4 - self.S3
        self.S_g_iu  = (self.S1 + self.S_a_iu_out) - (self.S4 + self.S_a_iu_in)

        self.X_a_iu_in = c_a * rho_a * self.V_a_iu * ((cu.C2K(self.T_a_iu_in) - T0_K) - T0_K * math.log(cu.C2K(self.T_a_iu_in)/T0_K))
        self.X_a_iu_out = c_a * rho_a * self.V_a_iu * ((cu.C2K(self.T_a_iu_out) - T0_K) - T0_K * math.log(cu.C2K(self.T_a_iu_out)/T0_K))
        self.X_a_ou_in = c_a * rho_a * self.V_a_ou * ((cu.C2K(self.T_a_ou_in) - T0_K) - T0_K * math.log(cu.C2K(self.T_a_ou_in)/T0_K))
        self.X_a_ou_out = c_a * rho_a * self.V_a_ou * ((cu.C2K(self.T_a_ou_out) - T0_K) - T0_K * math.log(cu.C2K(self.T_a_ou_out)/T0_K))

        # Exergy components (Simplify for optimization context)
        self.X1, self.X2 = _calc_X_flow(self.m_r, [self.h1, self.h2], [self.s1, self.s2], h0, s0, T0_K)
        self.X3, self.X4 = _calc_X_flow(self.m_r, [self.h3, self.h4], [self.s3, self.s4], h0, s0, T0_K)
        self.X_cmp = self.E_cmp
        self.X_fan_iu, self.X_fan_ou = self.E_fan_iu, self.E_fan_ou
        self.X_in_cmp, self.X_out_cmp, self.X_c_cmp = self.X_cmp + self.X1, self.X2, (self.X_cmp + self.X1 - self.X2)
        self.X_in_ou, self.X_out_ou, self.X_c_ou = self.X_fan_ou + self.X_a_ou_in + self.X2, self.X3 + self.X_a_ou_out, (self.X_fan_ou + self.X_a_ou_in + self.X2 - self.X3 - self.X_a_ou_out)
        self.X_in_exp, self.X_out_exp, self.X_c_exp = self.X3, self.X4, self.X3 - self.X4
        self.X_in_iu, self.X_out_iu, self.X_c_iu = self.X_fan_iu + self.X_a_iu_in + self.X4, self.X1 + self.X_a_iu_out, (self.X_fan_iu + self.X_a_iu_in + self.X4 - self.X1 - self.X_a_iu_out)

        # Outdoor unit of ASHP
        self.X_in_ou = self.X_fan_ou + self.X_a_ou_in + self.X2
        self.X_out_ou = self.X3 + self.X_a_ou_out
        self.X_c_ou = self.X_in_ou - self.X_out_ou

        # Expansion valve of ASHP
        self.X_in_exp = self.X3
        self.X_out_exp = self.X4
        self.X_c_exp = self.X_in_exp - self.X_out_exp

        # Indoor unit of ASHP
        self.X_in_iu = self.X_fan_iu + self.X_a_iu_in + self.X4
        self.X_out_iu = self.X1 + self.X_a_iu_out
        self.X_c_iu = self.X_in_iu - self.X_out_iu

        # Exergy efficiency (System level - modified to use only product exergy)
        self.X_in_tot = self.X_cmp + self.X_fan_iu + self.X_fan_ou
        self.X_out_tot = self.X_a_iu_out - self.X_a_iu_in # Exergy gain of indoor air
        self.X_eff = self.X_out_tot / self.X_in_tot

        self.energy_balance = {
            "Compressor": {"in": {"E_cmp": self.E_cmp, "E1": self.E1}, "out": {"E2": self.E2}},
            "Outdoor unit": {"in": {"E_fan_ou": self.E_fan_ou, "Q_a_ou_in": self.Q_a_ou_in, "E2": self.E2}, "out": {"Q_a_ou_out": self.Q_a_ou_out, "E3": self.E3}},
            "Expansion valve": {"in": {"E3": self.E3}, "out": {"E4": self.E4}},
            "Indoor unit": {"in": {"E_fan_iu": self.E_fan_iu, "Q_a_iu_in": self.Q_a_iu_in, "E4": self.E4}, "out": {"Q_a_iu_out": self.Q_a_iu_out, "E1": self.E1}}
        }

        self.entropy_balance = {
            "Compressor": {"in": {"S_cmp": self.S_cmp, "S1": self.S1}, "gen": {"S_g_cmp": self.S_g_cmp}, "out": {"S2": self.S2}},
            "Outdoor unit": {"in": {"S_fan_ou": self.S_fan_ou, "S_a_ou_in": self.S_a_ou_in, "S2": self.S2}, "gen": {"S_g_ou": self.S_g_ou}, "out": {"S_a_ou_out": self.S_a_ou_out, "S3": self.S3}},
            "Expansion valve": {"in": {"S3": self.S3}, "gen": {"S_g_exp": self.S_g_exp}, "out": {"S4": self.S4}},
            "Indoor unit": {"in": {"S_fan_iu": self.S_fan_iu, "S_a_iu_in": self.S_a_iu_in, "S4": self.S4}, "gen": {"S_g_iu": self.S_g_iu}, "out": {"S_a_iu_out": self.S_a_iu_out, "S1": self.S1}}
        }

        self.exergy_balance = {
            "Compressor": {"in": {"X_cmp": self.X_cmp, "X1": self.X1}, "con": {"X_c_cmp": self.X_c_cmp}, "out": {"X2": self.X2}},
            "Outdoor unit": {"in": {"X_fan_ou": self.X_fan_ou, "X_a_ou_in": self.X_a_ou_in, "X2": self.X2}, "con": {"X_c_ou": self.X_c_ou}, "out": {"X_a_ou_out": self.X_a_ou_out, "X3": self.X3}},
            "Expansion valve": {"in": {"X3": self.X3}, "con": {"X_c_exp": self.X_c_exp}, "out": {"X4": self.X4}},
            "Indoor unit": {"in": {"X_fan_iu": self.X_fan_iu, "X_a_iu_in": self.X_a_iu_in, "X4": self.X4}, "con": {"X_c_iu": self.X_c_iu}, "out": {"X_a_iu_out": self.X_a_iu_out, "X1": self.X1}}
        }

@dataclass
class AirSourceHeatPump_heating:
    """Air source heat pump model for heating mode (4 sub-systems)."""

    def __post_init__(self):
        # Refrigerant state
        self.ref = 'R32' # refrigerant
        self.SH = 6 # superheat [K]
        self.SC = 6 # subcooling [K]

        # Temperature
        self.T0 = 0 # environment temperature [°C]
        self.T_a_room = 21 # room air temperature [°C]

        # Load
        self.Q_r_iu = 6000 # thermal load [W] (Heating)

        # Heat exchanger UA
        self.UA_iu = 800 # indoor unit heat exchanger UA [W/K]
        self.UA_ou = 800 # outdoor unit heat exchanger UA [W/K]

        # Compressor parameter
        self.eta_is = 0.7 # isentropic efficiency [-]
        self.eta_el = 0.95 # electrical efficiency [-]

        # Fan parameter
        self.fan_iu = Fan().fan1 # indoor unit fan
        self.fan_ou = Fan().fan2 # outdoor unit fan

    def system_update(self):
        if self.Q_r_iu == 0:
            self.E_cmp, self.E_fan_iu, self.E_fan_ou, self.E_tot = 0.0, 0.0, 0.0, 0.0
            self.m_r, self.COP, self.COP_sys, self.X_eff = 0.0, 0.0, 0.0, 0.0
            self.T1, self.T2, self.T3, self.T4 = self.T0, self.T0, self.T0, self.T0
            self.p1, self.p2, self.p3, self.p4 = 101325, 101325, 101325, 101325
            self.h1, self.h2, self.h3, self.h4 = 0.0, 0.0, 0.0, 0.0
            self.s1, self.s2, self.s3, self.s4 = 0.0, 0.0, 0.0, 0.0
            self.T_s_iu, self.T_s_ou = self.T_a_room, self.T0
            self.V_a_iu, self.m_a_iu, self.C_a_iu, self.eps_iu, self.NTU_iu = 0.0, 0.0, 0.0, 0.0, 0.0
            self.V_a_ou, self.m_a_ou, self.C_a_ou, self.eps_ou, self.NTU_ou = 0.0, 0.0, 0.0, 0.0, 0.0
            return

        def solve_with_dT(dt_iu, dt_ou):
            res = {'T_a_iu_in': self.T_a_room, 'T_a_iu_out': self.T_a_room + dt_iu,
                   'T_a_ou_in': self.T0, 'T_a_ou_out': self.T0 - dt_ou}

            # Indoor unit (Condenser) air flows and properties
            C_a_iu = self.Q_r_iu / dt_iu
            m_a_iu = C_a_iu / c_a
            V_a_iu = m_a_iu / rho_a
            NTU_iu = self.UA_iu / C_a_iu
            eps_iu = epsilon_from_NTU(NTU_iu)
            T_s_iu = Ts_from_otherfluid(res['T_a_iu_in'], res['T_a_iu_out'], eps_iu)

            # Refrigerant states at condenser
            p2 = PropsSI('P', 'T', cu.C2K(T_s_iu), 'Q', 1, self.ref)
            T3 = T_s_iu - self.SC
            h3 = PropsSI('H', 'T', cu.C2K(T3), 'P', p2, self.ref)
            s3 = PropsSI('S', 'T', cu.C2K(T3), 'P', p2, self.ref)

            def residual(T_s_ou):
                try:
                    eps_ou = epsilon_from_temperature(res['T_a_ou_in'], res['T_a_ou_out'], T_s_ou)
                    if eps_ou <= 0 or eps_ou >= 1: return 1e9
                    NTU_ou = NTU_from_epsilon(eps_ou)
                    C_a_ou = self.UA_ou / NTU_ou
                    Q_air = C_a_ou * (res['T_a_ou_in'] - res['T_a_ou_out'])
                    p1 = PropsSI('P', 'T', cu.C2K(T_s_ou), 'Q', 1, self.ref)
                    T1 = T_s_ou + self.SH
                    h1 = PropsSI('H', 'T', cu.C2K(T1), 'P', p1, self.ref)
                    s1 = PropsSI('S', 'T', cu.C2K(T1), 'P', p1, self.ref)
                    h2s = PropsSI('H', 'P', p2, 'S', s1, self.ref)
                    h2 = h1 + (h2s - h1) / self.eta_is
                    m_r = self.Q_r_iu / (h2 - h3)
                    Q_ref = m_r * (h1 - h3) # m_r * (h1 - h4) where h4=h3
                    return Q_ref - Q_air
                except: return 1e9

            # Efficient bisection for inner T_s_ou
            lo, hi = -50.0, res['T_a_ou_out'] - 0.1
            T_s_ou = (lo + hi) / 2
            res_lo = residual(lo)
            for _ in range(15):
                mid = (lo + hi) / 2
                res_mid = residual(mid)
                if abs(res_mid) < 1.0: T_s_ou = mid; break
                if res_lo * res_mid < 0: hi = mid
                else: lo = mid; res_lo = res_mid
                T_s_ou = mid

            eps_ou = epsilon_from_temperature(res['T_a_ou_in'], res['T_a_ou_out'], T_s_ou)
            NTU_ou = NTU_from_epsilon(eps_ou)
            C_a_ou = self.UA_ou / NTU_ou
            m_a_ou = C_a_ou / c_a
            V_a_ou = m_a_ou / rho_a

            # Final properties at optimal T_s_ou point for this dT pair
            p1 = PropsSI('P', 'T', cu.C2K(T_s_ou), 'Q', 1, self.ref)
            T1_K = cu.C2K(T_s_ou) + self.SH
            h1 = PropsSI('H', 'T', T1_K, 'P', p1, self.ref)
            s1 = PropsSI('S', 'T', T1_K, 'P', p1, self.ref)
            h2s = PropsSI('H', 'P', p2, 'S', s1, self.ref)
            h2 = h1 + (h2s - h1) / self.eta_is
            m_r = self.Q_r_iu / (h2 - h3)

            E_cmp = (m_r * (h2 - h1)) / self.eta_el
            E_fan_iu = Fan().get_power(self.fan_iu, V_a_iu)
            E_fan_ou = Fan().get_power(self.fan_ou, V_a_ou)

            if E_cmp <= 0 or E_fan_iu <= 0 or E_fan_ou <= 0:
                return {'E_tot': 1e12}

            res.update({
                'E_tot': E_cmp + E_fan_iu + E_fan_ou, 'E_cmp': E_cmp, 'E_fan_iu': E_fan_iu, 'E_fan_ou': E_fan_ou,
                'm_r': m_r, 'h1': h1, 'h2': h2, 'h3': h3, 'h4': h3,
                's1': s1, 's2': PropsSI('S', 'H', h2, 'P', p2, self.ref), 's3': s3, 's4': PropsSI('S', 'H', h3, 'P', p1, self.ref),
                'p1': p1, 'p2': p2, 'p3': p2, 'p4': p1, 'T_s_iu': T_s_iu, 'T_s_ou': T_s_ou,
                'T1': T1_K-273.15, 'T2': PropsSI('T', 'H', h2, 'P', p2, self.ref)-273.15, 'T3': T3, 'T4': PropsSI('T', 'H', h3, 'P', p1, self.ref)-273.15,
                'V_a_iu': V_a_iu, 'm_a_iu': m_a_iu, 'C_a_iu': C_a_iu, 'eps_iu': eps_iu, 'NTU_iu': NTU_iu,
                'V_a_ou': V_a_ou, 'm_a_ou': m_a_ou, 'C_a_ou': C_a_ou, 'eps_ou': eps_ou, 'NTU_ou': NTU_ou
            })
            return res

        # 2D Grid Search for minimizing E_tot
        best_E = float('inf')
        best_res = None

        # Coarse Grid (Reduced to 3x3 for speed)
        dtiu_coarse = [7, 10, 13]
        dtou_coarse = [5, 9, 13]
        for dt_iu in dtiu_coarse:
            for dt_ou in dtou_coarse:
                try:
                    r = solve_with_dT(dt_iu, dt_ou)
                    if r['E_tot'] < best_E:
                        best_E = r['E_tot']
                        best_res = r
                        best_dt_iu, best_dt_ou = dt_iu, dt_ou
                except: continue

        # Fine Grid (Reduced to 3x3 around best)
        if best_res:
            dtiu_fine = [best_dt_iu-1.5, best_dt_iu, best_dt_iu+1.5]
            dtou_fine = [best_dt_ou-2.0, best_dt_ou, best_dt_ou+2.0]
            for dt_iu in dtiu_fine:
                for dt_ou in dtou_fine:
                    if dt_iu in dtiu_coarse and dt_ou in dtou_coarse: continue
                    try:
                        r = solve_with_dT(dt_iu, dt_ou)
                        if r['E_tot'] < best_E:
                            best_E = r['E_tot']
                            best_res = r
                    except: continue

        if not best_res:
            best_res = solve_with_dT(8, 5)

        # Apply results to instance
        for k, v in best_res.items(): setattr(self, k, v)

        # Calculate derived exergy/energy metrics for best point
        p0 = 101325
        h0 = PropsSI('H', 'P', p0, 'T', cu.C2K(self.T0), self.ref)
        s0 = PropsSI('S', 'P', p0, 'T', cu.C2K(self.T0), self.ref)
        T0_K = cu.C2K(self.T0)

        self.E_tot = self.E_cmp + self.E_fan_iu + self.E_fan_ou
        self.COP_sys = self.Q_r_iu / self.E_tot
        self.COP = self.Q_r_iu / self.E_cmp
        self.Q_ou = self.m_r * (self.h1 - self.h4)
        self.X_in_tot = self.E_tot

        # Energy and Entropy flows
        self.E1, self.E2, self.E3, self.E4 = self.m_r*self.h1, self.m_r*self.h2, self.m_r*self.h3, self.m_r*self.h4
        self.S1, self.S2, self.S3, self.S4 = self.m_r*self.s1, self.m_r*self.s2, self.m_r*self.s3, self.m_r*self.s4
        self.S_cmp, self.S_fan_iu, self.S_fan_ou = 0.0, 0.0, 0.0

        # Air exergy and energy/entropy flows (Use standardized attributes)
        self.Q_a_iu_in = c_a * rho_a * self.V_a_iu * cu.C2K(self.T_a_iu_in)
        self.Q_a_iu_out = c_a * rho_a * self.V_a_iu * cu.C2K(self.T_a_iu_out)
        self.Q_a_ou_in = c_a * rho_a * self.V_a_ou * cu.C2K(self.T_a_ou_in)
        self.Q_a_ou_out = c_a * rho_a * self.V_a_ou * cu.C2K(self.T_a_ou_out)

        self.S_a_iu_in = c_a * rho_a * self.V_a_iu * math.log(cu.C2K(self.T_a_iu_in))
        self.S_a_iu_out = c_a * rho_a * self.V_a_iu * math.log(cu.C2K(self.T_a_iu_out))
        self.S_a_ou_in = c_a * rho_a * self.V_a_ou * math.log(cu.C2K(self.T_a_ou_in))
        self.S_a_ou_out = c_a * rho_a * self.V_a_ou * math.log(cu.C2K(self.T_a_ou_out))

        # Entropy generation
        self.S_g_cmp = self.S2 - self.S1
        self.S_g_iu  = (self.S3 + self.S_a_iu_out) - (self.S2 + self.S_a_iu_in)
        self.S_g_exp = self.S4 - self.S3
        self.S_g_ou  = (self.S1 + self.S_a_ou_out) - (self.S4 + self.S_a_ou_in)

        self.X_a_iu_in = c_a * rho_a * self.V_a_iu * ((cu.C2K(self.T_a_iu_in) - T0_K) - T0_K * math.log(cu.C2K(self.T_a_iu_in)/T0_K))
        self.X_a_iu_out = c_a * rho_a * self.V_a_iu * ((cu.C2K(self.T_a_iu_out) - T0_K) - T0_K * math.log(cu.C2K(self.T_a_iu_out)/T0_K))
        self.X_a_ou_in = c_a * rho_a * self.V_a_ou * ((cu.C2K(self.T_a_ou_in) - T0_K) - T0_K * math.log(cu.C2K(self.T_a_ou_in)/T0_K))
        self.X_a_ou_out = c_a * rho_a * self.V_a_ou * ((cu.C2K(self.T_a_ou_out) - T0_K) - T0_K * math.log(cu.C2K(self.T_a_ou_out)/T0_K))

        # Exergy components (Simplify for optimization context)
        self.X1, self.X2 = _calc_X_flow(self.m_r, [self.h1, self.h2], [self.s1, self.s2], h0, s0, T0_K)
        self.X3, self.X4 = _calc_X_flow(self.m_r, [self.h3, self.h4], [self.s3, self.s4], h0, s0, T0_K)
        self.X_cmp = self.E_cmp
        self.X_fan_iu, self.X_fan_ou = self.E_fan_iu, self.E_fan_ou
        self.X_in_cmp, self.X_out_cmp, self.X_c_cmp = self.X_cmp + self.X1, self.X2, (self.X_cmp + self.X1 - self.X2)
        self.X_in_iu, self.X_out_iu, self.X_c_iu = self.X_fan_iu + self.X_a_iu_in + self.X2, self.X3 + self.X_a_iu_out, (self.X_fan_iu + self.X_a_iu_in + self.X2 - self.X3 - self.X_a_iu_out)
        self.X_in_exp, self.X_out_exp, self.X_c_exp = self.X3, self.X4, self.X3 - self.X4
        self.X_in_ou, self.X_out_ou, self.X_c_ou = self.X_fan_ou + self.X_a_ou_in + self.X4, self.X1 + self.X_a_ou_out, (self.X_fan_ou + self.X_a_ou_in + self.X4 - self.X1 - self.X_a_ou_out)

        # Exergy efficiency (System level)
        self.X_in_tot = self.X_cmp + self.X_fan_iu + self.X_fan_ou
        self.X_out_tot = self.X_a_iu_out - self.X_a_iu_in # Exergy gain of indoor air
        self.X_eff = self.X_out_tot / self.X_in_tot

        self.energy_balance = {
            "Compressor": {"in": {"E_cmp": self.E_cmp, "E1": self.E1}, "out": {"E2": self.E2}},
            "Outdoor unit": {"in": {"E_fan_ou": self.E_fan_ou, "Q_a_ou_in": self.Q_a_ou_in, "E4": self.E4}, "out": {"Q_a_ou_out": self.Q_a_ou_out, "E1": self.E1}},
            "Expansion valve": {"in": {"E3": self.E3}, "out": {"E4": self.E4}},
            "Indoor unit": {"in": {"E_fan_iu": self.E_fan_iu, "Q_a_iu_in": self.Q_a_iu_in, "E2": self.E2}, "out": {"Q_a_iu_out": self.Q_a_iu_out, "E3": self.E3}}
        }

        self.entropy_balance = {
            "Compressor": {"in": {"S_cmp": self.S_cmp, "S1": self.S1}, "gen": {"S_g_cmp": self.S_g_cmp}, "out": {"S2": self.S2}},
            "Outdoor unit": {"in": {"S_fan_ou": self.S_fan_ou, "S_a_ou_in": self.S_a_ou_in, "S4": self.S4}, "gen": {"S_g_ou": self.S_g_ou}, "out": {"S_a_ou_out": self.S_a_ou_out, "S1": self.S1}},
            "Expansion valve": {"in": {"S3": self.S3}, "gen": {"S_g_exp": self.S_g_exp}, "out": {"S4": self.S4}},
            "Indoor unit": {"in": {"S_fan_iu": self.S_fan_iu, "S_a_iu_in": self.S_a_iu_in, "S2": self.S2}, "gen": {"S_g_iu": self.S_g_iu}, "out": {"S_a_iu_out": self.S_a_iu_out, "S3": self.S3}}
        }

        self.exergy_balance = {
            "Compressor": {"in": {"X_cmp": self.X_cmp, "X1": self.X1}, "con": {"X_c_cmp": self.X_c_cmp}, "out": {"X2": self.X2}},
            "Outdoor unit": {"in": {"X_fan_ou": self.X_fan_ou, "X_a_ou_in": self.X_a_ou_in, "X4": self.X4}, "con": {"X_c_ou": self.X_c_ou}, "out": {"X_a_ou_out": self.X_a_ou_out, "X1": self.X1}},
            "Expansion valve": {"in": {"X3": self.X3}, "con": {"X_c_exp": self.X_c_exp}, "out": {"X4": self.X4}},
            "Indoor unit": {"in": {"X_fan_iu": self.X_fan_iu, "X_a_iu_in": self.X_a_iu_in, "X2": self.X2}, "con": {"X_c_iu": self.X_c_iu}, "out": {"X_a_iu_out": self.X_a_iu_out, "X3": self.X3}}
        }
