from dataclasses import dataclass

from scipy.optimize import fsolve

from .. import calc_util as cu
from ..constants import sigma
from ..heat_transfer import calc_h_vertical_plate


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
