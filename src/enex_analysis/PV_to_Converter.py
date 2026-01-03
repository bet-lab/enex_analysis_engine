#%%
from dataclasses import dataclass
from . import calc_util as cu
from .constants import k_D, k_d

@dataclass
class PV_to_Converter:
    """
    PV 시스템 (PV Cell -> Controller -> Battery -> DC/AC Converter)의
    에너지, 엔트로피, 엑서지 밸런스를 계산하는 클래스.
    
    모든 온도 입력은 켈빈(K) 단위입니다.
    """

    # --- 입력 인자 (가정값) ---
    # 환경 및 설치 조건
    def __post_init__(self):
        self.A_pv = 5.0       # Area of panel surface [m²]
        self.alp_pv = 0.9    # Absorptivity of PV panel surface [-]
        self.I_DN = 500.0      # Direct normal solar radiation [W/m²]
        self.I_dH = 150.0      # Diffuse radiation [W/m²]
        self.h_o = 15.0        # Overall outdoor heat transfer coefficient [W/(m²·K)]

        # 컴포넌트 특성 (효율)
        self.eta_pv = 0.20     # PV panel efficiency [-] 17% ~ 25%
        self.eta_ctrl = 0.95   # Controller efficiency [-] 98% ~ 99.5% 
 
        self.eta_batt = 0.90   # Battery efficiency [-] 90% ~ 98%
        self.eta_DC_AC = 0.95  # DC/AC converter efficiency [-] 95% ~ 99%

        # 컴포넌트 작동 온도 (가정)
        self.T0_C      = 20      # Environmental temperature [°C] (e.g., 25°C)
        self.T_ctrl_C  = 35  # Temperature of controller [°C] (e.g., 35°C)
        self.T_batt_C  = 40  # Temperature of battery [°C] (e.g., 40°C)
        self.T_DC_AC_C = 40  # Temperature of DC/AC converter [°C] (e.g., 40°C)
        
        # Unit conversion for temperatures
        self.T0      = cu.C2K(self.T0_C)
        self.T_ctrl  = cu.C2K(self.T_ctrl_C)
        self.T_batt  = cu.C2K(self.T_batt_C)
        self.T_DC_AC = cu.C2K(self.T_DC_AC_C)

    def system_update(self):
        """
        제공된 수식을 기반으로 전체 시스템의 에너지, 엔트로피,
        엑서지 밸런스 계산을 수행합니다.
        """

        # --- 단계 0: 초기 계산 ---
        self.I_sol = self.I_DN + self.I_dH

        # --- 단계 1: PV Cell ---
        # T_pv 계산 (에너지 밸런스 수식으로부터 유도)
        # A_pv*alp_pv*I_sol = E_pv0 + Q_l,pv
        # A_pv*alp_pv*I_sol = (A_pv*eta_pv*I_sol) + (2*A_pv*h_o*(T_pv - T_0))
        # I_sol * (alp_pv - eta_pv) = 2 * h_o * (T_pv - T_0)
        # T_pv = T_0 + (I_sol * (alp_pv - eta_pv)) / (2 * h_o)
        self.T_pv = self.T0 + (self.I_sol * (self.alp_pv - self.eta_pv)) / (2 * self.h_o)
        
        # 에너지 밸런스 (PV)
        self.E_pv0 = self.A_pv * self.eta_pv * self.I_sol
        self.Q_l_pv = 2 * self.A_pv * self.h_o * (self.T_pv - self.T0)
        
        # 엔트로피 밸런스 (PV)
        self.s_DN = k_D * self.I_DN ** 0.9
        self.s_dH = k_d * self.I_dH ** 0.9
        self.s_sol = self.s_DN + self.s_dH
        
        self.S_sol = self.A_pv * self.alp_pv * self.s_sol
        self.S_pv0 = (1 / float('inf')) * self.E_pv0
        self.S_l_pv = (1 / self.T_pv) * self.Q_l_pv
        self.S_g_pv = self.S_pv0 + self.S_l_pv - self.S_sol

        # 엑서지 밸런스 (PV)
        self.X_sol = self.A_pv * self.alp_pv * (self.I_sol - self.s_sol * self.T0)
        self.X_pv0 = self.E_pv0  
        self.X_l_pv = (1 - self.T0 / self.T_pv) * self.Q_l_pv
        self.X_c_pv = self.S_g_pv * self.T0

        # --- 단계 2: Controller ---
        # 에너지 밸런스 (Controller)
        self.E_pv1 = self.eta_ctrl * self.E_pv0
        self.Q_l_ctrl = (1 - self.eta_ctrl) * self.E_pv0

        # 엔트로피 밸런스 (Controller)
        self.S_pv1 = (1 / float('inf')) * self.E_pv1
        self.S_l_ctrl = (1 / self.T_ctrl) * self.Q_l_ctrl
        self.S_g_ctrl = self.S_pv1 + self.S_l_ctrl - self.S_pv0

        # 엑서지 밸런스 (Controller)
        self.X_pv1 = self.E_pv1 - self.S_pv1 * self.T0
        self.X_l_ctrl = self.Q_l_ctrl - self.S_l_ctrl * self.T0
        self.X_c_ctrl = self.S_g_ctrl * self.T0

        # --- 단계 3: Battery ---
        # 에너지 밸런스 (Battery)
        self.E_pv2 = self.eta_batt * self.E_pv1
        self.Q_l_batt = (1 - self.eta_batt) * self.E_pv1

        # 엔트로피 밸런스 (Battery)
        self.S_pv2 = (1 / float('inf')) * self.E_pv2
        self.S_l_batt = (1 / self.T_batt) * self.Q_l_batt
        self.S_g_batt = self.S_pv2 + self.S_l_batt - self.S_pv1

        # 엑서지 밸런스 (Battery)
        self.X_pv2 = self.E_pv2 - self.S_pv2 * self.T0
        self.X_l_batt = self.Q_l_batt - self.S_l_batt * self.T0
        self.X_c_batt = self.S_g_batt * self.T0

        # --- 단계 4: DC/AC Converter ---
        # 에너지 밸런스 (DC/AC)
        self.E_pv3 = self.eta_DC_AC * self.E_pv2
        # 수식 오타 수정: Q_l = (1-eta)*E_pv1 -> (1-eta)*E_pv2
        self.Q_l_DC_AC = (1 - self.eta_DC_AC) * self.E_pv2

        # 엔트로피 밸런스 (DC/AC)
        self.S_pv3 = (1 / float('inf')) * self.E_pv3
        self.S_l_DC_AC = (1 / self.T_DC_AC) * self.Q_l_DC_AC
        self.S_g_DC_AC = self.S_pv3 + self.S_l_DC_AC - self.S_pv2

        # 엑서지 밸런스 (DC/AC)
        self.X_pv3 = self.E_pv3 - self.S_pv3 * self.T0
        self.X_l_DC_AC = self.Q_l_DC_AC - self.S_l_DC_AC * self.T0
        self.X_c_DC_AC = self.S_g_DC_AC * self.T0

# %%

