#%%
import numpy as np
import math
from . import calc_util as cu
from dataclasses import dataclass
from scipy.optimize import minimize
from .enex_functions import (
    calc_ref_state,
)
import CoolProp.CoolProp as CP
from tqdm import tqdm
import pandas as pd

# Import constants from constants.py
from .constants import *

# Import functions from enex_functions.py
from .enex_functions import *

@dataclass
class GroundSourceHeatPumpBoiler:
    '''
    물리적 원리에 기반한 지열워 히트펌프 성능 계산 및 최적 운전점 탐색 클래스.
    '''
    def __init__(
        self,

        # 1. 냉매/사이클/압축기 파라미터 -------------------------------
        refrigerant    = 'R410A',
        V_disp_cmp     = 0.0005,
        eta_cmp_isen   = 0.7,

        # 2. 열교환기 파라미터 -----------------------------------------
        UA_cond = 500,   # W/K
        UA_evap = 500,   # W/K

        # 3. 탱크/제어/부하 파라미터 -----------------------------------
        #    (온도/제어 관련)
        T0                    = 0.0,    # [°C] 기준 외기 온도
        Ts                    = 16.0,   # [°C] 지중 온도/초기값
        T_tank_w_setpoint     = 65.0,   # [°C] 저탕조 설정 온도
        T_tank_w_lower_bound  = 55.0,   # [°C] 저탕조 하한 온도
        T_serv_w              = 40.0,   # [°C] 서비스 급탕 온도
        T_sup_w               = 15.0,   # [°C] 급수(상수도) 온도

        heater_capacity       = 8000.0,    # [W] 히터 최대 용량
        dV_w_serv_m3s         = 0.0001,    # [m³/s] 최대 급탕 유량

        #   (탱크/보온 관련)
        r0       = 0.2,   # [m] 탱크 반지름
        H        = 0.8,   # [m] 탱크 높이
        x_shell  = 0.01,  # [m] 탱크 외벽 두께
        x_ins    = 0.05,  # [m] 단열재 두께
        k_shell  = 25,    # [W/mK] 탱크 외벽 열전도도
        k_ins    = 0.03,  # [W/mK] 단열재 열전도도
        h_o      = 15,    # [W/m²K] 외부 대류 열전달계수

        # 4. 지중/보어홀/유체 파라미터 -------------------------------
        #   (보어홀)
        D_b = 0,          # [m] 보어홀 깊이 (unused/option)
        H_b = 200,        # [m] 보어홀 유효 길이
        r_b = 0.08,       # [m] 보어홀 반지름
        R_b = 0.108,      # [mK/W] 유효 보어홀 열저항

        #   (유체)
        dV_b_f_lpm = 24,  # [L/min] 지중 유체 유량

        #   (토양)
        k_s   = 2.0,      # [W/mK] 토양 열전도도
        c_s   = 800,      # [J/kgK] 토양 비열
        rho_s = 2000,     # [kg/m³] 토양 밀도

        #   (순환 펌프)
        E_pmp = 200,      # [W] 펌프 소비전력

        # 5. UV 램프 파라미터 -----------------------------------------
        lamp_power_watts = 0, # [W] 램프 소비 전력
        uv_lamp_exposure_duration_min = 0, # [min] 1회 UV램프 노출 기준시간
        num_switching_per_3hour = 1, # [개] 3시간 당 on 횟수

        # 6. 과열도 및 과냉각도 설정 (Default 3도)
        dT_superheat = 3.0,  # [K] 증발기 출구 과열도 (State 1* -> 1)
        dT_subcool   = 3.0,  # [K] 응축기 출구 과냉각도 (State 3* -> 3)
        ):

        # --- 1. 기본 물성/상수 ---
        # 열용량, 물 밀도 등 전역 상수는 상단 import 혹은 별도 config로 관리 권장
        self.tank_physical = {
            'r0': r0, 'H': H, 'x_shell': x_shell, 'x_ins': x_ins,
            'k_shell': k_shell, 'k_ins': k_ins, 'h_o': h_o,
        }
        self.UA_tank = calc_simple_tank_UA(**self.tank_physical)
        self.C_tank = c_w * rho_w * (math.pi * r0**2 * H)

        # --- 2. 냉매/사이클/압축기 파라미터 ---
        self.ref_params = {
            'refrigerant': refrigerant,
            'V_disp_cmp': V_disp_cmp,
            'eta_cmp_isen': eta_cmp_isen,
        }
        self.ref = refrigerant
        self.V_disp_cmp = V_disp_cmp
        self.eta_cmp_isen = eta_cmp_isen

        # --- 3. 열교환기 파라미터 ---
        self.heat_exchanger = {
            'UA_cond': UA_cond,
            'UA_evap': UA_evap,
        }
        self.UA_cond = UA_cond
        self.UA_evap = UA_evap

        # --- 4. 탱크/제어/부하 파라미터 (온도/제어/유량/설정) ---
        self.control_params = {
            'T0': T0,
            'Ts': Ts,
            'T_tank_w_setpoint': T_tank_w_setpoint,
            'T_tank_w_lower_bound': T_tank_w_lower_bound,
            'T_serv_w': T_serv_w,
            'T_sup_w': T_sup_w,
            'heater_capacity': heater_capacity,
            'dV_w_serv_m3s': dV_w_serv_m3s,
        }
        self.Ts = Ts
        self.T_b_f_in = Ts   # 초기값, 시뮬레이션 진행중 갱신

        self.heater_capacity = heater_capacity
        self.dV_w_serv_m3s = dV_w_serv_m3s
        self.T_tank_w_setpoint = T_tank_w_setpoint
        self.T_tank_w_lower_bound = T_tank_w_lower_bound

        # --- 5. UV 램프 파라미터 ---
        self.lamp_power_watts = lamp_power_watts
        self.uv_lamp_exposure_duration_min = uv_lamp_exposure_duration_min
        self.num_switching_per_3hour = num_switching_per_3hour
        # UV 램프 관련 계산 상수
        self.period_3hour_sec = 3 * cu.h2s  # 3시간을 초 단위로 변환
        self.uv_lamp_exposure_duration_sec = uv_lamp_exposure_duration_min * cu.m2s  # 분을 초로 변환

        # --- 6. 과열/과냉각 변수 저장 ---
        self.dT_superheat = dT_superheat
        self.dT_subcool = dT_subcool
        self.T_sup_w = T_sup_w
        self.T_serv_w = T_serv_w

        # --- 5. 탱크/보어홀/지중/유체 파라미터 ---
        self.borehole = {
            'D_b': D_b, 'H_b': H_b, 'r_b': r_b, 'R_b': R_b,
        }
        self.D_b = D_b
        self.H_b = H_b
        self.r_b = r_b
        self.R_b = R_b

        self.k_s    = k_s
        self.c_s    = c_s
        self.alp_s  = k_s / (c_s * rho_s)  # 토양 열확산계수 [m²/s]
        self.rho_s  = rho_s
        self.E_pmp  = E_pmp
        
        # ============================================================
        # 6단계: 단위 변환 및 추가 상수 설정
        # ============================================================
        # 온도 단위 변환: °C → K (냉매 계산용)
        self.T0_K       = cu.C2K(T0)
        self.Ts_K       = cu.C2K(self.Ts)
        self.T_b_f_in_K = cu.C2K(self.T_b_f_in)
        self.T_sup_w_K  = cu.C2K(T_sup_w)
        self.T_serv_w_K = cu.C2K(T_serv_w)
        self.dV_b_f_m3s = dV_b_f_lpm * cu.L2m3/cu.m2s  
        
        self.Q_cond_LOAD_OFF_TOL = 500.0     # [W] 이하면 완전 OFF
        
    def _calc_on_state(self, optimization_vars, T_tank_w, Q_cond_load, T0):
        """
        지열원 히트펌프 보일러(GSHPB)의 사이클 성능을 계산하는 메서드.
        
        이 메서드는 최적화 변수(optimization_vars)를 받아 히트펌프 사이클 성능을 계산합니다.
        최적화 과정에서 반복적으로 호출되어 목적 함수와 제약 조건을 평가하는 데 사용됩니다.
        
        주요 작업:
        1. 최적화 변수 언패킹 (온도차 추출)
        2. 증발 및 응축 온도 계산
        3. 공통 사이클 상태 계산
        4. 냉매 유량 및 성능 데이터 계산
        5. LMTD 기반 열량 계산 (응축기, 증발기)
        6. 지중열 교환 계산
        7. 엑서지 계산
        8. 최종 결과 딕셔너리 생성
        
        호출 관계:
        - 호출자: _optimize_operation (본 클래스)
        - 호출 함수: 
            - calc_ref_state (enex_functions.py)
        
        데이터 흐름:
        ──────────────────────────────────────────────────────────────────────────
        [optimization_vars, T_tank_w, Q_cond_load, T0]
            ↓
        증발/응축 온도 계산 (T_evap_K, T_cond_K)
            ↓
        calc_ref_state
            ↓ [State 1-4 물성치]
        냉매 유량 계산 (m_dot_ref)
            ↓
        성능 데이터 계산 (Q_ref_cond, Q_ref_evap, E_cmp)
            ↓
        LMTD 기반 열량 계산 및 지중열 교환 계산
            ↓
        [포맷팅된 결과 딕셔너리]
        
        Args:
            optimization_vars (list): 최적화 변수 배열 [dT_ref_HX, dT_ref_cond]
                - dT_ref_HX (float): 냉매-열교환기 온도차 [K]
                    증발 온도 = 지중 유체 입구 온도 - dT_ref_HX
                - dT_ref_cond (float): 냉매-저탕조 온도차 [K]
                    응축 온도 = 저탕조 온도 + dT_ref_cond
            
            T_tank_w (float): 저탕조 온도 [°C]
                현재 타임스텝의 저탕조 온도
            
            Q_cond_load (float): 저탕조 목표 열 교환율 [W]
                응축기가 저탕조에 전달해야 하는 목표 열량
                이 값을 만족하는 최적 운전점을 탐색
            
            T0 (float): 엑서지 분석 기준 온도(=외기온도) [°C]
                엑서지 계산의 기준점으로 사용되는 외기 온도
        
        Returns:
            dict: 사이클 성능 결과 딕셔너리 (단위 포함)
                None: 계산 실패 시 (예: h3 == h2인 경우)
        
        Notes:
            - 냉매 유량 계산: m_dot_ref = Q_cond_load / (h2 - h3)
                목표 열 교환율을 만족하기 위한 필요한 냉매 유량
            - 응축기 열량: Q_ref_cond = m_dot_ref * (h2 - h3)
                이 값은 Q_cond_load와 동일해야 함 (계산 검증)
            - 증발기 열량: Q_ref_evap = m_dot_ref * (h1 - h4)
                지중열교환기에서 흡수하는 열량
            - 압축기 전력: E_cmp = m_dot_ref * (h2 - h1)
                최적화의 목적 함수 (최소화 대상)
        """
        # ============================================================
        # 1단계: 최적화 변수 언패킹
        # ============================================================
        dT_ref_HX = optimization_vars[0]      # 냉매-열교환기 온도차 [K]
        dT_ref_cond = optimization_vars[1]    # 냉매-저탕조 온도차 [K]
        
        # ============================================================
        # 2단계: 온도 단위 변환 및 증발/응축 온도 계산
        # ============================================================
        T_tank_w_K = cu.C2K(T_tank_w)         # 저탕조 온도 [K]
        T0_K = cu.C2K(T0)                     # 기준 온도 [K]
        T_b_f_in_K = self.T_b_f_in_K          # 지중 유체 입구 온도 [K] (이전 타임스텝 값 사용)
        
        # 증발 온도 계산: 지중 유체 입구 온도에서 냉매-열교환기 온도차를 뺌
        T_evap_K = T_b_f_in_K - dT_ref_HX     # 증발 온도 [K]
        
        # 응축 온도 계산: 저탕조 온도에 냉매-저탕조 온도차를 더함
        T_cond_K = T_tank_w_K + dT_ref_cond   # 응축 온도 [K]
        
        # ============================================================
        # 3단계: 공통 사이클 상태 계산
        # ============================================================
        cycle_states = calc_ref_state(
            T_evap_K=T_evap_K,
            T_cond_K=T_cond_K,
            refrigerant=self.ref,
            eta_cmp_isen=self.eta_cmp_isen,
            T0_K=T0_K,
            P0=101325,
            dT_superheat=self.dT_superheat,
            dT_subcool=self.dT_subcool
        )
        
        # State 1의 밀도 (냉매 유량 계산에 사용)
        rho_ref_cmp_in = cycle_states['rho']
        
        # ============================================================
        # 4단계: 사이클 상태값 추출
        # ============================================================
        T1_K = cycle_states['T1_K']  # State 1 온도 [K]
        P1 = cycle_states['P1']      # State 1 압력 [Pa]
        h1 = cycle_states['h1']      # State 1 엔탈피 [J/kg]
        s1 = cycle_states['s1']      # State 1 엔트로피 [J/(kg·K)]
        
        T2_K = cycle_states['T2_K']  # State 2 온도 [K]
        P2 = cycle_states['P2']      # State 2 압력 [Pa]
        h2 = cycle_states['h2']      # State 2 엔탈피 [J/kg]
        s2 = cycle_states['s2']      # State 2 엔트로피 [J/(kg·K)]
        
        T3_K = cycle_states['T3_K']  # State 3 온도 [K]
        P3 = cycle_states['P3']      # State 3 압력 [Pa]
        h3 = cycle_states['h3']      # State 3 엔탈피 [J/kg]
        s3 = cycle_states['s3']      # State 3 엔트로피 [J/(kg·K)]
        
        T4_K = cycle_states['T4_K']  # State 4 온도 [K]
        P4 = cycle_states['P4']      # State 4 압력 [Pa]
        h4 = cycle_states['h4']      # State 4 엔탈피 [J/kg]
        s4 = cycle_states['s4']      # State 4 엔트로피 [J/(kg·K)]
        
        # 계산 불가능한 경우 체크 (h3 == h2인 경우 0으로 나누기 방지)
        if (h3 - h2) == 0:
            return None
        
        # ============================================================
        # 5단계: 냉매 유량 및 성능 데이터 계산
        # ============================================================
        # 냉매 유량 계산: 목표 열 교환율을 만족하기 위한 필요한 유량
        # Q_cond_load = m_dot_ref * (h2 - h3)
        m_dot_ref = Q_cond_load / (h2 - h3)  # 냉매 유량 [kg/s]
        
        # 사이클 열량 및 전력 계산
        Q_ref_cond = m_dot_ref * (h2 - h3)  # 응축기 열량 [W] (Q_cond_load와 동일해야 함)
        Q_ref_evap = m_dot_ref * (h1 - h4)  # 증발기 열량 [W] (지중열 교환량)
        E_cmp = m_dot_ref * (h2 - h1)       # 압축기 전력 [W] (목적 함수)
        
        # 압축기 회전수 계산
        # m_dot_ref = V_disp_cmp * rho_ref_cmp_in * cmp_rps * eta_vol
        # 여기서는 eta_vol = 1로 가정
        cmp_rps = m_dot_ref / (self.V_disp_cmp * rho_ref_cmp_in)  # 회전수 [1/s]
        
        # ============================================================
        # 6단계: LMTD 기반 열량 계산 (현실적 제약 조건)
        # ============================================================
        # 응축기(저탕조 측) LMTD 계산
        # 응축기에서 냉매는 과열 증기(State 2)에서 포화 액체(State 3)로 변화
        # 저탕조는 일정 온도로 유지됨 (T_tank_w_K)
        dT1_tank = T2_K - T_tank_w_K  # 응축기 입구 온도차 [K]
        dT2_tank = T3_K - T_tank_w_K  # 응축기 출구 온도차 [K]

        # LMTD 계산 가능 여부 확인 (0 또는 음수 온도차 방지)
        if dT1_tank <= 1e-6 or dT2_tank <= 1e-6 or abs(dT1_tank - dT2_tank) < 1e-6:
            # 물리적으로 불가능한 상태 (온도차가 없음)
            Q_LMTD_cond = -np.inf
        else:
            # 대수 평균 온도차 (LMTD) 계산
            LMTD_tank = (dT1_tank - dT2_tank) / np.log(dT1_tank / dT2_tank)
            # LMTD 기반 응축기 열량 계산: Q = UA * LMTD
            Q_LMTD_cond = self.UA_cond * LMTD_tank

        # ============================================================
        # 7단계: 지중열 교환 계산 (증발기 측)
        # ============================================================
        # 대향류(Counter-flow) 열교환기 모델
        # 지중 유체가 냉매로부터 열을 흡수하여 온도 상승
        
        # 지중열 교환량 계산 (증발기 열량 - 펌프 전력)
        Q_b = Q_ref_evap - self.E_pmp  # 지중열 교환량 [W]
        Q_b_unit = Q_b / self.H_b      # 단위 길이당 열량 [W/m]
        
        # 지중 유체 출구 온도 계산 (에너지 보존)
        # Q_ref_evap = m_dot_fluid * c_w * (T_out - T_in)
        T_b_f_out_K = T_b_f_in_K + Q_ref_evap / (c_w * rho_w * self.dV_b_f_m3s)
        T_b_f_out = cu.K2C(T_b_f_out_K)  # 유출수 온도 [°C]
        
        # 지중 유체 평균 온도
        T_b_f = (cu.K2C(T_b_f_in_K) + T_b_f_out) / 2  # 유체 평균 온도 [°C]
        
        # 토양 온도 계산 (보어홀 열저항 고려)
        T_b = T_b_f + Q_b_unit * self.R_b  # 토양 온도 [°C]

        # 증발기(지중열 측) LMTD 계산
        # 지중 유체는 T_b_f_in → T_b_f_out로 온도 하강
        # 냉매는 State 4(저온) → State 1(고온)로 온도 상승
        dT1_HX = T_b_f_in_K - T1_K   # 증발기 입구 온도차 [K] (유체 입구 - 냉매 출구)
        dT2_HX = T_b_f_out_K - T4_K  # 증발기 출구 온도차 [K] (유체 출구 - 냉매 입구)

        # LMTD 계산 가능 여부 확인
        if dT1_HX <= 1e-6 or dT2_HX <= 1e-6 or abs(dT1_HX - dT2_HX) < 1e-6:
            # 물리적으로 불가능한 상태
            Q_LMTD_evap = np.inf
        else:
            # 대수 평균 온도차 (LMTD) 계산
            LMTD_HX = (dT1_HX - dT2_HX) / np.log(dT1_HX / dT2_HX)
            # LMTD 기반 증발기 열량 계산: Q = UA * LMTD
            Q_LMTD_evap = self.UA_evap * LMTD_HX
        
        # ============================================================
        # 8단계: 엑서지 계산
        # ============================================================
        # 기준 상태(T0_K, P0) 대비 각 상태점의 엑서지 계산
        P0 = 101325           # 기준 압력 [Pa] (대기압)
        h0 = CP.PropsSI('H', 'T', T0_K, 'P', P0, self.ref)  # 기준 엔탈피 [J/kg]
        s0 = CP.PropsSI('S', 'T', T0_K, 'P', P0, self.ref)  # 기준 엔트로피 [J/kgK]
        
        # 기본 엑서지 값 (단위 질량당)
        x1 = (h1-h0) - T0_K*(s1 - s0)
        x2 = (h2-h0) - T0_K*(s2 - s0)
        x3 = (h3-h0) - T0_K*(s3 - s0)
        x4 = (h4-h0) - T0_K*(s4 - s0)
        
        # 포화점 물성치 추출
        T1_star_K = cycle_states.get('T1_star_K', np.nan)
        T2_star_K = cycle_states.get('T2_star_K', np.nan)
        T3_star_K = cycle_states.get('T3_star_K', np.nan)
        P2_star = cycle_states.get('P2_star', P2)
        
        # 포화점 엔탈피, 엔트로피, 엑서지 계산
        P1_star = P1  # 증발기 포화 압력
        h1_star = CP.PropsSI('H', 'P', P1_star, 'Q', 1, self.ref)
        s1_star = CP.PropsSI('S', 'P', P1_star, 'Q', 1, self.ref)
        x1_star = (h1_star-h0) - T0_K*(s1_star - s0)
        
        h2_star = cycle_states.get('h2_star', np.nan)
        s2_star = cycle_states.get('s2_star', np.nan)
        if np.isnan(h2_star) or np.isnan(s2_star):
            h2_star = CP.PropsSI('H', 'P', P2_star, 'Q', 1, self.ref)
            s2_star = CP.PropsSI('S', 'P', P2_star, 'Q', 1, self.ref)
        x2_star = (h2_star-h0) - T0_K*(s2_star - s0)
        
        P3_star = P3  # 응축기 포화 압력
        h3_star = CP.PropsSI('H', 'P', P3_star, 'Q', 0, self.ref)
        s3_star = CP.PropsSI('S', 'P', P3_star, 'Q', 0, self.ref)
        x3_star = (h3_star-h0) - T0_K*(s3_star - s0)
        
        # ============================================================
        # 9단계: 최종 결과 딕셔너리 생성 (단위 포함)
        # ============================================================
        T_b_f_in = cu.K2C(T_b_f_in_K)
        
        result = {
            'is_on': True,
            'converged': True,
            
            # === [온도: °C] =======================================
            # [NEW] Saturation Points
            'T1_star [°C]': cu.K2C(T1_star_K),
            'T2_star [°C]': cu.K2C(T2_star_K),
            'T3_star [°C]': cu.K2C(T3_star_K),
            
            # [Updated] Actual Points
            'T0 [°C]': T0,
            'T1 [°C]': cu.K2C(T1_K),
            'T2 [°C]': cu.K2C(T2_K),
            'T3 [°C]': cu.K2C(T3_K),
            'T4 [°C]': cu.K2C(T4_K),
            'T_cond [°C]': cu.K2C(T3_star_K if not np.isnan(T3_star_K) else T3_K),  # 대표 응축 온도는 포화 온도로 표시
            'T_tank_w [°C]': T_tank_w,
            'T_serv_w [°C]': self.T_serv_w,
            'T_sup_w [°C]': self.T_sup_w,
            'Ts [°C]': self.Ts,
            'T_b [°C]': T_b,
            'T_b_f [°C]': T_b_f,
            'T_b_f_in [°C]': T_b_f_in,
            'T_b_f_out [°C]': T_b_f_out,
            
            # === [체적유량: m3/s] ==================================
            'dV_b_f [m3/s]': self.dV_b_f_m3s,
            'dV_w_serv [m3/s]': self.dV_w_serv if hasattr(self, 'dV_w_serv') else 0.0,
            'dV_w_sup_tank [m3/s]': self.dV_w_sup_tank if hasattr(self, 'dV_w_sup_tank') else 0.0,
            'dV_w_sup_mix [m3/s]': self.dV_w_sup_mix if hasattr(self, 'dV_w_sup_mix') else 0.0,
            
            # === [압력: Pa] ========================================
            'P1 [Pa]': P1,
            'P2 [Pa]': P2,
            'P3 [Pa]': P3,
            'P4 [Pa]': P4,
            'P1_star [Pa]': P1_star,
            'P2_star [Pa]': P2_star,
            'P3_star [Pa]': P3_star,
            
            # === [질량유량: kg/s] ==================================
            'm_dot_ref [kg/s]': m_dot_ref,
            
            # === [rpm] =============================================
            'cmp_rpm [rpm]': cmp_rps * 60,
            
            # === [엔탈피: J/kg] ====================================
            'h1 [J/kg]': h1,
            'h2 [J/kg]': h2,
            'h3 [J/kg]': h3,
            'h4 [J/kg]': h4,
            'h1_star [J/kg]': h1_star,
            'h2_star [J/kg]': h2_star,
            'h3_star [J/kg]': h3_star,
            
            # === [엔트로피: J/(kg·K)] ==============================
            's1 [J/(kg·K)]': s1,
            's2 [J/(kg·K)]': s2,
            's3 [J/(kg·K)]': s3,
            's4 [J/(kg·K)]': s4,
            's1_star [J/(kg·K)]': s1_star,
            's2_star [J/(kg·K)]': s2_star,
            's3_star [J/(kg·K)]': s3_star,
            
            # === [엑서지 단위: J/kg] ===============================
            'x1 [J/kg]': x1,
            'x2 [J/kg]': x2,
            'x3 [J/kg]': x3,
            'x4 [J/kg]': x4,
            'x1_star [J/kg]': x1_star,
            'x2_star [J/kg]': x2_star,
            'x3_star [J/kg]': x3_star,
            
            # === [에너지/열량: W] ==================================
            'Q_cond_load [W]': Q_cond_load,
            'Q_ref_cond [W]': Q_ref_cond,
            'Q_ref_evap [W]': Q_ref_evap,
            'Q_LMTD_cond [W]': Q_LMTD_cond,
            'Q_LMTD_evap [W]': Q_LMTD_evap,
            'Q_b [W]': Q_b,
            
            # === [전력: W] =========================================
            'E_cmp [W]': E_cmp,
            'E_pmp [W]': self.E_pmp,
            'E_tot [W]': E_cmp + self.E_pmp,
        }
        
        return result
    
    def _calc_off_state(self, T_tank_w, T0):
        """
        OFF 상태 결과 포맷팅 함수.
        
        히트펌프가 OFF 상태일 때 사용되는 결과 딕셔너리를 생성합니다.
        모든 열량 및 전력 값은 0으로 설정하고, P-h 선도 플로팅을 위한
        기본 사이클 상태값은 포화점 기준으로 계산합니다.
        
        호출 관계:
        - 호출자: _optimize_operation (본 클래스)
            Q_cond_load가 임계값 이하일 때 호출
        
        주요 작업:
        1. ON 상태 템플릿 생성 (Q_cond_load=0으로 계산)
        2. 모든 숫자 값을 0으로 설정
        3. OFF 상태 플래그 및 필수 값 설정
        4. P-h 선도 플로팅용 포화점 계산
        
        Args:
            T_tank_w (float): 저탕조 온도 [°C]
                현재 타임스텝의 저탕조 온도
            
            T0 (float): 엑서지 분석 기준 온도(=외기온도) [°C]
                엑서지 계산의 기준점으로 사용되는 외기 온도
        
        Returns:
            dict: OFF 상태 결과 딕셔너리 (단위 포함)
                - 모든 열량 및 전력 값: 0.0
                - is_on: False
                - P1-4, h1-4, s1-4: 포화점 기준 계산값 (P-h 선도용)
                - 기타 상태값: 현재 시스템 상태 유지
        
        Notes:
            - P-h 선도 플로팅을 위해 기본 사이클 상태값을 계산합니다
            - 증발기 측은 지중 유체 입구 온도 기준 포화 증기
            - 응축기 측은 저탕조 온도 기준 포화 액체
        """
        # 1단계: ON 상태 템플릿 생성
        # Q_cond_load=0.0으로 계산하여 딕셔너리 구조만 얻음
        # 실제 계산은 필요 없지만 결과 딕셔너리 구조를 얻기 위해 실행
        result = self._calc_on_state(
            optimization_vars=[5.0, 5.0],  # 임의의 값 (계산 결과는 무시됨)
            T_tank_w=T_tank_w,
            Q_cond_load=0.0,
            T0=T0
        )
        
        if result is None:
            return None
        
        # 2단계: 모든 숫자 값을 0.0으로 설정
        # 히트펌프 OFF 상태이므로 모든 열량 및 전력 값은 0
        for key, value in result.items():
            if isinstance(value, (int, float)):
                result[key] = 0.0

        # 3~4단계: OFF 상태에 맞는 필수 값들과 P-h 선도 플로팅값을 result.update에서 한꺼번에 업데이트
        T_tank_w_K = cu.C2K(T_tank_w)
        T0_K = cu.C2K(T0)

        # 실제 서비스 온도 계산 (믹싱 밸브, OFF 상태에서도 일관성 유지)
        den = max(1e-6, T_tank_w_K - self.T_sup_w_K)
        alp = min(1.0, max(0.0, (self.T_serv_w_K - self.T_sup_w_K) / den))
        
        if alp >= 1.0:
            # 저탕조 온수를 그대로 사용하는 경우 (T_tank_w < T_serv_w 목표값)
            T_serv_w_actual = T_tank_w
        else:
            # 믹싱 밸브로 저탕조 온수와 상수도를 믹싱
            T_serv_w_actual_K = alp * T_tank_w_K + (1 - alp) * self.T_sup_w_K
            T_serv_w_actual = cu.K2C(T_serv_w_actual_K)

        # 증발기 측 포화 증기 (State 1, 4)
        P1_off = CP.PropsSI('P', 'T', self.T_b_f_in_K, 'Q', 1, self.ref)
        h1_off = CP.PropsSI('H', 'P', P1_off, 'Q', 1, self.ref)
        s1_off = CP.PropsSI('S', 'P', P1_off, 'Q', 1, self.ref)

        # 응축기 측 포화 액체 (State 2, 3)
        P3_off = CP.PropsSI('P', 'T', T_tank_w_K, 'Q', 0, self.ref)
        h3_off = CP.PropsSI('H', 'P', P3_off, 'Q', 0, self.ref)
        s3_off = CP.PropsSI('S', 'P', P3_off, 'Q', 0, self.ref)

        result.update({
            'is_on': False,
            'converged': True,  # OFF 상태는 정상적인 상태이므로 수렴 성공으로 간주
            
            'T_tank_w [°C]': T_tank_w,
            'T0 [°C]': T0,
            'T_serv_w [°C]': T_serv_w_actual,
            'T_sup_w [°C]': self.T_sup_w,
            'Ts [°C]': self.Ts,
            'T_b [°C]': self.Ts,
            'T_b_f [°C]': self.Ts,
            'T_b_f_in [°C]': self.Ts,
            'T_b_f_out [°C]': self.Ts,
            
            'dV_w_serv [m3/s]': self.dV_w_serv if hasattr(self, 'dV_w_serv') else 0.0,
            'dV_w_sup_tank [m3/s]': self.dV_w_sup_tank if hasattr(self, 'dV_w_sup_tank') else 0.0,
            'dV_w_sup_mix [m3/s]': self.dV_w_sup_mix if hasattr(self, 'dV_w_sup_mix') else 0.0,
            'dV_b_f [m3/s]': self.dV_b_f_m3s,
            
            'P1 [Pa]': P1_off,
            'P2 [Pa]': P3_off,
            'P3 [Pa]': P3_off,
            'P4 [Pa]': P1_off,
            
            'h1 [J/kg]': h1_off,
            'h2 [J/kg]': h1_off,
            'h3 [J/kg]': h3_off,
            'h4 [J/kg]': h3_off,
            
            's1 [J/(kg·K)]': s1_off,
            's2 [J/(kg·K)]': s1_off,
            's3 [J/(kg·K)]': s3_off,
            's4 [J/(kg·K)]': s3_off,
            
            'x1 [J/kg]': 0.0,
            'x2 [J/kg]': 0.0,
            'x3 [J/kg]': 0.0,
            'x4 [J/kg]': 0.0,
            
            'T1 [°C]': cu.K2C(self.T_b_f_in_K),
            'T2 [°C]': T_tank_w,
            'T3 [°C]': T_tank_w,
            'T4 [°C]': cu.K2C(self.T_b_f_in_K),
            'T_cond [°C]': T_tank_w,
            
            # 지중열 관련 값들 (OFF 상태이므로 0)
            'Q_b [W]': 0.0,
            'Q_ref_cond [W]': 0.0,
            'Q_ref_evap [W]': 0.0,
            'Q_LMTD_cond [W]': 0.0,
            'Q_LMTD_evap [W]': 0.0,
            'Q_cond_load [W]': 0.0,
            'E_cmp [W]': 0.0,
            'E_pmp [W]': 0.0,
            'E_tot [W]': 0.0,
            'm_dot_ref [kg/s]': 0.0,
            'cmp_rpm [rpm]': 0.0,
        })

        return result
    
    def _optimize_operation(self, T_tank_w, Q_cond_load, T0, method='SLSQP', callback=None):
        """
        히트펌프 최적 운전점 탐색을 수행하는 내부 메서드.
        
        이 메서드는 analyze_dynamic에서 사용되는 최적화 로직을 담당합니다.
        응축기 제약조건만 포함하여 최적화를 수행합니다.
        
        Args:
            T_tank_w (float): 저탕조 온도 [°C]
                현재 타임스텝의 저탕조 온도
            
            Q_cond_load (float): 저탕조 목표 열 교환율 [W]
                응축기가 저탕조에 전달해야 하는 목표 열량
            
            T0 (float): 엑서지 분석 기준 온도(=외기온도) [°C]
                엑서지 계산의 기준점으로 사용되는 외기 온도
            
            method (str): 최적화 알고리즘 선택
                - 'SLSQP': Sequential Least Squares Programming (기본값)
                - 'trust-constr': Trust-region constrained optimization
                - 'COBYLA': Constrained Optimization BY Linear Approximation
                기본값: 'SLSQP'
            
            callback (callable, optional): 각 반복마다 호출되는 콜백 함수
                콜백 함수는 현재 변수 벡터 x를 인자로 받습니다.
                기본값: None
        
        Returns:
            scipy.optimize.OptimizeResult: 최적화 결과 객체
                - x: 최적화된 변수 [dT_ref_HX, dT_ref_cond]
                - success: 최적화 성공 여부
                - iteration_history: 반복 이력 리스트 (각 반복의 변수 값)
                - 기타 최적화 메타데이터
        
        Notes:
            - 최적화 변수: [dT_ref_HX, dT_ref_cond]
                - dT_ref_HX: 냉매-열교환기 온도차 [K]
                - dT_ref_cond: 냉매-저탕조 온도차 [K]
            - 제약 조건:
                - Q_cond_load <= Q_LMTD_cond <= Q_cond_load * (1 + tolerance) (응축기 열전달 능력 범위)
            - 목적 함수: E_tot (E_cmp + E_pmp) 최소화
        """
        # tolerance 변수 정의
        tolerance = 0.01  # 1%
        
        # 최적화 변수 경계 조건 및 초기 추정값 설정
        bounds = [(1.0, 30.0), (1.0, 30.0)]  # [dT_ref_HX, dT_ref_cond]
        initial_guess = [5.0, 5.0]
        
        # 반복 이력 추적을 위한 리스트 초기화
        iteration_history = []
        
        # 콜백 래퍼 함수 생성
        def _callback_wrapper(xk):
            """각 반복마다 호출되는 콜백 함수"""
            iteration_history.append({
                'iteration': len(iteration_history),
                'x': np.array(xk).copy(),
                'dT_ref_HX': xk[0],
                'dT_ref_cond': xk[1]
            })
            if callback is not None:
                callback(xk)
        
        # 응축기 LMTD 제약 조건 함수 (하한): Q_LMTD_cond - Q_cond_load >= 0
        def _cond_LMTD_constraint_low(x):
            """
            응축기 LMTD 제약 조건 함수 (하한): Q_LMTD_cond - Q_cond_load >= 0
            perf가 None이면 (수렴 실패 등) 큰 제약 조건 위반 값을 반환하여
            최적화 알고리즘이 해당 변수 조합을 피하고 다른 조합을 시도하도록 함.
            """
            try:
                perf = self._calc_on_state(
                    optimization_vars=x,
                    T_tank_w=T_tank_w,
                    Q_cond_load=Q_cond_load,
                    T0=T0
                )
                if perf is None or not isinstance(perf, dict):
                    return -1e6  # ineq 제약이므로 음수 반환
                
                if "Q_LMTD_cond [W]" not in perf or np.isnan(perf["Q_LMTD_cond [W]"]):
                    return -1e6
                
                # 제약 조건: Q_LMTD_cond - Q_cond_load >= 0
                return perf["Q_LMTD_cond [W]"] - Q_cond_load
            except Exception as e:
                return -1e6
        
        # 응축기 LMTD 제약 조건 함수 (상한): Q_cond_load*(1+tolerance) - Q_LMTD_cond >= 0
        def _cond_LMTD_constraint_high(x):
            """
            응축기 LMTD 제약 조건 함수 (상한): Q_cond_load*(1+tolerance) - Q_LMTD_cond >= 0
            perf가 None이면 (수렴 실패 등) 큰 제약 조건 위반 값을 반환하여
            최적화 알고리즘이 해당 변수 조합을 피하고 다른 조합을 시도하도록 함.
            """
            try:
                perf = self._calc_on_state(
                    optimization_vars=x,
                    T_tank_w=T_tank_w,
                    Q_cond_load=Q_cond_load,
                    T0=T0
                )
                if perf is None or not isinstance(perf, dict):
                    return -1e6  # ineq 제약이므로 음수 반환
                
                if "Q_LMTD_cond [W]" not in perf or np.isnan(perf["Q_LMTD_cond [W]"]):
                    return -1e6
                
                # 제약 조건: Q_cond_load*(1+tolerance) - Q_LMTD_cond >= 0
                return Q_cond_load * (1 + tolerance) - perf["Q_LMTD_cond [W]"]
            except Exception as e:
                return -1e6
        
        # 제약 조건: 응축기만 두 개의 ineq 제약 (증발기 제약조건 제거)
        const_funcs = [
            {'type': 'ineq', 'fun': _cond_LMTD_constraint_low},   # Q_LMTD_cond - Q_cond_load >= 0
            {'type': 'ineq', 'fun': _cond_LMTD_constraint_high},  # Q_cond_load*(1+tolerance) - Q_LMTD_cond >= 0
        ]
        
        # COBYLA는 bounds를 직접 지원하지 않으므로 제약 조건으로 변환
        if method == 'COBYLA':
            def _bound_constraint_low_HX(x):
                return x[0] - bounds[0][0]  # dT_ref_HX >= 1.0
            def _bound_constraint_high_HX(x):
                return bounds[0][1] - x[0]  # dT_ref_HX <= 30.0
            def _bound_constraint_low_cond(x):
                return x[1] - bounds[1][0]  # dT_ref_cond >= 1.0
            def _bound_constraint_high_cond(x):
                return bounds[1][1] - x[1]  # dT_ref_cond <= 30.0
            
            const_funcs.extend([
                {'type': 'ineq', 'fun': _bound_constraint_low_HX},
                {'type': 'ineq', 'fun': _bound_constraint_high_HX},
                {'type': 'ineq', 'fun': _bound_constraint_low_cond},
                {'type': 'ineq', 'fun': _bound_constraint_high_cond},
            ])
            bounds_for_method = None  # COBYLA는 bounds를 사용하지 않음
        else:
            bounds_for_method = bounds

        # 목적 함수: E_tot (총 전력 소비) 최소화
        def _objective(x):  # x = [dT_ref_HX, dT_ref_cond]
            """
            목적 함수: E_tot (총 전력 소비) 최소화.
            perf가 None이면 (수렴 실패 등) 큰 penalty 값을 반환하여
            최적화 알고리즘이 해당 변수 조합을 피하고 다른 조합을 시도하도록 함.
            """
            try:
                perf = self._calc_on_state(
                    optimization_vars=x,
                    T_tank_w=T_tank_w,
                    Q_cond_load=Q_cond_load,
                    T0=T0
                )
                if perf is None or not isinstance(perf, dict):
                    return 1e6
                
                if "E_tot [W]" not in perf or np.isnan(perf["E_tot [W]"]):
                    return 1e6
                
                # 목적 함수: E_tot 최소화
                return perf["E_tot [W]"]
            except Exception as e:
                return 1e6
        
        # 알고리즘별 옵션 설정
        if method == 'SLSQP':
            options = {
                'disp': False,
                'maxiter': 100,
                'ftol': 10,      # 함수 값 수렴 허용 오차
                'eps': 0.01,      # 유한 차분 근사 스텝 크기
            }
        elif method == 'trust-constr':
            options = {
                'disp': False,
                'maxiter': 100,
                'gtol': 1e-6,    # 제약 조건 수렴 허용 오차
                'xtol': 1e-8,    # 변수 수렴 허용 오차
            }
        elif method == 'COBYLA':
            options = {
                'disp': False,
                'maxiter': 100,
            }
        else:
            # 기본 옵션
            options = {
                'disp': False,
                'maxiter': 100,
            }
        
        # 최적화 실행
        opt_result = minimize(
            _objective,           # 목적 함수 (E_tot = E_cmp + E_pmp 최소화)
            initial_guess,        # 초기 추정값
            method=method,        # 선택된 알고리즘 사용
            bounds=bounds_for_method,  # 변수 경계 조건 (COBYLA는 None)
            constraints=const_funcs,
            callback=_callback_wrapper if callback is not None else None,
            options=options
        )
        
        # 반복 이력 결과에 첨부
        opt_result.iteration_history = iteration_history
        
        return opt_result
    
    def analyze_dynamic(
        self, 
        simulation_period_sec, 
        dt_s, 
        T_tank_w_init_C,
        schedule_entries,
        T0_schedule,
        result_save_csv_path=None,
        optimization_method='SLSQP',  # 최적화 알고리즘 선택
        ):
        
        """
        동적 시뮬레이션을 실행합니다.
        
        Args:
            simulation_period_sec: 총 시뮬레이션 시간 [초]
            dt_s: 타임스텝 [초]
            T_tank_w_init_C: 저탕조 초기 온도 [°C]
            schedule_entries: 급탕 사용 스케줄
                [(시작시간_str, 종료시간_str, 사용비율_float), ...]
                예: [("6:00", "6:30", 0.5), ("6:30", "7:00", 0.9)]
            T0_schedule: 엑서지 분석 기준 온도(=외기온도) 스케줄 [°C]
                엑서지 계산의 기준점으로 사용되는 외기 온도의 시간별 스케줄
            result_save_csv_path: 결과 CSV 저장 경로
            optimization_method: 최적화 알고리즘 선택 ('SLSQP', 'trust-constr', 'COBYLA')
        
        Returns:
            pd.DataFrame: 시뮬레이션 타임스텝별 결과 데이터
        """
        
        # --- 0. 실행 조건 판단 ---
        
        if simulation_period_sec % dt_s != 0:
            raise ValueError("simulation_period_sec must be divisible by dt_s")
        
        if self.T_tank_w_setpoint < self.T_tank_w_lower_bound:
            raise ValueError("T_tank_w_setpoint must be greater than T_tank_w_lower_bound")
        
        if self.heater_capacity < 0:
            raise ValueError("heater_capacity must be greater than 0")
        
        if self.dV_w_serv_m3s < 0:
            raise ValueError("dV_w_serv_m3s must be greater than 0")
        
        if schedule_entries == []:
            raise ValueError("schedule_entries must be provided")
        
        time = np.arange(0, simulation_period_sec, dt_s)
        tN = len(time)
        T0_schedule = np.array(T0_schedule)
        if len(T0_schedule) != tN:
            raise ValueError(f"T0_schedule length ({len(T0_schedule)}) must match time array length ({tN})")
        
        results_data = []
        
        self.time = time
        self.dt = dt_s
        
        # --- 1. 시뮬레이션 초기화 ---
        self.T_b_f = self.Ts # 초기 지중열 교환기 유출수 온도
        self.T_b = self.Ts   # 초기 지중 온도
        self.T_b_f_in = self.Ts # 초기 지중열 교환기 유입수 온도
        self.T_b_f_out = self.Ts # 초기 지중열 교환기 유출수 온도
        self.Q_b = 0.0 # 초기 지중열 교환기 열 유량
        
        self.dV_w_serv = 0.0 
        self.dV_w_sup_tank = 0.0 
        self.dV_w_sup_mix = 0.0 
        
        self.w_use_frac = _build_schedule_ratios(schedule_entries, self.time)
        
        T_tank_w_K = cu.C2K(T_tank_w_init_C)
        Q_b_unit_pulse = np.zeros(tN)
        Q_b_unit_old = 0
        is_on_prev = False

        # --- 2. 시뮬레이션 루프 ---
        for n in tqdm(range(tN), desc="GSHPB Simulating"):
            step_results = {}
            T_tank_w = cu.K2C(T_tank_w_K)
            T0 = T0_schedule[n]
            T0_K = cu.C2K(T0)

            # 제어 상태 결정
            Q_tank_loss = self.UA_tank * (T_tank_w_K - T0_K)
            den = max(1e-6, T_tank_w_K - self.T_sup_w_K)
            alp = min(1.0, max(0.0, self.T_serv_w_K - self.T_sup_w_K) / den)

            self.dV_w_serv = self.w_use_frac[n] * self.dV_w_serv_m3s 
            self.dV_w_sup_tank = alp * self.dV_w_serv
            self.dV_w_sup_mix = (1 - alp) * self.dV_w_serv 

            Q_use_loss = c_w * rho_w * self.dV_w_sup_tank * (T_tank_w_K - self.T_sup_w_K)
            total_loss = Q_tank_loss + Q_use_loss
            
            # On/Off 결정
            if T_tank_w <= self.T_tank_w_lower_bound: is_on = True
            elif T_tank_w >= self.T_tank_w_setpoint: is_on = False
            else: is_on = is_on_prev
            
            is_transitioning_off_to_on = (not is_on_prev) and is_on # False to True
            Q_cond_load_n = self.heater_capacity if is_on else 0.0
            is_on_prev = is_on
            
            # OFF 상태 조기 체크: Q_cond_load_n이 임계값 이하이면 최적화 건너뛰기
            if abs(Q_cond_load_n) <= self.Q_cond_LOAD_OFF_TOL:
                
                result = self._calc_off_state(
                    T_tank_w=T_tank_w,
                    T0=T0
                )
            else:
                # 최적화 과정 진행
                opt_result = self._optimize_operation(
                    T_tank_w=T_tank_w,
                    Q_cond_load=Q_cond_load_n,
                    T0=T0,
                    method=optimization_method
                )
                    
                # opt_result.x로 결과 계산 시도 (성공/실패 무관)
                result = None
                try:
                    result = self._calc_on_state(
                        optimization_vars=opt_result.x,
                        T_tank_w=T_tank_w,
                        Q_cond_load=Q_cond_load_n,
                        T0=T0
                    )
                except Exception:
                    pass
                
                # result 검증 및 fallback
                if result is None or not isinstance(result, dict):
                    try:
                        result = self._calc_off_state(
                            T_tank_w=T_tank_w,
                            T0=T0
                        )
                    except Exception:
                        result = {
                            'is_on': False,
                            'converged': False,
                            'Q_ref_cond [W]': 0.0,
                            'Q_ref_evap [W]': 0.0,
                            'E_cmp [W]': 0.0,
                            'E_pmp [W]': 0.0,
                            'E_tot [W]': 0.0,
                            'T_tank_w [°C]': T_tank_w,
                            'T0 [°C]': T0
                        }
                
                # converged 플래그 설정
                if result is not None and isinstance(result, dict):
                    result['converged'] = opt_result.success
            
            if is_transitioning_off_to_on:
                # OFF→ON 전환 시점: 점진적 전환을 위해 이전 스텝의 지중 온도 값 사용
                # result는 ON 상태로 계산되었지만, 지중 온도는 점진적으로 업데이트
                step_results.update(result)
                step_results['is_on'] = is_on
                # 전환 시점임을 표시하는 플래그 추가
                step_results['is_transitioning'] = True
                
                # 전환 시점에서는 이전 스텝의 지중 온도 값 유지 (점진적 전환)
                step_results['T_b [°C]'] = self.T_b  # 이전 스텝 값
                step_results['T_b_f [°C]'] = self.T_b_f
                step_results['T_b_f_in [°C]'] = self.T_b_f_in
                step_results['T_b_f_out [°C]'] = self.T_b_f_out
                step_results['Q_b [W]'] = 0.0  # 전환 시점에서는 0으로 시작
            else:
                step_results.update(result)
                step_results['is_on'] = is_on
                step_results['is_transitioning'] = False

            # 지중 온도 업데이트
            if is_transitioning_off_to_on:
                # 전환 시점: 점진적 전환을 위해 Q_b_unit을 0으로 시작
                Q_b_unit = 0.0
                # 펄스 계산 건너뛰기 (전환 시점에서는 펄스 없음)
                # Q_b_unit_old를 0으로 업데이트하여 다음 스텝에서 정상 계산 시작
                Q_b_unit_old = 0.0
            else:
                # 일반 시점: 정상 계산
                Q_b_unit = (result.get('Q_ref_evap [W]', 0.0) - self.E_pmp) / self.H_b if result.get('is_on') else 0.0
            
            if abs(Q_b_unit - Q_b_unit_old) > 1e-6: # 만약 Q_b이 이전 스텝과 일정 수준 이상 차이가 난다면 펄스가 나타난 것으로 간주
                Q_b_unit_pulse[n] = Q_b_unit - Q_b_unit_old # 펄스는 이전 값과의 차이
                Q_b_unit_old = Q_b_unit # 업데이트
        
            # 펄스 계산 (전환 시점이 아닐 때만)
            if not is_transitioning_off_to_on:
                pulses_idx = np.flatnonzero(Q_b_unit_pulse[:n+1])
                dQ = Q_b_unit_pulse[pulses_idx]
                tau = self.time[n] - self.time[pulses_idx]
                
                # g-function 계산은 여전히 루프가 필요
                g_n = np.array([G_FLS(t, self.k_s, self.alp_s, self.r_b, self.H_b) for t in tau])
                dT_b = np.dot(dQ, g_n)
                
                self.T_b = self.Ts - dT_b
                self.T_b_f = self.T_b - Q_b_unit * self.R_b
                self.Q_b = Q_b_unit * self.H_b
                self.T_b_f_in  = self.T_b_f - self.Q_b / (2 * c_w * rho_w * self.dV_b_f_m3s) # °C
                self.T_b_f_out = self.T_b_f + self.Q_b / (2 * c_w * rho_w * self.dV_b_f_m3s) # °C
                self.T_b_f_in_K  = cu.C2K(self.T_b_f_in)
                self.T_b_f_out_K = cu.C2K(self.T_b_f_out)
            
                # step_results에 반영
                step_results['T_b [°C]'] = self.T_b
                step_results['T_b_f [°C]'] = self.T_b_f
                step_results['T_b_f_in [°C]'] = self.T_b_f_in
                step_results['T_b_f_out [°C]'] = self.T_b_f_out
                step_results['Q_b [W]'] = self.Q_b
            
            # UV 램프 전력 계산 (is_on과 무관)
            E_uv = 0
            if (
                self.num_switching_per_3hour > 0
                and self.lamp_power_watts > 0
            ):
                time_in_period = time[n] % self.period_3hour_sec
                interval = (
                    self.period_3hour_sec
                    - self.num_switching_per_3hour * self.uv_lamp_exposure_duration_sec
                ) / (self.num_switching_per_3hour + 1)
                for i in range(self.num_switching_per_3hour):
                    start_time = interval * (i + 1) + i * self.uv_lamp_exposure_duration_sec
                    if start_time <= time_in_period < start_time + self.uv_lamp_exposure_duration_sec:
                        E_uv = self.lamp_power_watts
                        break
            
            step_results['is_on'] = is_on
            if self.lamp_power_watts > 0:
                step_results['E_uv [W]'] = E_uv
            
            # 다음 스텝 탱크 온도 계산
            if n < tN - 1:
                # nan인 경우 0으로 처리 (탱크 온도 계산에는 실제 열량이 필요)
                Q_ref_cond_val = result.get('Q_ref_cond [W]', np.nan)
                Q_ref_cond_val = np.nan_to_num(Q_ref_cond_val, nan=0.0)
                E_uv_val = step_results.get('E_uv [W]', 0.0)
                Q_tank_in = Q_ref_cond_val + E_uv_val
                Q_net = Q_tank_in - total_loss
                T_tank_w_K += (Q_net / self.C_tank) * self.dt
            
            results_data.append(step_results)
            
        results_df = pd.DataFrame(results_data)

        if result_save_csv_path:
            results_df.to_csv(result_save_csv_path, index=False)

        return results_df

