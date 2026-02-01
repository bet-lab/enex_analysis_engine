import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm
import CoolProp.CoolProp as CP

from . import calc_util as cu
from .constants import *
from .enex_functions import *


@dataclass
class AirSourceHeatPumpBoiler:
    '''
    공기원 히트펌프 보일러 성능 계산 및 최적 운전점 탐색 클래스.
    (수정됨: 과열도(Superheating) 및 과냉각도(Subcooling) 고려 모델)
    '''
    def __init__(
        self,

        # 1. 냉매/사이클/압축기 파라미터 -------------------------------
        ref         = 'R134a',
        V_disp_cmp  = 0.0002,
        eta_cmp_isen = 0.8,

        # [NEW] 과열도 및 과냉각도 설정 (Default 3도)
        dT_superheat = 3.0,  # [K] 증발기 출구 과열도 (State 1* -> 1)
        dT_subcool   = 3.0,  # [K] 응축기 출구 과냉각도 (State 3* -> 3)

        # 2. 열교환기 파라미터 -----------------------------------------
        UA_cond_design = 2000.0,   # 응축기 열전달 계수 [W/K]
        UA_evap_design = 1000.0,   # 증발기 열전달 계수 [W/K]

        # 3. 실외기 팬 파라미터 ---------------------------------------
        dV_ou_design      = 1.5,     # 실외기 설계 풍량 [m3/s] (정풍량)
        dP_ou_design      = 90.0,    # 실외기 설계 정압 [Pa]
        A_cross_ou        = 0.25 ** 2 * np.pi,  # [m2] (정의된 값 사용)
        eta_fan_ou_design = 0.6,     # 실외기 팬 효율 [-]

        # 4. 탱크/제어/부하 파라미터 -----------------------------------
        T_tank_w_upper_bound = 65.0,   # [°C] 저탕조 설정 온도
        T_tank_w_lower_bound = 60.0,   # [°C] 저탕조 하한 온도
        T_serv_w             = 40.0,   # [°C] 서비스 급탕 온도
        T_sup_w              = 15.0,   # [°C] 급수(상수도) 온도

        hp_capacity   = 15000.0,   # [W] 히터 최대 용량
        dV_w_serv_max = 0.0045,    # [m3/s] 최대 급탕 유량

        #   (탱크/보온 관련)
        r0       = 0.2,      # [m] 탱크 반지름
        H        = 1.2,      # [m] 탱크 높이
        x_shell  = 0.005,    # [m] 탱크 외벽 두께
        x_ins    = 0.05,     # [m] 단열재 두께
        k_shell  = 25,       # [W/mK] 탱크 외벽 열전도도
        k_ins    = 0.03,     # [W/mK] 단열재 열전도도
        h_o      = 15,       # [W/m2K] 외부 대류 열전달계수
        
        # 5. UV 램프 파라미터 -----------------------------------------
        lamp_power_watts = 0, # [W] 램프 소비 전력
        uv_lamp_exposure_duration_min = 0, # [min] 1회 UV램프 노출 기준시간
        num_switching_per_3hour = 1, # [개] 3시간 당 on 횟수
        
        # 6. 저탕조 수위 관리 파라미터 ---------------------------------
        tank_always_full = True,  # [bool] True: 항상 100% 수위 유지, False: 수위 하한 기반 리필
        tank_level_lower_bound = 0.5,  # [float] 수위 하한 [0~1] (tank_always_full=False일 때만 사용)
        tank_level_upper_bound = 1.0,  # [float] 수위 상한 [0~1] (리필 종료 조건, tank_always_full=False일 때만 사용)
        dV_refill_m3s = 0.001,  # [m³/s] 리필 시 상수도 유량 (tank_always_full=False일 때만 사용)
        prevent_simultaneous_flow = False, #[bool] True: 입/출수 동시 발생 방지 (always_full=True일 때도 적용)
        
        # 7. 히트펌프 운전 구간 제어 -----------------------------------
        hp_on_schedule = [(0.0, 24.0)],  # [list] (시작 시, 종료 시) 구간 목록. 해당 구간에만 HP 운전 허용
        
        # 8. STC (Solar Thermal Collector) 파라미터 -------------------
        A_stc = 0.0,  # [m2] STC 집열판 면적 (0이면 STC 미사용)
        A_stc_pipe = 2.0,  # [m2] STC 파이프 면적
        alpha_stc = 0.95,  # [-] 흡수율
        h_o_stc = 15,  # [W/m2K] 외부 대류 열전달계수
        h_r_stc = 2,  # [W/m2K] 공기층 복사 열전달계수
        k_ins_stc = 0.03,  # [W/mK] 단열재 열전도도
        x_air_stc = 0.01,  # [m] 공기층 두께
        x_ins_stc = 0.05,  # [m] 단열재 두께
        
        # 9. STC 펌프 파라미터 -----------------------------------------
        preheat_start_hour = 6,  # [시] STC 순환 시작 시간
        preheat_end_hour = 18,  # [시] STC 순환 종료 시간
        dV_stc = 0.001,  # [m³/s] STC 유량
        E_pump = 50.0,  # [W] 펌프 출력
        
        # 10. STC 배치 선택 -------------------------------------------
        stc_placement = 'tank_circuit',  # ['tank_circuit' | 'mains_preheat'] STC 배치 위치
        
        # Reference:ASHRAE Standard 90.1 - 2022 (325 page)
        vsd_coeffs_ou = {
            'c1': 0.0013,
            'c2': 0.1470,
            'c3': 0.9506,
            'c4': -0.0998,
            'c5': 0.0,
        },
    
        ):
        '''
        AirSourceHeatPumpBoiler 초기화.
        '''

        # --- 1. 냉매/사이클/압축기 파라미터 ---
        self.ref = ref
        self.V_disp_cmp = V_disp_cmp
        self.eta_cmp_isen = eta_cmp_isen
        
        # [NEW] 과열/과냉각 변수 저장
        self.dT_superheat = dT_superheat
        self.dT_subcool = dT_subcool

        self.hp_capacity = hp_capacity

        # --- 2. 열교환기 파라미터 ---
        self.UA_cond_design = UA_cond_design
        self.UA_evap_design = UA_evap_design

        # --- 3. 실외기 팬 파라미터 ---
        self.dV_ou_design = dV_ou_design
        self.dP_ou_design = dP_ou_design
        self.eta_fan_ou_design = eta_fan_ou_design
        self.A_cross_ou = A_cross_ou
        
        # 팬 설계 전력 계산 (정풍량 기준)
        self.fan_power_ou_design = (self.dV_ou_design * self.dP_ou_design) / (self.eta_fan_ou_design)
        
        # VSD Curve 계수 VSD(Variable Speed Drive)
        self.vsd_coeffs_ou = vsd_coeffs_ou
        
        # 팬 파라미터 딕셔너리
        self.fan_params_ou = {
            'fan_design_flow_rate': self.dV_ou_design,
            'fan_design_power': self.fan_power_ou_design
        }
        
        # --- 4. 탱크 물리 파라미터 ---
        self.tank_physical = {
            'r0': r0, 'H': H, 'x_shell': x_shell, 'x_ins': x_ins,
            'k_shell': k_shell, 'k_ins': k_ins, 'h_o': h_o,
        }
        self.UA_tank = calc_simple_tank_UA(**self.tank_physical)
        self.V_tank_full = math.pi * r0**2 * H  # [m³] 저탕조 전체 체적
        self.C_tank = c_w * rho_w * self.V_tank_full  # [J/K] 저탕조 열용량 (100% 수위 기준)
        
        self.dV_w_serv_max = dV_w_serv_max
        self.T_tank_w_upper_bound = T_tank_w_upper_bound
        self.T_tank_w_lower_bound = T_tank_w_lower_bound
        self.T_sup_w = T_sup_w
        self.T_serv_w = T_serv_w
        
        self.T_sup_w_K = cu.C2K(T_sup_w)
        self.T_serv_w_K = cu.C2K(T_serv_w)
        
        # --- 5. UV 램프 파라미터 ---
        self.lamp_power_watts = lamp_power_watts
        self.uv_lamp_exposure_duration_min = uv_lamp_exposure_duration_min
        self.num_switching_per_3hour = num_switching_per_3hour
        # UV 램프 관련 계산 상수
        self.period_3hour_sec = 3 * cu.h2s  # 3시간을 초 단위로 변환
        self.uv_lamp_exposure_duration_sec = uv_lamp_exposure_duration_min * cu.m2s  # 분을 초로 변환
        
        # --- 6. 저탕조 수위 관리 파라미터 ---
        self.tank_always_full = tank_always_full
        self.tank_level_lower_bound = tank_level_lower_bound
        self.tank_level_upper_bound = tank_level_upper_bound
        self.dV_refill_m3s = dV_refill_m3s
        self.prevent_simultaneous_flow = prevent_simultaneous_flow
        
        # --- 7. 히트펌프 운전 구간 제어 ---
        self.hp_on_schedule = hp_on_schedule
        
        # --- 8. STC 파라미터 ---
        self.A_stc = A_stc
        self.A_stc_pipe = A_stc_pipe
        self.alpha_stc = alpha_stc
        self.h_o_stc = h_o_stc
        self.h_r_stc = h_r_stc
        self.k_ins_stc = k_ins_stc
        self.x_air_stc = x_air_stc
        self.x_ins_stc = x_ins_stc
        
        # --- 9. STC 펌프 파라미터 ---
        self.preheat_start_hour = preheat_start_hour
        self.preheat_end_hour = preheat_end_hour
        self.dV_stc = dV_stc
        self.E_pump = E_pump
        
        # --- 10. STC 배치 선택 ---
        self.stc_placement = stc_placement

        # [NEW] 최적화 초기값 저장을 위한 변수
        self.prev_opt_x = None
        
        # [NEW] Flow Rate Synchronization
        self.dV_w_tank_in = 0.0

    def _calc_state(self, optimization_vars, T_tank_w, Q_cond_load, T0):
        """
        공기원 히트펌프 보일러(ASHPB)의 사이클 성능을 계산하는 메서드.
        
        이 메서드는 최적화 변수(optimization_vars)를 받아 히트펌프 사이클 성능을 계산합니다.
        최적화 과정에서 반복적으로 호출되어 목적 함수와 제약 조건을 평가하는 데 사용됩니다.
        
        주요 작업:
        1. 최적화 변수 언패킹 (온도차 추출)
        2. 증발 및 응축 온도 계산
        3. 공통 사이클 상태 계산
        4. 냉매 유량 및 성능 데이터 계산
        5. LMTD 기반 열량 계산 (응축기, 증발기)
        6. 팬 전력 계산
        7. 엑서지 계산
        8. 최종 결과 딕셔너리 생성
        
        호출 관계:
        - 호출자: find_ref_loop_optimal_operation (enex_functions.py)
        - 호출 함수: 
            - calc_ref_state (enex_functions.py)
            - calc_HX_perf_for_target_heat (enex_functions.py)
            - calc_lmtd_fluid_and_constant_temp (enex_functions.py)
            - calc_fan_power_from_dV_fan (enex_functions.py)
        
        데이터 흐름:
        ──────────────────────────────────────────────────────────────────────────
        [optimization_vars, T_tank_w, Q_cond_load, T0]
            ↓
        증발/응축 포화 온도 계산 (T_evap_sat_K, T_cond_sat_K)
            ↓
        calc_ref_state (과열도/과냉각도 적용)
            ↓ [State 1-4 물성치, T1_star_K, T3_star_K]
        냉매 유량 계산 (m_dot_ref)
            ↓
        성능 데이터 계산 (Q_ref_cond, Q_ref_evap, E_cmp)
            ↓
        LMTD 기반 열량 계산 및 팬 전력 계산
            ↓
        [포맷팅된 결과 딕셔너리]
        
        Args:
            optimization_vars (list): 최적화 변수 배열 [dT_ref_evap, dT_ref_cond]
                - dT_ref_evap (float): 냉매-실외 공기 온도차 [K]
                    증발 온도 = 실외 공기 온도 - dT_ref_evap
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
            dict: 사이클 성능 결과 딕셔너리
                - 사이클 상태값 (P1-4, T1-4, h1-4, s1-4, x1-4)
                - 열량 (Q_ref_cond, Q_ref_evap, Q_LMTD_cond)
                - 전력 (E_cmp, E_fan_ou)
                - 유량 (m_dot_ref, dV_fan_ou, dV_w_serv , dV_w_sup_tank, dV_w_sup_mix)
                - 온도 (T0, T1-4, T_tank_w, T_serv_w, T_sup_w, T_a_ou_mid)
                - 기타 (cmp_rpm, hp_is_on)
                None: 계산 실패 시 (예: h3 == h2인 경우)
        
        Notes:
            - 냉매 유량 계산: m_dot_ref = Q_cond_load / (h2 - h3)
                목표 열 교환율을 만족하기 위한 필요한 냉매 유량
            - 응축기 열량: Q_ref_cond = m_dot_ref * (h2 - h3)
                이 값은 Q_cond_load와 동일해야 함 (계산 검증)
            - 증발기 열량: Q_ref_evap = m_dot_ref * (h1 - h4)
                실외 공기로부터 흡수하는 열량
            - 압축기 전력: E_cmp = m_dot_ref * (h2 - h1)
                최적화의 목적 함수 (최소화 대상)
            - LMTD 계산은 열교환기 물리적 제약 조건을 반영
            - Q_LMTD_cond와 Q_ref_cond는 최적화에서 일치해야 함
        """
        
        # 1단계: 최적화 변수 (온도차) -> 포화 온도 계산
        dT_ref_cond = optimization_vars[0]
        dT_ref_evap = optimization_vars[1]      
        
        # 2단계: 온도 단위 변환 및 증발/응축 포화 온도 계산
        T_tank_w_K = cu.C2K(T_tank_w)         
        T0_K = cu.C2K(T0)                 
        
        # [수정] 이 온도들은 '포화(Saturation) 온도'로 정의됩니다.
        T_evap_sat_K = T0_K - dT_ref_evap       
        T_cond_sat_K = T_tank_w_K + dT_ref_cond 
        
        # 활성화 상태 결정
        is_active = (Q_cond_load > 0.0)
        
        # 3단계: 공통 사이클 상태 계산 (과열도/과냉각도 적용)
        cycle_states = calc_ref_state(
            T_evap_K     = T_evap_sat_K,
            T_cond_K     = T_cond_sat_K,
            refrigerant  = self.ref,
            eta_cmp_isen = self.eta_cmp_isen,
            T0_K         = T0_K,
            mode         = 'heating',
            dT_superheat = self.dT_superheat,
            dT_subcool   = self.dT_subcool,
            is_active    = is_active,
        )
        
        # 4단계: 사이클 상태값 추출 (nan이면 자동으로 nan)
        T1_K = cycle_states['T1_K']  # State 1 온도 [K]
        P1   = cycle_states['P1']    # State 1 압력 [Pa]
        h1   = cycle_states['h1']    # State 1 엔탈피 [J/kg]
        s1   = cycle_states['s1']    # State 1 엔트로피 [J/(kg·K)]
        
        T2_K = cycle_states['T2_K']  # State 2 온도 [K]
        P2   = cycle_states['P2']    # State 2 압력 [Pa]
        h2   = cycle_states['h2']    # State 2 엔탈피 [J/kg]
        s2   = cycle_states['s2']    # State 2 엔트로피 [J/(kg·K)]
        
        T3_K = cycle_states['T3_K']  # State 3 온도 [K]
        P3   = cycle_states['P3']    # State 3 압력 [Pa]
        h3   = cycle_states['h3']    # State 3 엔탈피 [J/kg]
        s3   = cycle_states['s3']    # State 3 엔트로피 [J/(kg·K)]
        
        T4_K = cycle_states['T4_K']  # State 4 온도 [K]
        P4   = cycle_states['P4']    # State 4 압력 [Pa]
        h4   = cycle_states['h4']    # State 4 엔탈피 [J/kg]
        s4   = cycle_states['s4']    # State 4 엔트로피 [J/(kg·K)]
        
        rho_ref_cmp_in = cycle_states['rho']
        
        # 포화 온도 추출
        T1_star_K = cycle_states['T1_star_K']
        T2_star_K = cycle_states['T2_star_K']
        T3_star_K = cycle_states['T3_star_K']
        
        # 5단계: 냉매 유량 및 성능 데이터 계산 (nan이면 자동으로 nan)
        if is_active and not np.isnan(h2) and not np.isnan(h3) and (h2 - h3) != 0:
            m_dot_ref  = Q_cond_load / (h2 - h3)
            Q_ref_cond = m_dot_ref * (h2 - h3)
            Q_ref_evap = m_dot_ref * (h1 - h4)
            E_cmp      = m_dot_ref * (h2 - h1)
            cmp_rps    = m_dot_ref / (self.V_disp_cmp * rho_ref_cmp_in)
        else:
            m_dot_ref = np.nan
            Q_ref_cond = np.nan
            Q_ref_evap = np.nan
            E_cmp = np.nan
            cmp_rps = np.nan
        
        # 6단계: 실외기 열교환기 성능 계산
        T_a_ou_in_K = T0_K
        T_a_ou_in = cu.K2C(T_a_ou_in_K) if is_active else np.nan
        
        HX_perf_ou_dict = calc_HX_perf_for_target_heat(
            Q_ref_target  = Q_ref_evap if is_active else 0.0,
            T_a_ou_in_C   = T0,
            T1_star_K     = T1_star_K,
            T3_star_K     = T3_star_K,
            A_cross       = self.A_cross_ou,
            UA_design     = self.UA_evap_design,
            dV_fan_design = self.dV_ou_design,
            is_active     = is_active,
        )
        
        if HX_perf_ou_dict.get('converged', True) == False:
            return None
        
        # 수렴 성공 시 값 추출 (nan이면 자동으로 nan)
        dV_fan_ou   = HX_perf_ou_dict['dV_fan']
        v_fan_ou    = dV_fan_ou / self.A_cross_ou
        UA_evap     = HX_perf_ou_dict['UA']
        T_a_ou_mid  = HX_perf_ou_dict['T_a_ou_mid']
        Q_ou_air    = HX_perf_ou_dict['Q_ou_air']
        
        # T_a_ou_out 계산 (nan이면 자동으로 nan)
        if is_active and not np.isnan(T_a_ou_mid) and not np.isnan(dV_fan_ou) and dV_fan_ou > 0:
            # 8단계: 팬 전력 계산
            E_fan_ou = calc_fan_power_from_dV_fan(
                dV_fan     = dV_fan_ou,
                fan_params = self.fan_params_ou,
                vsd_coeffs = self.vsd_coeffs_ou,
                is_active  = is_active,
            )
            
            if not np.isnan(E_fan_ou) and E_fan_ou > 0:
                T_a_ou_out = T_a_ou_mid + E_fan_ou / (c_a * rho_a * dV_fan_ou)
                T_a_ou_out_K = cu.C2K(T_a_ou_out)
                fan_eff = self.eta_fan_ou_design * dV_fan_ou / E_fan_ou * 100
            else:
                T_a_ou_out = np.nan
                T_a_ou_out_K = np.nan
                fan_eff = np.nan
        else:
            E_fan_ou = np.nan
            T_a_ou_out = np.nan
            T_a_ou_out_K = np.nan
            fan_eff = np.nan
        
        # 7단계: LMTD 기반 열량 계산
        # 7단계: LMTD 대신 (저탕조 온수 온도 - 응축 온도) 차이 기반 열량 계산
        # LMTD_cond = calc_lmtd_fluid_and_constant_temp(...) -> 제거됨
        # dT_ref_cond = T_cond_sat_K - T_tank_w_K (Optimization Variable)
        
        UA_cond     = self.UA_cond_design  # constant assumption 
        
        # [MODIFIED] Q = UA * dT (Physical simplification per user request)
        # dT_ref_cond는 최적화 변수로, T_cond_sat - T_tank_w 차이임.
        Q_tank_w    = UA_cond * dT_ref_cond if is_active else 0.0
        
        # 9단계: 엑서지 계산
        P0 = 101325
        if is_active and not np.isnan(T0_K):
            h0 = CP.PropsSI('H', 'T', T0_K, 'P', P0, self.ref)
            s0 = CP.PropsSI('S', 'T', T0_K, 'P', P0, self.ref)
        else:
            h0 = s0 = np.nan
        
        # 기본 엑서지 값 (단위 질량당) - nan이면 자동으로 nan
        x1 = (h1-h0) - T0_K*(s1 - s0) if is_active and not np.isnan(h1) else np.nan
        x2 = (h2-h0) - T0_K*(s2 - s0) if is_active and not np.isnan(h2) else np.nan
        x3 = (h3-h0) - T0_K*(s3 - s0) if is_active and not np.isnan(h3) else np.nan
        x4 = (h4-h0) - T0_K*(s4 - s0) if is_active and not np.isnan(h4) else np.nan
        
        # T1_star, T2_star, T3_star의 물성치 계산 (포화점)
        if is_active and not np.isnan(P1) and not np.isnan(h0):
            # T1_star: 포화 증기 (Q=1)
            P1_star = P1  # 증발기 포화 압력
            h1_star = CP.PropsSI('H', 'P', P1_star, 'Q', 1, self.ref)
            s1_star = CP.PropsSI('S', 'P', P1_star, 'Q', 1, self.ref)
            x1_star = (h1_star-h0) - T0_K*(s1_star - s0)
            
            # T2_star: 응축기 포화 증기 (Q=1) - cycle_states에서 추출
            P2_star = cycle_states.get('P2_star', P2)  # 응축기 포화 압력
            h2_star = cycle_states.get('h2_star', np.nan)
            s2_star = cycle_states.get('s2_star', np.nan)
            if np.isnan(h2_star) or np.isnan(s2_star):
                # fallback: 직접 계산
                h2_star = CP.PropsSI('H', 'P', P2_star, 'Q', 1, self.ref)
                s2_star = CP.PropsSI('S', 'P', P2_star, 'Q', 1, self.ref)
            x2_star = (h2_star-h0) - T0_K*(s2_star - s0)
            
            # T3_star: 포화 액체 (Q=0)
            P3_star = P3  # 응축기 포화 압력
            h3_star = CP.PropsSI('H', 'P', P3_star, 'Q', 0, self.ref)
            s3_star = CP.PropsSI('S', 'P', P3_star, 'Q', 0, self.ref)
            x3_star = (h3_star-h0) - T0_K*(s3_star - s0)
        else:
            P1_star = P2_star = P3_star = np.nan
            h1_star = h2_star = h3_star = np.nan
            s1_star = s2_star = s3_star = np.nan
            x1_star = x2_star = x3_star = np.nan
        
        # 냉매 유량 기반 엑서지 (nan이면 자동으로 nan)
        X1       = m_dot_ref * x1 if is_active and not np.isnan(m_dot_ref) and not np.isnan(x1) else np.nan
        X2       = m_dot_ref * x2 if is_active and not np.isnan(m_dot_ref) and not np.isnan(x2) else np.nan
        X3       = m_dot_ref * x3 if is_active and not np.isnan(m_dot_ref) and not np.isnan(x3) else np.nan
        X4       = m_dot_ref * x4 if is_active and not np.isnan(m_dot_ref) and not np.isnan(x4) else np.nan
        X_cmp    = E_cmp if is_active and not np.isnan(E_cmp) else np.nan
        X_fan_ou = E_fan_ou if is_active and not np.isnan(E_fan_ou) else np.nan
        
        # 실외 공기 및 응축기 엑서지 (nan이면 자동으로 nan)
        X_a_ou_in = calc_exergy_flow(G=c_a * rho_a * dV_fan_ou, T=T_a_ou_in_K, T0=T0_K)
        X_a_ou_out = calc_exergy_flow(G=c_a * rho_a * dV_fan_ou, T=T_a_ou_out_K, T0=T0_K)
        X_ref_cond = Q_ref_cond * (1 - T0_K / T_tank_w_K) if is_active and not np.isnan(Q_ref_cond) else np.nan
        
        # 탱크 및 믹싱밸브 엑서지
        dV_w_tank_out = self.dV_w_tank_out
        dV_w_tank_in  = self.dV_w_tank_in
        dV_w_sup_mix  = self.dV_w_sup_mix
        dV_w_serv     = self.dV_w_serv
        
        
        # 실제 서비스 온도 계산 (믹싱 밸브)
        # dV_w_serv == 0이면 nan으로 설정
        if dV_w_serv == 0:
            T_serv_w_actual = np.nan
            T_serv_w_actual_K = np.nan
        else:
            # alp 계산: 저탕조 온수 비율
            den = max(1e-6, T_tank_w_K - self.T_sup_w_K)
            alp = min(1.0, max(0.0, (self.T_serv_w_K - self.T_sup_w_K) / den))
            
            # 실제 서비스 온도 계산
            if alp >= 1.0:
                # 저탕조 온수를 그대로 사용하는 경우 (T_tank_w < T_serv_w 목표값)
                T_serv_w_actual = T_tank_w
                T_serv_w_actual_K = T_tank_w_K
            else:
                # 믹싱 밸브로 저탕조 온수와 상수도를 믹싱
                T_serv_w_actual_K = alp * T_tank_w_K + (1 - alp) * self.T_sup_w_K
                T_serv_w_actual = cu.K2C(T_serv_w_actual_K)
        
        Q_tank_in    = calc_energy_flow(G=c_w * rho_w * dV_w_tank_in, T=self.T_sup_w_K, T0=T0_K)
        Q_tank_w_out = calc_energy_flow(G=c_w * rho_w * dV_w_tank_in, T=T_tank_w_K, T0=T0_K)
        Q_mix_sup_w  = calc_energy_flow(G=c_w * rho_w * dV_w_sup_mix, T=self.T_sup_w_K, T0=T0_K)
        Q_mix_serv_w = calc_energy_flow(G=c_w * rho_w * dV_w_serv, T=T_serv_w_actual_K, T0=T0_K)
        
        X_tank_w_in    = calc_exergy_flow(G=c_w * rho_w * dV_w_tank_in, T=self.T_sup_w_K, T0=T0_K)
        X_tank_w_out     = calc_exergy_flow(G=c_w * rho_w * dV_w_tank_in, T=T_tank_w_K, T0=T0_K)
        X_mix_sup_w  = calc_exergy_flow(G=c_w * rho_w * dV_w_sup_mix, T=self.T_sup_w_K, T0=T0_K)
        X_mix_serv_w = calc_exergy_flow(G=c_w * rho_w * dV_w_serv, T=T_serv_w_actual_K, T0=T0_K)
        
        # 10단계: 최종 결과 딕셔너리 생성
        # 포화 온도 추출 (OFF 상태에서는 nan)
        T1_star_K_result = cycle_states.get('T1_star_K', T1_star_K) if isinstance(cycle_states, dict) else T1_star_K
        T2_star_K_result = cycle_states.get('T2_star_K', T2_star_K) if isinstance(cycle_states, dict) else T2_star_K
        T3_star_K_result = cycle_states.get('T3_star_K', T3_star_K) if isinstance(cycle_states, dict) else T3_star_K
        
        result = {
            'hp_is_on': (Q_cond_load > 0),
            'converged': True,

            # === [온도: °C] =======================================
            # [NEW] Saturation Points
            'T1_star [°C]': cu.K2C(T1_star_K_result),
            'T2_star [°C]': cu.K2C(T2_star_K_result),
            'T3_star [°C]': cu.K2C(T3_star_K_result),
            
            # [Updated] Actual Points
            'T_a_ou_in [°C]': T_a_ou_in,
            'T_a_ou_out [°C]': T_a_ou_out,
            'T1 [°C]': cu.K2C(T1_K),
            'T2 [°C]': cu.K2C(T2_K),
            'T3 [°C]': cu.K2C(T3_K),
            'T4 [°C]': cu.K2C(T4_K),
            'T_tank_w [°C]': T_tank_w,
            'T_sup_w [°C]': self.T_sup_w,
            'T_serv_w [°C]': T_serv_w_actual,
            'T0 [°C]': T0,
            # 'LMTD_evap [K]' -> Removed
            # 'LMTD_cond [K]' -> Removed

            # === [체적유량: m3/s] ==================================
            'dV_fan_ou [m3/s]': dV_fan_ou,
            'v_fan_ou [m/s]': v_fan_ou, 
            'dV_w_serv [m3/s]': dV_w_serv if dV_w_serv > 0 else np.nan,
            'dV_w_tank_in [m3/s]': dV_w_tank_in if dV_w_tank_in > 0 else np.nan,
            'dV_w_sup_mix [m3/s]': dV_w_sup_mix if dV_w_sup_mix > 0 else np.nan,

            # === [압력: Pa] ========================================
            'P1 [Pa]': P1,
            'P2 [Pa]': P2,
            'P3 [Pa]': P3,
            'P4 [Pa]': P4,
            'P1_star [Pa]': P1_star,
            'P2_star [Pa]': P2_star,
            'P3_star [Pa]': P3_star,
            'dP_fan_ou_static [Pa]': self.dP_ou_design - 1/2 * rho_a * v_fan_ou**2,
            'dP_fan_ou_dynamic [Pa]': 1/2 * rho_a * v_fan_ou**2,

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

            # === [에너지/엑서지: W] ================================
            # ---- 실외측(공기, 증발기)
            'E_fan_ou [W]': E_fan_ou,
            'X_a_ou_in [W]': X_a_ou_in,
            'X_a_ou_out [W]': X_a_ou_out,

            # ---- 증발기
            'Q_ref_evap [W]': Q_ref_evap,
            'Q_ou_air [W]': Q_ou_air,

            # ---- 냉매 엑서지 (상태점)
            'X1 [W]': X1,
            'X2 [W]': X2,
            'X3 [W]': X3,
            'X4 [W]': X4,

            # ---- 압축기
            'E_cmp [W]': E_cmp,
            'X_cmp [W]': X_cmp,

            # ---- 응축기/저탕조
            'Q_cond_load [W]': Q_cond_load,
            'Q_ref_cond [W]': Q_ref_cond,
            'Q_tank_w [W]': Q_tank_w,
            'X_ref_cond [W]': X_ref_cond,

            # ---- 탱크
            'X_tank_w_in [W]': X_tank_w_in,
            'Q_tank_w_out [W]': Q_tank_w_out,
            'X_tank_w_out [W]': X_tank_w_out,

            # ---- 믹싱 밸브 & 온수 공급
            'Q_mix_serv_w [W]': Q_mix_serv_w,
            'X_mix_sup_w [W]': X_mix_sup_w,
            'X_mix_serv_w [W]': X_mix_serv_w,

            # ---- 총괄(손실, 총입력, 흐름 등)
            'X_fan_ou [W]': X_fan_ou,
            'E_tot [W]': (E_cmp + E_fan_ou) if is_active and not np.isnan(E_cmp) and not np.isnan(E_fan_ou) else np.nan,

            # === [무차원: 효율, COP] ==============================
            'fan_eff [%]': fan_eff,
        }
        
        return result
    
    def _optimize_operation(self, T_tank_w, Q_cond_load, T0):
        """
        히트펌프 최적 운전점 탐색을 수행하는 내부 메서드.
        
        이 메서드는 analyze_steady와 analyze_dynamic에서 공통으로 사용되는
        최적화 로직을 담당합니다. SLSQP 알고리즘을 사용하여 E_tot (총 전력 소비)를 최소화합니다.
        
        Args:
            T_tank_w (float): 저탕조 온도 [°C]
                현재 타임스텝의 저탕조 온도
            
            Q_cond_load (float): 저탕조 목표 열 교환율 [W]
                응축기가 저탕조에 전달해야 하는 목표 열량
            
            T0 (float): 엑서지 분석 기준 온도(=외기온도) [°C]
                엑서지 계산의 기준점으로 사용되는 외기 온도
        
        Returns:
            scipy.optimize.OptimizeResult: 최적화 결과 객체
                - x: 최적화된 변수 [dT_ref_cond, dT_ref_evap]
                - success: 최적화 성공 여부
                - 기타 최적화 메타데이터
        
        Notes:
            - 최적화 알고리즘: SLSQP (Sequential Least Squares Programming)
            - 최적화 변수: [dT_ref_cond, dT_ref_evap]
                - dT_ref_cond: 냉매-저탕조 온도차 [K]
                - dT_ref_evap: 냉매-실외 공기 온도차 [K]
            - 제약 조건:
                - Q_cond_load <= Q_LMTD_cond <= Q_cond_load * (1 + tolerance) (응축기 열전달 능력 범위)
                - Q_ref_evap * (1 - tolerance) <= Q_LMTD_evap <= Q_ref_evap * (1 + tolerance) (증발기 열전달 밸런스 범위)
            - 목적 함수: E_tot (E_cmp + E_fan_ou) 최소화
        """
        # 요구사항 A: tolerance 변수 정의
        tolerance = 0.001  # 0.1%
        
        # 최적화 변수 경계 조건 및 초기 추정값 설정
        bounds = [(1.0, 30.0), (1.0, 30.0)]  # [dT_ref_cond, dT_ref_evap]
        
        # [Updated] 이전에 최적화 된 값이 있다면 그 값을 초기값으로 할당
        if self.prev_opt_x is not None:
            initial_guess = self.prev_opt_x
        else:
            initial_guess = [5.0, 7.0]
        
        # Phase 2: 본 최적화 (E_tot 최소화)
        def _cond_LMTD_constraint_low(x):
            try:
                perf = self._calc_state(
                    optimization_vars=x,
                    T_tank_w=T_tank_w,
                    Q_cond_load=Q_cond_load,
                    T0=T0
                )
                if perf is None or not isinstance(perf, dict):
                    return -1e6  # ineq 제약이므로 음수 반환
                
                if "Q_tank_w [W]" not in perf or np.isnan(perf["Q_tank_w [W]"]):
                    return -1e6
                
                return perf["Q_tank_w [W]"] - Q_cond_load
            except Exception as e:
                return -1e6
        
        def _cond_LMTD_constraint_high(x):
            try:
                perf = self._calc_state(
                    optimization_vars=x,
                    T_tank_w=T_tank_w,
                    Q_cond_load=Q_cond_load,
                    T0=T0
                )
                if perf is None or not isinstance(perf, dict):
                    return -1e6  # ineq 제약이므로 음수 반환
                
                if "Q_tank_w [W]" not in perf or np.isnan(perf["Q_tank_w [W]"]):
                    return -1e6
                
                # 제약 조건: Q_cond_load*(1+tolerance) - Q_tank_w >= 0
                return Q_cond_load * (1 + tolerance) - perf["Q_tank_w [W]"]
            except Exception as e:
                return -1e6

        def _evap_LMTD_constraint_low(x):
            try:
                perf = self._calc_state(
                    optimization_vars=x,
                    T_tank_w=T_tank_w,
                    Q_cond_load=Q_cond_load,
                    T0=T0
                )
                if perf is None or not isinstance(perf, dict):
                    return -1e6  # ineq 제약이므로 음수 반환
                
                if ("Q_ou_air [W]" not in perf or "Q_ref_evap [W]" not in perf or
                    np.isnan(perf["Q_ou_air [W]"]) or np.isnan(perf["Q_ref_evap [W]"])):
                    return -1e6
                
                # 제약 조건: Q_ou_air - Q_ref_evap*(1-tolerance) >= 0
                return perf["Q_ou_air [W]"] - perf['Q_ref_evap [W]']
            except Exception as e:
                return -1e6
        
        def _evap_LMTD_constraint_high(x):
            try:
                perf = self._calc_state(
                    optimization_vars=x,
                    T_tank_w=T_tank_w,
                    Q_cond_load=Q_cond_load,
                    T0=T0
                )
                if perf is None or not isinstance(perf, dict):
                    return -1e6  # ineq 제약이므로 음수 반환
                
                if ("Q_ou_air [W]" not in perf or "Q_ref_evap [W]" not in perf or
                    np.isnan(perf["Q_ou_air [W]"]) or np.isnan(perf["Q_ref_evap [W]"])):
                    return -1e6
                
                # 제약 조건: Q_ref_evap*(1+tolerance) - Q_ou_air >= 0
                return perf['Q_ref_evap [W]'] * (1 + tolerance) - perf["Q_ou_air [W]"]
            except Exception as e:
                return -1e6
        
        const_funcs = [
            {'type': 'ineq', 'fun': _cond_LMTD_constraint_low},   # Q_tank_w - Q_cond_load >= 0
            {'type': 'ineq', 'fun': _cond_LMTD_constraint_high},  # Q_cond_load*(1+tolerance) - Q_tank_w >= 0
            {'type': 'ineq', 'fun': _evap_LMTD_constraint_low},   # Q_ou_air - Q_ref_evap*(1-tolerance) >= 0
            {'type': 'ineq', 'fun': _evap_LMTD_constraint_high},  # Q_ref_evap*(1+tolerance) - Q_ou_air >= 0
        ]

        def _objective(x):  # x = [dT_ref_cond, dT_ref_evap]
            """
            Phase 2 목적 함수: E_tot (총 전력 소비) 최소화.
            perf가 None이면 (수렴 실패 등) 큰 penalty 값을 반환하여
            최적화 알고리즘이 해당 변수 조합을 피하고 다른 조합을 시도하도록 함.
            """
            try:
                perf = self._calc_state(
                    optimization_vars=x,
                    T_tank_w=T_tank_w,
                    Q_cond_load=Q_cond_load,
                    T0=T0
                )
                if perf is None or not isinstance(perf, dict):
                    return 1e6
                
                if "E_tot [W]" not in perf or np.isnan(perf["E_tot [W]"]):
                    return 1e6
                
                return perf["E_tot [W]"]
            except Exception as e:
                return 1e6
        
        # SLSQP 알고리즘 옵션 설정
        options = {
            'disp': False,
            'maxiter': 1000,    # Iteration limit increased
            'ftol': 1e-5,      # Tightened tolerance (was 10)
            'eps': 1e-2,       # Finite difference step size
        }
        
        # 최적화 실행 (항상 SLSQP 사용)
        opt_result = minimize(
            _objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=const_funcs,
            callback=None,
            options=options
        )
        
        # [Updated] 최적화 성공 시 결과 저장
        if opt_result.success:
            self.prev_opt_x = opt_result.x
            
        return opt_result
    
    def analyze_steady(
        self,
        T_tank_w,
        T0,
        dV_w_serv=None,  # 급탕 유량 [m3/s], None이면 0으로 가정
        Q_cond_load=None,  # 저탕조 목표 열 교환율 [W], None이면 0으로 가정
        return_dict=True  # True면 dict 반환, False면 DataFrame 반환
    ):
        """
        정상상태 해석 함수.
        
        주어진 조건에서 히트펌프의 정상상태 성능을 계산합니다.
        
        두 가지 사용 모드를 지원합니다:
        1. dV_w_serv 제공: 급탕 유량 기반으로 열 손실을 계산하여 Q_cond_load를 자동 결정
        2. Q_cond_load 제공: 직접 주어진 응축기 열량으로 성능 계산
        
        Args:
            T_tank_w (float): 저탕조 온도 [°C]
            T0 (float): 엑서지 분석 기준 온도(=외기온도) [°C]
                엑서지 계산의 기준점으로 사용되는 외기 온도
            dV_w_serv (float, optional): 급탕 유량 [m3/s]. 
                Q_cond_load가 None인 경우 필수. 제공된 경우 열 손실을 계산하여 
                Q_cond_load를 자동 결정합니다.
            Q_cond_load (float, optional): 저탕조 목표 열 교환율 [W].
                dV_w_serv가 None인 경우 필수. 직접 주어진 응축기 열량으로 성능을 계산합니다.
            return_dict (bool): True면 dict 반환, False면 DataFrame 반환
        
        Returns:
            dict 또는 pd.DataFrame: 정상상태 해석 결과
                - run_simulation과 동일한 형식의 결과 딕셔너리
                - 최적화 실패 시 OFF 상태 결과 반환
        
        Raises:
            ValueError: dV_w_serv와 Q_cond_load가 둘 다 None이거나, 둘 다 값이 주어진 경우
        """
        # 입력 검증
        if dV_w_serv is None and Q_cond_load is None:
            raise ValueError("dV_w_serv와 Q_cond_load 중 하나는 반드시 제공되어야 합니다.")
        if dV_w_serv is not None and Q_cond_load is not None:
            raise ValueError("dV_w_serv와 Q_cond_load를 동시에 제공할 수 없습니다.")
        
        # 온도 단위 변환
        T_tank_w_K = cu.C2K(T_tank_w)
        T0_K = cu.C2K(T0)
        
        # 급탕 유량 설정
        if dV_w_serv is None:
            dV_w_serv = 0.0
        
        # 급탕 유량 관련 계산
        Q_tank_loss = self.UA_tank * (T_tank_w_K - T0_K)
        den = max(1e-6, T_tank_w_K - self.T_sup_w_K)
        alp = min(1.0, max(0.0, self.T_serv_w_K - self.T_sup_w_K) / den)
        
        self.dV_w_serv = dV_w_serv
        self.dV_w_tank_out = alp * dV_w_serv
        self.dV_w_sup_mix = (1 - alp) * dV_w_serv
        
        # Q_cond_load 계산
        if Q_cond_load is None:
            # dV_w_serv가 주어진 경우: 열 손실 계산하여 Q_cond_load 결정
            Q_use_loss = c_w * rho_w * self.dV_w_tank_out * (T_tank_w_K - self.T_sup_w_K)
            total_loss = Q_tank_loss + Q_use_loss
            # 정상상태: Q_cond_load = total_loss (에너지 밸런스)
            Q_cond_load = total_loss
        else:
            # Q_cond_load가 주어진 경우: 주어진 값 사용
            # Q_cond_load를 사용하므로 Q_use_loss는 계산하지 않음
            pass
        
        # ON/OFF 상태 결정
        if T_tank_w <= self.T_tank_w_lower_bound:
            hp_is_on = True
        elif T_tank_w > self.T_tank_w_upper_bound:
            hp_is_on = False
        else:
            # 정상상태에서는 Q_cond_load > 0이면 ON으로 가정
            hp_is_on = Q_cond_load > 0
        
        # OFF 상태 조기 체크: Q_cond_load가 0 이하이면 최적화 건너뛰기
        if Q_cond_load <= 0 or not hp_is_on:
            result = self._calc_state(
                optimization_vars=[5.0, 5.0],
                T_tank_w=T_tank_w,
                Q_cond_load=0.0,
                T0=T0
            )
        else:
            opt_result = self._optimize_operation(
                T_tank_w=T_tank_w,
                Q_cond_load=Q_cond_load,
                T0=T0
            )
            result = None
            try:
                result = self._calc_state(
                    optimization_vars=opt_result.x,
                    T_tank_w=T_tank_w,
                    T0=T0,
                    Q_cond_load=Q_cond_load,
                )
            except Exception:
                pass
            
            # result 검증 및 fallback
            if result is None or not isinstance(result, dict):
                try:
                    result = self._calc_state(
                        optimization_vars=[5.0, 5.0],
                        T_tank_w=T_tank_w,
                        Q_cond_load=0.0,
                        T0=T0
                    )
                except Exception:
                    result = {
                        'hp_is_on': False,
                        'converged': False,
                        'Q_cond_load [W]': Q_cond_load,
                        'Q_ref_cond [W]': 0.0,
                        'Q_ref_evap [W]': 0.0,
                        'E_cmp [W]': 0.0,
                        'E_fan_ou [W]': 0.0,
                        'E_tot [W]': 0.0,
                        'T_tank_w [°C]': T_tank_w,
                        'T0 [°C]': T0
                    }
            
            # converged 플래그 설정
            if result is not None and isinstance(result, dict):
                # opt_result가 정의되어 있고 success 속성이 있는 경우에만 설정
                if 'opt_result' in locals() and hasattr(opt_result, 'success'):
                    result['converged'] = opt_result.success
                    if result['converged'] is False:
                        print(f"Optimization failed")
                # opt_result가 없는 경우 (예외 상황), converged는 이미 result에 설정되어 있거나 False로 유지
        
        if return_dict:
            return result
        else:
            # DataFrame으로 변환
            return pd.DataFrame([result])
    
    def analyze_dynamic(
        self, 
        simulation_period_sec, 
        dt_s, 
        T_tank_w_init_C,
        schedule_entries,
        T0_schedule,
        I_DN_schedule=None,
        I_dH_schedule=None,
        tank_level_init = 1.0,    
        result_save_csv_path=None,
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
            I_DN_schedule: 직달일사 스케줄 [W/m2] (optional, STC 사용 시 필수)
            I_dH_schedule: 확산일사 스케줄 [W/m2] (optional, STC 사용 시 필수)
            result_save_csv_path: 결과 CSV 저장 경로
        
        Returns:
            pd.DataFrame: 시뮬레이션 타임스텝별 결과 데이터
        """
        
        time = np.arange(0, simulation_period_sec, dt_s)
        tN = len(time)
        T0_schedule = np.array(T0_schedule)
        if len(T0_schedule) != tN:
            raise ValueError(f"T0_schedule length ({len(T0_schedule)}) must match time array length ({tN})")
        
        # STC 사용 여부 확인
        use_stc = (self.A_stc > 0) and (I_DN_schedule is not None) and (I_dH_schedule is not None)
        if use_stc:
            I_DN_schedule = np.array(I_DN_schedule)
            I_dH_schedule = np.array(I_dH_schedule)
            if len(I_DN_schedule) != tN:
                raise ValueError(f"I_DN_schedule length ({len(I_DN_schedule)}) must match time array length ({tN})")
            if len(I_dH_schedule) != tN:
                raise ValueError(f"I_dH_schedule length ({len(I_dH_schedule)}) must match time array length ({tN})")
        
        results_data = []
        
        self.time = time
        self.dt = dt_s
        
        self.dV_w_serv = 0.0 
        self.dV_w_tank_out = 0.0 
        self.dV_w_sup_mix = 0.0 
        
        self.w_use_frac = build_schedule_ratios(schedule_entries, self.time)
        
        T_tank_w_K = cu.C2K(T_tank_w_init_C)
        T_tank_w_K_prev = cu.C2K(T_tank_w_init_C)  # 이전 타임스텝 온도 추적
        
        # 저탕조 수위 초기화
        tank_level = tank_level_init  # 초기 수위 100%
        
        # 리필 상태 추적 (히스테리시스 제어용)
        is_refilling = False

        hp_is_on_prev = False
        for n in tqdm(range(tN), desc="ASHPB Simulating"):
            # =================================================================
            # 1. TIMESTEP INITIALIZATION & CONTEXT
            # =================================================================
            # Time & Weather
            current_time_s = time[n]
            current_hour = current_time_s * cu.s2h
            T0 = T0_schedule[n]
            T0_K = cu.C2K(T0)
            
            # State Flags (Context)
            preheat_on = (
                current_hour >= self.preheat_start_hour
                and current_hour < self.preheat_end_hour
            )

            # Current Tank State (Start of Step)
            T_tank_w = cu.K2C(T_tank_w_K)
            
            # Initialize Result Container & Defaults
            step_results = {}
            
            # UV Lamp (Time-dependent only)
            E_uv = 0.0
            if (
                self.num_switching_per_3hour > 0
                and self.lamp_power_watts > 0
            ):
                time_in_period = current_time_s % self.period_3hour_sec
                interval = (
                    self.period_3hour_sec
                    - self.num_switching_per_3hour * self.uv_lamp_exposure_duration_sec
                ) / (self.num_switching_per_3hour + 1)
                for i in range(self.num_switching_per_3hour):
                    start_time = interval * (i + 1) + i * self.uv_lamp_exposure_duration_sec
                    if start_time <= time_in_period < start_time + self.uv_lamp_exposure_duration_sec:
                        E_uv = self.lamp_power_watts
                        break

            # STC Results Default Initialization
            stc_active = False
            T_tank_w_in_K = self.T_sup_w_K  # Default refill source is mains water
            T_stc_w_out_K = np.nan
            Q_stc_w_out = 0.0
            Q_stc_w_in = 0.0
            E_pump = self.E_pump
            Q_pump = E_pump
            stc_result = {}
            
            # Refill Volume Default
            V_refill = 0.0
            self.dV_w_tank_in = 0.0
            

            # =================================================================
            # 2. DEMAND & TANK OUTFLOW (MASS BALANCE 1)
            # =================================================================
            # Calculate demand fractions
            den = max(1e-6, T_tank_w_K - self.T_sup_w_K)
            alp = min(1.0, max(0.0, self.T_serv_w_K - self.T_sup_w_K) / den)

            self.dV_w_serv = self.w_use_frac[n] * self.dV_w_serv_max 
            self.dV_w_tank_out = alp * self.dV_w_serv
            tank_outlet_exist = self.dV_w_tank_out > 0
            self.dV_w_sup_mix = (1 - alp) * self.dV_w_serv

            # Tank Loss Calculation (Based on Pre-mix Temperature)
            # (Note: Original code calc Q_tank_loss at start, used unmixed T_tank_w_K)
            Q_tank_loss = self.UA_tank * (T_tank_w_K - T0_K)

            # Update Tank Level (Outflow)
            # [MODIFIED] always_full allows level drop if prevent_simultaneous_flow is ON
            if not self.tank_always_full or (self.tank_always_full and self.prevent_simultaneous_flow):
                tank_level -= (self.dV_w_tank_out * dt_s) / self.V_tank_full
                tank_level = max(0.0, tank_level)

            # =================================================================
            # 3. REFILL LOGIC (MASS BALANCE 2)
            # =================================================================

            # [CASE 1] Exclusive Flow Mode
            if self.tank_always_full and self.prevent_simultaneous_flow:
                # [CASE 1-1] Tank outlet exists
                if tank_outlet_exist:
                    self.dV_w_tank_in = 0.0
                    is_refilling = False 
                # [CASE 1-2] Tank outlet does not exist
                elif not tank_outlet_exist:
                    if tank_level < 1.0:
                        req_vol = (1.0 - tank_level) * self.V_tank_full
                        tank_will_overflow = self.dV_refill_m3s * dt_s > req_vol 
                        V_refill = self.dV_refill_m3s * dt_s if not tank_will_overflow else 0.0
                        self.dV_w_tank_in = V_refill / dt_s
                    else:
                        self.dV_w_tank_in = 0.0

            # [CASE 2] Classic Always Full
            elif self.tank_always_full:
                self.dV_w_tank_in = self.dV_w_tank_out # Simultaneous flow

            # [CASE 3] Classic Dynamic Level (Hysteretic)
            elif not self.tank_always_full:
                target_lower = self.tank_level_lower_bound
                target_upper = self.tank_level_upper_bound

                # Mains Preheat Special Condition: Keep full during solar window
                if use_stc and self.stc_placement == 'mains_preheat' and preheat_on:
                    target_lower = 1.0
                    target_upper = 1.0

                if not is_refilling:
                    if tank_level < target_lower - 1e-6:
                         is_refilling = True
                
                if is_refilling:
                    req_vol = (target_upper - tank_level) * self.V_tank_full
                    tank_will_overflow = self.dV_refill_m3s * dt_s > req_vol 
                    V_refill = self.dV_refill_m3s * dt_s if not tank_will_overflow else 0.0
                    self.dV_w_tank_in = V_refill / dt_s
                    
                    # Refill Stop Condition
                    if (tank_level + V_refill / self.V_tank_full) >= target_upper - 1e-6:
                        is_refilling = False
                else:
                    self.dV_w_tank_in = 0.0

            # =================================================================
            # 4. SOLAR THERMAL COLLECTOR (STC) & SOURCE TEMP
            # =================================================================
            if use_stc:
                # 4-1. Tank Circuit Mode
                if self.stc_placement == 'tank_circuit':
                    # Probing
                    stc_result_test = calc_stc_performance(
                        I_DN_stc=I_DN_schedule[n], I_dH_stc=I_dH_schedule[n],
                        T_stc_w_in_K=T_tank_w_K, T0_K=T0_K,
                        A_stc_pipe=self.A_stc_pipe, alpha_stc=self.alpha_stc,
                        h_o_stc=self.h_o_stc, h_r_stc=self.h_r_stc,
                        k_ins_stc=self.k_ins_stc, x_air_stc=self.x_air_stc,
                        x_ins_stc=self.x_ins_stc, dV_stc=self.dV_stc,
                        E_pump=self.E_pump,
                        is_active=True, 
                    )
                    
                    stc_active = preheat_on and stc_result_test['T_stc_w_out_K'] > T_tank_w_K
                    if stc_active:
                        stc_result = stc_result_test
                    else:
                        stc_result = calc_stc_performance(
                            I_DN_stc=I_DN_schedule[n], I_dH_stc=I_dH_schedule[n],
                            T_stc_w_in_K=T_tank_w_K, T0_K=T0_K,
                            A_stc_pipe=self.A_stc_pipe, alpha_stc=self.alpha_stc,
                            h_o_stc=self.h_o_stc, h_r_stc=self.h_r_stc,
                            k_ins_stc=self.k_ins_stc, x_air_stc=self.x_air_stc,
                            x_ins_stc=self.x_ins_stc, dV_stc=self.dV_stc,
                            E_pump=self.E_pump,
                            is_active=False
                        )

                    T_stc_w_out_K = stc_result['T_stc_w_out_K']
                    Q_stc_w_out   = stc_result.get('Q_stc_w_out', 0.0)
                    Q_stc_w_in    = stc_result.get('Q_stc_w_in', 0.0)
                
                # 4-2. Mains Preheat Mode
                elif self.stc_placement == 'mains_preheat':
                    # Only active if flow exists
                    if preheat_on and self.dV_w_tank_in > 0:
                        stc_result_test = calc_stc_performance(
                            I_DN_stc=I_DN_schedule[n], I_dH_stc=I_dH_schedule[n],
                            T_stc_w_in_K=self.T_sup_w_K, T0_K=T0_K,
                            A_stc_pipe=self.A_stc_pipe, alpha_stc=self.alpha_stc,
                            h_o_stc=self.h_o_stc, h_r_stc=self.h_r_stc,
                            k_ins_stc=self.k_ins_stc, x_air_stc=self.x_air_stc,
                            x_ins_stc=self.x_ins_stc, dV_stc=self.dV_w_tank_in,
                            E_pump=self.E_pump,
                            is_active=True,
                        )
                        
                        if stc_result_test['T_stc_w_out_K'] > self.T_sup_w_K:
                            stc_active = True
                            stc_result = stc_result_test
                        else:
                            stc_active = False
                            stc_result = calc_stc_performance(
                                I_DN_stc=I_DN_schedule[n], I_dH_stc=I_dH_schedule[n],
                                T_stc_w_in_K=self.T_sup_w_K, T0_K=T0_K,
                                A_stc_pipe=self.A_stc_pipe, alpha_stc=self.alpha_stc,
                                h_o_stc=self.h_o_stc, h_r_stc=self.h_r_stc,
                                k_ins_stc=self.k_ins_stc, x_air_stc=self.x_air_stc,
                                x_ins_stc=self.x_ins_stc, dV_stc=self.dV_w_tank_in,
                                E_pump=self.E_pump,
                                is_active=False
                            )
                    else:
                        stc_active = False
                        stc_result = calc_stc_performance(
                            I_DN_stc=I_DN_schedule[n], I_dH_stc=I_dH_schedule[n],
                            T_stc_w_in_K=self.T_sup_w_K, T0_K=T0_K,
                            A_stc_pipe=self.A_stc_pipe, alpha_stc=self.alpha_stc,
                            h_o_stc=self.h_o_stc, h_r_stc=self.h_r_stc,
                            k_ins_stc=self.k_ins_stc, x_air_stc=self.x_air_stc,
                            x_ins_stc=self.x_ins_stc, dV_stc=1.0, 
                            E_pump=self.E_pump,
                            is_active=False
                        )

                    T_stc_w_out_K = stc_result['T_stc_w_out_K']
                    Q_stc_w_out = stc_result.get('Q_stc_w_out', 0.0)
                    Q_stc_w_in = stc_result.get('Q_stc_w_in', 0.0)
                    
                    if stc_active:
                        T_tank_w_in_K = T_stc_w_out_K
                        E_pump = self.E_pump
                    else:
                        T_tank_w_in_K = self.T_sup_w_K
                        E_pump = 0.0

            # =================================================================
            # 5. REFILL MASS & ENERGY BALANCE
            # =================================================================
            # 5-1. Mass Balance Update (Refill)
            # Note: Outflow was already deducted in Step 2. Now add Inflow.
            V_new_fill = self.dV_w_tank_in * dt_s
            tank_level += V_new_fill / self.V_tank_full
            # Ensure strictly within bounds (prevent C_tank -> 0)
            tank_level = max(0.001, min(1.0, tank_level)) 
            
            # 5-2. Determine Refill Temperature (T_refill_K)
            # Fix potential issue where tank_circuit mode incorrectly sets T_refill = T_stc_out
            if use_stc and self.stc_placement == 'mains_preheat' and stc_active:
                T_refill_K = T_stc_w_out_K
            else:
                T_refill_K = self.T_sup_w_K

            # Update T_tank_w_in_K for result logging/Exergy (Enthalpy basis)
            T_tank_w_in_K = T_refill_K

            # 5-3. Energy Flux (Refill)
            # Q_refill_net = m_dot * Cp * (T_in - T_tank)
            # This accounts for the energy brought in by the refill water relative to current tank temp.
            # If mains_preheat is active, T_refill_K is boosted, so this Term INCLUDES STC gain.
            Q_refill_net = c_w * rho_w * self.dV_w_tank_in * (T_refill_K - T_tank_w_K)
            
            # 5-4. Use Loss & Enthalpy In (For Exergy/Reports ONLY, not for T update)
            # Q_use_loss: Energy leaving the system boundary relative to T0
            Q_use_loss = c_w * rho_w * self.dV_w_tank_out * (T_tank_w_K - T0_K)
            # Q_tank_w_in: Total enthalpy entering the system boundary relative to T0
            Q_tank_w_in = c_w * rho_w * self.dV_w_tank_in * (T_tank_w_in_K - T0_K)

            # =================================================================
            # 6. HP CONTROL & SIMULATION
            # =================================================================
            # Determine On/Off
            if T_tank_w <= self.T_tank_w_lower_bound:
                hp_is_on = True
            elif T_tank_w >= self.T_tank_w_upper_bound:
                hp_is_on = False
            else:
                hp_is_on = hp_is_on_prev
            
            hp_is_on_schedule = check_hp_schedule_active(current_hour, self.hp_on_schedule) 
            # Note: current_hour was calculated at top as current_time_s * cu.s2h

            hp_is_on = hp_is_on and hp_is_on_schedule
            hp_is_on_prev = hp_is_on
            
            Q_cond_load_n = self.hp_capacity if hp_is_on else 0.0

            # Optimize / Calculate State
            if Q_cond_load_n == 0:
                result = self._calc_state(
                    optimization_vars=[5.0, 5.0],
                    T_tank_w=T_tank_w,
                    Q_cond_load=0.0,
                    T0=T0
                )
            else:
                opt_result = self._optimize_operation(
                    T_tank_w=T_tank_w,
                    Q_cond_load=Q_cond_load_n,
                    T0=T0
                )
                result = self._calc_state(
                    optimization_vars=opt_result.x,
                    T_tank_w=T_tank_w,
                    Q_cond_load=Q_cond_load_n,
                    T0=T0
                )
                if not opt_result.success or result is None:
                    for k, v in result.items():
                        if isinstance(v, float): result[k] = round(v, 2)
                    raise ValueError(f"Optimization failed at timestep {n}: {result}")
                result['converged'] = opt_result.success
            
            # Note: E_uv already calculated at top.

            # =================================================================
            # 7. RESULTS & POST-PROCESS
            # =================================================================
            step_results.update(result)
            step_results['hp_is_on'] = hp_is_on
            if self.lamp_power_watts > 0:
                step_results['E_uv [W]'] = E_uv
            
            if not self.tank_always_full or (self.tank_always_full and self.prevent_simultaneous_flow):
                step_results['tank_level [-]'] = tank_level
            
            # STC Results
            if use_stc:
                step_results.update({
                    'stc_active [-]': stc_active,
                    'I_DN_stc [W/m2]': I_DN_schedule[n],
                    'I_dH_stc [W/m2]': I_dH_schedule[n],
                    'I_sol_stc [W/m2]': stc_result.get('I_sol_stc', np.nan),
                    'Q_sol_stc [W]': stc_result.get('Q_sol_stc', np.nan),
                    'Q_stc_w_out [W]': Q_stc_w_out,
                    'Q_stc_w_in [W]': Q_stc_w_in,
                    'Q_l_stc [W]': stc_result.get('Q_l_stc', np.nan),
                    'T_stc_w_out [°C]': cu.K2C(T_stc_w_out_K) if not np.isnan(T_stc_w_out_K) else np.nan,
                    'T_stc_w_in [°C]': cu.K2C(stc_result.get('T_stc_w_in_K', T_tank_w_K)) if 'T_stc_w_in_K' in stc_result else cu.K2C(T_tank_w_K),
                    'T_stc [°C]': cu.K2C(stc_result.get('T_stc_K', np.nan)),
                    'E_pump [W]': E_pump,
                    'Q_pump [W]': Q_pump,
                    'S_stc_w_in [W/K]': stc_result.get('S_stc_w_in', np.nan),
                    'S_DN_stc [W/K]': stc_result.get('S_DN_stc', np.nan),
                    'S_dH_stc [W/K]': stc_result.get('S_dH_stc', np.nan),
                    'S_sol_stc [W/K]': stc_result.get('S_sol_stc', np.nan),
                    'S_stc_w_out [W/K]': stc_result.get('S_stc_w_out', np.nan),
                    'S_l_stc [W/K]': stc_result.get('S_l_stc', np.nan),
                    'S_g_stc [W/K]': stc_result.get('S_g_stc', np.nan),
                    'X_stc_w_in [W]': stc_result.get('X_stc_w_in', np.nan),
                    'X_sol_stc [W]': stc_result.get('X_sol_stc', np.nan),
                    'X_stc_w_out [W]': stc_result.get('X_stc_w_out', np.nan),
                    'X_l_stc [W]': stc_result.get('X_l_stc', np.nan),
                    'Xc_stc [W]': stc_result.get('Xc_stc', np.nan),
                })
                if self.stc_placement == 'tank_circuit':
                    step_results['T_stc_w_final [°C]'] = cu.K2C(T_stc_w_final_K) if 'T_stc_w_final_K' in locals() else np.nan
            else:
                step_results['stc_active [-]'] = False

            # X Calculations (Exergy)
            if not self.tank_always_full or (self.tank_always_full and self.prevent_simultaneous_flow):
                C_tank_actual = self.C_tank * tank_level
            else:
                C_tank_actual = self.C_tank
            Xst_tank = (
                0.0 if n == 0
                else (1 - T0_K / T_tank_w_K) * C_tank_actual * (T_tank_w_K - T_tank_w_K_prev) / self.dt
            )

            X1             = result['X1 [W]']
            X2             = result['X2 [W]']
            X3             = result['X3 [W]']
            X4             = result['X4 [W]']

            X_a_ou_in      = result['X_a_ou_in [W]']
            X_a_ou_out     = result['X_a_ou_out [W]']

            X_stc_w_in    = stc_result.get('X_stc_w_in', np.nan)
            X_stc_w_out   = stc_result.get('X_stc_w_out', np.nan)
            X_stc_w_final = stc_result.get('X_stc_w_final', np.nan)
            
            X_uv           = E_uv
            X_cmp          = result['X_cmp [W]']
            X_fan_ou       = result['X_fan_ou [W]']

            X_tank_w_in    = result['X_tank_w_in [W]']
            X_tank_w_out   = result['X_tank_w_out [W]']
            X_mix_sup_w    = result['X_mix_sup_w [W]']
            X_mix_serv_w   = result['X_mix_serv_w [W]']
            X_tank_loss    = Q_tank_loss * (1 - T0_K / T_tank_w_K)

            Xc_ou   = np.nansum([X_a_ou_in, X4, X_fan_ou, -X1, -X_a_ou_out])
            Xc_cmp  = np.nansum([X_cmp, X1, -X2])
            Xc_exp  = np.nansum([X3, -X4])
            Xc_tank = np.nansum([X2, X_tank_w_in, X_uv, X_stc_w_final, -X_tank_w_out, -X3, -Xst_tank])
            Xc_mix  = np.nansum([X_tank_w_out, X_mix_sup_w, -X_mix_serv_w])
            Xc_stc  = stc_result.get('Xc_stc', np.nan)
            Xc_tot  = np.nansum([Xc_ou, Xc_cmp, Xc_exp, Xc_tank, Xc_mix, Xc_stc])
            

            E_cmp       = result['E_cmp [W]']
            E_fan_ou    = result['E_fan_ou [W]']
            Q_cond_load = result['Q_cond_load [W]']
            X_ref_cond  = result['X_ref_cond [W]']
            X_cmp       = result['X_cmp [W]']

            E_tot = np.nansum([E_cmp, E_fan_ou, E_uv, E_pump])
            X_tot = np.nansum([E_cmp, E_fan_ou, E_uv, E_pump])

            X_tank_loss = (1 - T0_K / T_tank_w_K) * Q_tank_loss

            step_results['Q_tank_loss [W]'] = Q_tank_loss
            step_results['E_tot [W]'] = E_tot
            
            cop_ref = Q_cond_load / E_cmp
            cop_sys = Q_cond_load / E_tot
            
            step_results.update({
                'X_tot [W]':       X_tot,
                'Xst_tank [W]':    Xst_tank,

                'Xc_tank [W]':     Xc_tank,
                'Xc_tot [W]':      Xc_tot,
                'Xc_ou [W]':       Xc_ou,
                'Xc_cmp [W]':      Xc_cmp,
                'Xc_exp [W]':      Xc_exp,
                'Xc_mix [W]':      Xc_mix,
                'Xc_stc [W]':      Xc_stc,

                'cop_ref [-]':     cop_ref,
                'cop_sys [-]':     cop_sys,
            })

            # Prepare for Next Step
            if n < tN - 1:
                T_tank_w_K_prev = T_tank_w_K
                
                Q_ref_cond = result.get('Q_ref_cond [W]', 0.0)
                E_uv = step_results.get('E_uv [W]', 0.0)
                
                # Consolidate Net Energy Gain (Q_net_gain):
                # 1. Heat Pump Condenser (Q_ref_cond)
                # 2. UV Lamp (E_uv)
                # 3. Pump Heat (Q_pump)
                # 4. Refill Net Gain (Q_refill_net)
                # 5. STC Gain (conditional):
                #    - If tank_circuit: Add (Q_stc_w_out - Q_stc_w_in)
                #    - If mains_preheat: Already included in Q_refill_net (via T_refill_K). DO NOT double count.
                
                Q_stc_net_gain = 0.0
                if use_stc and self.stc_placement == 'tank_circuit':
                    Q_stc_net_gain = Q_stc_w_out - Q_stc_w_in
                
                Q_net_gain = np.nansum([
                    Q_ref_cond, 
                    E_uv, 
                    Q_pump,
                    Q_refill_net,
                    Q_stc_net_gain
                ])
                
                # Update Tank Heat Capacity based on NEW level
                if not self.tank_always_full or (self.tank_always_full and self.prevent_simultaneous_flow):
                    C_tank_actual = self.C_tank * tank_level
                else:
                    C_tank_actual = self.C_tank
                
                # Update Tank Temperature
                # Q_net (for ODE) = Q_net_gain - Q_tank_loss (Wall Loss)
                # Note: Q_use_loss is NOT involved in dT/dt because it cancels out in the derivative 
                # (perfect mixing assumption: outflow temp = internal temp)
                
                T_tank_w_K = update_tank_temperature(
                    T_tank_w_K = T_tank_w_K,
                    Q_tank_in  = Q_net_gain,
                    total_loss = Q_tank_loss, # Only wall loss
                    C_tank     = C_tank_actual,
                    dt         = self.dt
                )

            if result is not None and isinstance(result, dict):
                prev_result = result.copy()
            
            results_data.append(step_results)
            
        results_df = pd.DataFrame(results_data)

        if result_save_csv_path:
            results_df.to_csv(result_save_csv_path, index=False)

        return results_df

