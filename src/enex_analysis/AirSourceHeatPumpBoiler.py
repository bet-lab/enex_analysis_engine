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
        dV_ou_fan_a_design   = 1.5,     # 실외기 설계 풍량 [m3/s] (정풍량)
        dP_ou_fan_design      = 90.0,    # 실외기 설계 정압 [Pa]
        A_cross_ou        = 0.25 ** 2 * np.pi,  # [m2] (정의된 값 사용)
        eta_ou_fan_design = 0.6,     # 실외기 팬 효율 [-]

        # 4. 탱크/제어/부하 파라미터 -----------------------------------
        T_tank_w_upper_bound = 65.0,   # [°C] 저탕조 설정 온도
        T_tank_w_lower_bound = 60.0,   # [°C] 저탕조 하한 온도
        T_mix_w_out          = 40.0,   # [°C] 서비스 급탕 온도
        T_tank_w_in          = 15.0,   # [°C] 급수(상수도) 온도

        hp_capacity   = 15000.0,   # [W] 히터 최대 용량
        dV_mix_w_out_max = 0.0045,    # [m3/s] 최대 급탕 유량

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
        dV_tank_w_in_refill = 0.001,  # [m³/s] 리필 시 상수도 유량 (tank_always_full=False일 때만 사용)
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
        dV_stc_w = 0.001,  # [m³/s] STC 유량
        E_stc_pump = 50.0,  # [W] 펌프 출력
        
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
        self.dV_ou_fan_a_design = dV_ou_fan_a_design
        self.dP_ou_fan_design = dP_ou_fan_design
        self.eta_ou_fan_design = eta_ou_fan_design
        self.A_cross_ou = A_cross_ou
        
        # 팬 설계 전력 계산 (정풍량 기준)
        self.E_ou_fan_design = (self.dV_ou_fan_a_design * self.dP_ou_fan_design) / (self.eta_ou_fan_design)
        
        # VSD Curve 계수 VSD(Variable Speed Drive)
        self.vsd_coeffs_ou = vsd_coeffs_ou
        
        # 팬 파라미터 딕셔너리
        self.fan_params_ou = {
            'fan_design_flow_rate': self.dV_ou_fan_a_design,
            'fan_design_power': self.E_ou_fan_design
        }
        
        # --- 4. 탱크 물리 파라미터 ---
        self.tank_physical = {
            'r0': r0, 'H': H, 'x_shell': x_shell, 'x_ins': x_ins,
            'k_shell': k_shell, 'k_ins': k_ins, 'h_o': h_o,
        }
        self.UA_tank = calc_simple_tank_UA(**self.tank_physical)
        self.V_tank_full = math.pi * r0**2 * H  # [m³] 저탕조 전체 체적
        self.C_tank = c_w * rho_w * self.V_tank_full  # [J/K] 저탕조 열용량 (100% 수위 기준)
        
        self.dV_mix_w_out_max = dV_mix_w_out_max
        self.T_tank_w_upper_bound = T_tank_w_upper_bound
        self.T_tank_w_lower_bound = T_tank_w_lower_bound
        self.T_tank_w_in = T_tank_w_in
        self.T_mix_w_out = T_mix_w_out
        
        self.T_tank_w_in_K = cu.C2K(T_tank_w_in)
        self.T_mix_w_out_K = cu.C2K(T_mix_w_out)
        
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
        self.dV_tank_w_in_refill = dV_tank_w_in_refill
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
        self.dV_stc_w = dV_stc_w
        self.E_stc_pump = E_stc_pump
        
        # --- 10. STC 배치 선택 ---
        self.stc_placement = stc_placement

        # [NEW] 최적화 초기값 저장을 위한 변수
        self.prev_opt_x = None
        
        # [NEW] Flow Rate Synchronization
        self.dV_tank_w_in = 0.0
        self.dV_tank_w_out = 0.0
        self.dV_mix_w_in_sup = 0.0
        self.dV_mix_w_out = 0.0

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
        T_ref_cmp_in_K  = cycle_states['T1_K']  # State 1 온도 [K]
        P_ref_cmp_in    = cycle_states['P1']    # State 1 압력 [Pa]
        h_ref_cmp_in    = cycle_states['h1']    # State 1 엔탈피 [J/kg]
        s_ref_cmp_in    = cycle_states['s1']    # State 1 엔트로피 [J/(kg·K)]
        
        T_ref_cmp_out_K = cycle_states['T2_K']  # State 2 온도 [K]
        P_ref_cmp_out   = cycle_states['P2']    # State 2 압력 [Pa]
        h_ref_cmp_out   = cycle_states['h2']    # State 2 엔탈피 [J/kg]
        s_ref_cmp_out   = cycle_states['s2']    # State 2 엔트로피 [J/(kg·K)]
        
        T_ref_exp_in_K  = cycle_states['T3_K']  # State 3 온도 [K]
        P_ref_exp_in    = cycle_states['P3']    # State 3 압력 [Pa]
        h_ref_exp_in    = cycle_states['h3']    # State 3 엔탈피 [J/kg]
        s_ref_exp_in    = cycle_states['s3']    # State 3 엔트로피 [J/(kg·K)]
        
        T_ref_exp_out_K = cycle_states['T4_K']  # State 4 온도 [K]
        P_ref_exp_out   = cycle_states['P4']    # State 4 압력 [Pa]
        h_ref_exp_out   = cycle_states['h4']    # State 4 엔탈피 [J/kg]
        s_ref_exp_out   = cycle_states['s4']    # State 4 엔트로피 [J/(kg·K)]
        
        rho_ref_cmp_in = cycle_states['rho']
        
        # 포화 온도 추출
        T_ref_evap_sat_K   = cycle_states['T1_star_K']
        T_ref_cond_sat_v_K = cycle_states['T2_star_K']
        T_ref_cond_sat_l_K = cycle_states['T3_star_K']
        
        # 5단계: 냉매 유량 및 성능 데이터 계산 (nan이면 자동으로 nan)
        m_dot_ref  = Q_cond_load / (h_ref_cmp_out - h_ref_exp_in)
        Q_ref_cond = m_dot_ref * (h_ref_cmp_out - h_ref_exp_in)
        Q_ref_evap = m_dot_ref * (h_ref_cmp_in - h_ref_exp_out)
        E_cmp      = m_dot_ref * (h_ref_cmp_out - h_ref_cmp_in)
        cmp_rps    = m_dot_ref / (self.V_disp_cmp * rho_ref_cmp_in)
        
        # 6단계: 실외기 열교환기 성능 계산
        T_a_ou_in_K = T0_K
        T_a_ou_in = cu.K2C(T_a_ou_in_K)
        
        HX_perf_ou_dict = calc_HX_perf_for_target_heat(
            Q_ref_target  = Q_ref_evap if is_active else 0.0,
            T_a_ou_in_C   = T0,
            T1_star_K     = T_ref_evap_sat_K,
            T3_star_K     = T_ref_cond_sat_l_K,
            A_cross       = self.A_cross_ou,
            UA_design     = self.UA_evap_design,
            dV_fan_design = self.dV_ou_fan_a_design,
            is_active     = is_active,
        )
        
        if HX_perf_ou_dict.get('converged', True) == False:
            return None
        
        # 수렴 성공 시 값 추출 (nan이면 자동으로 nan)
        dV_ou_a_fan   = HX_perf_ou_dict['dV_fan']
        v_ou_a_fan    = dV_ou_a_fan / self.A_cross_ou
        UA_evap     = HX_perf_ou_dict['UA']
        T_a_ou_mid  = HX_perf_ou_dict['T_a_ou_mid']
        Q_ou_a    = HX_perf_ou_dict['Q_ou_air']
        
        # 8단계: 팬 전력 계산 (nan이면 자동으로 nan)
        E_ou_fan = calc_fan_power_from_dV_fan(
            dV_fan     = dV_ou_a_fan,
            fan_params = self.fan_params_ou,
            vsd_coeffs = self.vsd_coeffs_ou,
            is_active  = is_active,
        )
        
        # T_a_ou_out 계산 (nan이면 자동으로 nan)
        T_a_ou_out = T_a_ou_mid + E_ou_fan / (c_a * rho_a * dV_ou_a_fan)
        T_a_ou_out_K = cu.C2K(T_a_ou_out)
        fan_eff = self.eta_ou_fan_design * dV_ou_a_fan / E_ou_fan * 100
        
        # 7단계: LMTD 기반 열량 계산
        # 7단계: LMTD 대신 (저탕조 온수 온도 - 응축 온도) 차이 기반 열량 계산
        # LMTD_cond = calc_lmtd_fluid_and_constant_temp(...) -> 제거됨
        # dT_ref_cond = T_cond_sat_K - T_tank_w_K (Optimization Variable)
        
        UA_cond     = self.UA_cond_design  # constant assumption 
        
        # [MODIFIED] Q = UA * dT (Physical simplification per user request)
        # dT_ref_cond는 최적화 변수로, T_cond_sat - T_tank_w 차이임.
        Q_tank_w    = UA_cond * dT_ref_cond
        
        # 탱크 및 믹싱밸브 변수
        dV_tank_w_out   = self.dV_tank_w_out
        dV_tank_w_in    = self.dV_tank_w_in
        dV_mix_w_in_sup = self.dV_mix_w_in_sup
        dV_mix_w_out    = self.dV_mix_w_out
        
        # 실제 서비스 온도 계산 (믹싱 밸브)
        # dV_mix_w_out == 0이면 nan으로 설정
        if dV_mix_w_out == 0:
            T_serv_w_actual = np.nan
            T_serv_w_actual_K = np.nan
        else:
            # alp 계산: 저탕조 온수 비율
            den = max(1e-6, T_tank_w_K - self.T_tank_w_in_K)
            alp = min(1.0, max(0.0, (self.T_mix_w_out_K - self.T_tank_w_in_K) / den))
            
            # 실제 서비스 온도 계산
            if alp >= 1.0:
                # 저탕조 온수를 그대로 사용하는 경우 (T_tank_w < T_serv_w 목표값)
                T_serv_w_actual = T_tank_w
                T_serv_w_actual_K = T_tank_w_K
            else:
                # [Case 2] 믹싱 운전: 저탕조 온도 >= 목표 급탕 온도
                # 3-way 믹싱 밸브를 조절하여 고온의 저탕조 물(T_tank_w)과 저온의 상수도(T_tank_w_in)를 혼합합니다.
                # - alp: 저탕조 물의 혼합 비율 (0 <= alp <= 1)
                # - (1-alp): 상수도 물의 혼합 비율
                # 에너지 보존 법칙(Enthalpy Mixing)에 따라 최종 급탕 온도를 물리적으로 재계산합니다.
                # (이론적으로 이 값은 목표 급탕 온도(T_mix_w_out)와 일치해야 합니다.)
                T_serv_w_actual_K = alp * T_tank_w_K + (1 - alp) * self.T_tank_w_in_K
                T_serv_w_actual = cu.K2C(T_serv_w_actual_K)
        
        Q_tank_w_in  = calc_energy_flow(G=c_w * rho_w * dV_tank_w_in, T=self.T_tank_w_in_K, T0=T0_K)
        Q_tank_w_out = calc_energy_flow(G=c_w * rho_w * dV_tank_w_out, T=T_tank_w_K, T0=T0_K)
        Q_mix_w_in_sup = calc_energy_flow(G=c_w * rho_w * dV_mix_w_in_sup, T=self.T_tank_w_in_K, T0=T0_K)
        Q_mix_w_out  = calc_energy_flow(G=c_w * rho_w * dV_mix_w_out, T=T_serv_w_actual_K, T0=T0_K)
        
        # 10단계: 최종 결과 딕셔너리 생성
        # 포화 온도 추출 (OFF 상태에서는 nan)
        T_ref_evap_sat_K_result   = cycle_states.get('T1_star_K', T_ref_evap_sat_K) if isinstance(cycle_states, dict) else T_ref_evap_sat_K
        T_ref_cond_sat_v_K_result = cycle_states.get('T2_star_K', T_ref_cond_sat_v_K) if isinstance(cycle_states, dict) else T_ref_cond_sat_v_K
        T_ref_cond_sat_l_K_result = cycle_states.get('T3_star_K', T_ref_cond_sat_l_K) if isinstance(cycle_states, dict) else T_ref_cond_sat_l_K
        
        result = {
            'hp_is_on': (Q_cond_load > 0),
            'converged': True,

            # === [온도: °C] =======================================
            # [NEW] Saturation Points
            'T_ref_evap_sat [°C]': cu.K2C(T_ref_evap_sat_K_result),
            'T_ref_cond_sat_v [°C]': cu.K2C(T_ref_cond_sat_v_K_result),
            'T_ref_cond_sat_l [°C]': cu.K2C(T_ref_cond_sat_l_K_result),
            
            # [Updated] Actual Points
            'T_a_ou_in [°C]': T_a_ou_in,
            'T_a_ou_out [°C]': T_a_ou_out,
            'T_ref_cmp_in [°C]': cu.K2C(T_ref_cmp_in_K),
            'T_ref_cmp_out [°C]': cu.K2C(T_ref_cmp_out_K),
            'T_ref_exp_in [°C]': cu.K2C(T_ref_exp_in_K),
            'T_ref_exp_out [°C]': cu.K2C(T_ref_exp_out_K),
            'T_tank_w [°C]': T_tank_w,
            'T_tank_w_in [°C]': self.T_tank_w_in,
            'T_mix_w_out [°C]': T_serv_w_actual,
            'T0 [°C]': T0,

            # === [체적유량: m3/s] ==================================
            'dV_ou_a_fan [m3/s]': dV_ou_a_fan,
            'v_ou_a_fan [m/s]': v_ou_a_fan, 
            'dV_mix_w_out [m3/s]': dV_mix_w_out if dV_mix_w_out > 0 else np.nan,
            'dV_tank_w_in [m3/s]': dV_tank_w_in if dV_tank_w_in > 0 else np.nan,
            'dV_mix_w_in_sup [m3/s]': dV_mix_w_in_sup if dV_mix_w_in_sup > 0 else np.nan,

            # === [압력: Pa] ========================================
            'P_ref_cmp_in [Pa]': P_ref_cmp_in,
            'P_ref_cmp_out [Pa]': P_ref_cmp_out,
            'P_ref_exp_in [Pa]': P_ref_exp_in,
            'P_ref_exp_out [Pa]': P_ref_exp_out,
            'P_ref_evap_sat [Pa]': cycle_states['P1'] if is_active else np.nan,
            'P_ref_cond_sat_v [Pa]': cycle_states['P2'] if is_active else np.nan,
            'P_ref_cond_sat_l [Pa]': cycle_states['P3'] if is_active else np.nan,
            'dP_ou_fan_static [Pa]': self.dP_ou_fan_design - 1/2 * rho_a * v_ou_a_fan**2,
            'dP_ou_fan_dynamic [Pa]': 1/2 * rho_a * v_ou_a_fan**2,

            # === [질량유량: kg/s] ==================================
            'm_dot_ref [kg/s]': m_dot_ref,

            # === [rpm] =============================================
            'cmp_rpm [rpm]': cmp_rps * 60,

            # === [엔탈피: J/kg] ====================================
            'h_ref_cmp_in [J/kg]': h_ref_cmp_in,
            'h_ref_cmp_out [J/kg]': h_ref_cmp_out,
            'h_ref_exp_in [J/kg]': h_ref_exp_in,
            'h_ref_exp_out [J/kg]': h_ref_exp_out,
            'h_ref_evap_sat [J/kg]': cycle_states.get('h1_star', np.nan) if is_active else np.nan,
            'h_ref_cond_sat_v [J/kg]': cycle_states.get('h2_star', np.nan) if is_active else np.nan,
            'h_ref_cond_sat_l [J/kg]': cycle_states.get('h3_star', np.nan) if is_active else np.nan,

            # === [에너지: W] =======================================
            # ---- 실외측(공기, 증발기)
            'E_ou_fan [W]': E_ou_fan,
    
            # ---- 증발기
            'Q_ref_evap [W]': Q_ref_evap,
            'Q_ou_a [W]': Q_ou_a,
    
            # ---- 압축기
            'E_cmp [W]': E_cmp,
    
            # ---- 응축기/저탕조
            'Q_cond_load [W]': Q_cond_load,
            'Q_ref_cond [W]': Q_ref_cond,
            'Q_tank_w [W]': Q_tank_w,
    
            # ---- 탱크
            'Q_tank_w_in [W]': Q_tank_w_in,
            'Q_tank_w_out [W]': Q_tank_w_out,
    
            # ---- 믹싱 밸브 & 온수 공급
            'Q_mix_w_in_sup [W]': Q_mix_w_in_sup,
            'Q_mix_w_out [W]': Q_mix_w_out,
    
            # ---- 총괄(총입력)
            'E_tot [W]': E_cmp + E_ou_fan,

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
                
                if ("Q_ou_a [W]" not in perf or "Q_ref_evap [W]" not in perf or
                    np.isnan(perf["Q_ou_a [W]"]) or np.isnan(perf["Q_ref_evap [W]"])):
                    return -1e6
                
                # 제약 조건: Q_ou_a - Q_ref_evap*(1-tolerance) >= 0
                return perf["Q_ou_a [W]"] - perf['Q_ref_evap [W]']
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
                
                if ("Q_ou_a [W]" not in perf or "Q_ref_evap [W]" not in perf or
                    np.isnan(perf["Q_ou_a [W]"]) or np.isnan(perf["Q_ref_evap [W]"])):
                    return -1e6
                
                # 제약 조건: Q_ref_evap*(1+tolerance) - Q_ou_a >= 0
                return perf['Q_ref_evap [W]'] * (1 + tolerance) - perf["Q_ou_a [W]"]
            except Exception as e:
                return -1e6
        
        const_funcs = [
            {'type': 'ineq', 'fun': _cond_LMTD_constraint_low},   # Q_tank_w - Q_cond_load >= 0
            {'type': 'ineq', 'fun': _cond_LMTD_constraint_high},  # Q_cond_load*(1+tolerance) - Q_tank_w >= 0
            {'type': 'ineq', 'fun': _evap_LMTD_constraint_low},   # Q_ou_a - Q_ref_evap*(1-tolerance) >= 0
            {'type': 'ineq', 'fun': _evap_LMTD_constraint_high},  # Q_ref_evap*(1+tolerance) - Q_ou_a >= 0
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
        den = max(1e-6, T_tank_w_K - self.T_tank_w_in_K)
        alp = min(1.0, max(0.0, self.T_mix_w_out_K - self.T_tank_w_in_K) / den)
        
        self.dV_mix_w_out = dV_w_serv
        self.dV_tank_w_out = alp * dV_w_serv
        self.dV_mix_w_in_sup = (1 - alp) * dV_w_serv
        
        # Q_cond_load 계산
        if Q_cond_load is None:
            # dV_w_serv가 주어진 경우: 열 손실 계산하여 Q_cond_load 결정
            Q_use_loss = c_w * rho_w * self.dV_tank_w_out * (T_tank_w_K - self.T_tank_w_in_K)
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
                        'E_ou_fan [W]': 0.0,
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
        
        self.dV_mix_w_out = 0.0 
        self.dV_tank_w_out = 0.0 
        self.dV_mix_w_in_sup = 0.0 
        
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
            T_tank_w_in_K = self.T_tank_w_in_K  # Default refill source is mains water
            T_stc_w_out_K = np.nan
            Q_stc_w_out = 0.0
            Q_stc_w_in = 0.0
            E_stc_pump_pwr = 0.0  # Default: no pump power (only active when STC is running)
            Q_pump = 0.0  # Default: no pump heat (only active when STC is running)
            stc_result = {}
            
            # Refill Volume Default
            V_refill = 0.0
            self.dV_tank_w_in = 0.0
            

            # =================================================================
            # 2. DEMAND & TANK OUTFLOW (MASS BALANCE 1)
            # =================================================================
            # Calculate demand fractions
            den = max(1e-6, T_tank_w_K - self.T_tank_w_in_K)
            alp = min(1.0, max(0.0, self.T_mix_w_out_K - self.T_tank_w_in_K) / den)

            self.dV_mix_w_out = self.w_use_frac[n] * self.dV_mix_w_out_max 
            self.dV_tank_w_out = alp * self.dV_mix_w_out
            tank_outlet_exist = self.dV_tank_w_out > 0
            self.dV_mix_w_in_sup = (1 - alp) * self.dV_mix_w_out

            # Tank Loss Calculation (Based on Pre-mix Temperature)
            # (Note: Original code calc Q_tank_loss at start, used unmixed T_tank_w_K)
            Q_tank_loss = self.UA_tank * (T_tank_w_K - T0_K)
            step_results['Q_tank_loss [W]'] = Q_tank_loss

            # Update Tank Level (Outflow)
            # [MODIFIED] always_full allows level drop if prevent_simultaneous_flow is ON
            if not self.tank_always_full or (self.tank_always_full and self.prevent_simultaneous_flow):
                tank_level -= (self.dV_tank_w_out * dt_s) / self.V_tank_full
                tank_level = max(0.0, tank_level)

            # =================================================================
            # 3. REFILL LOGIC (MASS BALANCE 2)
            # =================================================================

            # [CASE 1] Exclusive Flow Mode
            if self.tank_always_full and self.prevent_simultaneous_flow:
                # [CASE 1-1] Tank outlet exists
                if tank_outlet_exist:
                    self.dV_tank_w_in = 0.0
                    is_refilling = False 
                # [CASE 1-2] Tank outlet does not exist
                elif not tank_outlet_exist:
                    if tank_level < 1.0:
                        req_vol = (1.0 - tank_level) * self.V_tank_full
                        tank_will_overflow = self.dV_tank_w_in_refill * dt_s > req_vol 
                        V_refill = self.dV_tank_w_in_refill * dt_s if not tank_will_overflow else 0.0
                        self.dV_tank_w_in = V_refill / dt_s
                    else:
                        self.dV_tank_w_in = 0.0

            # [CASE 2] Classic Always Full
            elif self.tank_always_full:
                self.dV_tank_w_in = self.dV_tank_w_out # Simultaneous flow

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
                    tank_will_overflow = self.dV_tank_w_in_refill * dt_s > req_vol 
                    V_refill = self.dV_tank_w_in_refill * dt_s if not tank_will_overflow else 0.0
                    self.dV_tank_w_in = V_refill / dt_s
                    
                    # Refill Stop Condition
                    if (tank_level + V_refill / self.V_tank_full) >= target_upper - 1e-6:
                        is_refilling = False
                else:
                    self.dV_tank_w_in = 0.0

            # =================================================================
            # 4. SOLAR THERMAL COLLECTOR (STC) & SOURCE TEMP
            # =================================================================
            if use_stc:
                # 4-1. Tank Circuit Mode
                # Physical behavior: Water circulates from tank -> STC -> tank
                # - STC inlet temp = tank temp (T_tank_w_K)
                # - STC heats water using solar radiation
                # - Heated water returns to tank, increasing tank energy
                # - Flow rate is independent of tank refill (dV_stc_w)
                if self.stc_placement == 'tank_circuit':
                    # Probing: Check if STC can provide useful heating
                    stc_result_test = calc_stc_performance(
                        I_DN_stc=I_DN_schedule[n], I_dH_stc=I_dH_schedule[n],
                        T_stc_w_in_K=T_tank_w_K, T0_K=T0_K,
                        A_stc_pipe=self.A_stc_pipe, alpha_stc=self.alpha_stc,
                        h_o_stc=self.h_o_stc, h_r_stc=self.h_r_stc,
                        k_ins_stc=self.k_ins_stc, x_air_stc=self.x_air_stc,
                        x_ins_stc=self.x_ins_stc, dV_stc=self.dV_stc_w,
                        E_pump=self.E_stc_pump,
                        is_active=True, 
                    )
                    
                    # STC activates only if: (1) within preheat time window, (2) STC outlet > tank temp
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
                            x_ins_stc=self.x_ins_stc, dV_stc=self.dV_stc_w,
                            E_pump=self.E_stc_pump,
                            is_active=False
                        )

                    # Extract STC results
                    # Note: T_stc_w_out_K excludes pump heat; T_stc_w_final_K includes pump heat
                    # Q_stc_w_out is calculated from T_stc_w_out_K (solar heat only)
                    # Pump heat (E_stc_pump_pwr) is added separately in energy balance
                    T_stc_w_out_K = stc_result['T_stc_w_out_K']
                    T_stc_w_final_K = stc_result.get('T_stc_w_final_K', T_stc_w_out_K)
                    Q_stc_w_out   = stc_result.get('Q_stc_w_out', 0.0)
                    Q_stc_w_in    = stc_result.get('Q_stc_w_in', 0.0)
                    # Set pump power: active only when STC is actually heating
                    E_stc_pump_pwr = self.E_stc_pump if stc_active else 0.0
                
                # 4-2. Mains Preheat Mode
                # Physical behavior: Mains water -> STC -> tank
                # - STC preheats cold mains water before it enters tank
                # - STC inlet temp = mains temp (T_tank_w_in_K)
                # - Preheated water enters tank at T_stc_w_out_K
                # - STC energy gain is included in Q_refill_net (via T_refill_K)
                # - Flow rate equals tank refill rate (dV_tank_w_in)
                elif self.stc_placement == 'mains_preheat':
                    # Only active if: (1) within preheat time window, (2) refill flow exists
                    if preheat_on and self.dV_tank_w_in > 0:
                        stc_result_test = calc_stc_performance(
                            I_DN_stc=I_DN_schedule[n], I_dH_stc=I_dH_schedule[n],
                            T_stc_w_in_K=self.T_tank_w_in_K, T0_K=T0_K,
                            A_stc_pipe=self.A_stc_pipe, alpha_stc=self.alpha_stc,
                            h_o_stc=self.h_o_stc, h_r_stc=self.h_r_stc,
                            k_ins_stc=self.k_ins_stc, x_air_stc=self.x_air_stc,
                            x_ins_stc=self.x_ins_stc, dV_stc=self.dV_tank_w_in,
                            E_pump=self.E_stc_pump,
                            is_active=True,
                        )
                        
                        # STC activates only if outlet temp > inlet (mains) temp
                        if stc_result_test['T_stc_w_out_K'] > self.T_tank_w_in_K:
                            stc_active = True
                            stc_result = stc_result_test
                        else:
                            stc_active = False
                            stc_result = calc_stc_performance(
                                I_DN_stc=I_DN_schedule[n], I_dH_stc=I_dH_schedule[n],
                                T_stc_w_in_K=self.T_tank_w_in_K, T0_K=T0_K,
                                A_stc_pipe=self.A_stc_pipe, alpha_stc=self.alpha_stc,
                                h_o_stc=self.h_o_stc, h_r_stc=self.h_r_stc,
                                k_ins_stc=self.k_ins_stc, x_air_stc=self.x_air_stc,
                                x_ins_stc=self.x_ins_stc, dV_stc=self.dV_tank_w_in,
                                E_pump=self.E_stc_pump,
                                is_active=False
                            )
                    else:
                        stc_active = False
                        stc_result = calc_stc_performance(
                            I_DN_stc=I_DN_schedule[n], I_dH_stc=I_dH_schedule[n],
                            T_stc_w_in_K=self.T_tank_w_in_K, T0_K=T0_K,
                            A_stc_pipe=self.A_stc_pipe, alpha_stc=self.alpha_stc,
                            h_o_stc=self.h_o_stc, h_r_stc=self.h_r_stc,
                            k_ins_stc=self.k_ins_stc, x_air_stc=self.x_air_stc,
                            x_ins_stc=self.x_ins_stc, dV_stc=1.0, 
                            E_pump=self.E_stc_pump,
                            is_active=False
                        )

                    T_stc_w_out_K = stc_result['T_stc_w_out_K']
                    Q_stc_w_out = stc_result.get('Q_stc_w_out', 0.0)
                    Q_stc_w_in = stc_result.get('Q_stc_w_in', 0.0)
                    
                    # Set refill temp and pump power based on STC activation
                    # Note: T_stc_w_out_K excludes pump heat; pump heat added separately in energy balance
                    if stc_active:
                        T_tank_w_in_K = T_stc_w_out_K  # Will be used as T_refill_K
                        E_stc_pump_pwr = self.E_stc_pump
                    else:
                        T_tank_w_in_K = self.T_tank_w_in_K
                        E_stc_pump_pwr = 0.0

            # =================================================================
            # 5. REFILL MASS & ENERGY BALANCE
            # =================================================================
            # 5-1. Mass Balance Update (Refill)
            # Note: Outflow was already deducted in Step 2. Now add Inflow.
            V_new_fill = self.dV_tank_w_in * dt_s
            tank_level += V_new_fill / self.V_tank_full
            # Ensure strictly within bounds (prevent C_tank -> 0)
            tank_level = max(0.001, min(1.0, tank_level)) 
            
            # 5-2. Determine Refill Temperature (T_refill_K)
            # For mains_preheat mode: Use STC outlet temp if STC is active
            # For tank_circuit mode: STC does not affect refill temp (it circulates tank water)
            # Note: T_stc_w_out_K excludes pump heat; pump heat is added separately as E_stc_pump_pwr
            if use_stc and self.stc_placement == 'mains_preheat' and stc_active:
                T_refill_K = T_stc_w_out_K  # Preheated mains water enters tank
            else:
                T_refill_K = self.T_tank_w_in_K  # Cold mains water enters tank

            # Update T_tank_w_in_K for result logging/Exergy (Enthalpy basis)
            T_tank_w_in_K = T_refill_K

            # 5-3. Energy Flux (Refill)
            # Q_refill_net = m_dot * Cp * (T_in - T_tank)
            # This accounts for the energy brought in by the refill water relative to current tank temp.
            # If mains_preheat is active, T_refill_K is boosted, so this Term INCLUDES STC gain.
            Q_refill_net = c_w * rho_w * self.dV_tank_w_in * (T_refill_K - T_tank_w_K)
            
            # 5-4. Use Loss & Enthalpy In (For Exergy/Reports ONLY, not for T update)
            # Q_use_loss: Energy leaving the system boundary relative to T0
            Q_use_loss = c_w * rho_w * self.dV_tank_w_out * (T_tank_w_K - T0_K)
            # Q_tank_w_in: Total enthalpy entering the system boundary relative to T0
            Q_tank_w_in = c_w * rho_w * self.dV_tank_w_in * (T_tank_w_in_K - T0_K)

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
                    'E_stc_pump [W]': E_stc_pump_pwr,
                })
                if self.stc_placement == 'tank_circuit':
                    step_results['T_stc_w_final [°C]'] = cu.K2C(T_stc_w_final_K) if 'T_stc_w_final_K' in locals() else np.nan
            else:
                step_results['stc_active [-]'] = False

            # Prepare for Next Step
            if n < tN - 1:
                T_tank_w_K_prev = T_tank_w_K
                
                Q_ref_cond = result.get('Q_ref_cond [W]', 0.0)
                E_uv = step_results.get('E_uv [W]', 0.0)
                
                # Consolidate Net Energy Gain (Q_net_gain) for tank temperature update:
                # 1. Heat Pump Condenser (Q_ref_cond): HP heating
                # 2. UV Lamp (E_uv): UV lamp heat
                # 3. STC Pump Heat (E_stc_pump_pwr): Pump power converted to heat (added separately)
                #    Note: Q_stc_w_out is calculated from T_stc_w_out_K which excludes pump heat
                # 4. Refill Net Gain (Q_refill_net): Energy from refill water
                #    - For mains_preheat: Includes STC solar gain (via T_refill_K = T_stc_w_out_K)
                #    - Pump heat for mains_preheat is included in E_stc_pump_pwr
                # 5. STC Net Gain (Q_stc_net_gain): Only for tank_circuit mode
                #    - Q_stc_w_out - Q_stc_w_in: Net solar energy added to tank via circulation
                #    - Does NOT include pump heat (handled separately as E_stc_pump_pwr)
                #    - For mains_preheat: STC gain already in Q_refill_net, DO NOT double count
                
                Q_stc_net_gain = 0.0
                if use_stc and self.stc_placement == 'tank_circuit':
                    # Tank circuit: STC circulates tank water, net solar gain is (out - in)
                    Q_stc_net_gain = Q_stc_w_out - Q_stc_w_in
                
                Q_net_gain = np.nansum([
                    Q_ref_cond, 
                    E_uv, 
                    E_stc_pump_pwr,  # Pump heat (for both modes when STC active)
                    Q_refill_net,     # Refill energy (includes STC gain for mains_preheat)
                    Q_stc_net_gain    # STC circulation gain (tank_circuit only)
                ])
                
                C_tank_actual = self.C_tank * tank_level
                T_tank_w_K = update_tank_temperature(
                    T_tank_w_K = T_tank_w_K,
                    Q_tank_in  = Q_net_gain,
                    total_loss = Q_tank_loss,
                    C_tank     = C_tank_actual,
                    dt         = self.dt
                )

            if result is not None and isinstance(result, dict):
                prev_result = result.copy()
            
            results_data.append(step_results)
            
        results_df = pd.DataFrame(results_data)
        results_df = self.postprocess_exergy(results_df)

        if result_save_csv_path:
            results_df.to_csv(result_save_csv_path, index=False)

        return results_df

    def postprocess_exergy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        시뮬레이션 결과 DataFrame에 엑서지 변수를 계산하여 추가합니다.

        Parameters
        ----------
        df : pd.DataFrame
            analyze_dynamic() 결과 DataFrame

        Returns
        -------
        pd.DataFrame
            엑서지 변수가 추가된 DataFrame
        """
        df = df.copy()
        P0 = 101325

        def C2K(T_C):
            return T_C + 273.15

        T0_K = C2K(df['T0 [°C]'])
        T_tank_K = C2K(df['T_tank_w [°C]'])

        # 1. 냉매 엔트로피/엑서지
        state_map = {
            1: ('cmp_in', 'P_ref_cmp_in [Pa]', 'h_ref_cmp_in [J/kg]'),
            2: ('cmp_out', 'P_ref_cmp_out [Pa]', 'h_ref_cmp_out [J/kg]'),
            3: ('exp_in', 'P_ref_exp_in [Pa]', 'h_ref_exp_in [J/kg]'),
            4: ('exp_out', 'P_ref_exp_out [Pa]', 'h_ref_exp_out [J/kg]'),
        }

        for idx in df.index:
            t0_k = T0_K.iloc[idx]
            try:
                h0 = CP.PropsSI('H', 'T', t0_k, 'P', P0, self.ref)
                s0 = CP.PropsSI('S', 'T', t0_k, 'P', P0, self.ref)
            except:
                h0, s0 = np.nan, np.nan

            m_dot = df.loc[idx, 'm_dot_ref [kg/s]'] if 'm_dot_ref [kg/s]' in df.columns else np.nan

            for num, (name, P_col, h_col) in state_map.items():
                if P_col in df.columns and h_col in df.columns:
                    P = df.loc[idx, P_col]
                    h = df.loc[idx, h_col]
                    try:
                        if not np.isnan(P) and not np.isnan(h):
                            s_val = CP.PropsSI('S', 'P', P, 'H', h, self.ref)
                            x_val = (h - h0) - t0_k * (s_val - s0)
                            X_val = m_dot * x_val if not np.isnan(m_dot) else np.nan
                            df.loc[idx, f's_ref_{name} [J/(kg·K)]'] = s_val
                            df.loc[idx, f'x_ref_{name} [J/kg]'] = x_val
                            df.loc[idx, f'X_ref_{name} [W]'] = X_val
                    except:
                        pass

        # 2. 전력 = 엑서지
        df['X_cmp [W]'] = df['E_cmp [W]']
        if 'E_ou_fan [W]' in df.columns:
            df['X_ou_fan [W]'] = df['E_ou_fan [W]']

        # 3. 공기 엑서지
        if 'dV_ou_a_fan [m3/s]' in df.columns and 'T_a_ou_in [°C]' in df.columns:
            G_a = c_a * rho_a * df['dV_ou_a_fan [m3/s]']
            Tin = C2K(df['T_a_ou_in [°C]'])
            Tout = C2K(df['T_a_ou_out [°C]']) if 'T_a_ou_out [°C]' in df.columns else Tin
            df['X_a_ou_in [W]'] = calc_exergy_flow(G_a, Tin, T0_K)
            df['X_a_ou_out [W]'] = calc_exergy_flow(G_a, Tout, T0_K)

        # 4. 응축기 엑서지
        if 'Q_ref_cond [W]' in df.columns:
            df['X_ref_cond [W]'] = df['Q_ref_cond [W]'] * (1 - T0_K / T_tank_K)

        # 5. 탱크 물 엑서지
        if 'dV_tank_w_in [m3/s]' in df.columns:
            G_in = c_w * rho_w * df['dV_tank_w_in [m3/s]'].fillna(0)
            # self.T_tank_w_in이 scalar (초기 급수온도)라고 가정
            df['X_w_tank_in [W]'] = calc_exergy_flow(G_in, C2K(self.T_tank_w_in), T0_K)

        if 'E_uv [W]' in df.columns:
            df['X_uv [W]'] = df['E_uv [W]']

        if 'Q_tank_loss [W]' in df.columns:
            df['X_tank_loss [W]'] = df['Q_tank_loss [W]'] * (1 - T0_K / T_tank_K)

        # 6. 엑서지 축적
        tank_level = df['tank_level [-]'] if 'tank_level [-]' in df.columns else 1.0
        C_tank_actual = self.C_tank * tank_level
        T_tank_K_prev = T_tank_K.shift(1)
        # 벡터화된 연산 적용
        df['Xst_tank [W]'] = (1 - T0_K / T_tank_K) * C_tank_actual * (T_tank_K - T_tank_K_prev) / self.dt
        df.loc[df.index[0], 'Xst_tank [W]'] = 0.0

        # 7. 총 엑서지
        E_fan = df['E_ou_fan [W]'] if 'E_ou_fan [W]' in df.columns else 0
        E_pump = df['E_stc_pump [W]'] if 'E_stc_pump [W]' in df.columns else 0
        X_uv = df['X_uv [W]'] if 'X_uv [W]' in df.columns else 0
        df['X_tot [W]'] = df['E_cmp [W]'] + E_fan + X_uv + E_pump

        # 8. 엑서지 소비
        if all(c in df.columns for c in ['X_cmp [W]', 'X_ref_cmp_in [W]', 'X_ref_cmp_out [W]']):
            df['Xc_cmp [W]'] = df['X_cmp [W]'] + df['X_ref_cmp_in [W]'] - df['X_ref_cmp_out [W]']

        if all(c in df.columns for c in ['X_ref_exp_in [W]', 'X_ref_exp_out [W]']):
            df['Xc_exp [W]'] = df['X_ref_exp_in [W]'] - df['X_ref_exp_out [W]']

        # 9. COP
        if 'Q_cond_load [W]' in df.columns:
            df['cop_ref [-]'] = df['Q_cond_load [W]'] / df['E_cmp [W]'].replace(0, np.nan)
            df['cop_sys [-]'] = df['Q_cond_load [W]'] / df['E_tot [W]'].replace(0, np.nan)

        return df
