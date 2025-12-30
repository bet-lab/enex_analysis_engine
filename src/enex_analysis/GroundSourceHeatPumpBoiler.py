#%%
import numpy as np
import math
from . import calc_util as cu
from dataclasses import dataclass
from scipy.optimize import minimize
from .enex_functions import (
    compute_refrigerant_thermodynamic_states,
    create_lmtd_constraints,
    find_ref_loop_optimal_operation,
    plot_cycle_diagrams
)
import CoolProp.CoolProp as CP
from tqdm import tqdm
import pandas as pd

# Import constants from constants.py
from .constants import (
    c_a, rho_a, k_a, c_w, rho_w, mu_w, k_w, sigma, k_D, k_d, ex_eff_NG, SP
)

# Import functions from enex_functions.py
from .enex_functions import (
    calc_lmtd_fluid_and_constant_temp,
    calc_simple_tank_UA,
    G_FLS,
    _build_schedule_ratios,
)

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
        
    def _calculate_gshpb_next_step(self, optimization_vars, T_tank_w, Q_cond_load, **kwargs):
        """
        지열원 히트펌프 보일러(GSHPB)의 사이클 성능을 계산하는 메서드.
        
        이 메서드는 최적화 변수(optimization_vars)를 받아 히트펌프 사이클 성능을 계산합니다.
        최적화 과정에서 반복적으로 호출되어 목적 함수와 제약 조건을 평가하는 데 사용됩니다.
        
        주요 작업:
        1. 최적화 변수 언패킹 (온도차 추출)
        2. 증발 및 응축 온도 계산
        3. 공통 사이클 상태 계산 (cycle_performance.py 사용)
        4. 냉매 유량 및 성능 데이터 계산
        5. 클래스별 결과 포맷팅
        
        호출 관계:
        - 호출자: find_ref_loop_optimal_operation (cycle_performance.py)
        - 호출 함수: 
            - compute_refrigerant_thermodynamic_states (cycle_performance.py)
            - _format_gshpb_results_dict (본 클래스)
        
        데이터 흐름:
        ──────────────────────────────────────────────────────────────────────────
        [optimization_vars, T_tank_w, Q_cond_load]
            ↓
        증발/응축 온도 계산 (T_evap_K, T_cond_K)
            ↓
        compute_refrigerant_thermodynamic_states
            ↓ [State 1-4 물성치]
        냉매 유량 계산 (m_dot_ref)
            ↓
        성능 데이터 계산 (Q_ref_cond, Q_ref_evap, E_cmp)
            ↓
        _format_gshpb_results_dict
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
            
            **kwargs: 추가 파라미터 (현재 사용되지 않음)
        
        Returns:
            dict: 사이클 성능 결과 (포맷팅된 딕셔너리)
                _format_gshpb_results_dict의 반환값과 동일
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
        T_b_f_in_K = self.T_b_f_in_K          # 지중 유체 입구 온도 [K] (이전 타임스텝 값 사용)
        
        # 증발 온도 계산: 지중 유체 입구 온도에서 냉매-열교환기 온도차를 뺌
        T_evap_K = T_b_f_in_K - dT_ref_HX     # 증발 온도 [K]
        
        # 응축 온도 계산: 저탕조 온도에 냉매-저탕조 온도차를 더함
        T_cond_K = T_tank_w_K + dT_ref_cond   # 응축 온도 [K]
        
        # ============================================================
        # 3단계: 공통 사이클 상태 계산
        # ============================================================
        # cycle_performance.py의 공통 함수 사용
        cycle_states = compute_refrigerant_thermodynamic_states(
            T_evap_K=T_evap_K,
            T_cond_K=T_cond_K,
            refrigerant=self.ref,
            eta_cmp_isen=self.eta_cmp_isen,
            T0_K=self.T0_K,
            P0=101325
        )
        
        # State 1의 밀도 (냉매 유량 계산에 사용)
        rho_ref_cmp_in = cycle_states['rho']
        
        # ============================================================
        # 4단계: 냉매 유량 및 성능 데이터 계산
        # ============================================================
        # State 1-4의 엔탈피 추출
        h1 = cycle_states['h1']  # State 1 엔탈피 [J/kg]
        h2 = cycle_states['h2']  # State 2 엔탈피 [J/kg]
        h3 = cycle_states['h3']  # State 3 엔탈피 [J/kg]
        h4 = cycle_states['h4']  # State 4 엔탈피 [J/kg]
        
        # 계산 불가능한 경우 체크 (h3 == h2인 경우 0으로 나누기 방지)
        if (h3 - h2) == 0:
            return None
        
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
        
        # 성능 데이터 딕셔너리 생성
        performance_data = {
            'Q_ref_cond': Q_ref_cond,
            'Q_ref_evap': Q_ref_evap,
            'E_cmp': E_cmp,
            'm_dot_ref': m_dot_ref,
            'cmp_rps': cmp_rps,
            'cmp_rpm': cmp_rps * 60,  # 회전수 [rpm]
        }
        
        # ============================================================
        # 5단계: 클래스별 결과 포맷팅
        # ============================================================
        # GSHPB 클래스에 특화된 결과 딕셔너리로 포맷팅
        return self._format_gshpb_results_dict(cycle_states, performance_data, T_tank_w, Q_cond_load)
    
    def _format_gshpb_results_dict(self, cycle_states, performance_data, 
                                         T_tank_w, Q_cond_load):
        """
        GroundSourceHeatPumpBoiler 클래스별 사이클 성능 결과 포맷팅 함수.
        
        이 함수는 공통 사이클 계산 결과(cycle_states, performance_data)를 받아
        GSHPB 클래스에 특화된 결과 딕셔너리로 포맷팅합니다.
        
        주요 작업:
        1. 사이클 상태값 추출 (State 1-4 물성치)
        2. 성능 데이터 추출 (열량, 전력, 유량 등)
        3. LMTD 기반 열량 계산 (응축기, 증발기)
        4. 지중열 교환 계산 (보어홀, 토양 온도)
        5. 엑서지 계산
        6. 최종 결과 딕셔너리 생성
        
        호출 관계:
        - 호출자: _calculate_gshpb_next_step (본 클래스)
        - 사용 함수: compute_refrigerant_thermodynamic_states (cycle_performance.py, 간접 사용)
        
        Args:
            cycle_states (dict): compute_refrigerant_thermodynamic_states의 결과
                - P1, P2, P3, P4: 압력 [Pa]
                - T1_K, T2_K, T3_K, T4_K: 온도 [K]
                - h1, h2, h3, h4: 엔탈피 [J/kg]
                - s1, s2, s3, s4: 엔트로피 [J/kgK]
                - rho_ref_cmp_in: State 1의 밀도 [kg/m³]
            
            performance_data (dict): 사이클 성능 기본 데이터
                - Q_ref_cond (float): 사이클 계산 응축기 열량 [W]
                - Q_ref_evap (float): 사이클 계산 증발기 열량 [W]
                - E_cmp (float): 압축기 전력 [W]
                - m_dot_ref (float): 냉매 유량 [kg/s]
                - cmp_rps (float): 압축기 회전수 [1/s]
            
            T_tank_w (float): 저탕조 온도 [°C]
                현재 타임스텝의 저탕조 온도
            
            Q_cond_load (float): 저탕조 목표 열 교환율 [W]
                응축기가 저탕조에 전달해야 하는 목표 열량
        
        Returns:
            dict: GSHPB 클래스별 결과 딕셔너리
                - 사이클 상태값 (P1-4, T1-4, h1-4, s1-4, x1-4)
                - 열량 (Q_ref_cond, Q_ref_evap, Q_LMTD_cond, Q_LMTD_evap, Q_b)
                - 전력 (E_cmp, E_pmp)
                - 유량 (m_dot_ref, dV_b_f, dV_w_serv, dV_w_sup_tank, dV_w_sup_mix)
                - 온도 (T0, T1-4, T_tank_w, T_serv_w, T_sup_w, Ts, T_b, T_b_f, T_b_f_in, T_b_f_out)
                - 기타 (cmp_rpm, is_on)
        
        Notes:
            - LMTD 계산은 열교환기 물리적 제약 조건을 반영
            - Q_LMTD_cond와 Q_ref_cond는 최적화에서 일치해야 함
            - Q_LMTD_evap와 Q_ref_evap는 최적화에서 일치해야 함
        """
        T_tank_w_K = cu.C2K(T_tank_w)
        T_b_f_in_K = self.T_b_f_in_K
        T_b_f_in = cu.K2C(T_b_f_in_K)
        
        # ============================================================
        # 1단계: 사이클 상태값 추출
        # ============================================================
        # State 1-4의 열역학 물성치 추출
        T1_K = cycle_states['T1_K']  # State 1 온도 [K]
        P1 = cycle_states['P1']      # State 1 압력 [Pa]
        h1 = cycle_states['h1']      # State 1 엔탈피 [J/kg]
        s1 = cycle_states['s1']      # State 1 엔트로피 [J/kgK]
        
        T2_K = cycle_states['T2_K']  # State 2 온도 [K]
        P2 = cycle_states['P2']      # State 2 압력 [Pa]
        h2 = cycle_states['h2']      # State 2 엔탈피 [J/kg]
        s2 = cycle_states['s2']      # State 2 엔트로피 [J/kgK]
        
        T3_K = cycle_states['T3_K']  # State 3 온도 [K]
        P3 = cycle_states['P3']      # State 3 압력 [Pa]
        h3 = cycle_states['h3']      # State 3 엔탈피 [J/kg]
        s3 = cycle_states['s3']      # State 3 엔트로피 [J/kgK]
        
        T4_K = cycle_states['T4_K']  # State 4 온도 [K]
        P4 = cycle_states['P4']      # State 4 압력 [Pa]
        h4 = cycle_states['h4']      # State 4 엔탈피 [J/kg]
        s4 = cycle_states['s4']      # State 4 엔트로피 [J/kgK]
        
        # ============================================================
        # 2단계: 성능 데이터 추출
        # ============================================================
        # 사이클 계산으로부터 얻은 기본 성능 데이터
        Q_ref_cond = performance_data['Q_ref_cond']  # 사이클 계산 응축기 열량 [W]
        Q_ref_evap = performance_data['Q_ref_evap']  # 사이클 계산 증발기 열량 [W]
        E_cmp = performance_data['E_cmp']            # 압축기 전력 [W]
        m_dot_ref = performance_data['m_dot_ref']    # 냉매 유량 [kg/s]
        cmp_rps = performance_data['cmp_rps']        # 압축기 회전수 [1/s]
        
        # ============================================================
        # 3단계: LMTD 기반 열량 계산 (현실적 제약 조건)
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
        # 4단계: 지중열 교환 계산 (증발기 측)
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
        T_b_f = (T_b_f_in + T_b_f_out) / 2  # 유체 평균 온도 [°C]
        
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
        # 5단계: 엑서지 계산
        # ============================================================
        # 기준 상태(T0_K, P0) 대비 각 상태점의 엑서지 계산
        # 엑서지: 시스템에서 유용한 작업으로 변환 가능한 에너지
        T0_K = self.T0_K      # 기준 온도 [K]
        P0 = 101325           # 기준 압력 [Pa] (대기압)
        h0 = CP.PropsSI('H', 'T', T0_K, 'P', P0, self.ref)  # 기준 엔탈피 [J/kg]
        s0 = CP.PropsSI('S', 'T', T0_K, 'P', P0, self.ref)  # 기준 엔트로피 [J/kgK]
        
        result = {
            'is_on': True,
            
            'Q_ref_cond': Q_ref_cond,
            'Q_ref_evap': Q_ref_evap,
            'Q_LMTD_cond': Q_LMTD_cond,
            'Q_LMTD_evap': Q_LMTD_evap,
            
            'Q_b': self.Q_b,
            'Q_cond_load': Q_cond_load,
            'E_cmp': E_cmp,
            'E_pmp': self.E_pmp,
            'm_dot_ref': m_dot_ref,
            'cmp_rpm': cmp_rps * 60,
            
            'dV_b_f': self.dV_b_f_m3s,
            'dV_w_serv': self.dV_w_serv,
            'dV_w_sup_tank': self.dV_w_sup_tank,
            'dV_w_sup_mix': self.dV_w_sup_mix,
            
            'T_tank_w': T_tank_w,
            'T_serv_w': self.T_serv_w,
            'T_sup_w': self.T_sup_w,

            'T0': cu.K2C(T0_K),
            'T1': cu.K2C(T1_K),
            'T2': cu.K2C(T2_K),
            'T3': cu.K2C(T3_K),
            'T4': cu.K2C(T4_K),

            'Ts': self.Ts,
            'T_b': T_b,
            'T_b_f': T_b_f,
            'T_b_f_in': T_b_f_in,
            'T_b_f_out': T_b_f_out,

            'P1': P1,
            'P2': P2,
            'P3': P3,
            'P4': P4,
            
            'h1': h1,
            'h2': h2,
            'h3': h3,
            'h4': h4,
            
            's1': s1,
            's2': s2,
            's3': s3,
            's4': s4,
            
            'x1': (h1-h0) - T0_K*(s1 - s0),
            'x2': (h2-h0) - T0_K*(s2 - s0),
            'x3': (h3-h0) - T0_K*(s3 - s0),
            'x4': (h4-h0) - T0_K*(s4 - s0),
        }
        self.__dict__.update(result)
        return result
    

    
    def _format_gshpb_off_results_dict(self, T_tank_w, Q_cond_load=None, **kwargs):
        """
        OFF 상태 결과 포맷팅 함수.
        
        히트펌프가 OFF 상태일 때 사용되는 결과 딕셔너리를 생성합니다.
        모든 열량 및 전력 값은 0으로 설정하고, P-h 선도 플로팅을 위한
        기본 사이클 상태값은 포화점 기준으로 계산합니다.
        
        호출 관계:
        - 호출자: find_ref_loop_optimal_operation (cycle_performance.py)
            Q_cond_load가 임계값 이하일 때 호출
        
        주요 작업:
        1. ON 상태 템플릿 생성 (Q_cond_load=0으로 계산)
        2. 모든 숫자 값을 0으로 설정
        3. OFF 상태 플래그 및 필수 값 설정
        4. P-h 선도 플로팅용 포화점 계산
        
        Args:
            T_tank_w (float): 저탕조 온도 [°C]
                현재 타임스텝의 저탕조 온도
            
            Q_cond_load (float, optional): 저탕조 목표 열 교환율 [W]
                기본값: None (사용되지 않음, 항상 0.0으로 처리)
            
            **kwargs: 추가 파라미터 (사용되지 않음)
        
        Returns:
            dict: OFF 상태 결과 딕셔너리
                - 모든 열량 및 전력 값: 0.0
                - is_on: False
                - P1-4, h1-4, s1-4: 포화점 기준 계산값 (P-h 선도용)
                - 기타 상태값: 현재 시스템 상태 유지
        
        Notes:
            - P-h 선도 플로팅을 위해 기본 사이클 상태값을 계산합니다
            - 증발기 측은 지중 유체 입구 온도 기준 포화 증기
            - 응축기 측은 저탕조 온도 기준 포화 액체
        """
        # ============================================================
        # 1단계: ON 상태 템플릿 생성
        # ============================================================
        # Q_cond_load=0.0으로 계산하여 딕셔너리 구조만 얻음
        # 실제 계산은 필요 없지만 결과 딕셔너리 구조를 얻기 위해 실행
        result_template = self._calculate_gshpb_next_step(
            optimization_vars=[5.0, 5.0],  # 임의의 값 (계산 결과는 무시됨)
            T_tank_w=T_tank_w,
            Q_cond_load=0.0
        )
        
        if result_template is None:
            return None
        
        # ============================================================
        # 2단계: 모든 숫자 값을 0.0으로 설정
        # ============================================================
        # 히트펌프 OFF 상태이므로 모든 열량 및 전력 값은 0
        for key, value in result_template.items():
            if isinstance(value, (int, float)):
                result_template[key] = 0.0
        
        # ============================================================
        # 3단계: OFF 상태에 맞는 필수 값들 설정
        # ============================================================
        result_template['is_on'] = False  # OFF 상태 플래그
        result_template['Ts'] = self.Ts  # 지중 온도 유지
        result_template['T_tank_w'] = T_tank_w  # 저탕조 온도 유지
        
        # 유량 값 유지 (히트펌프 OFF 여부와 무관)
        result_template['dV_w_serv'] = self.dV_w_serv
        result_template['dV_w_sup_tank'] = self.dV_w_sup_tank
        result_template['dV_w_sup_mix'] = self.dV_w_sup_mix
        result_template['Q_b'] = 0.0  # 지중열 교환량 0
        
        # ============================================================
        # 4단계: P-h 선도 플로팅을 위한 포화점 값 계산
        # ============================================================
        # OFF 상태에서도 P-h 선도를 그리기 위해 기본 사이클 상태값 계산
        # 증발기 측: 지중 유체 입구 온도 기준 포화 증기
        # 응축기 측: 저탕조 온도 기준 포화 액체
        try:
            T_tank_w_K = cu.C2K(T_tank_w)
            
            # 증발기 측 포화 증기 (State 1, 4)
            P1_off = CP.PropsSI('P', 'T', self.T_b_f_in_K, 'Q', 1, self.ref)
            h1_off = CP.PropsSI('H', 'P', P1_off, 'Q', 1, self.ref)
            s1_off = CP.PropsSI('S', 'P', P1_off, 'Q', 1, self.ref)
            
            # 응축기 측 포화 액체 (State 2, 3)
            P3_off = CP.PropsSI('P', 'T', T_tank_w_K, 'Q', 0, self.ref)
            h3_off = CP.PropsSI('H', 'P', P3_off, 'Q', 0, self.ref)
            s3_off = CP.PropsSI('S', 'P', P3_off, 'Q', 0, self.ref)
            
            # OFF 상태 사이클: P1→P3→P3→P1, h1→h1→h3→h3
            result_template.update({
                'P1': P1_off, 'P2': P3_off, 'P3': P3_off, 'P4': P1_off,
                'h1': h1_off, 'h2': h1_off, 'h3': h3_off, 'h4': h3_off,
                's1': s1_off, 's2': s1_off, 's3': s3_off, 's4': s3_off,
            })
        except Exception:
            # 계산 실패 시 NaN으로 설정
            nan_keys = ['P1','P2','P3','P4','h1','h2','h3','h4','s1','s2','s3','s4']
            for k in nan_keys:
                result_template[k] = np.nan
        
        return result_template
    
    def run_simulation(
        self, 
        simulation_period_sec, 
        dt_s, 
        T_tank_w_init_C,
        schedule_entries,
        result_save_csv_path=None,
        save_ph_diagram=False,
        snapshot_save_path=None,
        ):
        
        # [run_simulation 내부 주요 흐름 (타임스텝 n 기준 구조 다이어그램)]
        # 
        # run_simulation (at time step n)
        # │
        # ├─ [1] 제어 상태 결정 (Control Logic)
        # │     └─ (입력) T_tank_w[n], (출력) is_on[n](=제어기 ON/OFF), Q_cond_load[n]
        # │
        # ├─ [2] 냉매루프 최적화 운전점 계산
        # │     └─ find_ref_loop_optimal_operation 호출
        # │          │
        # │          ├─ [내부 최적화 반복] ────────────────────────────────────────┐
        # │          │    └─ objective/constraints ▷ _calculate_gshpb_next(여러 번)
        # │          │             └─ self.T_b_f_in_K 사용 ← (n-1) 스텝의 값
        # │          └─ [최적값 도출 후] ─────────────────────────────────────────┘
        # │               └─ _calculate_gshpb_next(1회, 최종)
        # │                   └─ self.T_b_f_in_K 사용 ← (n-1) 스텝의 값
        # │
        # ├─ [3] 지중열 교환/지중 온도 계산 및 상태 업데이트
        # │     └─ (출력) self.T_b_f_in_K ← (n) 스텝 결과로 갱신
        # │
        # └─ [4] 저탕조 온도 등 내부 상태 업데이트
        #       └─ (출력) T_tank_w[n+1] 등 시스템 상태 계산
        
        """
        설정된 파라미터와 스케줄을 바탕으로 동적 시뮬레이션을 실행합니다.
        
        이 메서드는 시간에 따른 GSHPB 시스템의 동적 거동을 시뮬레이션합니다.
        각 타임스텝마다 제어 로직, 히트펌프 최적화, 지중 온도 업데이트,
        저탕조 온도 업데이트를 순차적으로 수행합니다.
        
        시뮬레이션 단계:
        ──────────────────────────────────────────────────────────────────────────
        각 타임스텝 n에서:
        1. 제어 상태 결정 (Control Logic)
           - 저탕조 온도 기반 ON/OFF 판단
           - 급탕 사용량 계산
           - 목표 열 교환율 결정
        
        2. 냉매루프 최적화 운전점 계산
           - find_ref_loop_optimal_operation 호출
           - 압축기 전력 최소화하는 최적 운전점 탐색
           - LMTD 기반 제약 조건 만족
        
        3. 지중열 교환/지중 온도 계산 및 상태 업데이트
           - g-function 기반 지중 온도 계산
           - 지중 유체 온도 업데이트
           - 다음 타임스텝에 사용할 T_b_f_in_K 갱신
        
        4. 저탕조 온도 등 내부 상태 업데이트
           - 열량 밸런스 기반 저탕조 온도 계산
           - 다음 타임스텝으로 전달
        
        Args:
            simulation_period_sec (float): 총 시뮬레이션 시간 (초)
            dt_s (int): 타임스텝 (초)
            T_tank_w_init_C (float): 저탕조 초기 온도 (°C)
            schedule_entries (list): 급탕 사용 스케줄
                [(시작시간_str, 종료시간_str, 사용비율_float), ...]
                예: [("6:00", "6:30", 0.5), ("6:30", "7:00", 0.9)]
            result_save_csv_path (str, optional): 결과 CSV 저장 경로.
            save_ph_diagram (bool, optional): P-h 선도 이미지 저장 여부.
            snapshot_save_path (str, optional): P-h 선도 이미지 저장 경로.

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
        
        # --- 1. 시뮬레이션 초기화 ---
        self.time = np.arange(0, simulation_period_sec, dt_s)
        self.dt = dt_s
        self.T_b_f = self.Ts # 초기 지중열 교환기 유출수 온도
        self.T_b = self.Ts   # 초기 지중 온도
        self.T_b_f_in = self.Ts # 초기 지중열 교환기 유입수 온도
        self.T_b_f_out = self.Ts # 초기 지중열 교환기 유출수 온도
        self.Q_b = 0.0 # 초기 지중열 교환기 열 유량
        
        self.dV_w_serv = 0.0 # 초기 서비스 유량
        self.dV_w_sup_tank = 0.0 # 초기 탱크 출수 유량
        self.dV_w_sup_mix = 0.0 # 초기 믹싱밸브 상수도 유량
        
        tN = len(self.time)
        
        # 스케줄 빌드
        self.serv_sched = _build_schedule_ratios(schedule_entries, self.time)
        
        results_data = [] # 결과를 딕셔너리 리스트로 저장
        
        # 동적 상태 변수 초기화
        T_tank_w_K = cu.C2K(T_tank_w_init_C)
        Q_b_unit_pulse = np.zeros(tN)
        Q_b_unit_old = 0
        is_on_prev = False

        # --- 2. 시뮬레이션 루프 ---
        for n in tqdm(range(tN), desc="GSHPB Simulating"):
            step_results = {}
            T_tank_w = cu.K2C(T_tank_w_K)

            # 2. 제어 상태 결정 (config -> self)
            Q_tank_loss = self.UA_tank * (T_tank_w_K - self.T0_K)
            den = max(1e-6, T_tank_w_K - self.T_sup_w_K)
            alp = min(1.0, max(0.0, self.T_serv_w_K - self.T_sup_w_K) / den)

            self.dV_w_serv = self.serv_sched[n] * self.dV_w_serv_m3s 
            self.dV_w_sup_tank = alp * self.dV_w_serv
            self.dV_w_sup_mix = (1 - alp) * self.dV_w_serv 

            Q_use_loss = c_w * rho_w * self.dV_w_sup_tank * (T_tank_w_K - self.T_sup_w_K)
            total_loss = Q_tank_loss + Q_use_loss
            
            # On/Off 결정 (config -> self)
            if T_tank_w < self.T_tank_w_lower_bound: is_on = True
            elif T_tank_w > self.T_tank_w_setpoint: is_on = False
            else: is_on = is_on_prev
            
            # OFF→ON 전환 시점 감지
            is_transitioning_off_to_on = (not is_on_prev) and is_on
            
            ###########################################################################
            # 현재 heater capacity로 고정되어있지만 나중에 가변 부하로 변경 가능
            Q_cond_load_n = self.heater_capacity if is_on else 0.0 # config -> self
            ###########################################################################
            is_on_prev = is_on
            
            # 3. 히트펌프 운전점 계산 (gshpb -> self)
            # OFF→ON 전환 시점과 정상 상태 모두에서 최적화 반복 호출만 다르고
            # 저장 여부만 다르므로 공통 부분 함수로 뺌
            def _run_ref_loop_dropt():
                return find_ref_loop_optimal_operation(
                    calculate_performance_func=self._calculate_gshpb_next_step,
                    T_tank_w=cu.K2C(T_tank_w_K),
                    Q_cond_load=Q_cond_load_n,
                    Q_cond_LOAD_OFF_TOL=self.Q_cond_LOAD_OFF_TOL,
                    bounds=[(0.1, 30.0), (0.1, 30.0)],
                    initial_guess=[5.0, 5.0],
                    constraint_funcs=create_lmtd_constraints(),
                    off_result_formatter=self._format_gshpb_off_results_dict
                )
            ref_result = _run_ref_loop_dropt()
            if is_transitioning_off_to_on:
                # OFF→ON 전환 시점: 이전 스텝의 값들을 그대로 유지하여 저장
                if len(results_data) > 0:
                    # 이전 스텝의 결과를 복사
                    step_results.update(results_data[-1].copy())
                    # is_on 상태만 현재 상태로 업데이트
                    step_results['is_on'] = is_on
                else:
                    # 첫 번째 스텝에서 전환되는 경우 (거의 없지만 안전을 위해)
                    # OFF 상태 결과를 사용
                    step_results.update(ref_result)
                    step_results['is_on'] = is_on
            else:
                # 정상 상태: 최적화 결과 저장
                step_results.update(ref_result)
                step_results['is_on'] = is_on

            # 4. 지중 온도 업데이트 (gshpb -> self)
            Q_b_unit = (ref_result.get('Q_ref_evap', 0.0) - self.E_pmp) / self.H_b if ref_result.get('is_on') else 0.0
            
            if abs(Q_b_unit - Q_b_unit_old) > 1e-6: # 만약 Q_b이 이전 스텝과 일정 수준 이상 차이가 난다면 펄스가 나타난 것으로 간주
                Q_b_unit_pulse[n] = Q_b_unit - Q_b_unit_old # 펄스는 이전 값과의 차이
                Q_b_unit_old = Q_b_unit # 업데이트
        
            pulses_idx = np.flatnonzero(Q_b_unit_pulse[:n+1])
            dQ = Q_b_unit_pulse[pulses_idx]
            tau = self.time[n] - self.time[pulses_idx]
            
            # g-function 계산은 여전히 루프가 필요
            g_n = np.array([G_FLS(t, self.k_s, self.alp_s, self.r_b, self.H_b) for t in tau])
            dT_b = np.dot(dQ, g_n)
            
            ################################################################################################
            '''
            본 데이터 중, T_b_f_in의 업데이트만 find_ref_loop_optimal_operation 함수에서의 데이터 계산에 활용되며
            실제 저장되는 데이터들은 find_ref_loop_optimal_operation 함수에 의해서만 결정된다.
            '''
            self.T_b = self.Ts - dT_b
            self.T_b_f = self.T_b - Q_b_unit * self.R_b
            self.Q_b = Q_b_unit * self.H_b
            self.T_b_f_in  = self.T_b_f - self.Q_b / (2 * c_w * rho_w * self.dV_b_f_m3s) # °C
            self.T_b_f_out = self.T_b_f + self.Q_b / (2 * c_w * rho_w * self.dV_b_f_m3s) # °C
            self.T_b_f_in_K  = cu.C2K(self.T_b_f_in)
            self.T_b_f_out_K = cu.C2K(self.T_b_f_out)
            
            # 업데이트된 지중 온도 관련 값들을 step_results에 반영
            # (전환 시점이 아닐 때만 반영 - 전환 시점은 이전 스텝 값을 유지해야 함)
            if not is_transitioning_off_to_on:
                step_results['T_b'] = self.T_b
                step_results['T_b_f'] = self.T_b_f
                step_results['T_b_f_in'] = self.T_b_f_in
                step_results['T_b_f_out'] = self.T_b_f_out
                step_results['Q_b'] = self.Q_b
            ################################################################################################
            
            
            # 5. 다음 스텝 탱크 온도 계산
            if n < tN - 1:
                Q_tank_in = ref_result.get('Q_ref_cond', 0.0)
                Q_net = Q_tank_in - total_loss
                T_tank_w_K += (Q_net / self.C_tank) * self.dt # config -> self

            # 6. 결과 저장 및 플롯 ##################################################################
            '''
            추후 완전한 plot 그래프 기능의 별도 기능 분리 필요
            '''
            # 결과 저장 (전환 시점에도 이전 스텝 값으로 저장)
            if save_ph_diagram: # P-h 선도 저장
                if snapshot_save_path is None:
                    raise ValueError("snapshot_save_path must be provided when ref_snapshot_on is True.")
                # 전환 시점이 아닐 때만 P-h 선도 저장 (전환 시점은 이전 스텝 값이므로)
                if not is_transitioning_off_to_on:
                    plot_cycle_diagrams(
                        result=ref_result,
                        refrigerant=self.ref,
                        show= False,
                        show_temp_limits=True,
                        save_path=snapshot_save_path+f'/{n:04d}.png',
                        T_tank_w=self.T_tank_w,
                        T_b_f_in=self.T_b_f_in, 
                        T_b_f_out=self.T_b_f_out,
                    )
            results_data.append(step_results)
            ################################################################################################
            
        # --- 3. 시뮬레이션 종료 및 반환 ---
        results_df = pd.DataFrame(results_data)

        if result_save_csv_path:
            results_df.to_csv(result_save_csv_path)

        return results_df

