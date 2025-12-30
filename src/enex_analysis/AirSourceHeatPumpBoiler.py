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
    _build_schedule_ratios,
    calc_UA_from_dV_fan,
    calc_fan_power_from_dV_fan,
    calc_HX_perf_for_target_heat,
    update_tank_temperature
)   


@dataclass
class AirSourceHeatPumpBoiler:
    '''
    공기원 히트펌프 보일러 성능 계산 및 최적 운전점 탐색 클래스.
    '''
    def __init__(
        self,

        # 1. 냉매/사이클/압축기 파라미터 -------------------------------
        ref         = 'R410A',
        V_disp_cmp  = 0.0005,
        eta_cmp_isen = 0.7,

        # 2. 열교환기 파라미터 -----------------------------------------
        UA_cond_design = 500.0,   # 응축기 열전달 계수 [W/K] (저탕조 온수와 응축기 열교환기 표면)
        UA_evap_design = 500.0,   # 증발기 열전달 계수 [W/K] (실외 공기와 증발기 열교환기 표면)

        # 3. 실외기 팬 파라미터 ---------------------------------------
        dV_ou_design = 2.5,   # 실외기 설계 풍량 [m³/s] (정풍량)
        dP_ou_design = 500.0, # 실외기 설계 정압 [Pa]
        A_cross_ou = 0.4,         # 실외기 단면적 [m²]
        eta_fan_ou   = 0.8,   # 실외기 팬 효율 [-]

        # 4. 탱크/제어/부하 파라미터 -----------------------------------
        T0                    = 0.0,    # [°C] 기준 외기 온도
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
        
        # --- 2. 열교환기 파라미터 ---
        self.UA_cond_design = UA_cond_design
        self.UA_evap_design = UA_evap_design
        
        # --- 3. 실외기 팬 파라미터 ---
        self.dV_ou_design = dV_ou_design
        self.dP_ou_design = dP_ou_design
        self.eta_fan_ou = eta_fan_ou
        self.A_cross_ou = A_cross_ou
        
        # 팬 설계 전력 계산 (정풍량 기준)
        self.fan_design_power_ou = (self.dV_ou_design * self.dP_ou_design) / (self.eta_fan_ou)
        
        # VSD Curve 계수 VSD(Variable Speed Drive)
        self.vsd_coeffs_ou = vsd_coeffs_ou
        
        # 팬 파라미터 딕셔너리
        self.fan_params_ou = {
            'fan_design_flow_rate': self.dV_ou_design,
            'fan_design_power': self.fan_design_power_ou
        }
        
        # --- 4. 기준 온도 ---
        self.T0 = T0
        
        self.tank_physical = {
            'r0': r0, 'H': H, 'x_shell': x_shell, 'x_ins': x_ins,
            'k_shell': k_shell, 'k_ins': k_ins, 'h_o': h_o,
        }
        self.UA_tank = calc_simple_tank_UA(**self.tank_physical)
        self.C_tank = c_w * rho_w * (math.pi * r0**2 * H)
        
        self.heater_capacity = heater_capacity
        self.dV_w_serv_m3s = dV_w_serv_m3s
        self.T_tank_w_setpoint = T_tank_w_setpoint
        self.T_tank_w_lower_bound = T_tank_w_lower_bound
        self.T_sup_w = T_sup_w
        self.T_serv_w = T_serv_w
        
        self.T_sup_w_K = cu.C2K(T_sup_w)
        self.T_serv_w_K = cu.C2K(T_serv_w)
        
        self.Q_cond_load_threshold = 500.0

    def _calc_on_state(self, optimization_vars, T_tank_w, Q_cond_load, T_oa):
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
            - compute_refrigerant_thermodynamic_states (enex_functions.py)
            - calc_HX_perf_for_target_heat (enex_functions.py)
            - calc_lmtd_fluid_and_constant_temp (enex_functions.py)
            - calc_fan_power_from_dV_fan (enex_functions.py)
        
        데이터 흐름:
        ──────────────────────────────────────────────────────────────────────────
        [optimization_vars, T_tank_w, Q_cond_load, T_oa]
            ↓
        증발/응축 온도 계산 (T_evap_K, T_cond_K)
            ↓
        compute_refrigerant_thermodynamic_states
            ↓ [State 1-4 물성치]
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
            
            T_oa (float): 실외 공기 온도 [°C] (기본값: 20)
                현재 타임스텝의 실외 공기 온도
        
        Returns:
            dict: 사이클 성능 결과 딕셔너리
                - 사이클 상태값 (P1-4, T1-4, h1-4, s1-4, x1-4)
                - 열량 (Q_ref_cond, Q_ref_evap, Q_LMTD_cond)
                - 전력 (E_cmp, E_fan_ou)
                - 유량 (m_dot_ref, dV_fan_ou, dV_w_serv, dV_w_sup_tank, dV_w_sup_mix)
                - 온도 (T0, T1-4, T_tank_w, T_serv_w, T_sup_w, T_oa, T_air_ou_out)
                - 기타 (cmp_rpm, is_on)
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
        
        # 1단계: 최적화 변수
        dT_ref_cond = optimization_vars[0]
        dT_ref_evap = optimization_vars[1]      
        
        # 2단계: 온도 단위 변환 및 증발/응축 온도 계산
        T_tank_w_K = cu.C2K(T_tank_w)         
        T_oa_K = cu.C2K(T_oa)                 
        
        T_evap_K = T_oa_K - dT_ref_evap       
        T_cond_K = T_tank_w_K + dT_ref_cond   
        
        T0_K = cu.C2K(self.T0) 
        
        # 3단계: 공통 사이클 상태 계산
        cycle_states = compute_refrigerant_thermodynamic_states(
            T_evap_K     = T_evap_K,
            T_cond_K     = T_cond_K,
            refrigerant  = self.ref,
            eta_cmp_isen = self.eta_cmp_isen,
            T0_K         = T0_K,
            mode         = 'heating'
        )
        
        # 4단계: 사이클 상태값 추출
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
        
        rho_ref_cmp_in = cycle_states['rho']
        
        # 5단계: 냉매 유량 및 성능 데이터 계산
        m_dot_ref = Q_cond_load / (h2 - h3) 
        Q_ref_cond = m_dot_ref * (h2 - h3) 
        Q_ref_evap = m_dot_ref * (h1 - h4) 
        E_cmp = m_dot_ref * (h2 - h1)      
        cmp_rps = m_dot_ref / (self.V_disp_cmp * rho_ref_cmp_in) 
        
        # 6단계: 실외기 열교환기 성능 계산
        T_ref_evap_avg_K = (T4_K + T1_K) / 2
        T_air_ou_in_K = T_oa_K
        
        # Q_cond_load가 0에 가까우면 열교환기 계산 생략 (root_scalar 에러 방지)
        if abs(Q_cond_load) < self.Q_cond_load_threshold:
            # 0 값으로 채운 결과 설정
            dV_fan_ou = 0.0
            UA_evap = self.UA_evap_design
            T_air_ou_out_K = T_oa_K  # 열교환이 없으므로 입구 온도와 동일
            LMTD_evap = 0.0
            Q_LMTD_evap = 0.0
        
        
        else:
            HX_perf_ou_dict = calc_HX_perf_for_target_heat(
                Q_ref_target=Q_ref_evap,
                T_air_in_C=T_oa,
                T_ref_avg_K=T_ref_evap_avg_K,
                A_cross=self.A_cross_ou,
                UA_design=self.UA_evap_design,
                dV_fan_design=self.dV_ou_design
            )
            # 수렴 실패 감지: converged 플래그 확인
            if HX_perf_ou_dict.get('converged', True) == False:
                # calc_HX_perf_for_target_heat 수렴 실패
                # None을 반환하여 최적화 알고리즘이 다른 변수 조합을 시도할 수 있도록 함
                return None
            
            # 수렴 성공 시 값 추출
            dV_fan_ou = HX_perf_ou_dict['dV_fan']
            UA_evap = HX_perf_ou_dict['UA']
            T_air_ou_out_K = HX_perf_ou_dict['T_air_out_K']
            LMTD_evap = HX_perf_ou_dict['LMTD']
            Q_LMTD_evap = HX_perf_ou_dict['Q_LMTD']
        
        # 7단계: LMTD 기반 열량 계산
        LMTD_tank = calc_lmtd_fluid_and_constant_temp(
            T_constant_K=T_tank_w_K, 
            T_fluid_in_K=T2_K, 
            T_fluid_out_K=T3_K
        )
        UA_cond = self.UA_cond_design # constant assumption 
        Q_LMTD_cond = UA_cond * LMTD_tank
        
        # 8단계: 팬 전력 계산
        E_fan_ou = calc_fan_power_from_dV_fan(
            dV_fan=dV_fan_ou, 
            fan_params=self.fan_params_ou, 
            vsd_coeffs=self.vsd_coeffs_ou
        )
        
        # 9단계: 엑서지 계산
        P0 = 101325
        h0 = CP.PropsSI('H', 'T', T0_K, 'P', P0, self.ref)
        s0 = CP.PropsSI('S', 'T', T0_K, 'P', P0, self.ref)
        
        # 10단계: 최종 결과 딕셔너리 생성
        result = {
            'is_on': True,
            'converged': True,  # 정상적으로 계산 완료
            
            'Q_cond_load': Q_cond_load,
            'Q_ref_cond': Q_ref_cond,
            'Q_ref_evap': Q_ref_evap,
            'Q_LMTD_cond': Q_LMTD_cond,
            'Q_LMTD_evap': Q_LMTD_evap,
            
            'E_cmp': E_cmp,
            'E_fan_ou': E_fan_ou,
            'E_tot': E_cmp + E_fan_ou,
            
            'cop': Q_cond_load / (E_cmp + E_fan_ou) if (E_cmp + E_fan_ou) > 0 else 0,
            
            'm_dot_ref': m_dot_ref,
            'cmp_rpm': cmp_rps * 60,
            
            'dV_fan_ou': dV_fan_ou,
            'dV_w_serv': self.dV_w_serv if hasattr(self, 'dV_w_serv') else 0.0,
            'dV_w_sup_tank': self.dV_w_sup_tank if hasattr(self, 'dV_w_sup_tank') else 0.0,
            'dV_w_sup_mix': self.dV_w_sup_mix if hasattr(self, 'dV_w_sup_mix') else 0.0,
            
            'T_oa': T_oa,
            'T_air_ou_out': cu.K2C(T_air_ou_out_K),
            'T_air_ou_in': cu.K2C(T_air_ou_in_K),
            
            'T_tank_w': T_tank_w,
            'T_serv_w': self.T_serv_w,
            'T_sup_w': self.T_sup_w,
            
            'T_cond': cu.K2C(T2_K),
            'T_evap': cu.K2C(T4_K),
            
            'T0': cu.K2C(T0_K),
            'T1': cu.K2C(T1_K),
            'T2': cu.K2C(T2_K),
            'T3': cu.K2C(T3_K),
            'T4': cu.K2C(T4_K),

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
        
        return result
    
    def _calc_off_state(self, T_tank_w, T_oa):
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
            
            T_oa (float, optional): 실외 공기 온도 [°C]
                기본값: 20.0
        
        Returns:
            dict: OFF 상태 결과 딕셔너리
                - 모든 열량 및 전력 값: 0.0
                - is_on: False
                - P1-4, h1-4, s1-4: 포화점 기준 계산값 (P-h 선도용)
                - 기타 상태값: 현재 시스템 상태 유지
        
        Notes:
            - P-h 선도 플로팅을 위해 기본 사이클 상태값을 계산합니다
            - 증발기 측은 실외 공기 온도 기준 포화 증기
            - 응축기 측은 저탕조 온도 기준 포화 액체
        """
        # T_oa는 파라미터로 전달받은 값을 사용 (기본값 20.0, 외부에서 전달된 값 우선)
        # self.T0로 덮어씌우지 않음
        
        
        # 1단계: ON 상태 템플릿 생성
        
        # Q_cond_load=0.0으로 계산하여 딕셔너리 구조만 얻음
        # 실제 계산은 필요 없지만 결과 딕셔너리 구조를 얻기 위해 실행
        result = self._calc_on_state(
                optimization_vars=[5.0, 5.0],  # 임의의 값 (계산 결과는 무시됨)
                T_tank_w=T_tank_w,
                Q_cond_load=0.0,
                T_oa=T_oa  # 파라미터로 받은 T_oa 사용
            )
        
        
        # 2단계: 모든 숫자 값을 0.0으로 설정
        for key, value in result.items():
            if isinstance(value, (int, float)):
                result[key] = 0.0

        # 3~4단계: OFF 상태에 맞는 필수 값들과 P-h 선도 플로팅값을 result.update에서 한꺼번에 업데이트
        T_tank_w_K = cu.C2K(T_tank_w)
        T_oa_K = cu.C2K(T_oa)

        # 증발기 측 포화 증기 (State 1, 4)
        P1_off = CP.PropsSI('P', 'T', T_oa_K, 'Q', 1, self.ref)
        h1_off = CP.PropsSI('H', 'P', P1_off, 'Q', 1, self.ref)
        s1_off = CP.PropsSI('S', 'P', P1_off, 'Q', 1, self.ref)

        # 응축기 측 포화 액체 (State 2, 3)
        P3_off = CP.PropsSI('P', 'T', T_tank_w_K, 'Q', 0, self.ref)
        h3_off = CP.PropsSI('H', 'P', P3_off, 'Q', 0, self.ref)
        s3_off = CP.PropsSI('S', 'P', P3_off, 'Q', 0, self.ref)

        result.update({
            'is_on': False,
            'converged': True,  # OFF 상태는 정상적인 상태이므로 수렴 성공으로 간주
            
            'T_tank_w': T_tank_w,
            'T_oa': T_oa,
            
            'dV_w_serv': self.dV_w_serv,
            'dV_w_sup_tank': self.dV_w_sup_tank,
            'dV_w_sup_mix': self.dV_w_sup_mix,
            
            'T0': self.T0,
            'T_serv_w': self.T_serv_w,
            'T_sup_w': self.T_sup_w,

            # P-h 선도 플로팅을 위한 포화점 값
            'T_cond': cu.K2C(T_tank_w_K),
            'T_evap': cu.K2C(T_oa_K),
            
            'T_air_ou_out': cu.K2C(T_oa_K),
            
            'P1': P1_off, 'P2': P3_off, 'P3': P3_off, 'P4': P1_off,
            'h1': h1_off, 'h2': h1_off, 'h3': h3_off, 'h4': h3_off,
            's1': s1_off, 's2': s1_off, 's3': s3_off, 's4': s3_off,
            
            'T1': cu.K2C(T_oa_K), 'T2': cu.K2C(T_tank_w_K),
            'T3': cu.K2C(T_tank_w_K), 'T4': cu.K2C(T_oa_K),
        })

        return result
    
    def analyze_steady(
        self,
        T_tank_w,
        T_oa,
        dV_w_serv=None,  # 급탕 유량 [m³/s], None이면 0으로 가정
        return_dict=True  # True면 dict 반환, False면 DataFrame 반환
    ):
        """
        정상상태 해석 함수.
        
        주어진 조건에서 히트펌프의 정상상태 성능을 계산합니다.
        저탕조의 에너지 밸런스를 만족하도록 Q_cond_load를 total_loss로 자동 계산합니다.
        
        Args:
            T_tank_w (float): 저탕조 온도 [°C]
            T_oa (float): 실외 공기 온도 [°C]
            dV_w_serv (float, optional): 급탕 유량 [m³/s]. None이면 0으로 가정
            return_dict (bool): True면 dict 반환, False면 DataFrame 반환
        
        Returns:
            dict 또는 pd.DataFrame: 정상상태 해석 결과
                - run_simulation과 동일한 형식의 결과 딕셔너리
                - 최적화 실패 시 OFF 상태 결과 반환
        """
        # 온도 단위 변환
        T_tank_w_K = cu.C2K(T_tank_w)
        T_oa_K = cu.C2K(T_oa)
        
        # 급탕 유량 설정
        if dV_w_serv is None:
            dV_w_serv = 0.0
        
        # 급탕 유량 관련 계산
        Q_tank_loss = self.UA_tank * (T_tank_w_K - T_oa_K)
        den = max(1e-6, T_tank_w_K - self.T_sup_w_K)
        alp = min(1.0, max(0.0, self.T_serv_w_K - self.T_sup_w_K) / den)
        
        self.dV_w_serv = dV_w_serv
        self.dV_w_sup_tank = alp * dV_w_serv
        self.dV_w_sup_mix = (1 - alp) * dV_w_serv
        
        # 열 손실 계산
        Q_use_loss = c_w * rho_w * self.dV_w_sup_tank * (T_tank_w_K - self.T_sup_w_K)
        total_loss = Q_tank_loss + Q_use_loss
        
        # 정상상태: Q_cond_load = total_loss (에너지 밸런스)
        Q_cond_load = total_loss
        
        # ON/OFF 상태 결정
        if T_tank_w < self.T_tank_w_lower_bound:
            is_on = True
        elif T_tank_w > self.T_tank_w_setpoint:
            is_on = False
        else:
            # 정상상태에서는 Q_cond_load > 0이면 ON으로 가정
            is_on = Q_cond_load > self.Q_cond_load_threshold
        
        # OFF 상태 조기 체크: Q_cond_load가 임계값 이하이면 최적화 건너뛰기
        if abs(Q_cond_load) <= self.Q_cond_load_threshold or not is_on:
            result = self._calc_off_state(
                T_tank_w=T_tank_w,
                T_oa=T_oa
            )
        else:
            # 최적화 과정 진행
            bounds = [(1.0, 30.0), (1.0, 30.0)]  # [dT_ref_evap, dT_ref_cond]
            initial_guess = [20.0, 1.0]
            
            def _cond_LMTD_constraint(x):
                """
                응축기 LMTD 제약 조건 함수.
                perf가 None이면 (수렴 실패 등) 큰 제약 조건 위반 값을 반환하여
                최적화 알고리즘이 해당 변수 조합을 피하고 다른 조합을 시도하도록 함.
                """
                try:
                    perf = self._calc_on_state(
                        optimization_vars=x,
                        T_tank_w=T_tank_w,
                        Q_cond_load=Q_cond_load,
                        T_oa=T_oa
                    )
                    # perf가 None이면 수렴 실패로 간주하여 큰 제약 조건 위반 값 반환
                    # 최적화 알고리즘이 다른 변수 조합을 시도하도록 함
                    if perf is None or not isinstance(perf, dict):
                        return 1e6
                    
                    # 필요한 키가 없거나 값이 nan이면 큰 제약 조건 위반 값 반환
                    if "Q_LMTD_cond" not in perf or np.isnan(perf["Q_LMTD_cond"]):
                        return 1e6
                    
                    # 제약 조건: Q_LMTD_cond - Q_cond_load >= 0 (Q_LMTD_cond >= Q_cond_load, 목표 부하보다 크거나 같으면 만족)
                    return perf["Q_LMTD_cond"] - Q_cond_load
                except Exception as e:
                    # 예외 발생 시 큰 제약 조건 위반 값 반환
                    return 1e6

            def _evap_LMTD_constraint(x):
                """
                증발기 LMTD 제약 조건 함수.
                perf가 None이면 (수렴 실패 등) 큰 제약 조건 위반 값을 반환하여
                최적화 알고리즘이 해당 변수 조합을 피하고 다른 조합을 시도하도록 함.
                """
                try:
                    perf = self._calc_on_state(
                        optimization_vars=x,
                        T_tank_w=T_tank_w,
                        Q_cond_load=Q_cond_load,
                        T_oa=T_oa
                    )
                    # perf가 None이면 수렴 실패로 간주하여 큰 제약 조건 위반 값 반환
                    # 최적화 알고리즘이 다른 변수 조합을 시도하도록 함
                    if perf is None or not isinstance(perf, dict):
                        return 1e6
                    
                    # 필요한 키가 없거나 값이 nan이면 큰 제약 조건 위반 값 반환
                    if ("Q_LMTD_evap" not in perf or "Q_ref_evap" not in perf or
                        np.isnan(perf["Q_LMTD_evap"]) or np.isnan(perf["Q_ref_evap"])):
                        return 1e6
                    
                    # 제약 조건: Q_LMTD_evap - Q_ref_evap = 0 (0이면 만족)
                    return perf["Q_LMTD_evap"] - perf['Q_ref_evap']
                except Exception as e:
                    # 예외 발생 시 큰 제약 조건 위반 값 반환
                    return 1e6
                
            const_funcs = [
                {'type': 'eq', 'fun': _cond_LMTD_constraint},  # Q_LMTD_cond >= Q_cond_load
                {'type': 'eq', 'fun': _evap_LMTD_constraint},
            ]

            def _objective(x):  # x = [dT_ref_evap, dT_ref_cond]
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
                        T_oa=T_oa
                    )
                    # perf가 None이면 수렴 실패로 간주하여 큰 penalty 반환
                    # 최적화 알고리즘이 다른 변수 조합을 시도하도록 함
                    if perf is None or not isinstance(perf, dict):
                        return 1e6
                    
                    # E_tot가 없거나 nan이면 큰 penalty 반환
                    if "E_tot" not in perf or np.isnan(perf["E_tot"]):
                        return 1e6
                    
                    # 목적 함수: E_tot 최소화
                    return perf["E_tot"]
                except Exception as e:
                    # 예외 발생 시 큰 penalty 반환
                    return 1e6
            
            opt_result = minimize(
                _objective,           # 목적 함수 (E_tot = E_cmp + E_fan_ou 최소화)
                initial_guess,        # 초기 추정값
                method='SLSQP',       # Sequential Least Squares Programming
                bounds=bounds,        # 변수 경계 조건
                constraints=const_funcs,
                    options={
                    'disp': False,
                    'maxiter': 100,
                    'ftol': 10,      # 함수 값 수렴 허용 오차
                    'eps': 0.01,      # 유한 차분 근사 스텝 크기
                }    # 상세 출력 비활성화, 최대 반복 횟수 1000회
            )
            
            # opt_result.x로 결과 계산 시도 (성공/실패 무관)
            result = None
            try:
                result = self._calc_on_state(
                    optimization_vars=opt_result.x,
                    T_tank_w=T_tank_w,
                    Q_cond_load=Q_cond_load,
                    T_oa=T_oa
                )
            except Exception:
                pass
            
            # result 검증 및 fallback
            if result is None or not isinstance(result, dict):
                try:
                    result = self._calc_off_state(
                        T_tank_w=T_tank_w,
                        T_oa=T_oa
                    )
                except Exception:
                    result = {
                        'is_on': False,
                        'converged': False,
                        'Q_ref_cond': 0.0,
                        'Q_ref_evap': 0.0,
                        'E_cmp': 0.0,
                        'E_fan_ou': 0.0,
                        'E_tot': 0.0,
                        'T_tank_w': T_tank_w,
                        'T_oa': T_oa
                    }
            
            # converged 플래그 설정
            if result is not None and isinstance(result, dict):
                result['converged'] = opt_result.success
        
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
        T_oa_schedule,
        heater_capacity_const = None,
        heater_capacity_schedule = None,
        heater_capacity_auto = False,
        result_save_csv_path=None,
        save_ph_diagram=False,
        snapshot_save_path=None,
        ):
        
        """
        동적 시뮬레이션을 실행합니다.
        
        Args:
            simulation_period_sec: 총 시뮬레이션 시간 [초]
            dt_s: 타임스텝 [초]
            T_tank_w_init_C: 저탕조 초기 온도 [°C]
            schedule_entries: 급탕 사용 스케줄
            T_oa_schedule: 실외 공기 온도 스케줄 [°C]
            result_save_csv_path: 결과 CSV 저장 경로
            save_ph_diagram: P-h 선도 이미지 저장 여부
            snapshot_save_path: P-h 선도 이미지 저장 경로
        
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
        T_oa_schedule = np.array(T_oa_schedule)
        if len(T_oa_schedule) != tN:
            raise ValueError(f"T_oa_schedule length ({len(T_oa_schedule)}) must match time array length ({tN})")
        
        results_data = []
        
        self.time = time
        self.dt = dt_s
        
        self.dV_w_serv = 0.0 
        self.dV_w_sup_tank = 0.0 
        self.dV_w_sup_mix = 0.0 
        
        self.w_use_frac = _build_schedule_ratios(schedule_entries, self.time)
        
        T_tank_w_K = cu.C2K(T_tank_w_init_C)

        is_on_prev = False
        for n in tqdm(range(tN), desc="ASHPB Simulating"):
            
            step_results = {}
            T_tank_w = cu.K2C(T_tank_w_K)
            T_oa = T_oa_schedule[n]
            T_oa_K = cu.C2K(T_oa)

            Q_tank_loss = self.UA_tank * (T_tank_w_K - T_oa_K)
            den = max(1e-6, T_tank_w_K - self.T_sup_w_K)
            alp = min(1.0, max(0.0, self.T_serv_w_K - self.T_sup_w_K) / den)

            self.dV_w_serv = self.w_use_frac[n] * self.dV_w_serv_m3s 
            self.dV_w_sup_tank = alp * self.dV_w_serv
            self.dV_w_sup_mix = (1 - alp) * self.dV_w_serv 

            Q_use_loss = c_w * rho_w * self.dV_w_sup_tank * (T_tank_w_K - self.T_sup_w_K)
            total_loss = Q_tank_loss + Q_use_loss
            
            if T_tank_w <= self.T_tank_w_lower_bound: is_on = True
            elif T_tank_w >= self.T_tank_w_setpoint: is_on = False
            else: is_on = is_on_prev
            
            is_transitioning_off_to_on = (not is_on_prev) and is_on # False to True
            
            if heater_capacity_schedule is not None:
                Q_cond_load_n = heater_capacity_schedule[n] if is_on else 0.0
            elif heater_capacity_const is not None:
                Q_cond_load_n = heater_capacity_const if is_on else 0.0
            elif heater_capacity_auto: 
                Q_cond_load_n = total_loss if is_on else 0.0
            
            is_on_prev = is_on
            
            # OFF 상태 조기 체크: Q_cond_load_n이 임계값 이하이면 최적화 건너뛰기
            if abs(Q_cond_load_n) <= self.Q_cond_load_threshold:
                
                result = self._calc_off_state(
                    T_tank_w=T_tank_w,
                    T_oa=T_oa
                )
            else:
                # 최적화 과정 진행
                bounds = [(1.0, 30.0), (1.0, 30.0)]  # [dT_ref_cond, dT_ref_evap]
                initial_guess = [5.0, 20.0]  
                
                def _cond_LMTD_constraint(x):
                    """
                    응축기 LMTD 제약 조건 함수.
                    perf가 None이면 (수렴 실패 등) 큰 제약 조건 위반 값을 반환하여
                    최적화 알고리즘이 해당 변수 조합을 피하고 다른 조합을 시도하도록 함.
                    """
                    try:
                        perf = self._calc_on_state(
                            optimization_vars=x,
                            T_tank_w=T_tank_w,
                            Q_cond_load=Q_cond_load_n,
                            T_oa=T_oa
                        )
                        # perf가 None이면 수렴 실패로 간주하여 큰 제약 조건 위반 값 반환
                        # 최적화 알고리즘이 다른 변수 조합을 시도하도록 함
                        if perf is None or not isinstance(perf, dict):
                            return 1e6
                        
                        # 필요한 키가 없거나 값이 nan이면 큰 제약 조건 위반 값 반환
                        if "Q_LMTD_cond" not in perf or np.isnan(perf["Q_LMTD_cond"]):
                            return 1e6
                        
                        # 제약 조건: Q_LMTD_cond - Q_cond_load_n >= 0 (Q_LMTD_cond >= Q_cond_load_n, 목표 부하보다 크거나 같으면 만족)
                        return perf["Q_LMTD_cond"] - Q_cond_load_n
                    except Exception as e:
                        # 예외 발생 시 큰 제약 조건 위반 값 반환
                        return 1e6

                def _evap_LMTD_constraint(x):
                    """
                    증발기 LMTD 제약 조건 함수.
                    perf가 None이면 (수렴 실패 등) 큰 제약 조건 위반 값을 반환하여
                    최적화 알고리즘이 해당 변수 조합을 피하고 다른 조합을 시도하도록 함.
                    """
                    try:
                        perf = self._calc_on_state(
                            optimization_vars=x,
                            T_tank_w=T_tank_w,
                            Q_cond_load=Q_cond_load_n,
                            T_oa=T_oa
                        )
                        # perf가 None이면 수렴 실패로 간주하여 큰 제약 조건 위반 값 반환
                        # 최적화 알고리즘이 다른 변수 조합을 시도하도록 함
                        if perf is None or not isinstance(perf, dict):
                            return 1e6
                        
                        # 필요한 키가 없거나 값이 nan이면 큰 제약 조건 위반 값 반환
                        if ("Q_LMTD_evap" not in perf or "Q_ref_evap" not in perf or
                            np.isnan(perf["Q_LMTD_evap"]) or np.isnan(perf["Q_ref_evap"])):
                            return 1e6
                        
                        # 제약 조건: Q_LMTD_evap - Q_ref_evap = 0 (0이면 만족)
                        return perf["Q_LMTD_evap"] - perf['Q_ref_evap']
                    except Exception as e:
                        # 예외 발생 시 큰 제약 조건 위반 값 반환
                        return 1e6
                    
                const_funcs = [
                    {'type': 'ineq', 'fun': _cond_LMTD_constraint},
                    {'type': 'ineq', 'fun': _evap_LMTD_constraint},
                               ]

                def _objective(x): # x = [dT_ref_evap, dT_ref_cond]
                    """
                    목적 함수: E_tot (총 전력 소비) 최소화.
                    perf가 None이면 (수렴 실패 등) 큰 penalty 값을 반환하여
                    최적화 알고리즘이 해당 변수 조합을 피하고 다른 조합을 시도하도록 함.
                    """
                    try:
                        perf = self._calc_on_state(
                            optimization_vars=x,
                            T_tank_w=T_tank_w,
                            Q_cond_load=Q_cond_load_n,
                            T_oa=T_oa
                        )
                        # perf가 None이면 수렴 실패로 간주하여 큰 penalty 반환
                        # 최적화 알고리즘이 다른 변수 조합을 시도하도록 함
                        if perf is None or not isinstance(perf, dict):
                            return 1e6
                        
                        # E_tot가 없거나 nan이면 큰 penalty 반환
                        if "E_tot" not in perf or np.isnan(perf["E_tot"]):
                            return 1e6
                        
                        # 목적 함수: E_tot 최소화
                        return perf["E_tot"]
                    except Exception as e:
                        # 예외 발생 시 큰 penalty 반환
                        return 1e6  
                
                opt_result = minimize(
                    _objective   ,           # 목적 함수 (E_tot = E_cmp + E_fan_ou 최소화)
                    initial_guess,           # 초기 추정값
                    method       ='SLSQP',   # Sequential Least Squares Programming
                    bounds       = bounds,    # 변수 경계 조건
                    constraints  = const_funcs,
                    options      = {'disp': False, 'maxiter': 100} 
                )
                    
                # opt_result.x로 결과 계산 시도 (성공/실패 무관)
                result = None
                try:
                    result = self._calc_on_state(
                        optimization_vars=opt_result.x,
                        T_tank_w=T_tank_w,
                        Q_cond_load=Q_cond_load_n,
                        T_oa=T_oa
                    )
                except Exception:
                    pass
                
                # result 검증 및 fallback
                if result is None or not isinstance(result, dict):
                    try:
                        result = self._calc_off_state(
                            T_tank_w=T_tank_w,
                            T_oa=T_oa
                        )
                    except Exception:
                        result = {
                            'is_on': False,
                            'converged': False,
                            'Q_ref_cond': 0.0,
                            'Q_ref_evap': 0.0,
                            'E_cmp': 0.0,
                            'E_fan_ou': 0.0,
                            'E_tot': 0.0,
                            'T_tank_w': T_tank_w,
                            'T_oa': T_oa
                        }
                
                # converged 플래그 설정
                if result is not None and isinstance(result, dict):
                    result['converged'] = opt_result.success
            
            if is_transitioning_off_to_on:
                # OFF→ON 전환 시점: 이전 스텝의 값들을 그대로 유지하여 저장
                if len(results_data) > 0:
                    step_results.update(results_data[-1].copy())
                    step_results['is_on'] = is_on
                else:
                    step_results.update(result)
                    step_results['is_on'] = is_on
            else:
                step_results.update(result)
                step_results['is_on'] = is_on
            
            if n < tN - 1:
                T_tank_w_K = update_tank_temperature(
                    T_tank_w_K = T_tank_w_K,
                    Q_tank_in  = result.get('Q_ref_cond', 0.0),
                    total_loss = Q_tank_loss + Q_use_loss,
                    C_tank     = self.C_tank,
                    dt         = self.dt
                ) 

            if save_ph_diagram: # P-h 선도 저장
                if snapshot_save_path is None:
                    raise ValueError("snapshot_save_path must be provided when save_ph_diagram is True.")
                # 전환 시점이 아닐 때만 P-h 선도 저장 (전환 시점은 이전 스텝 값이므로)
                if not is_transitioning_off_to_on:
                    plot_cycle_diagrams(
                        result=result,
                        refrigerant=self.ref,
                        show=False,
                        show_temp_limits=True,
                        save_path=snapshot_save_path+f'/{n:04d}.png',
                        T_tank_w=T_tank_w,
                        T_oa=T_oa,  # 실외 공기 온도 전달 (지중 관련 파라미터 제거)
                    )
            results_data.append(step_results)
            
        results_df = pd.DataFrame(results_data)

        if result_save_csv_path:
            results_df.to_csv(result_save_csv_path, index=False)

        return results_df

