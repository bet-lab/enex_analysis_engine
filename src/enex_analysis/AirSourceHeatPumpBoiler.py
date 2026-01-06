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
    update_tank_temperature,
    calc_exergy_flow,
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
        dV_ou_design = 2.5,   # 실외기 설계 풍량 [m3/s] (정풍량)
        dP_ou_design = 500.0, # 실외기 설계 정압 [Pa]
        A_cross_ou = 0.4,         # 실외기 단면적 [m²]
        eta_fan_ou_design   = 0.8,   # 실외기 팬 효율 [-]

        # 4. 탱크/제어/부하 파라미터 -----------------------------------
        T_tank_w_setpoint     = 65.0,   # [°C] 저탕조 설정 온도
        T_tank_w_lower_bound  = 55.0,   # [°C] 저탕조 하한 온도
        T_serv_w              = 40.0,   # [°C] 서비스 급탕 온도
        T_sup_w               = 15.0,   # [°C] 급수(상수도) 온도

        heater_capacity       = 8000.0,    # [W] 히터 최대 용량
        dV_w_serv_m3s         = 0.0001,    # [m3/s] 최대 급탕 유량

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

    def _calc_on_state(self, optimization_vars, T_tank_w, Q_cond_load, T0):
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
        [optimization_vars, T_tank_w, Q_cond_load, T0]
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
            
            T0 (float): 엑서지 분석 기준 온도(=외기온도) [°C]
                엑서지 계산의 기준점으로 사용되는 외기 온도
        
        Returns:
            dict: 사이클 성능 결과 딕셔너리
                - 사이클 상태값 (P1-4, T1-4, h1-4, s1-4, x1-4)
                - 열량 (Q_ref_cond, Q_ref_evap, Q_LMTD_cond)
                - 전력 (E_cmp, E_fan_ou)
                - 유량 (m_dot_ref, dV_fan_ou, dV_w_serv, dV_w_sup_tank, dV_w_sup_mix)
                - 온도 (T0, T1-4, T_tank_w, T_serv_w, T_sup_w, T_air_ou_out)
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
        T0_K = cu.C2K(T0)                 
        
        T_evap_K = T0_K - dT_ref_evap       
        T_cond_K = T_tank_w_K + dT_ref_cond 
        
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
        
        rho_ref_cmp_in = cycle_states['rho']
        
        # 5단계: 냉매 유량 및 성능 데이터 계산
        m_dot_ref = Q_cond_load / (h2 - h3) 
        Q_ref_cond = m_dot_ref * (h2 - h3) 
        Q_ref_evap = m_dot_ref * (h1 - h4) 
        E_cmp = m_dot_ref * (h2 - h1)      
        cmp_rps = m_dot_ref / (self.V_disp_cmp * rho_ref_cmp_in) 
        
        # 6단계: 실외기 열교환기 성능 계산
        T_ref_evap_avg_K = (T4_K + T1_K) / 2
        T_air_ou_in_K = T0_K
        
        # Q_cond_load가 0에 가까우면 열교환기 계산 생략 (root_scalar 에러 방지)
        if abs(Q_cond_load) < self.Q_cond_load_threshold:
            # 0 값으로 채운 결과 설정
            dV_fan_ou = 0.0
            UA_evap = self.UA_evap_design
            T_air_ou_out_K = T0_K  # 열교환이 없으므로 입구 온도와 동일
            LMTD_evap = 0.0
            Q_LMTD_evap = 0.0
        
        
        else:
            HX_perf_ou_dict = calc_HX_perf_for_target_heat(
                Q_ref_target=Q_ref_evap,
                T_air_in_C=T0,
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
        
        fan_eff = self.eta_fan_ou_design * dV_fan_ou / E_fan_ou
        
        # 9단계: 엑서지 계산
        P0 = 101325
        h0 = CP.PropsSI('H', 'T', T0_K, 'P', P0, self.ref)
        s0 = CP.PropsSI('S', 'T', T0_K, 'P', P0, self.ref)
        
        # 기본 엑서지 값 (단위 질량당)
        x1 = (h1-h0) - T0_K*(s1 - s0)
        x2 = (h2-h0) - T0_K*(s2 - s0)
        x3 = (h3-h0) - T0_K*(s3 - s0)
        x4 = (h4-h0) - T0_K*(s4 - s0)
        
        # 냉매 유량 기반 엑서지
        X1 = m_dot_ref * x1
        X2 = m_dot_ref * x2
        X3 = m_dot_ref * x3
        X4 = m_dot_ref * x4
        X_cmp = E_cmp  # 압축기 엑서지 입력
        X_fan_ou = E_fan_ou  # 팬 엑서지 입력
        
        # 실외 공기 및 응축기 엑서지
        X_a_ou_in = calc_exergy_flow(G=c_a * rho_a * dV_fan_ou, T=T_air_ou_in_K, T0=T0_K)
        X_a_ou_out = calc_exergy_flow(G=c_a * rho_a * dV_fan_ou, T=T_air_ou_out_K, T0=T0_K)
        X_ref_cond = (1 - T0_K / T_tank_w_K) * Q_ref_cond  # 응축기(냉매→저탕조) 엑서지 유입량
        
        # 탱크 및 믹싱밸브 엑서지
        dV_w_sup_tank = self.dV_w_sup_tank if hasattr(self, 'dV_w_sup_tank') else 0.0
        dV_w_sup_mix = self.dV_w_sup_mix if hasattr(self, 'dV_w_sup_mix') else 0.0
        dV_w_serv = self.dV_w_serv if hasattr(self, 'dV_w_serv') else 0.0
        
        # 실제 서비스 온도 계산 (믹싱 밸브)
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
        
        X_tank_sup_w = calc_exergy_flow(G=c_w * rho_w * dV_w_sup_tank, T=self.T_sup_w_K, T0=T0_K)
        X_tank_w = calc_exergy_flow(G=c_w * rho_w * dV_w_sup_tank, T=T_tank_w_K, T0=T0_K)
        X_mix_sup_w = calc_exergy_flow(G=c_w * rho_w * dV_w_sup_mix, T=self.T_sup_w_K, T0=T0_K)
        X_mix_serv_w = calc_exergy_flow(G=c_w * rho_w * dV_w_serv, T=T_serv_w_actual_K, T0=T0_K)
        
        # 컴포넌트별 엑서지 손실
        Xc_evap = (X_a_ou_in + X4) - X1  # 증발기 전후 손실
        Xc_cmp = (X1 + X_cmp) - X2  # 압축기 통과 후 손실
        Xc_ref_cond = X2 - X3 - X_ref_cond  # 응축기에서의 손실
        Xc_exp = X3 - X4  # 팽창밸브 손실
        Xc_tank = X_ref_cond + X_tank_sup_w - X_tank_w  # 탱크 전후 손실
        Xc_mix = (X_tank_w + X_mix_sup_w - X_mix_serv_w)  # 믹싱밸브 엑서지 손실
        Xc_tot = Xc_evap + Xc_cmp + Xc_ref_cond + Xc_exp + Xc_tank + Xc_mix  # 총 엑서지 손실
        
        # 순환 흐름 변수들
        X_flow_X1_to_X2 = X1 + X_cmp
        X_flow_X2_to_X3 = X_flow_X1_to_X2 - X_ref_cond - Xc_ref_cond
        X_flow_X3_to_X4 = X_flow_X2_to_X3 - Xc_exp
        X_flow_X4_to_X1 = X_flow_X3_to_X4 + X_a_ou_in - Xc_evap
        
        # 총 입력 엑서지 및 효율
        X_tot = E_cmp + E_fan_ou  # 총 입력 엑서지
        X_eff = X_mix_serv_w / X_tot if X_tot > 0 else 0.0  # 엑서지 효율
        
        # 10단계: 최종 결과 딕셔너리 생성
        result = {
            'is_on': True,
            'converged': True,

            # === [온도: °C] =======================================
            'T_air_ou_in [°C]': cu.K2C(T_air_ou_in_K),
            'T_air_ou_out [°C]': cu.K2C(T_air_ou_out_K),
            'T1 [°C]': cu.K2C(T1_K),
            'T2 [°C]': cu.K2C(T2_K),
            'T3 [°C]': cu.K2C(T3_K),
            'T4 [°C]': cu.K2C(T4_K),
            'T_cond [°C]': cu.K2C(T2_K),
            'T_tank_w [°C]': T_tank_w,
            'T_sup_w [°C]': self.T_sup_w,
            'T_serv_w [°C]': T_serv_w_actual,
            'T0 [°C]': T0,

            # === [체적유량: m3/s] ==================================
            'dV_fan_ou [m3/s]': dV_fan_ou,
            'dV_w_serv [m3/s]': self.dV_w_serv if hasattr(self, 'dV_w_serv') else 0.0,
            'dV_w_sup_tank [m3/s]': self.dV_w_sup_tank if hasattr(self, 'dV_w_sup_tank') else 0.0,
            'dV_w_sup_mix [m3/s]': self.dV_w_sup_mix if hasattr(self, 'dV_w_sup_mix') else 0.0,

            # === [압력: Pa] ========================================
            'P1 [Pa]': P1,
            'P2 [Pa]': P2,
            'P3 [Pa]': P3,
            'P4 [Pa]': P4,

            # === [질량유량: kg/s] ==================================
            'm_dot_ref [kg/s]': m_dot_ref,

            # === [rpm] =============================================
            'cmp_rpm [rpm]': cmp_rps * 60,

            # === [엔탈피: J/kg] ====================================
            'h1 [J/kg]': h1,
            'h2 [J/kg]': h2,
            'h3 [J/kg]': h3,
            'h4 [J/kg]': h4,

            # === [엔트로피: J/(kg·K)] ==============================
            's1 [J/(kg·K)]': s1,
            's2 [J/(kg·K)]': s2,
            's3 [J/(kg·K)]': s3,
            's4 [J/(kg·K)]': s4,

            # === [엑서지 단위: J/kg] ===============================
            'x1 [J/kg]': x1,
            'x2 [J/kg]': x2,
            'x3 [J/kg]': x3,
            'x4 [J/kg]': x4,

            # === [에너지/엑서지: W] ================================
            # ---- 실외측(공기, 증발기)
            'E_fan_ou [W]': E_fan_ou,
            'X_a_ou_in [W]': X_a_ou_in,
            'X_a_ou_out [W]': X_a_ou_out,

            # ---- 증발기
            'Q_ref_evap [W]': Q_ref_evap,
            'Q_LMTD_evap [W]': Q_LMTD_evap,
            'Xc_evap [W]': Xc_evap,

            # ---- 냉매 엑서지 (상태점)
            'X1 [W]': X1,
            'X2 [W]': X2,
            'X3 [W]': X3,
            'X4 [W]': X4,

            # ---- 압축기
            'E_cmp [W]': E_cmp,
            'X_cmp [W]': X_cmp,
            'Xc_cmp [W]': Xc_cmp,

            # ---- 응축기/저탕조
            'Q_cond_load [W]': Q_cond_load,
            'Q_ref_cond [W]': Q_ref_cond,
            'Q_LMTD_cond [W]': Q_LMTD_cond,
            'X_ref_cond [W]': X_ref_cond,
            'Xc_ref_cond [W]': Xc_ref_cond,

            # ---- 팽창밸브
            'Xc_exp [W]': Xc_exp,

            # ---- 탱크
            'X_tank_sup_w [W]': X_tank_sup_w,
            'X_tank_w [W]': X_tank_w,
            'Xc_tank [W]': Xc_tank,

            # ---- 믹싱 밸브 & 온수 공급
            'X_mix_sup_w [W]': X_mix_sup_w,
            'X_mix_serv_w [W]': X_mix_serv_w,
            'Xc_mix [W]': Xc_mix,

            # ---- 총괄(손실, 총입력, 흐름 등)
            'Xc_tot [W]': Xc_tot,
            'E_tot [W]': E_cmp + E_fan_ou,
            'X_fan_ou [W]': X_fan_ou,
            'X_flow_X1_to_X2 [W]': X_flow_X1_to_X2,
            'X_flow_X2_to_X3 [W]': X_flow_X2_to_X3,
            'X_flow_X3_to_X4 [W]': X_flow_X3_to_X4,
            'X_flow_X4_to_X1 [W]': X_flow_X4_to_X1,
            'X_tot [W]': X_tot,

            # === [무차원: 효율, COP] ==============================
            'fan_eff [-]': fan_eff,
            'cop [-]': Q_cond_load / (E_cmp + E_fan_ou) if (E_cmp + E_fan_ou) > 0 else 0,
            'X_eff [-]': X_eff,
        }
        
        return result
    
    def _calc_off_state(self, T_tank_w, T0, prev_result=None):
        """
        OFF 상태 결과 포맷팅 함수.
        
        히트펌프가 OFF 상태일 때 사용되는 결과 딕셔너리를 생성합니다.
        모든 열량 및 전력 값은 0으로 설정하고, 냉매 상태값(온도, 압력, 엔탈피, 엔트로피, 엑서지)은
        이전 타임스텝의 값을 사용합니다. 이전 값이 없으면 포화점 기준으로 계산합니다.
        
        호출 관계:
        - 호출자: find_ref_loop_optimal_operation (cycle_performance.py)
            Q_cond_load가 임계값 이하일 때 호출
        
        주요 작업:
        1. ON 상태 템플릿 생성 (Q_cond_load=0으로 계산)
        2. 모든 숫자 값을 0으로 설정
        3. OFF 상태 플래그 및 필수 값 설정
        4. 이전 타임스텝의 냉매 상태값 사용 (있는 경우)
        5. 이전 값이 없으면 P-h 선도 플로팅용 포화점 계산
        
        Args:
            T_tank_w (float): 저탕조 온도 [°C]
                현재 타임스텝의 저탕조 온도
            
            T0 (float): 엑서지 분석 기준 온도(=외기온도) [°C]
                엑서지 계산의 기준점으로 사용되는 외기 온도
            
            prev_result (dict, optional): 이전 타임스텝의 결과 딕셔너리
                이전 타임스텝의 냉매 상태값을 포함하는 딕셔너리
                None이면 포화점 기준으로 계산하거나 0으로 설정
        
        Returns:
            dict: OFF 상태 결과 딕셔너리
                - 모든 열량 및 전력 값: 0.0
                - is_on: False
                - P1-4, h1-4, s1-4, T1-4, x1-4: 이전 타임스텝 값 또는 포화점 기준 계산값
                - 기타 상태값: 현재 시스템 상태 유지
        
        Notes:
            - 이전 타임스텝 값이 있으면 그 값을 사용합니다
            - 이전 값이 없으면 포화점 기준으로 계산합니다 (P-h 선도용)
            - 첫 번째 타임스텝이 OFF일 때는 냉매 상태값을 0으로 설정합니다
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
        
        
        # 2단계: 모든 숫자 값을 0.0으로 설정
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
        
        # 필요한 키 목록 정의
        required_keys = [
            'T1 [°C]', 'T2 [°C]', 'T3 [°C]', 'T4 [°C]',
            'P1 [Pa]', 'P2 [Pa]', 'P3 [Pa]', 'P4 [Pa]',
            'h1 [J/kg]', 'h2 [J/kg]', 'h3 [J/kg]', 'h4 [J/kg]',
            's1 [J/(kg·K)]', 's2 [J/(kg·K)]', 's3 [J/(kg·K)]', 's4 [J/(kg·K)]',
            'x1 [J/kg]', 'x2 [J/kg]', 'x3 [J/kg]', 'x4 [J/kg]'
        ]
        
        # 이전 결과에서 값 추출 (있는 경우)
        if prev_result is not None and all(key in prev_result for key in required_keys):
            # 이전 타임스텝의 냉매 상태값 사용
            refrigerant_state_values = {
                'T1 [°C]': prev_result['T1 [°C]'],
                'T2 [°C]': prev_result['T2 [°C]'],
                'T3 [°C]': prev_result['T3 [°C]'],
                'T4 [°C]': prev_result['T4 [°C]'],
                'P1 [Pa]': prev_result['P1 [Pa]'],
                'P2 [Pa]': prev_result['P2 [Pa]'],
                'P3 [Pa]': prev_result['P3 [Pa]'],
                'P4 [Pa]': prev_result['P4 [Pa]'],
                'h1 [J/kg]': prev_result['h1 [J/kg]'],
                'h2 [J/kg]': prev_result['h2 [J/kg]'],
                'h3 [J/kg]': prev_result['h3 [J/kg]'],
                'h4 [J/kg]': prev_result['h4 [J/kg]'],
                's1 [J/(kg·K)]': prev_result['s1 [J/(kg·K)]'],
                's2 [J/(kg·K)]': prev_result['s2 [J/(kg·K)]'],
                's3 [J/(kg·K)]': prev_result['s3 [J/(kg·K)]'],
                's4 [J/(kg·K)]': prev_result['s4 [J/(kg·K)]'],
                'x1 [J/kg]': prev_result['x1 [J/kg]'],
                'x2 [J/kg]': prev_result['x2 [J/kg]'],
                'x3 [J/kg]': prev_result['x3 [J/kg]'],
                'x4 [J/kg]': prev_result['x4 [J/kg]'],
            }
        else:
            # 이전 값이 없으면 nan으로 설정 (첫 번째 타임스텝이 OFF인 경우)
            import numpy as np
            refrigerant_state_values = {key: np.nan for key in required_keys}
            
            # P-h 선도 플로팅을 위한 포화점 계산 (참고용, 실제로는 0이 사용됨)
            # 증발기 측 포화 증기 (State 1, 4)
            P1_off = CP.PropsSI('P', 'T', T0_K, 'Q', 1, self.ref)
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
            
            'dV_w_serv [m3/s]': self.dV_w_serv,
            'dV_w_sup_tank [m3/s]': self.dV_w_sup_tank,
            'dV_w_sup_mix [m3/s]': self.dV_w_sup_mix,
            
            'T_serv_w [°C]': T_serv_w_actual,
            'T_sup_w [°C]': self.T_sup_w,
        })
        
        # 냉매 상태값 업데이트 (이전 값 또는 0)
        result.update(refrigerant_state_values)

        return result
    
    def _optimize_operation(self, T_tank_w, Q_cond_load, T0, method='SLSQP', callback=None):
        """
        히트펌프 최적 운전점 탐색을 수행하는 내부 메서드.
        
        이 메서드는 analyze_steady와 analyze_dynamic에서 공통으로 사용되는
        최적화 로직을 담당합니다. 두 단계로 구성됩니다:
        1. Phase 1: Feasible point 탐색 (제약 위반량 최소화)
        2. Phase 2: 본 최적화 (E_tot 최소화)
        
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
                - x: 최적화된 변수 [dT_ref_cond, dT_ref_evap]
                - success: 최적화 성공 여부
                - iteration_history: 반복 이력 리스트 (각 반복의 변수 값)
                - 기타 최적화 메타데이터
        
        Notes:
            - 최적화 변수: [dT_ref_cond, dT_ref_evap]
                - dT_ref_cond: 냉매-저탕조 온도차 [K]
                - dT_ref_evap: 냉매-실외 공기 온도차 [K]
            - Phase 2 제약 조건:
                - Q_cond_load <= Q_LMTD_cond <= Q_cond_load * (1 + tolerance) (응축기 열전달 능력 범위)
                - Q_ref_evap * (1 - tolerance) <= Q_LMTD_evap <= Q_ref_evap * (1 + tolerance) (증발기 열전달 밸런스 범위)
            - 목적 함수: E_tot (E_cmp + E_fan_ou) 최소화
        """
        # 요구사항 A: tolerance 변수 정의
        tolerance = 0.01  # 1%
        
        # 최적화 변수 경계 조건 및 초기 추정값 설정
        bounds = [(1.0, 30.0), (1.0, 30.0)]  # [dT_ref_cond, dT_ref_evap]
        initial_guess = [5.0, 20.0]
        
        # 반복 이력 추적을 위한 리스트 초기화
        iteration_history = []
        
        # 콜백 래퍼 함수 생성
        def _callback_wrapper(xk):
            """각 반복마다 호출되는 콜백 함수"""
            iteration_history.append({
                'iteration': len(iteration_history),
                'x': np.array(xk).copy(),
                'dT_ref_cond': xk[0],
                'dT_ref_evap': xk[1]
            })
            if callback is not None:
                callback(xk)
        
        # Phase 2: 본 최적화 (E_tot 최소화)
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

        def _evap_LMTD_constraint_low(x):
            """
            증발기 LMTD 제약 조건 함수 (하한): Q_LMTD_evap - Q_ref_evap*(1-tolerance) >= 0
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
                
                if ("Q_LMTD_evap [W]" not in perf or "Q_ref_evap [W]" not in perf or
                    np.isnan(perf["Q_LMTD_evap [W]"]) or np.isnan(perf["Q_ref_evap [W]"])):
                    return -1e6
                
                # 제약 조건: Q_LMTD_evap - Q_ref_evap*(1-tolerance) >= 0
                return perf["Q_LMTD_evap [W]"] - perf['Q_ref_evap [W]'] * (1 - tolerance)
            except Exception as e:
                return -1e6
        
        def _evap_LMTD_constraint_high(x):
            """
            증발기 LMTD 제약 조건 함수 (상한): Q_ref_evap*(1+tolerance) - Q_LMTD_evap >= 0
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
                
                if ("Q_LMTD_evap [W]" not in perf or "Q_ref_evap [W]" not in perf or
                    np.isnan(perf["Q_LMTD_evap [W]"]) or np.isnan(perf["Q_ref_evap [W]"])):
                    return -1e6
                
                # 제약 조건: Q_ref_evap*(1+tolerance) - Q_LMTD_evap >= 0
                return perf['Q_ref_evap [W]'] * (1 + tolerance) - perf["Q_LMTD_evap [W]"]
            except Exception as e:
                return -1e6
        
        # Phase 2 제약 조건: 응축기와 증발기 모두 두 개의 ineq 제약
        const_funcs = [
            {'type': 'ineq', 'fun': _cond_LMTD_constraint_low},   # Q_LMTD_cond - Q_cond_load >= 0
            {'type': 'ineq', 'fun': _cond_LMTD_constraint_high},  # Q_cond_load*(1+tolerance) - Q_LMTD_cond >= 0
            {'type': 'ineq', 'fun': _evap_LMTD_constraint_low},   # Q_LMTD_evap - Q_ref_evap*(1-tolerance) >= 0
            {'type': 'ineq', 'fun': _evap_LMTD_constraint_high},  # Q_ref_evap*(1+tolerance) - Q_LMTD_evap >= 0
        ]

        def _objective(x):  # x = [dT_ref_cond, dT_ref_evap]
            """
            Phase 2 목적 함수: E_tot (총 전력 소비) 최소화.
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
        
        # Phase 2 최적화 실행
        opt_result = minimize(
            _objective,         
            initial_guess,
            method=method,       
            bounds=bounds,  
            constraints=const_funcs,
            callback=_callback_wrapper if callback is not None else None,
            options=options
        )
        
        # 반복 이력 결과에 첨부
        opt_result.iteration_history = iteration_history
        
        return opt_result
    
    def analyze_steady(
        self,
        T_tank_w,
        T0,
        dV_w_serv=None,  # 급탕 유량 [m3/s], None이면 0으로 가정
        return_dict=True,  # True면 dict 반환, False면 DataFrame 반환
        optimization_method='SLSQP'  # 최적화 알고리즘 선택
    ):
        """
        정상상태 해석 함수.
        
        주어진 조건에서 히트펌프의 정상상태 성능을 계산합니다.
        저탕조의 에너지 밸런스를 만족하도록 Q_cond_load를 total_loss로 자동 계산합니다.
        
        Args:
            T_tank_w (float): 저탕조 온도 [°C]
            T0 (float): 엑서지 분석 기준 온도(=외기온도) [°C]
                엑서지 계산의 기준점으로 사용되는 외기 온도
            dV_w_serv (float, optional): 급탕 유량 [m3/s]. None이면 0으로 가정
            return_dict (bool): True면 dict 반환, False면 DataFrame 반환
        
        Returns:
            dict 또는 pd.DataFrame: 정상상태 해석 결과
                - run_simulation과 동일한 형식의 결과 딕셔너리
                - 최적화 실패 시 OFF 상태 결과 반환
        """
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
        print(f'alp: {alp:.2f}')
        
        self.dV_w_serv = dV_w_serv
        self.dV_w_sup_tank = alp * dV_w_serv
        self.dV_w_sup_mix = (1 - alp) * dV_w_serv
        
        # 열 손실 계산
        Q_use_loss = c_w * rho_w * self.dV_w_sup_tank * (T_tank_w_K - self.T_sup_w_K)
        total_loss = Q_tank_loss + Q_use_loss
        
        # 정상상태: Q_cond_load = total_loss (에너지 밸런스)
        Q_cond_load = total_loss
        print(f"Q_cond_load [W]: {Q_cond_load}")
        
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
                T0=T0
            )
        else:
            # 최적화 과정 진행
            opt_result = self._optimize_operation(
                T_tank_w=T_tank_w,
                Q_cond_load=Q_cond_load,
                T0=T0
            )
            
            # opt_result.x로 결과 계산 시도 (성공/실패 무관)
            result = None
            try:
                result = self._calc_on_state(
                    optimization_vars=opt_result.x,
                    T_tank_w=T_tank_w,
                    Q_cond_load=Q_cond_load,
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
        T0_schedule,
        result_save_csv_path=None,
        save_ph_diagram=False,
        snapshot_save_path=None,
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
        T0_schedule = np.array(T0_schedule)
        if len(T0_schedule) != tN:
            raise ValueError(f"T0_schedule length ({len(T0_schedule)}) must match time array length ({tN})")
        
        results_data = []
        
        self.time = time
        self.dt = dt_s
        
        self.dV_w_serv = 0.0 
        self.dV_w_sup_tank = 0.0 
        self.dV_w_sup_mix = 0.0 
        
        self.w_use_frac = _build_schedule_ratios(schedule_entries, self.time)
        
        T_tank_w_K = cu.C2K(T_tank_w_init_C)

        is_on_prev = False
        prev_result = None  # 이전 타임스텝 결과 저장용
        for n in tqdm(range(tN), desc="ASHPB Simulating"):
            
            step_results = {}
            T_tank_w = cu.K2C(T_tank_w_K)
            T0 = T0_schedule[n]
            T0_K = cu.C2K(T0)

            Q_tank_loss = self.UA_tank * (T_tank_w_K - T0_K)
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
            Q_cond_load_n = self.heater_capacity if is_on else 0.0
            is_on_prev = is_on
            
            # OFF 상태 조기 체크: Q_cond_load_n이 임계값 이하이면 최적화 건너뛰기
            if abs(Q_cond_load_n) <= self.Q_cond_load_threshold:
                
                result = self._calc_off_state(
                    T_tank_w=T_tank_w,
                    T0=T0,
                    prev_result=prev_result  # 이전 결과 전달
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
                            T0=T0,
                            prev_result=prev_result  # 이전 결과 전달
                        )
                    except Exception:
                        result = {
                            'is_on': False,
                            'converged': False,
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
                    result['converged'] = opt_result.success
            
            if is_transitioning_off_to_on:
                # OFF→ON 전환 시점: 실제 계산된 ON 상태 값 저장
                step_results.update(result)
                step_results['is_on'] = is_on
                # 전환 시점임을 표시하는 플래그 추가
                step_results['is_transitioning'] = True
            else:
                step_results.update(result)
                step_results['is_on'] = is_on
                step_results['is_transitioning'] = False
            
            if n < tN - 1:
                T_tank_w_K = update_tank_temperature(
                    T_tank_w_K = T_tank_w_K,
                    Q_tank_in  = result.get('Q_ref_cond [W]', 0.0),
                    total_loss = Q_tank_loss + Q_use_loss,
                    C_tank     = self.C_tank,
                    dt         = self.dt
                ) 

            if save_ph_diagram: # P-h 선도 저장
                if snapshot_save_path is None:
                    raise ValueError("snapshot_save_path must be provided when save_ph_diagram is True.")
                # 전환 시점이 아닐 때만 P-h 선도 저장 (전환 시점은 이전 스텝 값이므로)
                plot_cycle_diagrams(
                    result=result,
                    refrigerant=self.ref,
                    show=False,
                    show_temp_limits=True,
                    save_path=snapshot_save_path+f'/{n:04d}.png',
                    temp_limits=[('Tank water', T_tank_w)],
                    # T0=T0,  # 엑서지 분석 기준 온도(=외기온도) 전달
                )
            
            # 결과 저장 후 prev_result 업데이트 (다음 타임스텝에서 사용)
            if result is not None and isinstance(result, dict):
                prev_result = result.copy()
            
            results_data.append(step_results)
            
        results_df = pd.DataFrame(results_data)

        if result_save_csv_path:
            results_df.to_csv(result_save_csv_path, index=False)

        return results_df

