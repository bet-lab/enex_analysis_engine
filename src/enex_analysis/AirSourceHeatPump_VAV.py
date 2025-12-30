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

# Import constants from constants.py
from .constants import (
    c_a, rho_a, k_a, c_w, rho_w, mu_w, k_w, sigma, k_D, k_d, ex_eff_NG, SP
)

# Import functions from enex_functions.py
from .enex_functions import (
    calc_lmtd_one_fluid_constant_temp,
    get_fan_flow_for_target_heat,
    calculate_fan_power,
)

@dataclass
class AirSourceHeatPump_VAV:
    '''
    (미완성)###################################################################################################################################
    
    물리적 원리에 기반한 공기원 히트펌프(ASHP) 성능 계산 및 최적 운전점 탐색 클래스.
    
    GroundSourceHeatPumpBoiler와 동일한 구조로 설계되어, 최적화 변수 [dT_ref_iu, dT_ref_evap]를
    사용하여 압축기 전력 + 실내기 팬 전력 + 실외기 팬 전력을 최소화하는 최적 운전점을 탐색합니다.
    #######################################################################################################################################
    '''
    def __init__(self,
                 # 1. 냉매/사이클/압축기 파라미터
                 ref         = 'R410A',
                 V_disp_cmp  = 0.0005,
                 eta_cmp_isen = 0.7,

                # 2. 열교환기 파라미터
                A_iu        = 15.0,   # 실내기 전열 면적 [m²]
                A_ou        = 20.0,   # 실외기 전열 면적 [m²]
                UA_iu       = 150.0,  # 실내기 열전달 계수 [W/K] (상수, GSHP와 동일한 방식)
                UA_ou       = 160.0,  # 실외기 열전달 계수 [W/K] (상수, GSHP와 동일한 방식)
                
                # A_cross는 더 이상 사용하지 않지만 호환성을 위해 유지
                A_cross_iu  = 0.03,   # 실내기 단면적 [m²] (deprecated)
                A_cross_ou  = 0.04,   # 실외기 단면적 [m²] (deprecated)

                 # 3. 실내기 팬 파라미터
                 dV_iu_design = 2.0,   # 실내기 설계 풍량 [m³/s]
                 dP_iu_design = 500.0, # 실내기 설계 정압 [Pa]

                 ############################################################
                 # 우선 고려하지 않는 변수들 추후 완성되면 추가 구현 예정
                 eta_motor_iu = 0.8,   # 실내기 모터 효율 [-]
                 eta_fan_iu   = 0.8,   # 실내기 팬 효율 [-]
                 f_motor_air_iu = 0.8, # 실내기 모터 발열 공기전달비율 [-]
                 ############################################################

                 # 4. 실외기 팬 파라미터
                 dV_ou_design = 2.5,   # 실외기 설계 풍량 [m³/s]
                 dP_ou_design = 500.0, # 실외기 설계 정압 [Pa]

                 ############################################################
                 # 우선 고려하지 않는 변수들 추후 완성되면 추가 구현 예정
                 eta_motor_ou = 0.8,   # 실외기 모터 효율 [-] 
                 eta_fan_ou   = 0.8,   # 실외기 팬 효율 [-]
                 f_motor_air_ou = 0.8, # 실외기 모터 발열 공기전달비율 [-]
                 ############################################################
                 
                # 6. 기준 온도
                T0 = 20  # 기준 온도 [°C] 
                 ):
        '''
        공기원 히트펌프(AirSourceHeatPump)의 고정 물리 파라미터를 초기화합니다.

        Args:
            ref (str): 사용할 냉매 이름 (CoolProp 형식).
            V_disp_cmp (float): 압축기 토출 스트로크 체적(1회전 당 흡입량) [m³].
            eta_cmp_isen (float): 압축기 단열 효율 [-].
            
            A_iu (float): 실내기(Indoor Unit, IU) 열교환기 전열 면적 [m²].
            A_ou (float): 실외기(Outdoor Unit, OU) 열교환기 전열 면적 [m²].
            UA_iu (float): 실내기 열전달 계수 [W/K] (상수, GSHP와 동일한 방식).
            UA_ou (float): 실외기 열전달 계수 [W/K] (상수, GSHP와 동일한 방식).
            
            A_cross_iu (float): 실내기 단면적 [m²] (deprecated, 호환성 유지용).
            A_cross_ou (float): 실외기 단면적 [m²] (deprecated, 호환성 유지용).

            dV_iu_design (float): 실내기 팬 설계 풍량 [m³/s] (ASHRAE 90.1 VSD Curve).
                기본값: 2.0 m³/s
            dP_iu_design (float): 실내기 팬 설계 정압 [Pa] (ASHRAE 90.1 VSD Curve).
                기본값: 500.0 Pa
            eta_motor_iu (float): 실내기 팬 모터 효율 [-].
            eta_fan_iu (float): 실내기 팬 효율 [-].
            f_motor_air_iu (float): 실내기 모터 발열의 공기 전달 비율 [-].

            dV_ou_design (float): 실외기 팬 설계 풍량 [m³/s] (ASHRAE 90.1 VSD Curve).
                기본값: 2.5 m³/s
            dP_ou_design (float): 실외기 팬 설계 정압 [Pa] (ASHRAE 90.1 VSD Curve).
                기본값: 500.0 Pa
            eta_motor_ou (float): 실외기 팬 모터 효율 [-].
            eta_fan_ou (float): 실외기 팬 효율 [-].
            f_motor_air_ou (float): 실외기 모터 발열의 공기 전달 비율 [-].

            T0 (float, optional): 기준 온도 [°C] (엑서지 계산용). 기본값: None

        Note:
            - Q_iu_load의 부호로 작동 모드를 자동 판별:
                Q_iu_load >= 0: 난방(실내기=응축기, 실외기=증발기)
                Q_iu_load < 0: 냉방(실내기=증발기, 실외기=응축기)

        Notes:
            - 팬 전력은 ASHRAE 90.1 Appendix G VSD Curve(EnergyPlus Variable Volume Fan Model)로 계산
            - 설계 팬 전력은 P_design = (dV_design * dP_design) / (효율)로 자동 산출
            - 팬 발열은 공기 엔탈피 증가로 반영되어 열교환기 입구 온도 보정에 활용
            - 열전달 계수는 상수 UA 값으로 단순화 (GSHP와 동일한 방식)
            - 수렴 안정성을 위해 풍량에 따른 U 변화를 고려하지 않음
        '''

        # --- 1. 냉매/사이클/압축기 파라미터 ---
        self.ref = ref
        self.V_disp_cmp = V_disp_cmp
        self.eta_cmp_isen = eta_cmp_isen
        
        # --- 2. 열교환기 파라미터 ---
        self.A_iu = A_iu
        self.A_ou = A_ou
        self.UA_iu = UA_iu  # 상수 UA 값 (GSHP와 동일한 방식)
        self.UA_ou = UA_ou  # 상수 UA 값 (GSHP와 동일한 방식)
        
        # A_cross는 더 이상 사용하지 않지만 호환성을 위해 유지
        self.A_cross_iu = A_cross_iu
        self.A_cross_ou = A_cross_ou
        
        # --- 3. 실내기 팬 파라미터 ---
        self.dV_iu_design = dV_iu_design
        self.dP_iu_design = dP_iu_design
        self.eta_motor_iu = eta_motor_iu
        self.eta_fan_iu = eta_fan_iu
        self.f_motor_air_iu = f_motor_air_iu
        
        # --- 4. 실외기 팬 파라미터 ---
        self.dV_ou_design = dV_ou_design
        self.dP_ou_design = dP_ou_design
        self.eta_motor_ou = eta_motor_ou
        self.eta_fan_ou = eta_fan_ou
        self.f_motor_air_ou = f_motor_air_ou
        
        # --- 5. 기준 온도 (엑서지 계산용) ---
        self.T0 = T0
        self.T0_K = cu.C2K(T0)
        
        # --- 6. ASHRAE 90.1 VSD Curve 팬 설계 파라미터 및 설계 전력 계산 ---
        # 실내기 팬 설계 파라미터
        self.dP_iu_design = dP_iu_design
        self.eta_motor_iu = eta_motor_iu
        self.eta_fan_iu = eta_fan_iu
        self.f_motor_air_iu = f_motor_air_iu
        
        # 실외기 팬 설계 파라미터
        self.dP_ou_design = dP_ou_design
        self.eta_motor_ou = eta_motor_ou
        self.eta_fan_ou = eta_fan_ou
        self.f_motor_air_ou = f_motor_air_ou
        
        # 설계 전력 계산: P_design = (V_design * ΔP_design) / η_tot
        # 실내기 팬 설계 전력 [W]
        self.fan_design_power_iu = (self.dV_iu_design * self.dP_iu_design) / (self.eta_motor_iu * self.eta_fan_iu)
        
        # 실외기 팬 설계 전력 [W]
        self.fan_design_power_ou = (self.dV_ou_design * self.dP_ou_design) / (self.eta_motor_ou * self.eta_fan_ou)
        
        # --- 7. 열전달 계수 파라미터 (상수 UA 사용) ---
        # 더 이상 사용하지 않지만 호환성을 위해 유지
        # self.U_coeff_iu, self.U_coeff_ou는 제거됨 (상수 UA 사용)
        
        # --- 8. 팬 파라미터 딕셔너리 생성 ---
        self.fan_params_iu = {
            'fan_design_flow_rate': self.dV_iu_design,
            'fan_design_power': self.fan_design_power_iu
        }
        self.fan_params_ou = {
            'fan_design_flow_rate': self.dV_ou_design,
            'fan_design_power': self.fan_design_power_ou
        }
        
        # --- 9. 압축기 체적 효율 (기본값) ---
        self.eta_cmp_vol = 1.0  # 기본값: 이상적 효율
        
        # --- 10. VSD Curve 계수 초기화 (기본값) ---
        self.c1 = 0.0013
        self.c2 = 0.1470
        self.c3 = 0.9506
        self.c4 = -0.0998
        self.c5 = 0.0
        
    def _calculate_ashp_next_step_cycle_dict(self, optimization_vars, T_ia, T_oa, Q_iu_load, **kwargs):
        """
        공기원 히트펌프(ASHP)의 사이클 성능을 계산하는 메서드.

        최적화 변수(optimization_vars)를 받아 히트펌프 사이클 성능을 계산하며,
        클래스별 결과 포맷팅까지 한번에 처리합니다.
        
        Args:
            - optimization_vars (list): 최적화 변수 [dT_ref_iu, dT_ref_evap]
            - T_ia (float): 실내기 평균 온도 [°C]
            - T_oa (float): 실외기 평균 온도 [°C]
            - Q_iu_load (float): 실내기 부하 [W]
            - **kwargs: 추가 파라미터
        
        Returns:
            - dict or None: 히트펌프 사이클 성능 딕셔너리
        """

        # 1. 모드 판단 (Q_iu_load 부호로 자동 판단)
        mode = 'heating' if Q_iu_load >= 0 else 'cooling'

        # 2. 최적화 변수 언패킹
        dT_ref_iu = optimization_vars[0]
        dT_ref_evap = optimization_vars[1]

        # 3. 온도 단위 변환 및 증발/응축 온도 계산
        T_ia_K = cu.C2K(T_ia)
        T_oa_K = cu.C2K(T_oa)

        if mode == 'heating':
            T_cond_K = T_ia_K + dT_ref_iu
            T_evap_K = T_oa_K - dT_ref_evap
        elif mode == 'cooling':
            T_evap_K = T_ia_K - dT_ref_iu
            T_cond_K = T_oa_K + dT_ref_evap
        else:
            raise ValueError("Invalid mode: must be 'heating' or 'cooling'")

        # 4. 공통 사이클 상태 계산
        cycle_states = compute_refrigerant_thermodynamic_states(
            T_evap_K     = T_evap_K,
            T_cond_K     = T_cond_K,
            refrigerant  = self.ref,
            eta_cmp_isen = self.eta_cmp_isen,
            T0_K         = self.T0_K,
            P0           = 101325,
            mode         = mode
        )

        # 5단계: 냉매 유량 및 성능 데이터 계산 (모드별)
        h1 = cycle_states['h1']
        h2 = cycle_states['h2']
        h3 = cycle_states['h3']
        h4 = cycle_states['h4']
        rho_cmp_in = cycle_states['rho']

        # 6. 부가 변수 계산
        if mode       == 'heating':
           m_dot_ref   = Q_iu_load / (h2 - h3)
           Q_ref_evap  = m_dot_ref * (h4 - h1)
           Q_ref_cond  = m_dot_ref * (h2 - h3)
           E_cmp       = m_dot_ref * (h2 - h1)
            
        elif mode       == 'cooling':
            m_dot_ref  = Q_iu_load / (h3 - h2)
            Q_ref_evap = m_dot_ref * (h3 - h2)
            Q_ref_cond = m_dot_ref * (h1 - h4)
            E_cmp      = m_dot_ref * (h1 - h2)
            
        cmp_rps = m_dot_ref / (rho_cmp_in * self.V_disp_cmp)
        cmp_rpm = cmp_rps * 60

        if m_dot_ref < 0:
            raise ValueError("m_dot_ref is negative")
        if Q_ref_evap > 0:
            raise ValueError("Q_ref_evap is positive")
        if Q_ref_cond < 0:
            raise ValueError("Q_ref_cond is negative")
        if E_cmp < 0:
            raise ValueError("E_cmp is negative")

        # 7. 목표 열교환량 만족을 위한 필요 풍량 계산
        T1_K = cycle_states['T1_K']
        T2_K = cycle_states['T2_K']
        T3_K = cycle_states['T3_K']
        T4_K = cycle_states['T4_K']
        
        if mode == 'heating':
            T_ref_cond_in  = T2_K
            T_ref_cond_out = T3_K
            T_ref_evap_in  = T4_K
            T_ref_evap_out = T1_K
            
            dV_fan_iu = get_fan_flow_for_target_heat(
                Q_ref_target = Q_ref_cond,
                T_air_in_C   = T_ia,
                T_ref_in_K   = T_ref_cond_in,
                T_ref_out_K  = T_ref_cond_out,
                A_cross      = self.A_cross_iu,
                UA_design    = self.UA_iu,
                dV_fan_design = self.dV_iu_design,
            )
            dV_fan_ou = get_fan_flow_for_target_heat(
                Q_ref_target = Q_ref_evap,
                T_air_in_C   = T_oa,
                T_ref_in_K   = T_ref_evap_in,
                T_ref_out_K  = T_ref_evap_out,
                A_cross      = self.A_cross_ou,
                UA_design    = self.UA_ou,
                dV_fan_design = self.dV_ou_design,
            )
            
        elif mode == 'cooling':
            T_ref_cond_in  = T1_K
            T_ref_cond_out = T4_K
            T_ref_evap_in  = T3_K
            T_ref_evap_out = T2_K
            
            dV_fan_iu = get_fan_flow_for_target_heat(
                Q_ref_target = Q_ref_evap,
                T_air_in_C   = T_ia,
                T_ref_in_K   = T_ref_evap_in,
                T_ref_out_K  = T_ref_evap_out,
                A_cross      = self.A_cross_iu,
                UA_design    = self.UA_iu,
                dV_fan_design = self.dV_iu_design,
            )
            dV_fan_ou = get_fan_flow_for_target_heat(
                Q_ref_target = Q_ref_cond,
                T_air_in_C   = T_oa,
                T_ref_in_K   = T_ref_cond_in,
                T_ref_out_K  = T_ref_cond_out,
                A_cross      = self.A_cross_ou,
                UA_design    = self.UA_ou,
                dV_fan_design = self.dV_ou_design,
            )
        

        if dV_fan_iu is None or dV_fan_ou is None:
            raise ValueError('Fan airflow calculation failed')

        # 7. 팬 전력 계산 (ASHRAE 90.1 VSD Curve)
        vsd_coeffs = {
            'c1': self.c1,
            'c2': self.c2,
            'c3': self.c3,
            'c4': self.c4,
            'c5': self.c5
        }
        fan_result_iu = calculate_fan_power(
            dV_fan_iu, 
            self.fan_params_iu,
            vsd_coeffs
        )
        E_fan_iu = fan_result_iu['required_power_W']

        fan_result_ou = calculate_fan_power(
            dV_fan_ou, 
            self.fan_params_ou,
            vsd_coeffs
        )
        E_fan_ou = fan_result_ou['required_power_W']
        

        # 8. 결과 포맷 생성(기존 format 함수 통합)
        T1_K = cycle_states['T1_K']
        P1   = cycle_states['P1']
        h1   = cycle_states['h1']
        s1   = cycle_states['s1']

        T2_K = cycle_states['T2_K']
        P2   = cycle_states['P2']
        h2   = cycle_states['h2']
        s2   = cycle_states['s2']

        T3_K = cycle_states['T3_K']
        P3   = cycle_states['P3']
        h3   = cycle_states['h3']
        s3   = cycle_states['s3']

        T4_K = cycle_states['T4_K']
        P4   = cycle_states['P4']
        h4   = cycle_states['h4']
        s4   = cycle_states['s4']

        x1 = cycle_states.get('x1', np.nan)
        x2 = cycle_states.get('x2', np.nan)
        x3 = cycle_states.get('x3', np.nan)
        x4 = cycle_states.get('x4', np.nan)

        E_tot = E_cmp + E_fan_iu + E_fan_ou
        cop = Q_iu_load / E_tot if E_tot > 0 else 0

        result = {
            'is_on': True,
            'mode': mode,

            # 열량
            'Q_ref_cond': Q_ref_cond,
            'Q_ref_evap': Q_ref_evap,
            'Q_iu_load': Q_iu_load,

            # 전력
            'E_cmp': E_cmp,
            'E_fan_iu': E_fan_iu,
            'E_fan_ou': E_fan_ou,
            'E_tot': E_tot,
            'cop': cop,
            
            # 압축기
            'cmp_rps': cmp_rps,
            'cmp_rpm': cmp_rpm,
            
            # 팬
            'fan_result_iu': fan_result_iu,
            'fan_result_ou': fan_result_ou,

            # 유량
            'm_dot_ref': m_dot_ref,
            'dV_fan_iu': dV_fan_iu,
            'dV_fan_ou': dV_fan_ou,

            # 온도
            'T_ia': T_ia,
            'T_oa': T_oa,
            'T_cond': cu.K2C(T_cond_K),
            'T_evap': cu.K2C(T_evap_K),
            'T1': cu.K2C(T1_K),
            'T2': cu.K2C(T2_K),
            'T3': cu.K2C(T3_K),
            'T4': cu.K2C(T4_K),

            # 압력
            'P1': P1,
            'P2': P2,
            'P3': P3,
            'P4': P4,

            # 엔탈피
            'h1': h1,
            'h2': h2,
            'h3': h3,
            'h4': h4,

            # 엔트로피
            's1': s1,
            's2': s2,
            's3': s3,
            's4': s4,

            # 엑서지
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'x4': x4,
        }

        return result
    
    def find_optimal_operation(self, Q_iu_load, T_ia, T_oa, **kwargs):
        '''
        주어진 실내기 부하와 실내/실외 온도 조건에서 총 전력 사용량을 최소화하는
        최적 운전점을 찾고 결과 딕셔너리를 반환합니다.
        
        내부 상태를 업데이트하지 않고, 주어진 조건에서 최적화를 수행한 후 결과만 반환합니다.
        
        목적 함수: E_total = E_cmp + E_fan_iu + E_fan_ou (최소화)
        최적화 변수: [dT_ref_iu, dT_ref_evap] (온도차 [K])
        제약 조건: Q_ref_cond - Q_iu_load = 0 (난방 모드) 또는 Q_ref_evap - Q_iu_load = 0 (냉방 모드)
        
        호출 관계:
        - 호출자: 사용자 코드, EnergyPlus 연동 코드
        - 호출 함수: _calculate_ashp_next_step_cycle_dict (본 클래스)
        
        Args:
            - Q_iu_load (float): 실내기 목표 열 교환율 [W]
                실내기에 전달해야 하는 목표 열량
                부호로 모드 자동 판단: >= 0 (난방), < 0 (냉방)
            - T_ia (float): 실내 공기 온도 [°C]
                현재 타임스텝의 실내 온도
            - T_oa (float): 실외 공기 온도 [°C]
                현재 타임스텝의 실외 온도
            - **kwargs: 추가 파라미터
                - bounds (list, optional): 최적화 변수 경계 조건
                  기본값: [(0.1, 30.0), (0.1, 30.0)] [K]
                - initial_guess (list, optional): 초기 추정값
                  기본값: [5.0, 5.0] [K]
        
        Returns:
            - dict: 타임스텝 계산 결과 딕셔너리
                성공 시: _calculate_ashp_next_step_cycle_dict의 반환값과 동일한 구조
                실패 시: {'is_on': False, 'mode': 'heating' 또는 'cooling'} 포함
                
                주요 키:
                - is_on (bool): 히트펌프 작동 여부
                - mode (str): 작동 모드 ('heating' 또는 'cooling')
                - Q_ref_cond, Q_ref_evap, Q_iu_load: 열량 [W]
                - E_cmp, E_fan_iu, E_fan_ou, E_tot: 전력 [W]
                - cop: 성능계수 [-]
                - m_dot_ref: 냉매 유량 [kg/s]
                - cmp_rpm: 압축기 회전수 [rpm]
                - dV_fan_iu, dV_fan_ou: 팬 풍량 [m³/s]
                - T_ia, T_oa, T_cond, T_evap: 온도 [°C]
                - T1, T2, T3, T4: 사이클 상태점 온도 [°C]
                - P1, P2, P3, P4: 사이클 상태점 압력 [Pa]
                - h1, h2, h3, h4: 사이클 상태점 엔탈피 [J/kg]
                - s1, s2, s3, s4: 사이클 상태점 엔트로피 [J/kgK]
                - x1, x2, x3, x4: 사이클 상태점 엑서지 [J/kg] (T0_K가 설정된 경우)
        
        Notes:
            - 내부 상태를 업데이트하지 않으므로, 외부 시뮬레이터와 연동 시 상태 관리는 호출자가 담당
            - 최적화 실패 시에도 기본 정보(mode 등)를 포함한 딕셔너리를 반환
            - GroundSourceHeatPumpBoiler의 run_simulation과 달리, 타임스텝 간 상태 전달 없음
            - SLSQP 알고리즘 사용
        '''
        # 경계 조건 및 초기 추정값 설정
        # 물리적 제약을 고려한 bounds 설정
        # 난방 모드: T_cond = T_ia + dT_ref_iu, T_evap = T_oa - dT_ref_evap
        #   - 응축기: T_ref_out > T_air_ou_in → T_cond > T_ia (항상 만족)
        #   - 증발기: T_ref_in < T_air_ou_out → T_evap < T_oa (항상 만족)
        # 냉방 모드: T_evap = T_ia - dT_ref_iu, T_cond = T_oa + dT_ref_evap
        #   - 증발기: T_ref_out < T_air_ou_in → T_evap < T_ia (항상 만족)
        #   - 응축기: T_ref_in > T_air_ou_out → T_cond > T_oa (항상 만족)
        # 일반적으로 dT_ref는 1~20K 범위가 적절함
        bounds = kwargs.get('bounds', [(1.0, 20.0), (1.0, 20.0)])  # [K]
        initial_guess = kwargs.get('initial_guess', [5.0, 5.0])  # [K]
        
        # 1. 목적 함수: 총 전력 사용량 (최소화 대상)
        def objective(x):  # x = [dT_ref_iu, dT_ref_evap]
            try:
                result = self._calculate_ashp_next_step_cycle_dict(
                    optimization_vars = x,
                    T_ia              = T_ia,
                    T_oa              = T_oa,
                    Q_iu_load         = Q_iu_load
                )
                if result is None:
                    error_msg = (
                        f'사이클 성능 계산 실패: result가 None입니다.\n'
                        f'  optimization_vars: {x}\n'
                        f'  T_ia: {T_ia:.2f} [°C], T_oa: {T_oa:.2f} [°C]\n'
                        f'  Q_iu_load: {Q_iu_load:.2f} [W]'
                    )
                    print(f'ERROR: {error_msg}')
                    raise ValueError(error_msg)
                return result['E_tot']
            except Exception as e:
                error_msg = (
                    f'목적 함수 계산 중 오류 발생: {type(e).__name__}: {str(e)}\n'
                    f'  optimization_vars: {x}\n'
                    f'  T_ia: {T_ia:.2f} [°C], T_oa: {T_oa:.2f} [°C]\n'
                    f'  Q_iu_load: {Q_iu_load:.2f} [W]'
                )
                print(f'ERROR: {error_msg}')
                raise ValueError(error_msg) from e
        
        # 2. 제약 조건: 계산된 실내기 열량이 목표 부하와 같아야 함
        def constraint(x):
            try:
                result = self._calculate_ashp_next_step_cycle_dict(
                    optimization_vars = x,
                    T_ia              = T_ia,
                    T_oa              = T_oa,
                    Q_iu_load         = Q_iu_load
                )
                if result is None:
                    error_msg = (
                        f'사이클 성능 계산 실패: result가 None입니다.\n'
                        f'  optimization_vars: {x}\n'
                        f'  T_ia: {T_ia:.2f} [°C], T_oa: {T_oa:.2f} [°C]\n'
                        f'  Q_iu_load: {Q_iu_load:.2f} [W]'
                    )
                    print(f'ERROR: {error_msg}')
                    raise ValueError(error_msg)
                
                # 모드에 따라 제약 조건 다르게 설정 (Q_iu_load 부호로 판단)
                mode = 'heating' if Q_iu_load >= 0 else 'cooling'
                if mode == 'heating':
                    return result['Q_ref_cond'] - Q_iu_load
                elif mode == 'cooling':
                    return result['Q_ref_evap'] - Q_iu_load
            except Exception as e:
                error_msg = (
                    f'제약 조건 계산 중 오류 발생: {type(e).__name__}: {str(e)}\n'
                    f'  optimization_vars: {x}\n'
                    f'  T_ia: {T_ia:.2f} [°C], T_oa: {T_oa:.2f} [°C]\n'
                    f'  Q_iu_load: {Q_iu_load:.2f} [W]'
                )
                print(f'ERROR: {error_msg}')
                raise ValueError(error_msg) from e
        
        # 제약 조건 설정
        cons = ({'type': 'eq', 'fun': constraint})
        
        # 최적화 실행 (SLSQP 알고리즘 사용)
        try:
            opt_result = minimize(
                objective,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'disp': False, 'maxiter': 1000}
            )
        except ValueError as e:
            # 물리적 타당성 위반 또는 계산 실패
            mode = 'heating' if Q_iu_load >= 0 else 'cooling'
            error_msg = (
                f'최적화 과정에서 오류 발생: {str(e)}\n'
                f'  T_ia: {T_ia:.2f} [°C], T_oa: {T_oa:.2f} [°C]\n'
                f'  Q_iu_load: {Q_iu_load:.2f} [W]\n'
                f'  mode: {mode}'
            )
            print(f'ERROR: {error_msg}')
            return {
                'is_on': False,
                'mode': mode,
                'error': error_msg
            }
        except Exception as e:
            # 기타 예상치 못한 오류
            mode = 'heating' if Q_iu_load >= 0 else 'cooling'
            error_msg = (
                f'최적화 과정에서 예상치 못한 오류 발생: {type(e).__name__}: {str(e)}\n'
                f'  T_ia: {T_ia:.2f} [°C], T_oa: {T_oa:.2f} [°C]\n'
                f'  Q_iu_load: {Q_iu_load:.2f} [W]\n'
                f'  mode: {mode}'
            )
            print(f'ERROR: {error_msg}')
            return {
                'is_on': False,
                'mode': mode,
                'error': error_msg
            }
        
        if opt_result.success:
            # 최적 운전점에서의 성능 계산
            print(opt_result.x)
            try:
                optimal_result = self._calculate_ashp_next_step_cycle_dict(
                    optimization_vars = opt_result.x,
                    T_ia              = T_ia,
                    T_oa              = T_oa,
                    Q_iu_load         = Q_iu_load
                )
                
                if optimal_result is None:
                    error_msg = 'optimal_result is None after successful optimization'
                    print(f'ERROR: {error_msg}')
                    raise ValueError(error_msg)
                
                return optimal_result
            except Exception as e:
                error_msg = (
                    f'최적 운전점 계산 중 오류 발생: {type(e).__name__}: {str(e)}\n'
                    f'  optimization_vars: {opt_result.x}\n'
                    f'  T_ia: {T_ia:.2f} [°C], T_oa: {T_oa:.2f} [°C]\n'
                    f'  Q_iu_load: {Q_iu_load:.2f} [W]'
                )
                print(f'ERROR: {error_msg}')
                mode = 'heating' if Q_iu_load >= 0 else 'cooling'
                return {
                    'is_on': False,
                    'mode': mode,
                    'error': error_msg
                }
        else:
            # 최적화 실패 시
            mode = 'heating' if Q_iu_load >= 0 else 'cooling'
            error_msg = f'Optimization failed: {opt_result.message}'
            print(f'ERROR: 최적화 실패 - {error_msg}')
            return {
                'is_on': False,
                'mode': mode,
                'error': error_msg
            }

