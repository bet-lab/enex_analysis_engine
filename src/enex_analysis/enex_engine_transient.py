#%%
import numpy as np
import math
from . import calc_util as cu
# import calc_util as cu
from dataclasses import dataclass
import dartwork_mpl as dm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import root_scalar
import CoolProp.CoolProp as CP
from tqdm import tqdm
from scipy.optimize import minimize
import pandas as pd
from . import cycle_performance as cp
from . import utility_functions as uf
import os

# Import constants and functions from utility_functions
from .utility_functions import (
    c_a, rho_a, k_a, c_w, rho_w, mu_w, k_w, sigma, k_D, k_d, ex_eff_NG, SP,
    calc_lmtd_constant_refrigerant_temp,
    calc_simple_tank_UA,
    G_FLS,
    _build_schedule_ratios,
    find_fan_airflow_for_heat_transfer,
    calculate_fan_power,
    calculate_heat_transfer_coefficient
)

@dataclass
class AirSourceHeatPump:
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
        cycle_states = cp.compute_refrigerant_thermodynamic_states(
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
            
            dV_fan_iu = find_fan_airflow_for_heat_transfer(
                Q_ref_target = Q_ref_cond,
                T_air_ou_in_C   = T_ia,
                T_ref_in_K   = T_ref_cond_in,
                T_ref_out_K  = T_ref_cond_out,
                UA           = self.UA_iu,
            )
            dV_fan_ou = find_fan_airflow_for_heat_transfer(
                Q_ref_target = Q_ref_evap,
                T_air_ou_in_C   = T_oa,
                T_ref_in_K   = T_ref_evap_in,
                T_ref_out_K  = T_ref_evap_out,
                UA           = self.UA_ou,
            )
            
        elif mode == 'cooling':
            T_ref_cond_in  = T1_K
            T_ref_cond_out = T4_K
            T_ref_evap_in  = T3_K
            T_ref_evap_out = T2_K
            
            dV_fan_iu = find_fan_airflow_for_heat_transfer(
                Q_ref_target = Q_ref_evap,
                T_air_ou_in_C   = T_ia,
                T_ref_in_K   = T_ref_evap_in,
                T_ref_out_K  = T_ref_evap_out,
                UA           = self.UA_iu,
            )
            dV_fan_ou = find_fan_airflow_for_heat_transfer(
                Q_ref_target = Q_ref_cond,
                T_air_ou_in_C   = T_oa,
                T_ref_in_K   = T_ref_cond_in,
                T_ref_out_K  = T_ref_cond_out,
                UA           = self.UA_ou,
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
        cycle_states = cp.compute_refrigerant_thermodynamic_states(
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
                return cp.find_ref_loop_optimal_operation(
                    calculate_performance_func=self._calculate_gshpb_next_step,
                    T_tank_w=cu.K2C(T_tank_w_K),
                    Q_cond_load=Q_cond_load_n,
                    Q_cond_LOAD_OFF_TOL=self.Q_cond_LOAD_OFF_TOL,
                    bounds=[(0.1, 30.0), (0.1, 30.0)],
                    initial_guess=[5.0, 5.0],
                    constraint_funcs=cp.create_lmtd_constraints(),
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
                    cp.plot_cycle_diagrams(
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
        UA_cond     = 500.0,  # 응축기 열전달 계수 [W/K] (상수)
        UA_evap     = 500.0,  # 증발기 열전달 계수 [W/K] (상수)

        # 3. 실외기 팬 파라미터 ---------------------------------------
        dV_ou_design = 2.5,   # 실외기 설계 풍량 [m³/s] (정풍량)
        dP_ou_design = 500.0, # 실외기 설계 정압 [Pa]
        eta_motor_ou = 0.8,   # 실외기 모터 효율 [-] 
        eta_fan_ou   = 0.8,   # 실외기 팬 효율 [-]

        # 4. 탱크/제어/부하 파라미터 -----------------------------------
        #    (온도/제어 관련)
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
        ):
        '''
        AirSourceHeatPumpBoiler 초기화.
        '''

        # --- 1. 냉매/사이클/압축기 파라미터 ---
        self.ref = ref
        self.V_disp_cmp = V_disp_cmp
        self.eta_cmp_isen = eta_cmp_isen
        
        # --- 2. 열교환기 파라미터 ---
        self.UA_cond = UA_cond
        self.UA_evap = UA_evap
        
        # --- 3. 실외기 팬 파라미터 ---
        self.dV_ou_design = dV_ou_design
        self.dP_ou_design = dP_ou_design
        self.eta_motor_ou = eta_motor_ou
        self.eta_fan_ou = eta_fan_ou
        
        # 팬 설계 전력 계산 (정풍량 기준)
        self.fan_design_power_ou = (self.dV_ou_design * self.dP_ou_design) / (self.eta_motor_ou * self.eta_fan_ou)
        
        # VSD Curve 계수 (정풍량이므로 사용하지 않지만 호환성 유지)
        self.c1 = 0.0013
        self.c2 = 0.1470
        self.c3 = 0.9506
        self.c4 = -0.0998
        self.c5 = 0.0
        
        # 팬 파라미터 딕셔너리
        self.fan_params_ou = {
            'fan_design_flow_rate': self.dV_ou_design,
            'fan_design_power': self.fan_design_power_ou
        }
        
        # --- 4. 기준 온도 ---
        self.T0 = T0
        self.T0_K = cu.C2K(T0)
        
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
        
        self.Q_cond_LOAD_OFF_TOL = 500.0

    def _calculate_ashpb_next_step(self, optimization_vars, T_tank_w, Q_cond_load, T_oa=20):
        """
        공기원 히트펌프 보일러(ASHPB)의 사이클 성능을 계산하는 메서드.
        
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
            - _format_ashpb_results_dict (본 클래스)
        
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
        _format_ashpb_results_dict
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
            
            **kwargs: 추가 파라미터
                - T_oa (float): 실외 공기 온도 [°C] (필수)
        
        Returns:
            dict: 사이클 성능 결과 (포맷팅된 딕셔너리)
                _format_ashpb_results_dict의 반환값과 동일
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
        """
        # ============================================================
        # 1단계: 최적화 변수 언패킹
        # ============================================================
        dT_ref_evap = optimization_vars[0]      # 냉매-실외 공기 온도차 [K]
        dT_ref_cond = optimization_vars[1]      # 냉매-저탕조 온도차 [K]
        
        # ============================================================
        # 2단계: 온도 단위 변환 및 증발/응축 온도 계산
        # ============================================================
        T_tank_w_K = cu.C2K(T_tank_w)         # 저탕조 온도 [K]
        T_oa_K = cu.C2K(T_oa)                 # 실외 공기 온도 [K]
        
        # 증발 온도 계산: 실외 공기 온도에서 냉매-실외 공기 온도차를 뺌
        T_evap_K = T_oa_K - dT_ref_evap        # 증발 온도 [K]
        
        # 응축 온도 계산: 저탕조 온도에 냉매-저탕조 온도차를 더함
        T_cond_K = T_tank_w_K + dT_ref_cond    # 응축 온도 [K]
        
        # ============================================================
        # 3단계: 공통 사이클 상태 계산
        # ============================================================
        # cycle_performance.py의 공통 함수 사용
        cycle_states = cp.compute_refrigerant_thermodynamic_states(
            T_evap_K=T_evap_K,
            T_cond_K=T_cond_K,
            refrigerant=self.ref,
            eta_cmp_isen=self.eta_cmp_isen,
            T0_K=self.T0_K,
            P0=101325,
            mode='heating'
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
        if abs(h2 - h3) < 1e-6:
            return None
        
        # 냉매 유량 계산: 목표 열 교환율을 만족하기 위한 필요한 유량
        m_dot_ref = Q_cond_load / (h2 - h3)  # 냉매 유량 [kg/s]
        # 사이클 열량 및 전력 계산
        Q_ref_cond = m_dot_ref * (h2 - h3)  # 응축기 열량 [W] (Q_cond_load와 동일해야 함)
        Q_ref_evap = m_dot_ref * (h1 - h4)  # 증발기 열량 [W] (실외 공기로부터 흡수)
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
        # ASHPB 클래스에 특화된 결과 딕셔너리로 포맷팅
        return self._format_ashpb_results_dict(cycle_states, performance_data, T_tank_w, Q_cond_load, T_oa)
    
    def _format_ashpb_results_dict(self, cycle_states, performance_data, 
                                         T_tank_w, Q_cond_load, T_oa):
        """
        AirSourceHeatPumpBoiler 클래스별 사이클 성능 결과 포맷팅 함수.
        
        이 함수는 공통 사이클 계산 결과(cycle_states, performance_data)를 받아
        ASHPB 클래스에 특화된 결과 딕셔너리로 포맷팅합니다.
        
        주요 작업:
        1. 사이클 상태값 추출 (State 1-4 물성치)
        2. 성능 데이터 추출 (열량, 전력, 유량 등)
        3. LMTD 기반 열량 계산 (응축기, 증발기)
        4. 실외 공기 온도 계산 (증발기 측)
        5. 팬 전력 계산 (정풍량 기준 고정값)
        6. 엑서지 계산
        7. 최종 결과 딕셔너리 생성
        
        호출 관계:
        - 호출자: _calculate_ashpb_next_step (본 클래스)
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
            
            T_oa (float): 실외 공기 온도 [°C]
                현재 타임스텝의 실외 공기 온도
        
        Returns:
            dict: ASHPB 클래스별 결과 딕셔너리
                - 사이클 상태값 (P1-4, T1-4, h1-4, s1-4, x1-4)
                - 열량 (Q_ref_cond, Q_ref_evap, Q_LMTD_cond, Q_LMTD_evap)
                - 전력 (E_cmp, E_fan_ou)
                - 유량 (m_dot_ref, dV_fan_ou, dV_w_serv, dV_w_sup_tank, dV_w_sup_mix)
                - 온도 (T0, T1-4, T_tank_w, T_serv_w, T_sup_w, T_oa, T_air_ou_out)
                - 기타 (cmp_rpm, is_on)
        
        Notes:
            - LMTD 계산은 열교환기 물리적 제약 조건을 반영
            - Q_LMTD_cond와 Q_ref_cond는 최적화에서 일치해야 함
            - Q_LMTD_evap와 Q_ref_evap는 최적화에서 일치해야 함
        """
        T_tank_w_K = cu.C2K(T_tank_w)
        T_oa_K = cu.C2K(T_oa)
        
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
        # 4단계: 증발기(실외 공기 측) LMTD 계산
        # ============================================================
        # 대향류(Counter-flow) 열교환기 모델
        # 실외 공기가 냉매로부터 열을 흡수하여 온도 하강
        
        # 실외 공기 출구 온도 계산 (에너지 보존)
        # 냉매 평균 온도 (증발기 측)
        T_ref_evap_avg_K = (T4_K + T1_K) / 2  # 냉매 평균 온도 [K]
        T_air_ou_in_K = T_oa_K
        T_air_ou_in = cu.K2C(T_air_ou_in_K)
        T_air_ou_out_K = T_air_ou_in_K + (1 - np.exp(-self.UA_evap / (c_a * rho_a * self.dV_ou_design))) * (T_ref_evap_avg_K - T_air_ou_in_K)
        T_air_ou_out = cu.K2C(T_air_ou_out_K)  # 실외 공기 출구 온도 [°C]
        ##############################################################################################################################
        # print(f"T_air_ou_in: {T_air_ou_in}")
        # print(f"T_air_ou_out: {T_air_ou_out}")
        ##############################################################################################################################
        
        # 증발기(실외 공기 측) LMTD 계산
        # 실외 공기는 T_air_ou_in_K → T_air_ou_out_K로 온도 하강
        # 냉매는 State 4(저온) → State 1(고온)로 온도 상승
        dT1_evap = T_air_ou_in_K - T_ref_evap_avg_K      # 증발기 입구 온도차 [K] (공기 입구 - 냉매 출구)
        dT2_evap = T_air_ou_out_K - T_ref_evap_avg_K # 증발기 출구 온도차 [K] (공기 출구 - 냉매 입구)

        # LMTD 계산 가능 여부 확인
        if dT1_evap <= 1e-6 or dT2_evap <= 1e-6 or abs(dT1_evap - dT2_evap) < 1e-6:
            print(f"dT error: dT1_evap: {dT1_evap}, dT2_evap: {dT2_evap}")
            # 물리적으로 불가능한 상태
            Q_LMTD_evap = np.inf
        else:
            # 대수 평균 온도차 (LMTD) 계산
            LMTD_evap = (dT1_evap - dT2_evap) / np.log(dT1_evap / dT2_evap)
            Q_LMTD_evap = self.UA_evap * LMTD_evap
            ##############################################################################################################################
            # print(f"LMTD_evap: {LMTD_evap}")
            # print(f"self.UA_evap: {self.UA_evap}")
            # print(f"Q_LMTD_evap: {Q_LMTD_evap}")
            ##############################################################################################################################
        
        # ============================================================
        # 5단계: 팬 전력 계산 (정풍량 기준 고정값)
        # ============================================================
        # 정풍량이므로 설계 전력을 그대로 사용
        E_fan_ou = self.fan_design_power_ou  # 팬 전력 [W]
        
        # ============================================================
        # 6단계: 엑서지 계산
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
            
            'Q_cond_load': Q_cond_load,
            'E_cmp': E_cmp,
            'E_fan_ou': E_fan_ou,
            'E_tot': E_cmp + E_fan_ou,
            'cop': Q_cond_load / (E_cmp + E_fan_ou) if (E_cmp + E_fan_ou) > 0 else 0,
            'm_dot_ref': m_dot_ref,
            'cmp_rpm': cmp_rps * 60,
            
            'dV_fan_ou': self.dV_ou_design,  # 정풍량
            'dV_w_serv': self.dV_w_serv if hasattr(self, 'dV_w_serv') else 0.0,
            'dV_w_sup_tank': self.dV_w_sup_tank if hasattr(self, 'dV_w_sup_tank') else 0.0,
            'dV_w_sup_mix': self.dV_w_sup_mix if hasattr(self, 'dV_w_sup_mix') else 0.0,
            
            'T_tank_w': T_tank_w,
            'T_serv_w': self.T_serv_w,
            'T_sup_w': self.T_sup_w,
            'T_oa': T_oa,
            'T_air_ou_out': T_air_ou_out,

            'T0': cu.K2C(T0_K),
            'T1': cu.K2C(T1_K),
            'T2': cu.K2C(T2_K),
            'T3': cu.K2C(T3_K),
            'T4': cu.K2C(T4_K),
            'T_cond': cu.K2C(T2_K),  # 응축 온도 [°C] (State 2 기준)
            'T_evap': cu.K2C(T4_K),  # 증발 온도 [°C] (State 4 기준)

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
    
    def _format_ashpb_off_results_dict(self, T_tank_w, Q_cond_load=None, T_oa=20.0):
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
        
        # ============================================================
        # 1단계: ON 상태 템플릿 생성
        # ============================================================
        # Q_cond_load=0.0으로 계산하여 딕셔너리 구조만 얻음
        # 실제 계산은 필요 없지만 결과 딕셔너리 구조를 얻기 위해 실행
        try:
            result_template = self._calculate_ashpb_next_step(
                optimization_vars=[5.0, 5.0],  # 임의의 값 (계산 결과는 무시됨)
                T_tank_w=T_tank_w,
                Q_cond_load=0.0,
                T_oa=T_oa  # 파라미터로 받은 T_oa 사용
            )
        except Exception:
            result_template = {}
        
        if result_template is None:
            result_template = {}
        
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
        result_template['T_tank_w'] = T_tank_w  # 저탕조 온도 유지
        result_template['T_oa'] = T_oa  # 실외 공기 온도 유지
        
        # 유량 값 유지 (히트펌프 OFF 여부와 무관)
        result_template['dV_w_serv'] = self.dV_w_serv if hasattr(self, 'dV_w_serv') else 0.0
        result_template['dV_w_sup_tank'] = self.dV_w_sup_tank if hasattr(self, 'dV_w_sup_tank') else 0.0
        result_template['dV_w_sup_mix'] = self.dV_w_sup_mix if hasattr(self, 'dV_w_sup_mix') else 0.0
        result_template['dV_fan_ou'] = self.dV_ou_design  # 정풍량 유지
        
        # 필수 키 확인 및 추가
        required_keys = {
            'Q_ref_cond': 0.0,
            'Q_ref_evap': 0.0,
            'Q_LMTD_cond': 0.0,
            'Q_LMTD_evap': 0.0,
            'Q_cond_load': 0.0,
            'E_cmp': 0.0,
            'E_fan_ou': 0.0,
            'E_tot': 0.0,
            'cop': 0.0,
            'm_dot_ref': 0.0,
            'cmp_rpm': 0.0,
            'T_serv_w': self.T_serv_w,
            'T_sup_w': self.T_sup_w,
            'T0': self.T0,
        }
        
        for key, default_value in required_keys.items():
            if key not in result_template:
                result_template[key] = default_value

        # ============================================================
        # 4단계: P-h 선도 플로팅을 위한 포화점 값 계산
        # ============================================================
        # OFF 상태에서도 P-h 선도를 그리기 위해 기본 사이클 상태값 계산
        # 증발기 측: 실외 공기 온도 기준 포화 증기
        # 응축기 측: 저탕조 온도 기준 포화 액체
        try:
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

            # OFF 상태 사이클: P1→P3→P3→P1, h1→h1→h3→h3
            result_template.update({
                'P1': P1_off, 'P2': P3_off, 'P3': P3_off, 'P4': P1_off,
                'h1': h1_off, 'h2': h1_off, 'h3': h3_off, 'h4': h3_off,
                's1': s1_off, 's2': s1_off, 's3': s3_off, 's4': s3_off,
                'T1': cu.K2C(T_oa_K), 'T2': cu.K2C(T_tank_w_K), 
                'T3': cu.K2C(T_tank_w_K), 'T4': cu.K2C(T_oa_K),
                'T_cond': cu.K2C(T_tank_w_K),
                'T_evap': cu.K2C(T_oa_K),
                'T_air_ou_out': cu.K2C(T_oa_K),
                'T_oa': T_oa
            })
        except Exception:
            # 계산 실패 시 NaN으로 설정
            nan_keys = ['P1','P2','P3','P4','h1','h2','h3','h4','s1','s2','s3','s4','T1','T2','T3','T4','T_cond','T_evap','T_air_ou_out']
            for k in nan_keys:
                result_template[k] = np.nan

        return result_template
    
    def run_simulation(
        self, 
        simulation_period_sec, 
        dt_s, 
        T_tank_w_init_C,
        schedule_entries,
        T_oa_schedule,
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
        
        # T_oa_schedule 검증
        time = np.arange(0, simulation_period_sec, dt_s)
        tN = len(time)
        T_oa_schedule = np.array(T_oa_schedule)
        if len(T_oa_schedule) != tN:
            raise ValueError(f"T_oa_schedule length ({len(T_oa_schedule)}) must match time array length ({tN})")
        
        # --- 1. 시뮬레이션 초기화 ---
        self.time = time
        self.dt = dt_s
        
        self.dV_w_serv = 0.0 # 초기 서비스 유량
        self.dV_w_sup_tank = 0.0 # 초기 탱크 출수 유량
        self.dV_w_sup_mix = 0.0 # 초기 믹싱밸브 상수도 유량
        
        # 스케줄 빌드
        self.serv_sched = _build_schedule_ratios(schedule_entries, self.time)
        
        results_data = [] # 결과를 딕셔너리 리스트로 저장
        
        # 동적 상태 변수 초기화
        T_tank_w_K = cu.C2K(T_tank_w_init_C)
        is_on_prev = False

        # --- 2. 시뮬레이션 루프 ---
        for n in tqdm(range(tN), desc="ASHPB Simulating"):
            step_results = {}
            T_tank_w = cu.K2C(T_tank_w_K)
            T_oa = T_oa_schedule[n]  # 현재 타임스텝의 실외 공기 온도

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
            
            # 3. 히트펌프 운전점 계산 (ashpb -> self)
            # OFF→ON 전환 시점과 정상 상태 모두에서 최적화 반복 호출만 다르고
            # 저장 여부만 다르므로 공통 부분 함수로 뺌
            def _run_ref_loop_dropt():
                return cp.find_ref_loop_optimal_operation(
                    calculate_performance_func=self._calculate_ashpb_next_step,
                    T_tank_w=cu.K2C(T_tank_w_K),
                    Q_cond_load=Q_cond_load_n,
                    Q_cond_LOAD_OFF_TOL=self.Q_cond_LOAD_OFF_TOL,
                    bounds=[(0.1, 30.0), (0.1, 30.0)],
                    initial_guess=[5.0, 5.0],
                    constraint_funcs=cp.create_lmtd_constraints(),
                    off_result_formatter=self._format_ashpb_off_results_dict,
                    T_oa=T_oa  # kwargs로 전달
                )
            ref_result = _run_ref_loop_dropt()
            
            if ref_result is None:
                try:
                    ref_result = self._format_ashpb_off_results_dict(
                        T_tank_w=cu.K2C(T_tank_w_K),
                        Q_cond_load=Q_cond_load_n,
                        T_oa=T_oa
                    )
                    print(f'경고: 타임스텝 {n}에서 최적화 실패, OFF 상태 결과 사용')
                except Exception as e:
                    print(f'에러: 타임스텝 {n}에서 OFF 상태 결과 생성 실패: {e}')
                    ref_result = {
                        'is_on': False,
                        'Q_ref_cond': 0.0,
                        'Q_ref_evap': 0.0,
                        'E_cmp': 0.0,
                        'E_fan_ou': 0.0,
                        'E_tot': 0.0,
                        'T_tank_w': cu.K2C(T_tank_w_K),
                        'T_oa': T_oa
                    }
            
            if ref_result is None or not isinstance(ref_result, dict):
                print(f'에러: 타임스텝 {n}에서 ref_result가 유효하지 않음: {type(ref_result)}')
                ref_result = {
                    'is_on': False,
                    'Q_ref_cond': 0.0,
                    'Q_ref_evap': 0.0,
                    'E_cmp': 0.0,
                    'E_fan_ou': 0.0,
                    'E_tot': 0.0,
                    'T_tank_w': cu.K2C(T_tank_w_K),
                    'T_oa': T_oa
                }
            
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
            
            # 4. 다음 스텝 탱크 온도 계산
            if n < tN - 1:
                Q_tank_in = ref_result.get('Q_ref_cond', 0.0)
                Q_net = Q_tank_in - total_loss
                T_tank_w_K += (Q_net / self.C_tank) * self.dt # config -> self

            # 5. 결과 저장 및 플롯 ##################################################################
            '''
            추후 완전한 plot 그래프 기능의 별도 기능 분리 필요
            '''
            # 결과 저장 (전환 시점에도 이전 스텝 값으로 저장)
            if save_ph_diagram: # P-h 선도 저장
                if snapshot_save_path is None:
                    raise ValueError("snapshot_save_path must be provided when save_ph_diagram is True.")
                # 전환 시점이 아닐 때만 P-h 선도 저장 (전환 시점은 이전 스텝 값이므로)
                if not is_transitioning_off_to_on:
                    cp.plot_cycle_diagrams(
                        result=ref_result,
                        refrigerant=self.ref,
                        show=False,
                        show_temp_limits=True,
                        save_path=snapshot_save_path+f'/{n:04d}.png',
                        T_tank_w=T_tank_w,
                        T_oa=T_oa,  # 실외 공기 온도 전달 (지중 관련 파라미터 제거)
                    )
            results_data.append(step_results)
            ################################################################################################
            
        # --- 3. 시뮬레이션 종료 및 반환 ---
        results_df = pd.DataFrame(results_data)

        if result_save_csv_path:
            results_df.to_csv(result_save_csv_path)

        return results_df

@dataclass
class AirSourceHeatPumpBoiler2:
    '''
    공기열원 히트펌프 보일러 동적 최적화 클래스.
    
    주어진 저탕조 온도(T_tank_w)와 외기온도(T_oa) 조건에서,
    fan 풍량과 냉매 운전점 온도 조합을 최적화하여 총 투입 전력(fan + compressor)을 최소화합니다.
    '''
    
    def __init__(
        self,
        
        # 1. 냉매/사이클/압축기 파라미터 -------------------------------
        ref         = 'R410A',
        V_disp_cmp  = 0.0005,
        eta_cmp_isen = 0.7,
        
        # 2. 열교환기 파라미터 -----------------------------------------
        A_ou        = 20.0,   # 실외기 전열 면적 [m²]
        UA_cond     = 500.0,  # 응축기 열전달 계수 [W/K] (상수)
        UA_evap     = 500.0,  # 증발기 열전달 계수 [W/K] (기준값, 풍량 보정용)
        A_cross_ou  = 0.4,    # 실외기 단면적 [m²]
        
        # 3. 실외기 팬 파라미터 ---------------------------------------
        dV_ou_design = 2.5,   # 실외기 설계 풍량 [m³/s]
        dP_ou_design = 500.0, # 실외기 설계 정압 [Pa]
        eta_motor_ou = 0.8,   # 실외기 모터 효율 [-]
        eta_fan_ou   = 0.8,   # 실외기 팬 효율 [-]
        
        # 4. VSD Curve 계수 ------------------------------------------
        c1 = 0.0013,
        c2 = 0.1470,
        c3 = 0.9506,
        c4 = -0.0998,
        c5 = 0.0,
        
        # 5. 기준 온도 -----------------------------------------------
        T0 = 0.0,    # [°C] 기준 외기 온도
        ):
        '''
        AirSourceHeatPumpBoilerOptimizer 초기화.
        
        Args:
            ref: 냉매 종류 (기본값: 'R410A')
            V_disp_cmp: 압축기 배기량 [m³]
            eta_cmp_isen: 압축기 등엔트로피 효율 [-]
            A_ou: 실외기 전열 면적 [m²]
            UA_cond: 응축기 열전달 계수 [W/K]
            UA_evap: 증발기 열전달 계수 [W/K] (기준값)
            A_cross_ou: 실외기 단면적 [m²]
            dV_ou_design: 실외기 설계 풍량 [m³/s]
            dP_ou_design: 실외기 설계 정압 [Pa]
            eta_motor_ou: 실외기 모터 효율 [-]
            eta_fan_ou: 실외기 팬 효율 [-]
            c1~c5: VSD Curve 계수
            T0: 기준 외기 온도 [°C]
        '''
        
        # 냉매/사이클/압축기 파라미터
        self.ref = ref
        self.V_disp_cmp = V_disp_cmp
        self.eta_cmp_isen = eta_cmp_isen
        
        # 열교환기 파라미터
        self.A_ou = A_ou
        self.UA_cond = UA_cond
        self.UA_evap = UA_evap
        self.A_cross_ou = A_cross_ou
        
        # 실외기 팬 파라미터
        self.dV_ou_design = dV_ou_design
        self.dP_ou_design = dP_ou_design
        self.eta_motor_ou = eta_motor_ou
        self.eta_fan_ou = eta_fan_ou
        
        # 기준 온도
        self.T0 = T0
        self.T0_K = cu.C2K(T0)
        
        # 팬 설계 전력 계산
        self.fan_design_power_ou = (self.dV_ou_design * self.dP_ou_design) / (self.eta_motor_ou * self.eta_fan_ou)
        
        # 팬 파라미터 딕셔너리
        self.fan_params_ou = {
            'fan_design_flow_rate': self.dV_ou_design,
            'fan_design_power': self.fan_design_power_ou
        }
        
        # VSD Curve 계수
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.c5 = c5
    
    def _calculate_cycle_performance(self, optimization_vars, T_tank_w, Q_cond_load, T_oa, **kwargs):
        """
        사이클 성능을 계산합니다.
        
        Args:
            optimization_vars: [dT_ref_cond, dT_ref_evap, dV_fan_ou] [K, K, m³/s]
            T_tank_w: 저탕조 온도 [°C]
            Q_cond_load: 목표 열 교환율 [W]
            T_oa: 실외 공기 온도 [°C]
        
        Returns:
            dict: 사이클 성능 결과 딕셔너리 (None: 계산 실패 시)
        """
        # 입력 검증
        if len(optimization_vars) != 3:
            return None
        
        dT_ref_cond = optimization_vars[0]
        dT_ref_evap = optimization_vars[1]
        dV_fan_ou = optimization_vars[2]
        
        # 물리적 제약 조건 검증
        if dT_ref_cond < 0 or dT_ref_evap < 0 or dV_fan_ou < 0:
            return None
        
        # 온도 범위 검증
        T_tank_w_K = cu.C2K(T_tank_w)
        T_oa_K = cu.C2K(T_oa)
        
        if T_tank_w < 0 or T_tank_w > 100 :  # 0°C ~ 100°C 범위
            return None
        if T_oa < -40 or T_oa > 50:  # -40°C ~ 50°C 범위
            return None
        
        T_tank_w_K = cu.C2K(T_tank_w)
        T_oa_K = cu.C2K(T_oa)
        
        # 증발/응축 온도 계산
        T_cond_K = T_tank_w_K + dT_ref_cond
        T_evap_K = T_oa_K - dT_ref_evap
        
        T_cond = cu.K2C(T_cond_K)
        T_evap = cu.K2C(T_evap_K)
        
        # 증발/응축 온도 물리적 제약 검증
        if T_evap >= T_cond:
            return None  # 증발 온도는 응축 온도보다 낮아야 함
        
        if T_evap < -40 or T_cond > 100:  # 합리적인 온도 범위
            return None
        
        # 사이클 상태 계산
        try:
            cycle_states = cp.compute_refrigerant_thermodynamic_states(
                T_evap_K     = T_evap_K,
                T_cond_K     = T_cond_K,
                refrigerant  = self.ref,
                eta_cmp_isen = self.eta_cmp_isen,
                T0_K         = self.T0_K,
                P0           = 101325,
                mode         = 'heating'
            )
        except Exception:
            # 사이클 상태 계산 실패
            return None
        
        if cycle_states is None:
            return None
        
        rho_ref_cmp_in = cycle_states.get('rho')
        if rho_ref_cmp_in is None or rho_ref_cmp_in <= 0:
            return None
        
        h1 = cycle_states.get('h1')
        h2 = cycle_states.get('h2')
        h3 = cycle_states.get('h3')
        h4 = cycle_states.get('h4')
        
        # 엔탈피 값 검증
        if any(h is None or np.isnan(h) for h in [h1, h2, h3, h4]):
            return None
        
        # 계산 불가능한 경우 체크
        if abs(h2 - h3) < 1e-6 or abs(h4 - h1) < 1e-6:
            return None
        
        T1_K = cycle_states.get('T1_K')
        T2_K = cycle_states.get('T2_K')
        T3_K = cycle_states.get('T3_K')
        T4_K = cycle_states.get('T4_K')
        
        # 온도 값 검증
        if any(T is None or np.isnan(T) for T in [T1_K, T2_K, T3_K, T4_K]):
            return None
        
        # dV_fan_ou를 사용하여 동적 UA 계산
        try:
            UA_evap_calc = calculate_heat_transfer_coefficient(
                dV_fan=dV_fan_ou,
                dV_fan_design=self.dV_ou_design,
                A_cross=self.A_cross_ou,
                UA=self.UA_evap
            )
            if UA_evap_calc <= 0 or np.isnan(UA_evap_calc):
                return None
        except Exception:
            return None
        
        # T_ref_evap_avg_K 계산
        T_ref_evap_avg_K = (T4_K + T1_K) / 2 # 냉매 평균 온도가 열교환기 전체가 constant 한 온도 값과 유사하다 가정(냉매 in out 온도변화가 매우 작은 증발기 한정)
        
        # 반복 계산: T_air_ou_out_K와 Q_LMTD_evap을 수렴시킴
        # T_air_ou_out_K → Q_LMTD_evap → m_dot_ref → Q_ref_evap → T_air_ou_out_K (순환)
        if dV_fan_ou < 1e-6:
            Q_LMTD_evap = 0.0
            m_dot_ref = 0.0
            T_air_ou_out_K = T_oa_K
        else:
            def T_air_ou_out_residual(T_air_ou_out_K):
                # LMTD 계산
                LMTD_evap = calc_lmtd_constant_refrigerant_temp(
                    T_ref_avg_K = T_ref_evap_avg_K,
                    T_air_ou_in_K  = T_oa_K,
                    T_air_ou_out_K = T_air_ou_out_K
                )
                
                # 계산 불가(비정상) 상황 처리
                if np.isnan(LMTD_evap) or LMTD_evap <= 1e-6 or abs(h4 - h1) < 1e-6:
                    return 1e6

                Q_LMTD_evap = UA_evap_calc * LMTD_evap
                m_dot_ref = Q_LMTD_evap / (h4 - h1)
                Q_ref_evap = m_dot_ref * (h4 - h1)
                T_air_ou_out_K_calc = T_oa_K + (1 - np.exp(-UA_evap_calc / (c_a * rho_a * dV_fan_ou))) * (T_ref_evap_avg_K - T_oa_K) # NTU method
                return T_air_ou_out_K_calc - T_air_ou_out_K

            # T_air_ou_out_K는 T_oa_K ~ T_oa_K + 30K(실외유입공기->최대냉각 예상)
            try:
                sol = root_scalar(
                    T_air_ou_out_residual, 
                    bracket=[T_oa_K - 30.0, T_oa_K], 
                    method='brentq',
                    xtol=1e-3
                )

                if sol.converged:
                    T_air_ou_out_K = sol.root
                    print(f"T_air_ou_out_C: {T_air_ou_out_K - 273.15}")
                    print(f"T_oa_C: {T_oa_K - 273.15}")
                    print(f"T_ref_evap_avg_C: {T_ref_evap_avg_K - 273.15}")
                    print(f"T_oa: {T_oa}")
                    print(f"T_evap: {T_evap}")
                    print(f"T_cond: {T_cond}")
                    print(f"UA_evap_calc: {UA_evap_calc}")
                    print(f"dV_fan_ou: {dV_fan_ou}")
                    # 수렴한 root에 대해 LMTD/열량 계산
                    LMTD_evap = calc_lmtd_constant_refrigerant_temp(
                        T_ref_avg_K = T_ref_evap_avg_K,
                        T_air_ou_in_K  = T_oa_K,
                        T_air_ou_out_K = T_air_ou_out_K
                    )
                    if np.isnan(LMTD_evap) or LMTD_evap <= 1e-6 or abs(h4 - h1) < 1e-6:
                        Q_LMTD_evap = 0.0
                        m_dot_ref = 0.0
                    else:
                        Q_LMTD_evap = UA_evap_calc * LMTD_evap
                        m_dot_ref = Q_LMTD_evap / (h4 - h1)
                else:
                    # 수렴 실패시 None 반환
                    return None
            except ValueError:
                # bracket 범위 문제 등으로 수렴 실패
                return None
        
        # 사이클 열량 및 전력 계산
        Q_ref_cond = m_dot_ref * (h2 - h3)
        Q_ref_evap = m_dot_ref * (h4 - h1)
        E_cmp = m_dot_ref * (h2 - h1)
        cmp_rps = m_dot_ref / (self.V_disp_cmp * rho_ref_cmp_in) if m_dot_ref > 0 else 0.0
        
        # 계산 결과 검증
        if any(np.isnan(val) or np.isinf(val) for val in [Q_ref_cond, Q_ref_evap, E_cmp, m_dot_ref]):
            return None
        
        if E_cmp < 0 or Q_ref_cond < 0:
            return None
        
        performance_data = {
            'Q_ref_cond': Q_ref_cond,
            'Q_ref_evap': Q_ref_evap,
            'E_cmp': E_cmp,
            'm_dot_ref': m_dot_ref,
            'cmp_rps': cmp_rps,
            'cmp_rpm': cmp_rps * 60,
            'T_cond_K': T_cond_K,
            'T_evap_K': T_evap_K,
        }
        
        vsd_coeffs = {
            'c1': self.c1,
            'c2': self.c2,
            'c3': self.c3,
            'c4': self.c4,
            'c5': self.c5
        }
        
        # 팬 전력 계산
        try:
            E_fan_ou = calculate_fan_power(
                dV_fan_ou, 
                self.fan_params_ou,
                vsd_coeffs
            )
            if E_fan_ou < 0 or np.isnan(E_fan_ou) or np.isinf(E_fan_ou):
                return None
        except (ValueError, TypeError):
            # 팬 전력 계산 실패
            return None
        
        performance_data['E_fan_ou'] = E_fan_ou
        performance_data['dV_fan_ou'] = dV_fan_ou
        performance_data['UA_evap'] = self.UA_evap
        performance_data['UA_evap_calc'] = UA_evap_calc
        performance_data['Q_LMTD_evap'] = Q_LMTD_evap
        performance_data['T_ref_evap_avg_K'] = T_ref_evap_avg_K
        performance_data['T_air_ou_out_K'] = T_air_ou_out_K
        
        return self._format_results(cycle_states, performance_data, T_tank_w, Q_cond_load, T_oa)
    
    def _format_results(self, cycle_states, performance_data, T_tank_w, Q_cond_load, T_oa):
        """
        사이클 성능 결과를 포맷팅합니다.
        
        Args:
            cycle_states: 사이클 상태값 딕셔너리
            performance_data: 성능 데이터 딕셔너리
            T_tank_w: 저탕조 온도 [°C]
            Q_cond_load: 목표 열 교환율 [W]
            T_oa: 실외 공기 온도 [°C]
        
        Returns:
            dict: 포맷팅된 결과 딕셔너리
        """
        T_tank_w_K = cu.C2K(T_tank_w)
        T_oa_K = cu.C2K(T_oa)
        
        # 사이클 상태값 추출
        T1_K = cycle_states['T1_K']
        P1 = cycle_states['P1']
        h1 = cycle_states['h1']
        s1 = cycle_states['s1']
        
        T2_K = cycle_states['T2_K']
        P2 = cycle_states['P2']
        h2 = cycle_states['h2']
        s2 = cycle_states['s2']
        
        T3_K = cycle_states['T3_K']
        P3 = cycle_states['P3']
        h3 = cycle_states['h3']
        s3 = cycle_states['s3']
        
        T4_K = cycle_states['T4_K']
        P4 = cycle_states['P4']
        h4 = cycle_states['h4']
        s4 = cycle_states['s4']
        
        # 성능 데이터 추출
        Q_ref_cond = performance_data['Q_ref_cond']
        Q_ref_evap = performance_data['Q_ref_evap']
        E_cmp = performance_data['E_cmp']
        E_fan_ou = performance_data['E_fan_ou']
        m_dot_ref = performance_data['m_dot_ref']
        cmp_rps = performance_data['cmp_rps']
        dV_fan_ou = performance_data['dV_fan_ou']
        
        E_tot = E_cmp + E_fan_ou
        
        # 응축기 LMTD 계산
        dT1_tank = T2_K - T_tank_w_K
        dT2_tank = T3_K - T_tank_w_K

        if dT1_tank <= 1e-6 or dT2_tank <= 1e-6 or abs(dT1_tank - dT2_tank) < 1e-6:
            Q_LMTD_cond = -np.inf
        else:
            LMTD_tank = (dT1_tank - dT2_tank) / np.log(dT1_tank / dT2_tank)
            Q_LMTD_cond = self.UA_cond * LMTD_tank

        # performance_data에서 이미 계산된 값들 가져오기
        T_ref_evap_avg_K = performance_data.get('T_ref_evap_avg_K', (T4_K + T1_K) / 2)
        T_air_ou_out_K = performance_data.get('T_air_ou_out_K', T_oa_K)
        Q_LMTD_evap = performance_data.get('Q_LMTD_evap', 0.0)
        
        # 엑서지 계산
        T0_K = self.T0_K
        P0 = 101325
        h0 = CP.PropsSI('H', 'T', T0_K, 'P', P0, self.ref)
        s0 = CP.PropsSI('S', 'T', T0_K, 'P', P0, self.ref)
        
        result = {
            'is_on': True,
            
            'Q_ref_cond': Q_ref_cond,
            'Q_ref_evap': Q_ref_evap,
            'Q_LMTD_cond': Q_LMTD_cond,
            'Q_LMTD_evap': Q_LMTD_evap,
            
            'Q_cond_load': Q_cond_load,
            'E_cmp': E_cmp,
            'E_fan_ou': E_fan_ou,
            'E_tot': E_tot,
            'cop': Q_cond_load / E_tot if E_tot > 0 else 0,
            'm_dot_ref': m_dot_ref,
            'cmp_rpm': cmp_rps * 60,
            
            'dV_fan_ou': dV_fan_ou,
            
            'T_tank_w': T_tank_w,
            'T_oa': T_oa,

            'T0': cu.K2C(T0_K),
            'T1': cu.K2C(T1_K),
            'T2': cu.K2C(T2_K),
            'T3': cu.K2C(T3_K),
            'T4': cu.K2C(T4_K),
            'T_cond': cu.K2C(performance_data.get('T_cond_K', T2_K)),
            'T_evap': cu.K2C(performance_data.get('T_evap_K', T4_K)),
            'T_air_ou_out': cu.K2C(T_air_ou_out_K),

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
    
    def find_optimal_operation(self, T_tank_w, T_oa, Q_cond_load, **kwargs):
        """
        주어진 저탕조 온도와 외기 온도 조건에서 총 전력 사용량을 최소화하는 최적 운전점을 찾습니다.
        
        목적 함수: E_total = E_cmp + E_fan_ou (최소화)
        최적화 변수: [dT_ref_cond, dT_ref_evap, dV_fan_ou] [K, K, m³/s]
        제약 조건: Q_ref_cond - Q_cond_load = 0 (등식 제약)
        
        Args:
            T_tank_w (float): 저탕조 온도 [°C]
            T_oa (float): 실외 공기 온도 [°C]
            Q_cond_load (float): 목표 열 교환율 [W] (응축기가 저탕조에 전달해야 하는 열량)
            **kwargs: 추가 파라미터
                - bounds (list, optional): 최적화 변수 경계 조건
                  기본값: [(1.0, 20.0), (1.0, 20.0), (0.1, 5.0)] [K, K, m³/s]
                - initial_guess (list, optional): 초기 추정값
                  기본값: [5.0, 5.0, 2.5] [K, K, m³/s]
        
        Returns:
            dict: 최적 운전점 결과 딕셔너리
                성공 시: _format_results의 반환값과 동일한 구조
                실패 시: {'is_on': False, 'error': error_msg} 포함
        """
        # 경계 조건 및 초기 추정값 설정
        bounds = kwargs.get('bounds', [(1.0, 20.0), (1.0, 20.0), (0.1, 5.0)])  # [K, K, m³/s]
        initial_guess = kwargs.get('initial_guess', [5.0, 5.0, 2.5])  # [K, K, m³/s]
        
        # 입력 검증
        if Q_cond_load is None or np.isnan(Q_cond_load) or np.isinf(Q_cond_load):
            return {
                'is_on': False,
                'error': f'Invalid Q_cond_load: {Q_cond_load}',
                'Q_cond_load': Q_cond_load,
                'T_tank_w': T_tank_w,
                'T_oa': T_oa
            }
        
        # 목표 열 교환율이 너무 작으면 OFF
        if abs(Q_cond_load) < 500.0:
            return {
                'is_on': False,
                'error': f'Q_cond_load too small: {Q_cond_load:.2f} [W]',
                'Q_cond_load': Q_cond_load,
                'T_tank_w': T_tank_w,
                'T_oa': T_oa
            }
        
        # 온도 검증
        if T_tank_w is None or T_oa is None:
            return {
                'is_on': False,
                'error': 'T_tank_w or T_oa is None',
                'T_tank_w': T_tank_w,
                'T_oa': T_oa,
                'Q_cond_load': Q_cond_load
            }
        
        # 1. 목적 함수: 총 전력 사용량 (최소화 대상)
        def objective(x):  # x = [dT_ref_cond, dT_ref_evap, dV_fan_ou]
            try:
                result = self._calculate_cycle_performance(
                    optimization_vars = x,
                    T_tank_w         = T_tank_w,
                    Q_cond_load      = Q_cond_load,
                    T_oa             = T_oa
                )
                if result is None:
                    # 계산 실패 시 큰 값을 반환하여 해당 점을 피하도록 함
                    return 1e10
                return result['E_tot']
            except Exception as e:
                # 예외 발생 시 큰 값을 반환
                return 1e10
        
        # 2. 제약 조건: 계산된 응축기 열량이 목표 부하와 같아야 함
        def constraint(x):
            try:
                result = self._calculate_cycle_performance(
                    optimization_vars = x,
                    T_tank_w         = T_tank_w,
                    Q_cond_load      = Q_cond_load,
                    T_oa             = T_oa
                )
                if result is None:
                    # 계산 실패 시 제약 조건을 만족하지 않음으로 표시
                    return 1e6
                return result['Q_ref_cond'] - Q_cond_load
            except Exception as e:
                # 예외 발생 시 제약 조건을 만족하지 않음으로 표시
                return 1e6
        
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
                options={'disp': False, 'maxiter': 1000, 'ftol': 1e-6}
            )
        except Exception as e:
            error_msg = (
                f'최적화 과정에서 오류 발생: {type(e).__name__}: {str(e)}\n'
                f'  T_tank_w: {T_tank_w:.2f} [°C], T_oa: {T_oa:.2f} [°C]\n'
                f'  Q_cond_load: {Q_cond_load:.2f} [W]'
            )
            return {
                'is_on': False,
                'error': error_msg,
                'T_tank_w': T_tank_w,
                'T_oa': T_oa,
                'Q_cond_load': Q_cond_load
            }
        
        if opt_result.success:
            # 최적 운전점에서의 성능 계산
            try:
                optimal_result = self._calculate_cycle_performance(
                    optimization_vars = opt_result.x,
                    T_tank_w         = T_tank_w,
                    Q_cond_load      = Q_cond_load,
                    T_oa             = T_oa
                )
                
                if optimal_result is None:
                    error_msg = 'optimal_result is None after successful optimization'
                    return {
                        'is_on': False,
                        'error': error_msg,
                        'T_tank_w': T_tank_w,
                        'T_oa': T_oa,
                        'Q_cond_load': Q_cond_load
                    }
                
                # 최적화 변수 정보 추가
                optimal_result['optimization_vars'] = opt_result.x
                optimal_result['dT_ref_cond'] = opt_result.x[0]
                optimal_result['dT_ref_evap'] = opt_result.x[1]
                optimal_result['dV_fan_ou_optimal'] = opt_result.x[2]
                
                return optimal_result
            except Exception as e:
                error_msg = (
                    f'최적 운전점 계산 중 오류 발생: {type(e).__name__}: {str(e)}\n'
                    f'  optimization_vars: {opt_result.x}\n'
                    f'  T_tank_w: {T_tank_w:.2f} [°C], T_oa: {T_oa:.2f} [°C]\n'
                    f'  Q_cond_load: {Q_cond_load:.2f} [W]'
                )
                return {
                    'is_on': False,
                    'error': error_msg,
                    'T_tank_w': T_tank_w,
                    'T_oa': T_oa,
                    'Q_cond_load': Q_cond_load
                }
        else:
            # 최적화 실패 시
            error_msg = f'Optimization failed: {opt_result.message}'
            return {
                'is_on': False,
                'error': error_msg,
                'T_tank_w': T_tank_w,
                'T_oa': T_oa,
                'Q_cond_load': Q_cond_load
            }


@dataclass
class PVSystemAnalyzer:
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
