import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm
import CoolProp.CoolProp as CP

from . import calc_util as cu
from .constants import (
    c_a, rho_a, k_a, c_w, rho_w, mu_w, k_w, sigma, k_D, k_d, ex_eff_NG, SP
)
from .enex_functions import (
    compute_refrigerant_thermodynamic_states,
    create_lmtd_constraints,
    find_ref_loop_optimal_operation,
    plot_cycle_diagrams,
    calc_lmtd_fluid_and_constant_temp,
    calc_simple_tank_UA,
    _build_schedule_ratios,
    calc_UA_from_dV_fan,
    calc_fan_power_from_dV_fan,
    calc_HX_perf_for_target_heat,
    update_tank_temperature,
    calc_energy_flow,
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
        eta_cmp_isen = 0.8,

        # 2. 열교환기 파라미터 -----------------------------------------
        UA_cond_design = None,   # 응축기 열전달 계수 [W/K]: 자동 산정 (kW당 250)
        UA_evap_design = None,   # 증발기 열전달 계수 [W/K]: 자동 산정 (kW당 80)

        # 3. 실외기 팬 파라미터 ---------------------------------------
        dV_ou_design      = 2.5,     # 실외기 설계 풍량 [m3/s] (정풍량)
        dP_ou_design      = 500.0,   # 실외기 설계 정압 [Pa]
        A_cross_ou        = 0.4,     # 실외기 단면적 [m²]
        eta_fan_ou_design = 0.8,     # 실외기 팬 효율 [-]

        # 4. 탱크/제어/부하 파라미터 -----------------------------------
        T_tank_w_upper_bound = 65.0,   # [°C] 저탕조 설정 온도
        T_tank_w_lower_bound = 55.0,   # [°C] 저탕조 하한 온도
        T_serv_w             = 40.0,   # [°C] 서비스 급탕 온도
        T_sup_w              = 15.0,   # [°C] 급수(상수도) 온도

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
        
        # 5. UV 램프 파라미터 -----------------------------------------
        lamp_power_watts = 0, # [W] 램프 소비 전력
        uv_lamp_exposure_duration_min = 0, # [min] 1회 UV램프 노출 기준시간
        num_switching_per_3hour = 1, # [개] 3시간 당 on 횟수
        
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

        self.heater_capacity = heater_capacity

        # --- 2. 열교환기 파라미터 ---
        '''
        Asloune, H., & Riviere, P. (2018). A simplified model for assessing improvement potential of air-to-air conditioners and heat pumps.
        International Refrigeration and Air Conditioning Conference. Paper 1878. https://docs.lib.purdue.edu/iracc/1878
        '''
        _hc_kw = self.heater_capacity / 1000.0
        if UA_cond_design is None:
            self.UA_cond_design = 400.0 * _hc_kw    # 250 W/K per kW
        else:
            self.UA_cond_design = UA_cond_design
        if UA_evap_design is None:
            self.UA_evap_design = 280.0 * _hc_kw     # 80 W/K per kW
        else:
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
        
        self.dV_w_serv_m3s = dV_w_serv_m3s
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
                - 온도 (T0, T1-4, T_tank_w, T_serv_w, T_sup_w, T_a_ou_out)
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
        
        # Q_cond_load가 0일 때는 CoolProp 호출을 건너뛰고 기본값 사용
        if Q_cond_load == 0.0:
            # OFF 상태용 기본값 설정 (냉매 상태값은 모두 nan)
            T1_K = T2_K = T3_K = T4_K = np.nan
            P1 = P2 = P3 = P4 = np.nan
            h1 = h2 = h3 = h4 = np.nan
            s1 = s2 = s3 = s4 = np.nan
            x1 = x2 = x3 = x4 = np.nan
            rho_ref_cmp_in = np.nan
            m_dot_ref = 0.0
            Q_ref_cond = 0.0
            Q_ref_evap = 0.0
            E_cmp = 0.0
            cmp_rps = 0.0
            dV_fan_ou = 0.0
            UA_evap = np.nan
            T_a_ou_out = T0_K
            LMTD_evap = np.nan
            Q_LMTD_evap = 0.0
            LMTD_tank = np.nan
            UA_cond = self.UA_cond_design
            Q_LMTD_cond = 0.0
            E_fan_ou = 0.0
            fan_eff = 0.0
            
            # 엑서지 관련 기본값
            P0 = 101325
            h0 = s0 = np.nan
            X1 = X2 = X3 = X4 = 0.0
            T_a_ou_in = T0
            T_a_ou_out = T0
            X_cmp = 0.0
            X_fan_ou = 0.0
            X_a_ou_in = 0.0
            X_a_ou_out = 0.0
            X_ref_cond = 0.0
            X_ref_evap = 0.0
        else:
            # 정상 계산 수행
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
            T_a_ou_in_K = T0_K
            T_a_ou_in = cu.K2C(T_a_ou_in_K)
            
            HX_perf_ou_dict = calc_HX_perf_for_target_heat(
                Q_ref_target=Q_ref_evap,
                T_a_ou_in_C=T0,
                T_ref_avg_K=T_ref_evap_avg_K,
                A_cross=self.A_cross_ou,
                UA_design=self.UA_evap_design,
                dV_fan_design=self.dV_ou_design
            )
            
            if HX_perf_ou_dict.get('converged', True) == False:
                return None
            
            # 수렴 성공 시 값 추출
            dV_fan_ou = HX_perf_ou_dict['dV_fan']
            UA_evap = HX_perf_ou_dict['UA']
            T_a_ou_out = HX_perf_ou_dict['T_a_ou_out']
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
            
            fan_eff = self.eta_fan_ou_design * dV_fan_ou / E_fan_ou * 100
            
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
            X_a_ou_in = calc_exergy_flow(G=c_a * rho_a * dV_fan_ou, T=T_a_ou_in_K, T0=T0_K)
            X_a_ou_out = calc_exergy_flow(G=c_a * rho_a * dV_fan_ou, T=T_a_ou_out, T0=T0_K)
            X_ref_cond = Q_ref_cond * (1 - T0_K / T_tank_w_K)
            X_ref_evap = Q_ref_evap * (1 - T0_K / T_a_ou_out)
        
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
        
        Q_tank_sup_w = calc_energy_flow(G=c_w * rho_w * dV_w_sup_tank, T=self.T_sup_w_K, T0=T0_K)
        Q_tank_w = calc_energy_flow(G=c_w * rho_w * dV_w_sup_tank, T=T_tank_w_K, T0=T0_K)
        Q_mix_sup_w = calc_energy_flow(G=c_w * rho_w * dV_w_sup_mix, T=self.T_sup_w_K, T0=T0_K)
        Q_mix_serv_w = calc_energy_flow(G=c_w * rho_w * dV_w_serv, T=T_serv_w_actual_K, T0=T0_K)
        
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
        
        # 총 입력 엑서지
        X_tot = E_cmp + E_fan_ou  # 총 입력 엑서지
        # 효율 계산(cop_ref, cop_sys, X_eff_ref, X_eff_sys)은 analyze_dynamic에서 UV 램프 전력을 고려하여 계산
        
        # 10단계: 최종 결과 딕셔너리 생성
        result = {
            'is_on': (Q_cond_load > 0),
            'converged': True,

            # === [온도: °C] =======================================
            'T_a_ou_in [°C]': T_a_ou_in,
            'T_a_ou_out [°C]': T_a_ou_out,
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
            'Q_mix_serv_w [W]': Q_mix_serv_w,
            'X_mix_sup_w [W]': X_mix_sup_w,
            'X_mix_serv_w [W]': X_mix_serv_w,
            'Xc_mix [W]': Xc_mix,

            # ---- 총괄(손실, 총입력, 흐름 등)
            'Xc_tot [W]': Xc_tot,
            'E_tot [W]': E_cmp + E_fan_ou,
            'X_fan_ou [W]': X_fan_ou,
            # 'X_flow_X1_to_X2 [W]': X_flow_X1_to_X2,
            # 'X_flow_X2_to_X3 [W]': X_flow_X2_to_X3,
            # 'X_flow_X3_to_X4 [W]': X_flow_X3_to_X4,
            # 'X_flow_X4_to_X1 [W]': X_flow_X4_to_X1,
            'X_tot [W]': X_tot,

            # === [무차원: 효율, COP] ==============================
            'fan_eff [%]': fan_eff,
            # cop_ref, cop_sys, X_eff_ref, X_eff_sys는 analyze_dynamic에서 UV 램프 전력을 고려하여 계산
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
        tolerance = 0.01  # 1%
        
        # 최적화 변수 경계 조건 및 초기 추정값 설정
        bounds = [(1.0, 30.0), (1.0, 30.0)]  # [dT_ref_cond, dT_ref_evap]
        initial_guess = [5.0, 20.0]
        
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
                
                if "Q_LMTD_cond [W]" not in perf or np.isnan(perf["Q_LMTD_cond [W]"]):
                    return -1e6
                
                return perf["Q_LMTD_cond [W]"] - Q_cond_load
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
                
                if "Q_LMTD_cond [W]" not in perf or np.isnan(perf["Q_LMTD_cond [W]"]):
                    return -1e6
                
                # 제약 조건: Q_cond_load*(1+tolerance) - Q_LMTD_cond >= 0
                return Q_cond_load * (1 + tolerance) - perf["Q_LMTD_cond [W]"]
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
                
                if ("Q_LMTD_evap [W]" not in perf or "Q_ref_evap [W]" not in perf or
                    np.isnan(perf["Q_LMTD_evap [W]"]) or np.isnan(perf["Q_ref_evap [W]"])):
                    return -1e6
                
                # 제약 조건: Q_LMTD_evap - Q_ref_evap*(1-tolerance) >= 0
                return perf["Q_LMTD_evap [W]"] - perf['Q_ref_evap [W]']
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
                
                if ("Q_LMTD_evap [W]" not in perf or "Q_ref_evap [W]" not in perf or
                    np.isnan(perf["Q_LMTD_evap [W]"]) or np.isnan(perf["Q_ref_evap [W]"])):
                    return -1e6
                
                # 제약 조건: Q_ref_evap*(1+tolerance) - Q_LMTD_evap >= 0
                return perf['Q_ref_evap [W]'] * (1 + tolerance) - perf["Q_LMTD_evap [W]"]
            except Exception as e:
                return -1e6
        
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
            'maxiter': 100,
            'ftol': 10, 
            'eps': 0.01,
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
        self.dV_w_sup_tank = alp * dV_w_serv
        self.dV_w_sup_mix = (1 - alp) * dV_w_serv
        
        # Q_cond_load 계산
        if Q_cond_load is None:
            # dV_w_serv가 주어진 경우: 열 손실 계산하여 Q_cond_load 결정
            Q_use_loss = c_w * rho_w * self.dV_w_sup_tank * (T_tank_w_K - self.T_sup_w_K)
            total_loss = Q_tank_loss + Q_use_loss
            # 정상상태: Q_cond_load = total_loss (에너지 밸런스)
            Q_cond_load = total_loss
        else:
            # Q_cond_load가 주어진 경우: 주어진 값 사용
            # Q_cond_load를 사용하므로 Q_use_loss는 계산하지 않음
            pass
        
        # ON/OFF 상태 결정
        if T_tank_w <= self.T_tank_w_lower_bound:
            is_on = True
        elif T_tank_w > self.T_tank_w_upper_bound:
            is_on = False
        else:
            # 정상상태에서는 Q_cond_load > 0이면 ON으로 가정
            is_on = Q_cond_load > 0
        
        # OFF 상태 조기 체크: Q_cond_load가 0 이하이면 최적화 건너뛰기
        if Q_cond_load <= 0 or not is_on:
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
            result_save_csv_path: 결과 CSV 저장 경로
            save_ph_diagram: P-h 선도 이미지 저장 여부
            snapshot_save_path: P-h 선도 이미지 저장 경로
        
        Returns:
            pd.DataFrame: 시뮬레이션 타임스텝별 결과 데이터
        """
        
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
            
            if T_tank_w <= self.T_tank_w_lower_bound: is_on = True
            elif T_tank_w >= self.T_tank_w_upper_bound: is_on = False
            else: is_on = is_on_prev
            
            Q_cond_load_n = self.heater_capacity if is_on else 0.0
            is_on_prev = is_on
            
            # OFF 상태 조기 체크: Q_cond_load_n이 0 이하이면 최적화 건너뛰기
            if Q_cond_load_n == 0:
                result = self._calc_state(
                    optimization_vars=[5.0, 5.0],
                    T_tank_w=T_tank_w,
                    Q_cond_load=0.0,
                    T0=T0
                )
            else:
                # 최적화 과정 진행
                opt_result = self._optimize_operation(
                    T_tank_w=T_tank_w,
                    Q_cond_load=Q_cond_load_n,
                    T0=T0
                )
                
                # 최적화 결과 계산
                result = self._calc_state(
                    optimization_vars=opt_result.x,
                    T_tank_w=T_tank_w,
                    Q_cond_load=Q_cond_load_n,
                    T0=T0
                )
                
                # 수렴 실패 시 에러 발생
                if not opt_result.success or result is None:
                    raise ValueError(f"Optimization failed at timestep {n}: success={opt_result.success}, result={result}")
                
                result['converged'] = opt_result.success
            
            # UV 램프 전력 계산 (is_on과 독립적으로 작동)
            E_uv_lamp = 0
            if self.num_switching_per_3hour > 0 and self.lamp_power_watts > 0:
                time_in_period = time[n] % self.period_3hour_sec
                interval = (self.period_3hour_sec - self.num_switching_per_3hour * self.uv_lamp_exposure_duration_sec) / (self.num_switching_per_3hour + 1)
                
                for i in range(self.num_switching_per_3hour):
                    start_time = interval * (i + 1) + i * self.uv_lamp_exposure_duration_sec
                    if start_time <= time_in_period < start_time + self.uv_lamp_exposure_duration_sec:
                        E_uv_lamp = self.lamp_power_watts
                        break
            
            step_results.update(result)
            step_results['is_on'] = is_on
            
            if self.lamp_power_watts > 0:
                step_results['E_uv_lamp [W]'] = E_uv_lamp
            
            # 효율 계산을 위한 변수 추출
            E_cmp = step_results.get('E_cmp [W]', 0.0)
            E_fan_ou = step_results.get('E_fan_ou [W]', 0.0)
            Q_cond_load = step_results.get('Q_cond_load [W]', 0.0)
            X_ref_cond = step_results.get('X_ref_cond [W]', 0.0)
            X_cmp = step_results.get('X_cmp [W]', 0.0)
            X_tot_base = step_results.get('X_tot [W]', 0.0)
            
            E_total = E_cmp + E_fan_ou + E_uv_lamp
            X_tot_with_uv = X_tot_base + E_uv_lamp
            
            step_results['Q_tank_loss [W]'] = Q_tank_loss
            step_results['E_tot [W]'] = E_total
            step_results['cop_ref [-]'] = Q_cond_load / E_cmp if E_cmp > 0 else 0.0
            step_results['cop_sys [-]'] = Q_cond_load / E_total if E_total > 0 else 0.0
            step_results['X_eff_ref [%]'] = (X_ref_cond / X_cmp * 100) if X_cmp > 0 else 0.0
            step_results['X_eff_sys [%]'] = (X_ref_cond / X_tot_with_uv * 100) if X_tot_with_uv > 0 else 0.0
            step_results['X_tot [W]'] = X_tot_with_uv
            
            if n < tN - 1:
                Q_tank_in = result.get('Q_ref_cond [W]', 0.0) + step_results.get('E_uv_lamp [W]', 0.0)
                T_tank_w_K = update_tank_temperature(
                    T_tank_w_K = T_tank_w_K,
                    Q_tank_in  = Q_tank_in,
                    total_loss = Q_tank_loss + Q_use_loss,
                    C_tank     = self.C_tank,
                    dt         = self.dt
                ) 
            
            if result is not None and isinstance(result, dict):
                prev_result = result.copy()
            
            results_data.append(step_results)
            
        results_df = pd.DataFrame(results_data)

        if result_save_csv_path:
            results_df.to_csv(result_save_csv_path, index=False)

        return results_df

