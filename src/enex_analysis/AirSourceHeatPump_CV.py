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
from scipy.optimize import root_scalar
import CoolProp.CoolProp as CP

# Import constants from constants.py
from .constants import (
    c_a, rho_a, k_a, c_w, rho_w, mu_w, k_w, sigma, k_D, k_d, ex_eff_NG, SP
)

# Import functions from enex_functions.py
from .enex_functions import (
    calc_lmtd_one_fluid_constant_temp,
    get_fan_flow_for_target_heat,
    calculate_fan_power,
    calculate_heat_transfer_coefficient
)

@dataclass
class AirSourceHeatPump_CV:
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
            cycle_states = compute_refrigerant_thermodynamic_states(
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
                LMTD_evap = calc_lmtd_one_fluid_constant_temp(
                    T_constant_K = T_ref_evap_avg_K,
                    T_fluid_in_K  = T_oa_K,
                    T_fluid_out_K = T_air_ou_out_K
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
                    LMTD_evap = calc_lmtd_one_fluid_constant_temp(
                        T_constant_K = T_ref_evap_avg_K,
                        T_fluid_in_K  = T_oa_K,
                        T_fluid_out_K = T_air_ou_out_K
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