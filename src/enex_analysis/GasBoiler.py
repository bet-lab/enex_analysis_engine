#%%
import numpy as np
import math
from . import calc_util as cu
from scipy.optimize import minimize
from tqdm import tqdm
import pandas as pd

# Import constants from constants.py
from .constants import (
    c_a, rho_a, k_a, c_w, rho_w, mu_w, k_w, sigma, k_D, k_d, ex_eff_NG, SP
)

# Import functions from enex_functions.py
from .enex_functions import (
    _build_schedule_ratios,
)


class GasBoiler:
    '''
    가스 보일러 성능 계산 및 동적 분석 클래스.
    
    저탕조 없이 직접 급탕 공급 구조로 동작합니다.
    Combustion chamber → Mixing valve → Service water
    '''
    
    def __init__(
        self,
        
        # 가스 보일러 파라미터
        eta_comb = 0.9,  # 연소 효율 [-]
        
        # 온도 파라미터
        T_serv_w = 45.0, # 서비스 급탕 온도 [°C]
        T_sup_w = 15.0,  # 급수(상수도) 온도 [°C]
        T_exh = 70.0,    # 배기가스 온도 [°C]
        
        # 제어 파라미터
        T_comb_setpoint = 60.0,  # 보일러 출수 설정 온도 [°C]
        dV_w_serv_m3s = 0.0001,  # 최대 급탕 유량 [m3/s]
        
        ):
        '''
        GasBoiler 초기화.
        '''
        
        # 가스 보일러 파라미터
        self.eta_comb = eta_comb
        
        # 온도 파라미터
        self.T_serv_w = T_serv_w
        self.T_sup_w = T_sup_w
        self.T_exh = T_exh
        self.T_comb_setpoint = T_comb_setpoint
        
        # 온도 단위 변환 (Kelvin)
        self.T_serv_w_K = cu.C2K(self.T_serv_w)
        self.T_sup_w_K = cu.C2K(self.T_sup_w)
        self.T_exh_K = cu.C2K(self.T_exh)
        self.T_comb_setpoint_K = cu.C2K(self.T_comb_setpoint)
        
        # 제어 파라미터
        self.dV_w_serv_m3s = dV_w_serv_m3s
        
        # 임계값
        self.Q_comb_load_threshold = 100.0  # 최소 연소 부하 [W]
    
    def _calc_on_state(self, Q_comb_load, T0, dV_w_serv):
        """
        가스 보일러 ON 상태 계산 메서드.
        
        Args:
            Q_comb_load (float): 연소 부하 [W] (보일러 유량 기준 필요한 열량)
            T0 (float): 엑서지 분석 기준 온도(=외기온도) [°C]
                엑서지 계산의 기준점으로 사용되는 외기 온도
            dV_w_serv (float): 급탕 유량 [m3/s]
        
        Returns:
            dict: ON 상태 결과 딕셔너리
        """
        # 온도 단위 변환
        T0_K = cu.C2K(T0)
        
        # Mixing valve 계산: 보일러 출수 온도와 상수도를 믹싱하여 서비스 온도로 공급
        # alp: 보일러 출수 비율, (1-alp): 상수도 직접 공급 비율
        # AirSourceHeatPumpBoiler와 동일한 로직
        den = max(1e-6, self.T_comb_setpoint_K - self.T_sup_w_K)
        alp = min(1.0, max(0.0, (self.T_serv_w_K - self.T_sup_w_K) / den))
        
        # 유량 계산
        dV_w_sup_comb = alp * dV_w_serv  # 보일러로 들어가는 급수 유량
        dV_w_sup_mix = (1 - alp) * dV_w_serv  # 믹싱밸브로 직접 들어가는 상수도 유량
        
        # 보일러 출수 온도 (고정 설정 온도 사용)
        T_w_comb_out = self.T_comb_setpoint
        
        # 실제 서비스 온도 계산 (믹싱 밸브)
        T_serv_w_actual_K = alp * self.T_comb_setpoint_K + (1 - alp) * self.T_sup_w_K
        T_serv_w_actual = cu.K2C(T_serv_w_actual_K)
        
        # 연소실 계산
        # Q_comb_load는 이미 보일러 유량 기준으로 계산되어 전달됨
        # 실제 필요한 가스 에너지: E_NG = Q_comb_load / eta_comb
        E_NG = Q_comb_load / self.eta_comb if self.eta_comb > 0 else 0.0
        
        # 실제 보일러 출수 열량
        Q_w_comb_out = c_w * rho_w * dV_w_sup_comb * (self.T_comb_setpoint_K - T0_K)
        
        # 배기가스 열량
        Q_exh = (1 - self.eta_comb) * E_NG
        
        # 급수 열량
        Q_w_sup = c_w * rho_w * dV_w_sup_comb * (self.T_sup_w_K - T0_K)
        
        # 믹싱밸브 계산
        Q_w_sup_mix = c_w * rho_w * dV_w_sup_mix * (self.T_sup_w_K - T0_K)
        Q_w_serv = c_w * rho_w * dV_w_serv * (T_serv_w_actual_K - T0_K)
        
        # 엔트로피 계산
        T_NG = T0_K / (1 - ex_eff_NG)  # 천연가스 온도 [K]
        
        S_NG = (1 / T_NG) * E_NG
        S_w_sup = c_w * rho_w * dV_w_sup_comb * math.log(self.T_sup_w_K / T0_K)
        S_w_comb_out = c_w * rho_w * dV_w_sup_comb * math.log(self.T_comb_setpoint_K / T0_K)
        S_exh = (1 / self.T_exh_K) * Q_exh
        S_g_comb = (S_w_comb_out + S_exh) - (S_NG + S_w_sup)
        
        S_w_sup_mix = c_w * rho_w * dV_w_sup_mix * math.log(self.T_sup_w_K / T0_K)
        S_w_serv = c_w * rho_w * dV_w_serv * math.log(T_serv_w_actual_K / T0_K)
        S_g_mix = S_w_serv - (S_w_comb_out + S_w_sup_mix)
        
        # 엑서지 계산
        X_NG = ex_eff_NG * E_NG
        X_w_sup = c_w * rho_w * dV_w_sup_comb * ((self.T_sup_w_K - T0_K) - T0_K * math.log(self.T_sup_w_K / T0_K))
        X_w_comb_out = c_w * rho_w * dV_w_sup_comb * ((self.T_comb_setpoint_K - T0_K) - T0_K * math.log(self.T_comb_setpoint_K / T0_K))
        X_exh = (1 - T0_K / self.T_exh_K) * Q_exh
        X_c_comb = S_g_comb * T0_K
        
        X_w_sup_mix = c_w * rho_w * dV_w_sup_mix * ((self.T_sup_w_K - T0_K) - T0_K * math.log(self.T_sup_w_K / T0_K))
        X_w_serv = c_w * rho_w * dV_w_serv * ((T_serv_w_actual_K - T0_K) - T0_K * math.log(T_serv_w_actual_K / T0_K))
        X_c_mix = S_g_mix * T0_K
        
        # 총 엑서지 소멸
        X_c_tot = X_c_comb + X_c_mix
        X_eff = X_w_serv / X_NG if X_NG > 0 else 0.0
        
        # 결과 딕셔너리 생성
        result = {
            'is_on': True,
            'converged': True,
            
            'Q_comb_load [W]': Q_comb_load,
            'E_NG [W]': E_NG,
            'Q_w_comb_out [W]': Q_w_comb_out,
            'Q_exh [W]': Q_exh,
            'Q_w_sup [W]': Q_w_sup,
            'Q_w_sup_mix [W]': Q_w_sup_mix,
            'Q_w_serv [W]': Q_w_serv,
            
            'dV_w_serv [m3/s]': dV_w_serv,
            'dV_w_sup_comb [m3/s]': dV_w_sup_comb,
            'dV_w_sup_mix [m3/s]': dV_w_sup_mix,
            
            'T0 [°C]': T0,
            'T_serv_w [°C]': T_serv_w_actual,
            'T_sup_w [°C]': self.T_sup_w,
            'T_w_comb_out [°C]': T_w_comb_out,
            'T_exh [°C]': self.T_exh,
            
            'alp [-]': alp,
            
            # 엔트로피
            'S_NG [W/K]': S_NG,
            'S_w_sup [W/K]': S_w_sup,
            'S_w_comb_out [W/K]': S_w_comb_out,
            'S_exh [W/K]': S_exh,
            'S_g_comb [W/K]': S_g_comb,
            'S_w_sup_mix [W/K]': S_w_sup_mix,
            'S_w_serv [W/K]': S_w_serv,
            'S_g_mix [W/K]': S_g_mix,
            
            # 엑서지
            'X_NG [W]': X_NG,
            'X_w_sup [W]': X_w_sup,
            'X_w_comb_out [W]': X_w_comb_out,
            'X_exh [W]': X_exh,
            'X_c_comb [W]': X_c_comb,
            'X_w_sup_mix [W]': X_w_sup_mix,
            'X_w_serv [W]': X_w_serv,
            'X_c_mix [W]': X_c_mix,
            'X_c_tot [W]': X_c_tot,
            'X_eff [-]': X_eff,
        }
        
        # 밸런스 딕셔너리 생성
        result['energy_balance'] = {
            "combustion chamber": {
                "in": {
                    "E_NG": E_NG,
                    "Q_w_sup": Q_w_sup
                },
                "out": {
                    "Q_w_comb_out": Q_w_comb_out,
                    "Q_exh": Q_exh
                }
            },
            "mixing valve": {
                "in": {
                    "Q_w_comb_out": Q_w_comb_out,
                    "Q_w_sup_mix": Q_w_sup_mix
                },
                "out": {
                    "Q_w_serv": Q_w_serv
                }
            }
        }
        
        result['entropy_balance'] = {
            "combustion chamber": {
                "in": {
                    "S_NG": S_NG,
                    "S_w_sup": S_w_sup
                },
                "out": {
                    "S_w_comb_out": S_w_comb_out,
                    "S_exh": S_exh
                },
                "gen": {
                    "S_g_comb": S_g_comb
                }
            },
            "mixing valve": {
                "in": {
                    "S_w_comb_out": S_w_comb_out,
                    "S_w_sup_mix": S_w_sup_mix
                },
                "out": {
                    "S_w_serv": S_w_serv
                },
                "gen": {
                    "S_g_mix": S_g_mix
                }
            }
        }
        
        result['exergy_balance'] = {
            "combustion chamber": {
                "in": {
                    "X_NG": X_NG,
                    "X_w_sup": X_w_sup
                },
                "out": {
                    "X_w_comb_out": X_w_comb_out,
                    "X_exh": X_exh
                },
                "con": {
                    "X_c_comb": X_c_comb
                }
            },
            "mixing valve": {
                "in": {
                    "X_w_comb_out": X_w_comb_out,
                    "X_w_sup_mix": X_w_sup_mix
                },
                "out": {
                    "X_w_serv": X_w_serv
                },
                "con": {
                    "X_c_mix": X_c_mix
                }
            }
        }
        
        return result

    def _calc_off_state(self, T0):
        """
        가스 보일러 OFF 상태 계산 메서드.
        
        Args:
            T0 (float): 엑서지 분석 기준 온도(=외기온도) [°C]
                엑서지 계산의 기준점으로 사용되는 외기 온도
        
        Returns:
            dict: OFF 상태 결과 딕셔너리
        """
        # 기본값으로 ON 상태 템플릿 생성
        result = self._calc_on_state(
            Q_comb_load=0.0,
            T0=T0,
            dV_w_serv=0.0
        )
        
        # 모든 숫자 값을 0으로 설정
        for key, value in result.items():
            if isinstance(value, (int, float)) and 'T_' not in key and 'alp' not in key:
                result[key] = 0.0
        
        # OFF 상태 플래그 설정
        result['is_on'] = False
        result['converged'] = True
        
        # 실제 서비스 온도 계산 (믹싱 밸브, OFF 상태에서도 일관성 유지)
        den = max(1e-6, self.T_comb_setpoint_K - self.T_sup_w_K)
        alp = min(1.0, max(0.0, (self.T_serv_w_K - self.T_sup_w_K) / den))
        T_serv_w_actual_K = alp * self.T_comb_setpoint_K + (1 - alp) * self.T_sup_w_K
        T_serv_w_actual = cu.K2C(T_serv_w_actual_K)
        
        # 온도 값은 유지
        result['T0 [°C]'] = T0
        result['T_serv_w [°C]'] = T_serv_w_actual
        result['T_sup_w [°C]'] = self.T_sup_w
        result['T_w_comb_out [°C]'] = self.T_comb_setpoint
        result['T_exh [°C]'] = self.T_exh
        
        return result

    def analyze_steady(
        self,
        T0,
        dV_w_serv=None,  # 급탕 유량 [m3/s], None이면 0으로 가정
        return_dict=True,  # True면 dict 반환, False면 DataFrame 반환
    ):
        """
        정상상태 해석 함수.
        
        주어진 조건에서 가스 보일러의 정상상태 성능을 계산합니다.
        필요한 열량을 계산하여 보일러를 작동시킵니다.
        
        Args:
            T0 (float): 엑서지 분석 기준 온도(=외기온도) [°C]
                엑서지 계산의 기준점으로 사용되는 외기 온도
            dV_w_serv (float, optional): 급탕 유량 [m3/s]. None이면 0으로 가정
            return_dict (bool): True면 dict 반환, False면 DataFrame 반환
        
        Returns:
            dict 또는 pd.DataFrame: 정상상태 해석 결과
        """
        # 급탕 유량 설정
        if dV_w_serv is None:
            dV_w_serv = 0.0
        
        # Mixing valve 계산: 보일러 출수 온도와 상수도를 믹싱하여 서비스 온도로 공급
        # alp: 보일러 출수 비율, (1-alp): 상수도 직접 공급 비율
        den = max(1e-6, self.T_comb_setpoint_K - self.T_sup_w_K)
        alp = min(1.0, max(0.0, (self.T_serv_w_K - self.T_sup_w_K) / den))
        
        # 보일러로 들어가는 유량 계산
        dV_w_sup_comb = alp * dV_w_serv
        
        # 필요한 열량 계산: 보일러 유량 기준으로 계산
        # 보일러는 dV_w_sup_comb만큼의 물을 T_comb_setpoint로 가열해야 함
        Q_comb_load = c_w * rho_w * dV_w_sup_comb * (self.T_comb_setpoint_K - self.T_sup_w_K)
        
        # ON/OFF 상태 결정
        is_on = Q_comb_load > self.Q_comb_load_threshold
        
        # OFF 상태 체크
        if abs(Q_comb_load) <= self.Q_comb_load_threshold or not is_on:
            result = self._calc_off_state(T0=T0)
        else:
            result = self._calc_on_state(
                Q_comb_load=Q_comb_load,
                T0=T0,
                dV_w_serv=dV_w_serv
            )
        
        if return_dict:
            return result
        else:
            # DataFrame으로 변환
            return pd.DataFrame([result])
    
    def analyze_dynamic(
        self,
        simulation_period_sec,
        dt_s,
        schedule_entries,
        T0_schedule,
        heater_capacity_const=None,
        heater_capacity_schedule=None,
        result_save_csv_path=None,
    ):
        """
        동적 시뮬레이션을 실행합니다.
        
        Args:
            simulation_period_sec: 총 시뮬레이션 시간 [초]
            dt_s: 타임스텝 [초]
            schedule_entries: 급탕 사용 스케줄
                [(시작시간_str, 종료시간_str, 사용비율_float), ...]
                예: [("6:00", "6:30", 0.5), ("6:30", "7:00", 0.9)]
            T0_schedule: 엑서지 분석 기준 온도(=외기온도) 스케줄 [°C]
                엑서지 계산의 기준점으로 사용되는 외기 온도의 시간별 스케줄
            heater_capacity_const: 고정 히터 용량 [W] (None이면 자동 계산)
            heater_capacity_schedule: 히터 용량 스케줄 [W]
            result_save_csv_path: 결과 CSV 저장 경로
        
        Returns:
            pd.DataFrame: 시뮬레이션 타임스텝별 결과 데이터
        """
        # 실행 조건 판단
        if simulation_period_sec % dt_s != 0:
            raise ValueError("simulation_period_sec must be divisible by dt_s")
        
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
        self.dV_w_sup_comb = 0.0
        self.dV_w_sup_mix = 0.0
        
        # 스케줄 빌드
        self.w_use_frac = _build_schedule_ratios(schedule_entries, self.time)
        
        for n in tqdm(range(tN), desc="GasBoiler Simulating"):
            step_results = {}
            T0 = T0_schedule[n]
            
            # 급탕 유량 계산
            self.dV_w_serv = self.w_use_frac[n] * self.dV_w_serv_m3s
            
            # Mixing valve 계산: 보일러 출수 온도와 상수도를 믹싱하여 서비스 온도로 공급
            # alp: 보일러 출수 비율, (1-alp): 상수도 직접 공급 비율
            den = max(1e-6, self.T_comb_setpoint_K - self.T_sup_w_K)
            alp = min(1.0, max(0.0, (self.T_serv_w_K - self.T_sup_w_K) / den))
            
            # 보일러로 들어가는 유량 계산
            self.dV_w_sup_comb = alp * self.dV_w_serv
            self.dV_w_sup_mix = (1 - alp) * self.dV_w_serv
            
            # 필요한 열량 계산: 보일러 유량 기준으로 계산
            # 보일러는 dV_w_sup_comb만큼의 물을 T_comb_setpoint로 가열해야 함
            Q_comb_load = c_w * rho_w * self.dV_w_sup_comb * (self.T_comb_setpoint_K - self.T_sup_w_K)
            
            # ON/OFF 상태 결정
            is_on = (self.T_serv_w > self.T_sup_w) and (self.dV_w_sup_comb > 0)
            
            result = self._calc_on_state(
                Q_comb_load=Q_comb_load,
                T0=T0,
                dV_w_serv=self.dV_w_serv
            )
            
            step_results.update(result)
            step_results['is_on'] = is_on
            step_results['time [s]'] = time[n]
            
            results_data.append(step_results)
        
        results_df = pd.DataFrame(results_data)
        
        if result_save_csv_path:
            results_df.to_csv(result_save_csv_path, index=False)
        
        return results_df

