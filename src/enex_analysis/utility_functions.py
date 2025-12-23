#%%
import numpy as np
import math
from . import calc_util as cu
from scipy import integrate
from scipy.special import erf
from scipy.optimize import root_scalar

#%%
# constant
c_a = 1005 # Specific heat capacity of air [J/kgK]
rho_a = 1.225 # Density of air [kg/m³]
k_a = 0.0257 # Thermal conductivity of air [W/mK]

c_w   = 4186 # Water specific heat [J/kgK]
rho_w = 1000
mu_w = 0.001 # Water dynamic viscosity [Pa.s]
k_w = 0.606 # Water thermal conductivity [W/mK]

sigma = 5.67*10**-8 # Ste_fan-Boltzmann constant [W/m²K⁴]

# https://www.notion.so/betlab/Scattering-of-photon-particles-coming-from-the-sun-and-their-energy-entropy-exergy-b781821ae9a24227bbf1a943ba9df51a?pvs=4#1ea6947d125d80ddb0a5caec50031ae3
k_D = 0.000462 # direct solar entropy coefficient [-]
k_d = 0.0014 # diffuse solar entropy coefficient [-]

# Shukuya - Exergy theory and applications in the built environment, 2013
# The ratio of chemical exergy to higher heating value of liquefied natural gas (LNG) is 0.93.
ex_eff_NG   = 0.93 # exergy efficiency of natural gas [-]

SP = np.sqrt(np.pi) # Square root of pi

#%%
# function
def air_dynamic_viscosity(T_K):
    '''
    Calculate air dynamic viscosity using Sutherland's formula.
    
    Parameters:
    T_K (float): Temperature [K]
    
    Returns:
    float: Dynamic viscosity [Pa·s]
    
    Reference: Sutherland's formula for air
    mu = mu0 * (T/T0)^1.5 * (T0 + S) / (T + S)
    where mu0 = 1.716e-5 Pa·s at T0 = 273.15 K, S = 110.4 K
    '''
    T0 = 273.15  # Reference temperature [K]
    mu0 = 1.716e-5  # Reference viscosity [Pa·s] at T0
    S = 110.4  # Sutherland constant [K] for air
    
    mu = mu0 * ((T_K / T0)**1.5) * ((T0 + S) / (T_K + S))
    return mu

def air_prandtl_number(T_K):
    '''
    Calculate air Prandtl number.
    
    Parameters:
    T_K (float): Temperature [K]
    
    Returns:
    float: Prandtl number [-]
    
    Note: Pr ≈ 0.71 for air at typical temperatures (20-50°C)
    Temperature dependence is weak, so using constant value.
    '''
    # Pr = mu * cp / k
    # For air: Pr ≈ 0.71 (weak temperature dependence)
    return 0.71

def darcy_friction_factor(Re, e_d):
    '''
    Darcy 마찰계수(Darcy friction factor)를 계산합니다. 

    사용 수식:
        - Laminar flow (Re < 2300):
            f = 64 / Re

        - Turbulent flow (Re >= 2300):
            Haaland 근사식(Haaland equation, 1983) 사용:
            f = 0.25 / [log10(e/D/3.7 + 5.74/Re^0.9)]^2

        [참고 논문]
        - Haaland, S.E., "Simple and explicit formulas for the friction factor in turbulent pipe flow," 
          J. Fluids Engineering, 1983.
        - Colebrook-White 식을 수치 근사한 공식

    입력 변수:
    - Re (float): Reynolds 수
    - e_d (float): 상대 거칠기 (e/D, roughness ratio)
        - e (float): 거칠기 (e, roughness)
        - D (float): 직경 (D, diameter)
         
    반환값:
        - float: Darcy 마찰계수
    '''
    if Re < 2300:
        # Laminar flow
        return 64 / Re
    else:
        # Turbulent flow (Haaland equation 참조)
        return 0.25 / (math.log10(e_d / 3.7 + 5.74 / Re ** 0.9)) ** 2

def linear_function(x, a, b):
    return a * x + b

def quadratic_function(x, a, b, c):
    return a * x ** 2 + b * x + c

def cubic_function(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d

def quartic_function(x, a, b, c, d, e):
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e

def f(x):
    return x*erf(x) - (1-np.exp(-x**2))/SP

def chi(s, rb, H, z0=0):
    h = H * s
    d = z0 * s
    
    temp = np.exp(-(rb*s)**2) / (h * s)
    Is = 2*f(h) + 2*f(h+2*d) - f(2*h+2*d) - f(2*d)
    
    return temp * Is

_g_func_cache = {}
def G_FLS(t, ks, as_, rb, H):
    key = (round(t, 0), round(ks, 2), round(as_, 6), round(rb, 2), round(H, 0))
    if key in _g_func_cache:
        return _g_func_cache[key]

    factor = 1 / (4 * np.pi * ks)
    
    lbs = 1 / np.sqrt(4*as_*t)
    
    # Scalar 값인 경우 shape == (,).
    single = len(lbs.shape) == 0
    # 0차원에 1차원으로 변경.
    lbs = lbs.reshape(-1)
        
    # 0 부터 inf 까지의 적분값 미리 계산.
    total = integrate.quad(chi, 0, np.inf, args=(rb, H))[0]
    # ODE 초기값.
    first = integrate.quad(chi, 0, lbs[0], args=(rb, H))[0]
   
    # Scipy의 ODE solver의 인자의 함수 형태는 dydx = f(y, s).
    def func(y, s):
        return chi(s, rb, H, z0=0)
    
    values = total - integrate.odeint(func, first, lbs)[:, 0]
    
    # Single time 값은 첫 번째 값만 선택하여 float를 리턴하도록 함.
    if single:
        values = values[0]

    result = factor * values
    _g_func_cache[key] = result
    return result


def _build_schedule_ratios(entries, t_array):
    """
    스케줄 엔트리로부터 각 타임스텝(t_array)에 대한 사용 비율 배열을 생성합니다.

    Parameters
    ----------
    entries : list of tuple
        스케줄 엔트리 리스트. 각 항목은 (start_str, end_str, frac) 형태입니다.
        - start_str, end_str: "H:M" 또는 "H" 형식의 문자열 (예: "6:00", "23:30", "24:00" 등).
          "24:00"은 하루의 끝(= 24*cu.h2s초)로 특별 처리됩니다.
        - frac: 해당 구간에서의 사용 비율(부동소수). 0.0 ~ 1.0 범위로 클립됩니다.
        구간은 반개구간 [start, end)으로 처리됩니다.

    t_array : numpy.ndarray
        초 단위의 타임스텝 배열(예: np.arange(0, sim_seconds, dt)). 배열의 각 요소에 대해
        하루(24*cu.h2s)로 모듈로 연산을 하여 같은 하루 내 시간으로 맵핑합니다.

    Returns
    -------
    numpy.ndarray
        t_array와 같은 shape의 배열. 각 타임스텝에서의 스케줄 비율(0.0 ~ 1.0).
        여러 엔트리가 겹칠 경우 해당 위치에서의 값은 엔트리별 frac의 최댓값으로 결정됩니다.

    동작 요약
    ----------
    - 시간 문자열은 내부적으로 _time_str_to_sec를 사용해 초로 변환합니다.
    - end가 24*cu.h2s인 경우(예: "24:00")에는 하루의 마지막 직전 값으로 조정합니다.
    - 구간이 자정을 가로지를 수 있다면(예: start=23:00, end=02:00) OR 마스크로 처리하여
      자정을 넘어가는 구간도 올바르게 커버합니다.
    - 비율(frac)은 np.clip으로 0~1 범위로 제한됩니다.
    - 반환값은 np.clip으로 최종적으로 0~1 범위를 보장합니다.

    Examples
    --------
    entries = [("6:00","7:00",0.5), ("6:30","8:00",0.8)]
    -> 6:30~7:00 구간은 max(0.5,0.8)=0.8가 적용됩니다.

    Notes
    -----
    - t_array는 초 단위 연속시간 배열이어야 하며, 내부적으로 24*cu.h2s으로 모듈로 연산됩니다.
    - 동일한 시간대에 여러 엔트리가 있을 때 우선순위는 frac의 최대값입니다(덮어쓰기 대신 병합).
    """
    day = 24*cu.h2s
    secs_mod = (t_array % day)
    sched = np.zeros_like(t_array, dtype=float)

    def _time_str_to_sec(time_str):
        """
        시간 문자열 형태(ex. "H", "H:M")를 하루 단위의 정수 초(0 ~ 86400)로 변환합니다.

        Parameters
        ----------
        time_str : str
            "H" 또는 "H:M" 형식의 시간 문자열.
            - "H"는 시(hour)를 나타내며, 0~24 범위의 정수입니다.
            - "H:M"는 시와 분(minute)을 콜론(:)으로 구분한 형식입니다.
                시는 0~24, 분은 0~59 범위의 정수입니다.
            - "24:00"은 하루의 끝(= 24*cu.h2s초)으로 특별 처리됩니다.

        동작
        - 시와 분을 분리하여 초로 변환: seconds = (h % 24) * 3600 + m * 60
        - 입력이 "24:00" (문자열이 '24'로 시작하고 h%24 == 0인 경우) 이면 24*cu.h2s을 반환한다.
        - 시(h)는 24로 모듈로 연산되어 24 이상의 표기는 일단 0..23로 매핑된다.
        
        Returns
        -------
        int
            하루 단위의 정수 초 (0 ~ 86400).
        """ 
        h, m = (time_str.split(':') + ['0'])[:2]
        h = int(h) % 24
        m = int(m)
        return 24*cu.h2s if (h == 0 and time_str.strip().startswith('24')) else h*cu.h2s + m*60
        
    # 스케줄 엔트리 처리
    for start_str, end_str, frac in entries:
        s = _time_str_to_sec(start_str)
        e = _time_str_to_sec(end_str)
        if e == 24*cu.h2s: e = 24*cu.h2s - 1e-9
        ratio = np.clip(frac, 0.0, 1.0)
        if s == e: continue
        
        if s < e:
            mask = (secs_mod >= s) & (secs_mod < e)
        else:
            mask = (secs_mod >= s) | (secs_mod < e)
        sched[mask] = np.maximum(sched[mask], ratio)

    return np.clip(sched, 0.0, 1.0)

#%%
def calc_simple_tank_UA(
        # Tank size [m]
        r0 = 0.2,
        H = 0.8,
        # Tank layer thickness [m]
        x_shell = 0.01,
        x_ins   = 0.10,
        # Tank thermal conductivity [W/mK]
        k_shell = 25,  
        k_ins   = 0.03,
        # External convective heat transfer coefficient [W/m²K]
        h_o     = 10,
        ):
    
        r1 = r0 + x_shell
        r2 = r1 + x_ins
        
        # Tank surface areas [m²]
        A_side = 2 * math.pi * r2 * H
        A_base = math.pi * r0**2
        R_base_unit = x_shell / k_shell + x_ins / k_ins # [m2K/W]
        R_side_unit = math.log(r1 / r0) / (2 * math.pi * k_shell) + math.log(r2 / r1) / (2 * math.pi * k_ins) # [mK/W]
        
        # Thermal resistances [K/W]
        R_base = R_base_unit / A_base # [K/W]
        R_side = R_side_unit / H # [K/W]
        
        # Thermal resistances [K/W]
        R_base_ext = 1 / (h_o * A_base)
        R_side_ext = 1 / (h_o * A_side)

        # Total thermal resistances [K/W]
        R_base_tot = R_base + R_base_ext
        R_side_tot = R_side + R_side_ext

        # U-value [W/K]
        U_tank = 2/R_base_tot + 1/R_side_tot 
        return U_tank


#%%
def calc_lmtd_counter_flow(T_hot_in_K, T_hot_out_K, T_cold_in_K, T_cold_out_K):
    '''
    대향류(counter-flow) 열교환기의 LMTD를 계산합니다.
    
    Args:
        T_hot_in_K (float): 고온 유체 입구 온도 [K]
        T_hot_out_K (float): 고온 유체 출구 온도 [K]
        T_cold_in_K (float): 저온 유체 입구 온도 [K]
        T_cold_out_K (float): 저온 유체 출구 온도 [K]
    
    Returns:
        float: LMTD [K]
    '''
    # 대향류: 고온 입구 ↔ 저온 출구, 고온 출구 ↔ 저온 입구
    dT1 = T_hot_in_K - T_cold_out_K
    dT2 = T_hot_out_K - T_cold_in_K
    
    if dT1 <= 0 or dT2 <= 0:
        return np.nan
    
    if abs(dT1 - dT2) < 1e-4:
        return (dT1 + dT2) / 2
    else:
        return (dT1 - dT2) / np.log(dT1 / dT2)

def calc_lmtd_parallel_flow(T_hot_in_K, T_hot_out_K, T_cold_in_K, T_cold_out_K):
    '''
    병류(parallel-flow) 열교환기의 LMTD를 계산합니다.
    
    Args:
        T_hot_in_K (float): 고온 유체 입구 온도 [K]
        T_hot_out_K (float): 고온 유체 출구 온도 [K]
        T_cold_in_K (float): 저온 유체 입구 온도 [K]
        T_cold_out_K (float): 저온 유체 출구 온도 [K]
    
    Returns:
        float: LMTD [K]
    '''
    # 병류: 고온 입구 ↔ 저온 입구, 고온 출구 ↔ 저온 출구
    dT1 = T_hot_in_K - T_cold_in_K
    dT2 = T_hot_out_K - T_cold_out_K
    
    if dT1 <= 0 or dT2 <= 0:
        return np.nan
    
    if abs(dT1 - dT2) < 1e-4:
        return (dT1 + dT2) / 2
    else:
        return (dT1 - dT2) / np.log(dT1 / dT2)

def calc_lmtd_constant_refrigerant_temp(T_ref_avg_K, T_air_in_K, T_air_out_K):
    '''
    냉매 온도가 일정하게 유지되는 경우의 LMTD를 계산합니다.
    
    냉매는 입출구 평균온도로 일정하게 유지되고, 공기는 입구부터 출구까지 온도가 변화합니다.
    이는 응축기나 증발기에서 냉매가 상변화를 하는 경우에 해당합니다.
    
    Args:
        T_ref_avg_K (float): 냉매 평균 온도 [K] (일정하게 유지됨)
        T_air_in_K (float): 공기 입구 온도 [K]
        T_air_out_K (float): 공기 출구 온도 [K]
    
    Returns:
        float: LMTD [K]
    
    Notes:
        - 냉매 온도가 일정하므로 LMTD는 단순화된 형태로 계산됩니다.
        - Q>0 (냉매가 열 방출): T_ref_avg > T_air_in, T_ref_avg > T_air_out
          → dT_in = T_ref_avg - T_air_in, dT_out = T_ref_avg - T_air_out
        - Q<0 (냉매가 열 흡수): T_ref_avg < T_air_in, T_ref_avg < T_air_out
          → dT_in = T_air_in - T_ref_avg, dT_out = T_air_out - T_ref_avg
    '''
    # 온도차 계산 (부호를 유지하여 계산)
    dT_in = T_ref_avg_K - T_air_in_K
    dT_out = T_ref_avg_K - T_air_out_K
    
    # 물리적 타당성 검증: dT_in과 dT_out의 부호가 일치해야 함
    if dT_in * dT_out <= 0:
        # 냉매 온도가 공기 입출구 온도 사이에 있는 경우 (물리적으로 불가능)
        return np.nan
    
    # 절댓값으로 LMTD 계산
    dT_in_abs = abs(dT_in)
    dT_out_abs = abs(dT_out)
    
    if dT_in_abs <= 0 or dT_out_abs <= 0:
        return np.nan
    
    # LMTD 계산
    if abs(dT_in_abs - dT_out_abs) < 1e-4:
        return (dT_in_abs + dT_out_abs) / 2
    else:
        return (dT_in_abs - dT_out_abs) / np.log(dT_in_abs / dT_out_abs)

def calculate_heat_transfer_coefficient(dV_fan, dV_fan_design, A_cross, UA):
    '''
    Dittus-Boelter 방정식을 기반으로 한 열전달 계수를 계산합니다.
    
    이 함수는 공기 유속에 따른 열전달 계수를 계산합니다.
    Dittus-Boelter 방정식: Nu = 0.023 * Re^0.8 * Pr^n
    유속의 0.8승에 비례하는 형태로 계산됩니다.
    
    Args:
        dV_fan (float): 풍량 [m³/s]
        A_cross (float): 열교환기 단면적 [m²]
        UA (float): 냉매 측 저항 및 보정 계수 [W/K]
            Dittus-Boelter 방정식의 계수로, U = UA * V^0.8 형태
    
    Returns:
        float: 전체 열전달 계수 U [W/K]
    
    Notes:
        - 속도 계산: v = dV_fan / A_cross
        - 열전달 계수: U = UA * v^0.8
    '''
    # 속도 계산
    v = dV_fan / A_cross if A_cross > 0 else 0
    v_design = dV_fan_design / A_cross if A_cross > 0 else 0
    return UA * (v / v_design) ** 0.8

def find_fan_airflow_for_heat_transfer(Q_ref_target, T_air_in_C, T_ref_in_K, T_ref_out_K, A_cross, UA_design):
    '''
    목표 열교환량을 만족하는 팬 풍량을 계산합니다.

    풍속에 따라 동적으로 변화하는 열전달 계수(UA)를 고려하여 계산합니다.
    UA는 Dittus-Boelter 방정식에 따라 풍속의 0.8승에 비례합니다.

    Args:
        Q_ref_target (float): 목표 열교환량 [W]
        T_air_in_C (float): 공기 입구 온도 [°C]
        T_ref_in_K (float): 냉매 입구 온도 [K]
        T_ref_out_K (float): 냉매 출구 온도 [K]
        A_cross (float): 열교환기 단면적 [m²]
        UA_design (float): 설계 풍량에서의 열전달 계수 [W/K]

    Returns:
        float: 필요 풍량 [m³/s]
    '''
    T_air_in_K = cu.C2K(T_air_in_C)

    def error_function(dV_fan):
        if dV_fan <= 0:
            return 1e8
        T_air_out_K = T_air_in_K + Q_ref_target / (c_a * rho_a * dV_fan)
        max_dT_air = 30.0
        if abs(T_air_out_K - T_air_in_K) > max_dT_air:
            return 1e8
        T_ref_avg_K = (T_ref_in_K + T_ref_out_K) / 2
        TEMP_TOLERANCE_K = 1e-6
        if Q_ref_target > 0:
            if not (T_ref_avg_K > max(T_air_in_K, T_air_out_K) - TEMP_TOLERANCE_K):
                return 1e8
        else:
            if not (T_ref_avg_K < min(T_air_in_K, T_air_out_K) + TEMP_TOLERANCE_K):
                return 1e8

        UA = calculate_heat_transfer_coefficient(dV_fan, A_cross, UA_design)
        LMTD = calc_lmtd_constant_refrigerant_temp(
            T_ref_avg_K=T_ref_avg_K,
            T_air_in_K=T_air_in_K,
            T_air_out_K=T_air_out_K
        )
        Q_calculated = UA * LMTD
        return Q_calculated - Q_ref_target

    max_dT_air = 30.0
    if abs(Q_ref_target) > 0:
        dV_min = abs(Q_ref_target) / (c_a * rho_a * max_dT_air)
        dV_min = max(0.01, dV_min)
    else:
        dV_min = 0.01
    dV_max = 10.0

    sol = root_scalar(error_function, bracket=[dV_min, dV_max], method='bisect')
    if sol.converged:
        return sol.root
    else:
        raise ValueError("Fan flow rate solution did not converge.")

def calculate_fan_power(dV_fan, fan_params, vsd_coeffs):
    '''
    ASHRAE 90.1 VSD Curve를 사용하여 팬 전력을 계산합니다.
    
    Args:
        dV_fan (float): 현재 풍량 [m³/s]
        fan_params (dict): 팬 파라미터 (fan_design_flow_rate, fan_design_power)
        vsd_coeffs (dict): VSD Curve 계수 (c1~c5)
    
    Returns:
        float: 팬 전력 [W]
    '''
    # 설계 파라미터 추출
    fan_design_flow_rate = fan_params.get('fan_design_flow_rate', None)
    fan_design_power = fan_params.get('fan_design_power', None)
    
    # 설계 파라미터가 없는 경우 에러
    if fan_design_flow_rate is None or fan_design_power is None:
        raise ValueError("fan_design_flow_rate and fan_design_power must be provided in fan_params")
    
    # 풍량 검증
    if dV_fan <= 0:
        raise ValueError("fan flow rate must be greater than 0")
    
    # VSD Curve 계수 추출
    c1 = vsd_coeffs.get('c1', 0.0013)
    c2 = vsd_coeffs.get('c2', 0.1470)
    c3 = vsd_coeffs.get('c3', 0.9506)
    c4 = vsd_coeffs.get('c4', -0.0998)
    c5 = vsd_coeffs.get('c5', 0.0)
    
    # 풍량 비율 계산 (flow fraction)
    flow_fraction = dV_fan / fan_design_flow_rate
    
    # Part-load 비율 계산: PLR = c1 + c2*x + c3*x² + c4*x³ + c5*x⁴
    x = flow_fraction
    PLR = c1 + c2*x + c3*x**2 + c4*x**3 + c5*x**4
    
    # Part-load 비율이 음수가 되지 않도록 보정
    PLR = max(0.0, PLR)
    
    # 팬 전력 계산
    fan_power = fan_design_power * PLR
    
    return fan_power

# %%

