import numpy as np
import math
from . import calc_util as cu
# import calc_util as cu
from dataclasses import dataclass
import dartwork_mpl as dm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import integrate
from scipy.special import erf
dm.style.use('scientific')

#%%
# constant
c_a = 1005 # Specific heat capacity of air [J/kgK]
rho_a = 1.225 # Density of air [kg/m³]
k_a = 0.0257 # Thermal conductivity of air [W/mK]

c_w   = 4186 # Water specific heat [J/kgK]
rho_w = 1000
mu_w = 0.001 # Water dynamic viscosity [Pa.s]
k_w = 0.606 # Water thermal conductivity [W/mK]
g = 9.81         # 중력가속도 [m/s²]
beta = 2.07e-4   # 물의 체적팽창계수 [1/K] (약 20°C 기준)

sigma = 5.67*10**-8 # Stefan-Boltzmann constant [W/m²K⁴]

# https://www.notion.so/betlab/Scattering-of-photon-particles-coming-from-the-sun-and-their-energy-entropy-exergy-b781821ae9a24227bbf1a943ba9df51a?pvs=4#1ea6947d125d80ddb0a5caec50031ae3
k_D = 0.000462 # direct solar entropy coefficient [-]
k_d = 0.0014 # diffuse solar entropy coefficient [-]

# Shukuya - Exergy theory and applications in the built environment, 2013
# The ratio of chemical exergy to higher heating value of liquefied natural gas (LNG) is 0.93.
ex_eff_NG   = 0.93 # exergy efficiency of natural gas [-]

SP = np.sqrt(np.pi) # Square root of pi

#%%
# function
def darcy_friction_factor(Re, e_d):
    '''
    Calculate the Darcy friction factor for given Reynolds number and relative roughness.
    
    Parameters:
    Re (float): Reynolds number
    e_d (float): Relative roughness (e/D)
    
    Returns:
    float: Darcy friction factor
    '''
    # Laminar flow
    if Re < 2300:
        return 64 / Re
    # Turbulent flow
    else:
        return 0.25 / (math.log10(e_d / 3.7 + 5.74 / Re ** 0.9)) ** 2

def calc_h_vertical_plate(T_s, T_inf, L):
    '''
    📌 Function: compute_natural_convection_h_cp
    이 함수는 자연 대류에 의한 열전달 계수를 계산합니다.
    🔹 Parameters
        - T_s (float): 표면 온도 [K]
        - T_inf (float): 유체 온도 [K]
        - L (float): 특성 길이 [m]
    🔹 Return
        - h_c_w (float): 열전달 계수 [W/m²K]
    🔹 Example
        ```
        h_c_w = compute_natural_convection_h_cp(T_s, T_inf, L)
        ```
    🔹 Note
        - 이 함수는 자연 대류에 의한 열전달 계수를 계산하는 데 사용됩니다.
        - L은 특성 길이로, 일반적으로 물체의 길이나 직경을 사용합니다.
        - 이 함수는 Churchill & Chu 식을 사용하여 열전달 계수를 계산합니다.
    '''
    # 공기 물성치 @ 40°C
    nu = 1.6e-5  # 0.000016 m²/s
    k_air = 0.027 # W/m·K
    Pr = 0.7 # Prandtl number 
    beta = 1 / ((T_s + T_inf)/2) # 1/K
    g = 9.81 # m/s²

    # Rayleigh 수 계산
    delta_T = T_s - T_inf
    Ra_L = g * beta * delta_T * L**3 / (nu**2) * Pr

    # Churchill & Chu 식 https://doi.org/10.1016/0017-9310(75)90243-4
    Nu_L = (0.825 + (0.387 * Ra_L**(1/6)) / (1 + (0.492/Pr)**(9/16))**(8/27))**2
    h_c_w = Nu_L * k_air / L  # [W/m²K]
    
    return h_c_w

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
   
    # Scipy의 ODE solver의 인자의 함수 형태는 dydx = f(y, x).
    def func(y, s):
        return chi(s, rb, H, z0=0)
    
    values = total - integrate.odeint(func, first, lbs)[:, 0]
    
    # Single time 값은 첫 번째 값만 선택하여 float를 리턴하도록 함.
    if single:
        values = values[0]

    result = factor * values
    _g_func_cache[key] = result
    return result

def generate_entropy_exergy_term(energy_term, Tsys, T0, fluid = None):
    """
    Calculates the entropy and exergy terms based on the provided energy term and temperatures.
    Parameters:
        energy_term (float): The energy value for which entropy and exergy are to be calculated.
        Tsys (float): The system temperature [K].
        T0 (float): The reference (environment) temperature [K].
        fluid (optional): If provided, modifies the entropy calculation using a logarithmic relation.
    Returns:
        tuple:
            entropy_term (float): The calculated entropy term.
            exergy_term (float): The calculated exergy term.
    """
    entropy_term = energy_term / Tsys
    
    if fluid:
        if Tsys - T0 != 0:
            entropy_term = energy_term * math.log(Tsys/T0) / (Tsys - T0)
        elif Tsys - T0 == 0:
            entropy_term = 0
            
    exergy_term = energy_term - entropy_term * T0

    if not fluid and Tsys < T0: # Cool exergy (fluid의 경우 항상 exergy term이 양수임 엑서지 항을 구성하는 {(A-B)-ln(A/B)*B} 구조는 항상 A>0, B>0일 때 양수일 수 밖에 없기 때문)
        exergy_term = -exergy_term
    return entropy_term, exergy_term

def calc_exergy_flow(G, T, T0):
    """
    Description:
    물질 흐름(advection)에 의한 엑서지율을 계산.
    Xf = G * ((T - T0) - T0 * ln(T/T0) )
    
    Parameters:
        G : 열용량 유량 = 비열 x 밀도 x 유량 [W/K]
        T : 흐름의 온도 [K]
        T0 : 기준(환경) 온도 (T_dead_state) [K]

    Returns: 엑서지 유량 [W]
    """
    
    # G * ( (T - T0) - T0 * ln(T/T0) )
    return G * ((T - T0) - T0 * np.log(T / T0))

######################################################################################### 미완
def calc_Orifice_flow_coefficient(D0, D1):
    """
    Calculate the orifice flow coefficient based on the diameters.

    ---------------
     ->      |
     D0     D1 ->
     ->      |
    ---------------

    Parameters:
    - D0 : float
        pipe diameter [m]
    - D1 : float
        hole diameter [m]

    Returns:
    - C_d : float
        Orifice flow coefficient (dimensionless)
    """
    
    m = D1 / D0 # 개공비
    return (m)**2
#########################################################################################

def calc_boussinessq_mixing_flow(T_upper, T_lower, A, dz, C_d=0.1):
    """
    # To do
    C_d 값을 물리적 수식에 기반해 계산하는 알고리즘 추가 필요. 
    
    
    두 인접 노드 간의 부시네스크 근사에 기반한 혼합 유량을 계산합니다.
    혼합은 하단 노드의 온도가 상단 노드보다 높아 중력적으로 불안정할 때만 발생합니다.

    Parameters:
    -----------
    T_upper : float
        상단 노드의 온도 [K]
    T_lower : float
        하단 노드의 온도 [K]
    A : float
        탱크 단면적 [m²]
    dz : float
        노드 높이 [m]
    C_d : float, optional
        유량 계수 (경험적 상수), 기본값 0.1

    Returns:
    --------
    dV_mix : float
        두 노드 간 교환되는 체적 유량 [m³/s]
    """
    if T_upper < T_lower:
        # 상단이 더 차가우면 (밀도가 높으면) 불안정하여 혼합 발생
        delta_T = T_lower - T_upper
        dV_mix = C_d * A * math.sqrt(2 * g * beta * delta_T * dz)
        return dV_mix # 위에서 아래
    else:
        # 안정적인 상태에서는 혼합 없음
        return 0.0

def calc_UA_tank_arr(r0, x_shell, x_ins, k_shell, k_ins, H, N, h_w, h_o):
    """
    Overall heat-loss UA per vertical segment of a cylindrical tank (radial through side;
    planar through bottom/top). Side applies to all nodes; bottom/top add in parallel for node 1 and N.

    Calculate the overall heat transfer coefficient (U-value) of a cylindrical tank.
    Parameters:
    r0 : Inner radius of the tank [m]
    x_shell : Thickness of the tank shell [m]
    x_ins : Thickness of the insulation layer [m]
    k_shell : Thermal conductivity of the tank shell material [W/mK]
    k_ins : Thermal conductivity of the insulation material [W/mK]
    H : Height of the tank [m]
    N : Number of segments 
    h_w : Internal convective heat transfer coefficient [W/m²K]
    h_o : External convective heat transfer coefficient [W/m²K]
    Returns:
    UA_arr : Array of overall heat transfer coefficients for each segment [W/K]
    """
    dz = H / N
    r1 = r0 + x_shell
    r2 = r1 + x_ins

    # --- Areas ---
    # Side (per segment)
    A_side_in_seg  = 2.0 * math.pi * r0 * dz   # inner wetted area (for h_w)
    A_side_out_seg = 2.0 * math.pi * r2 * dz   # outer area (for h_o)
    # Bases (single discs)
    A_base_in  = math.pi * r0**2               # internal disc area (for h_w)
    A_base_out = math.pi * r2**2               # external disc area (for h_o)

    # --- Side: convection (in/out) + cylindrical conduction (shell + insulation) ---
    # Conduction (cylindrical) per segment
    R_side_cond_shell = math.log(r1 / r0) / (2.0 * math.pi * k_shell * dz)
    R_side_cond_ins   = math.log(r2 / r1) / (2.0 * math.pi * k_ins   * dz)
    R_side_cond = R_side_cond_shell + R_side_cond_ins  # [K/W]

    R_side_w   = 1.0 / (h_w * A_side_in_seg)          # [K/W]
    R_side_ext = 1.0 / (h_o * A_side_out_seg)         # [K/W]
    R_side_tot = R_side_w + R_side_cond + R_side_ext  # [K/W]  (series)

    # --- Bottom/Top discs: convection (in/out) + planar conduction (shell + insulation) ---
    # 권장: 각 층의 면적을 구분하여 직렬 합
    R_base_cond_shell = x_shell / (k_shell * A_base_in)   # [K/W]  (inner metal plate)
    R_base_cond_ins   = x_ins   / (k_ins   * A_base_out)  # [K/W]  (outer insulation plate)
    R_base_cond = R_base_cond_shell + R_base_cond_ins

    R_base_w   = 1.0 / (h_w * A_base_in)   # [K/W]
    R_base_ext = 1.0 / (h_o * A_base_out)  # [K/W]
    R_base_tot = R_base_w + R_base_cond + R_base_ext  # [K/W] (series through the base)

    # --- Equivalent node-to-ambient resistances ---
    # Middle nodes: side only
    R_mid = R_side_tot

    # Node 1 (bottom) and Node N (top): side || base
    R_end = (R_side_tot * R_base_tot) / (R_side_tot + R_base_tot)  # [K/W] (parallel)

    R_arr = np.array([R_end] + [R_mid]*(N-2) + [R_end], dtype=float)
    UA_arr = 1.0 / R_arr  # [W/K]
    return UA_arr


# Re-run after reset: build and execute the TDMA-based stratified tank demo

def TDMA(a,b,c,d) -> np.ndarray:
    """
    TDMA (Tri-Diagonal Matrix Algorithm)를 사용하여 온도를 업데이트합니다.
    
    Reference: https://doi.org/10.1016/j.ijheatmasstransfer.2017.09.057 [Appendix B - Eq.(B7)]
    
    만약 boundary condition이 None이 아닌 경우, 각각 추가된 최좌측, 최우측 열저항에 종합 열저항을 추가하여 계산하고 이를 바탕으로 TDMA 알고리즘을 적용함.
    
    즉, 대류 경계층을 boundary layer 함수를 통해 지정한 경우 Construction의 표면온도를 계산할 때, 정상상태를 가정하에 경계층 열저항을 고려하여 표면온도를 다시 구해줘야함
    
    Parameters:
    -----------
    a : np.ndarray
        하부 대각선 요소 (길이 N-1)
    b : np.ndarray
        주 대각선 요소 (길이 N)
    c : np.ndarray
        상부 대각선 요소 (길이 N-1)
    d : np.ndarray
        우변 벡터 (길이 N)
    Returns:
    --------
    np.ndarray
        다음 시간 단계의 온도 배열
    """
    n = len(b)

    A_mat = np.zeros((n, n))
    np.fill_diagonal(A_mat[1:], a[1:])
    np.fill_diagonal(A_mat, b)
    np.fill_diagonal(A_mat[:, 1:], c[:-1])
    A_inv = np.linalg.inv(A_mat)

    T_new = np.dot(A_inv, d).flatten() # Flatten the result to 1D array
    return T_new

def _add_loop_advection_terms(a, b, c, d, in_idx, out_idx, G_loop, T_loop_in):
    """
    지정 구간(in_idx -> out_idx)으로 흐르는 강제 대류를 TDMA 계수(a,b,c,d)에 더함.
    - 인덱스는 0-based (노드 1 -> idx 0).
    - 방향: in_idx > out_idx 이면 '상향'(아래→위), 반대면 '하향'(위→아래).
    """
    
    # 유효하지 않은 경우 무시
    if G_loop <= 0 or in_idx == out_idx:
        return print("Warning: negative loop flow rate or identical in/out loop nodes.")

    # inlet 노드 (공통)
    b[in_idx] += G_loop
    d[in_idx] += G_loop * T_loop_in  # 유입 스트림 온도
    
    # 상향: in(N쪽) -> ... -> out(1쪽)
    if in_idx > out_idx:
        # 경로 내부 노드 (out_idx+1 .. in_idx-1)
        for k in range(in_idx - 1, out_idx, -1):
            b[k] += G_loop
            c[k] -= G_loop
        # outlet 노드 (out_idx)
        b[out_idx] += G_loop
        c[out_idx] -= G_loop

    # 하향: in(1쪽) -> ... -> out(N쪽)
    else:
        for k in range(in_idx + 1, out_idx):
            a[k] -= G_loop 
            b[k] += G_loop
        # outlet 노드 (out_idx)
        a[out_idx] -= G_loop
        b[out_idx] += G_loop


class StratifiedTankTDMA:
    """
    TDMA-based 1D stratified hot-water tank model (vertical discretization).

    This class models a cylindrical storage tank split into N vertical layers (nodes).
    Each node enforces an energy balance that includes:
    - Storage term via node thermal capacitance (C).
    - Effective thermal conduction between adjacent nodes using effective
      conductivity (k_eff) that accounts for both molecular conduction and
      natural convection driven by buoyancy forces.
    - Advection due to draw/inlet flow (G = rho_w * c_w * dV).
    - Heat loss to ambient through a per-node UA array (side for all nodes;
      bottom/top discs additionally for the end nodes).
    - Optional point heater applied at a single node.
    - Optional external loop advection across a node range in either direction,
      with loop heat input Q_loop.

    Effective Thermal Conductivity Approach:
    ----------------------------------------
    The model uses an effective thermal conductivity (k_eff) approach to
    integrate molecular conduction and natural convection effects. For each
    node pair (i, i+1), the effective conductivity is calculated based on:
    - Temperature difference (dT = T[i+1] - T[i])
    - Rayleigh number (Ra), which characterizes the buoyancy-driven flow
    - Nusselt number (Nu), which relates effective to molecular conductivity
    
    Stable stratification (dT < 0, upper warmer than lower):
    - Convection is suppressed, primarily molecular conduction
    - Nu ≈ 1.0 with small correction terms
    
    Unstable stratification (dT > 0, lower warmer than upper):
    - Natural convection enhances heat transfer
    - Nu > 1.0, increasing with Rayleigh number
    - Laminar (Ra < 1e7): Nu ∝ Ra^0.25
    - Turbulent (Ra ≥ 1e7): Nu ∝ Ra^0.33
    
    The effective conduction coefficient between nodes is:
    K_eff = k_eff * A / dh, where k_eff = k_molecular * Nu

    The semi-implicit time advance assembles a tri-diagonal linear system
    a, b, c, d and solves it with a TDMA routine (see TDMA()) to obtain
    next-step temperatures.

    Units
    - Temperatures: K
    - Geometry: m
    - Volumetric flow: m³/s
    - UA, K: W/K
    - Heater power, heat flows: W

    Parameters
    ----------
    H : float
        Tank height [m].
    N : int
        Number of vertical layers (nodes).
    r0 : float
        Inner radius of tank [m] (D = 2*r0).
    x_shell : float
        Shell thickness [m].
    x_ins : float
        Insulation thickness [m].
    k_shell : float
        Shell thermal conductivity [W/mK].
    k_ins : float
        Insulation thermal conductivity [W/mK].
    h_w : float
        Internal convective heat transfer coefficient (water side) [W/m²K].
    h_o : float
        External convective heat transfer coefficient (ambient side) [W/m²K].
    C_d_mix : float
        Empirical discharge coefficient for buoyancy-driven mixing [-].
        (Note: This parameter is retained for compatibility but is not used
        in the effective conductivity approach.)

    Attributes
    ----------
    H, D, N : float, float, int
        Geometry and discretization (D = 2*r0).
    A : float
        Cross-sectional area [m²].
    dh : float
        Layer height [m].
    V : float
        Per-node volume [m³].
    UA : np.ndarray
        Node-to-ambient UA per node [W/K], shape (N,).
    K : float
        Reference axial conduction equivalent between nodes [W/K]
        (based on molecular conductivity only, for reference).
    C : float
        Per-node thermal capacitance [J/K].
    C_d_mix : float
        Mixing discharge coefficient [-] (retained for compatibility).
    g : float
        Gravitational acceleration [m/s²].
    beta : float
        Volumetric expansion coefficient of water [1/K].
    nu : float
        Kinematic viscosity of water [m²/s].
    alpha : float
        Thermal diffusivity of water [m²/s].
    Pr : float
        Prandtl number [-].
    k_molecular : float
        Molecular thermal conductivity of water [W/m·K].
    Ra_critical : float
        Critical Rayleigh number for stable stratification (≈1708).
    k_eff : np.ndarray
        Effective thermal conductivity between node pairs [W/m·K],
        shape (N-1,), updated during each time step.
    K_eff : np.ndarray
        Effective conduction coefficient between node pairs [W/K],
        shape (N-1,), updated during each time step.
    G_use, G_loop : float
        Flow-related terms cached from the last update step.

    Methods
    -------
    effective_conductivity(T_upper, T_lower)
        Calculate effective thermal conductivity between two nodes based on
        temperature difference and Rayleigh number.
    update_tank_temp(...)
        Advance temperatures by one time step using the TDMA scheme.
        Heater and optional external loop can be applied. Heater and loop
        node indices are 1-based in the public API.
    info(as_dict=False, precision=3)
        Print or return a concise summary of model geometry and thermal properties.

    Notes
    -----
    - Heater/loop node indices are 1-based (converted to 0-based internally).
    - External loop terms are added to the TDMA coefficients to represent
      directed advection across a node range.
    - The effective conductivity approach replaces the previous Boussinesq
      mixing flow model, providing a more physically consistent representation
      of heat transfer in stratified tanks.
    """
    def __init__(self, H, N, r0, x_shell, x_ins, k_shell, k_ins, h_w, h_o, C_d_mix):
        self.H = H; self.D = 2*r0; self.N = N
        self.A = np.pi * (self.D**2) / 4.0
        self.dh = H / N
        self.V = self.A * self.dh
        self.UA = calc_UA_tank_arr(r0, x_shell, x_ins, k_shell, k_ins, H, N, h_w, h_o)
        self.K = k_w * self.A / self.dh
        self.C = c_w * rho_w
        self.C_d_mix = C_d_mix
        
        # 물성값 속성 (유효 열전도율 계산용)
        self.g = g  # 중력가속도 [m/s²]
        self.beta = beta  # 물의 체적팽창계수 [1/K]
        self.nu = mu_w / rho_w  # 동점성계수 [m²/s]
        self.alpha = k_w / (rho_w * c_w)  # 열확산율 [m²/s]
        self.Pr = (mu_w / rho_w) / (k_w / (rho_w * c_w))  # Prandtl 수 [-]
        self.k_molecular = k_w  # 분자 열전도율 [W/m·K]
        self.Ra_critical = 1708  # 안정 성층 임계 Rayleigh 수 (수평 평판 간 유체)
        
    def effective_conductivity(self, T_upper, T_lower):
        """
        온도 구배에 따른 유효 열전도율(effective thermal conductivity)을 계산합니다.
        
        이 메서드는 성층화된 탱크 내에서 인접한 두 노드 간의 열전달을 모델링합니다.
        순수 분자 전도와 부력 구동 자연 대류를 통합적으로 고려하여 유효 열전도율을 계산합니다.
        
        원리:
        -----
        유체 내에서 온도 구배가 존재할 때, 두 가지 메커니즘이 열전달에 기여합니다:
        1. 분자 전도 (Molecular conduction): 확산에 의한 열전달
        2. 자연 대류 (Natural convection): 부력에 의한 유체 운동으로 인한 열전달
        
        안정 성층 (Stable stratification, dT < 0):
        - 위쪽 노드가 더 뜨거워 밀도 구배가 안정적일 때
        - 대류가 억제되고 주로 분자 전도만 발생
        - Nu ≈ 1.0에 가까우며, 약한 확산만 고려
        
        불안정 성층 (Unstable stratification, dT > 0):
        - 아래쪽 노드가 더 뜨거워 밀도 구배가 불안정할 때
        - 부력에 의해 자연 대류가 발생하여 열전달이 강화됨
        - Rayleigh 수에 따라 대류 강도가 결정됨
        - Nu > 1.0으로 증가하여 유효 열전도율이 분자 전도보다 큼
        
        수식:
        -----
        Rayleigh 수 (Ra):
            Ra = (g * beta * |ΔT| * L_char³) / (ν * α)
        
        여기서:
            g: 중력가속도 [m/s²]
            beta: 체적팽창계수 [1/K]
            ΔT: 온도 차이 [K] (|T_lower - T_upper|)
            L_char: 특성 길이 [m] (노드 높이 dh)
            ν: 동점성계수 [m²/s]
            α: 열확산율 [m²/s]
        
        Nusselt 수 (Nu):
            - 안정 성층 (dT < 0): Nu = 1.0 + 0.1 * (Ra/Ra_critical)^0.25 (Ra > 0일 때)
            - 불안정 성층 (dT > 0):
                * Ra < 1e3: Nu = 1.0 (주로 전도)
                * 1e3 ≤ Ra < 1e7: Nu = 0.2 * Ra^0.25 (층류 대류)
                * Ra ≥ 1e7: Nu = 0.1 * Ra^0.33 (난류 대류)
        
        유효 열전도율:
            k_eff = k_molecular * Nu
        
        참고 문헌:
        ---------
        - Incropera & DeWitt, "Fundamentals of Heat and Mass Transfer", 7th ed.
        - Bejan, "Convection Heat Transfer", 4th ed.
        - 수평 평판 간 유체의 자연 대류에 대한 실험적 상관식
        
        Parameters:
        -----------
        T_upper : float
            상단 노드의 온도 [K]
        T_lower : float
            하단 노드의 온도 [K]
        
        Returns:
        --------
        k_eff : float
            유효 열전도율 [W/m·K]
        """
        # 기본 분자 열전도율
        k_molecular = self.k_molecular  # W/m·K
        
        # 온도 차이 계산
        dT = T_lower - T_upper  # [K]
        
        # 특성 길이 (노드 높이)
        L_char = self.dh  # [m]
        
        # Rayleigh 수 계산
        # Ra = (g * beta * |dT| * L_char³) / (nu * alpha)
        # Rayleigh 수는 부력과 점성력의 비율을 나타내며, 자연 대류의 강도를 결정
        Ra = abs(self.g * self.beta * dT * L_char**3) / (self.nu * self.alpha)
        
        # 안정 성층 (위가 더 뜨거움, dT < 0)
        # 이 경우 대류가 억제되고 주로 분자 전도만 발생
        if dT < 0:
            # 안정 성층에서는 대류가 거의 발생하지 않지만,
            # 약한 확산 효과를 고려하여 Nu를 1.0보다 약간 크게 설정
            # Ra_critical (약 1708)을 기준으로 정규화하여 작은 보정항 추가
            if Ra > 0:
                # 안정 성층에서도 작은 온도 구배가 있을 수 있으므로
                # 매우 약한 대류 효과를 고려 (0.25 지수는 실험적 상관식)
                Nu = 1.0 + 0.1 * (Ra / self.Ra_critical)**0.25
            else:
                # dT = 0인 경우 순수 전도
                Nu = 1.0
        
        # 불안정 성층 (아래가 더 뜨거움, dT > 0)
        # 이 경우 부력에 의해 자연 대류가 발생하여 열전달이 강화됨
        else:
            if Ra < 1e3:
                # 매우 작은 Ra에서는 대류 효과가 미미하여 주로 전도만 발생
                Nu = 1.0
            elif Ra < 1e7:
                # 중간 정도의 Ra에서 층류 대류 발생
                # 실험적 상관식: Nu ∝ Ra^0.25 (층류 영역)
                # 계수 0.2는 수직 평판이나 수평 평판 간 유체에 대한 실험적 값
                Nu = 0.2 * Ra**0.25
            else:
                # 높은 Ra에서 난류 대류 발생
                # 실험적 상관식: Nu ∝ Ra^0.33 (난류 영역)
                # 계수 0.1은 난류 영역에서의 실험적 값
                Nu = 0.1 * Ra**0.33
        
        # 유효 열전도율 계산
        # Nusselt 수는 유효 열전도율과 분자 열전도율의 비율을 나타냄
        # Nu = k_eff / k_molecular 이므로, k_eff = k_molecular * Nu
        k_eff = k_molecular * Nu
        
        return k_eff
        
    # --- 추가: 유틸리티 헬퍼 (클래스 바깥에 둬도 됨) -----------------------------
    def update_tank_temp(self,
             T , dt, T_in, dV_use, T_amb, T0,
             heater_node=None, heater_capacity=None,
             loop_outlet_node=None, loop_inlet_node=None,
             dV_loop=0.0, Q_loop=0.0):
        """
        주어진 시간 간격 dt 동안 탱크의 온도를 업데이트합니다.
        
        Parameters:
        -----------
        T : np.ndarray
            현재 노드 온도 배열 [K]
        dt : float
            시간 간격 [s]
        T_in : float
            유입수 온도 [K]
        dV_use : float
            온수 사용에 의해 유입/유출되는 물의 부피 [m³/s]
        T_amb : float
            주변 온도 [K]
        T_0 : float
            기준(환경) 온도 [K]
        heater_node_arr : np.ndarray, optional
            히터가 설치된 노드 번호 배열 (1부터 N까지), 기본값은 None (히터 없음)
        heater_capacity_arr : np.ndarray, optional
            각 heater node array에 대응되는 히터 출력 [W], 기본값은 0.0
        loop_outlet_node : int, optional
            외부 루프 유출 노드 번호 (1부터 N까지), 기본값은 None (루프 없음)
        loop_inlet_node : int, optional
            외부 루프 유입 노드 번호 (1부터 N까지), 기본값은 None (루프 없음)
        dV_loop : float, optional
            외부 루프를 통한 부피 유량 [m³/s], 기본값은 0.0
        Q_loop : float, optional
            외부 루프를 통한 열 유량 [W], 기본값은 0.0
            
        Returns:
        --------
        np.ndarray
            다음 시간 단계의 노드 온도 배열 [K]
        """
        self.T0 = T0  # 기준 온도 저장
        N = self.N
        UA = self.UA
        G_use = c_w * rho_w * dV_use
        eps = 1e-12
        G_loop = c_w * rho_w * max(dV_loop, 0.0) 

        # ---- 유효 열전도율 계산 (노드 간) ------------------------------------------------
        # 각 노드 쌍(i, i+1)에 대해 유효 열전도율 계산
        # k_eff[i]는 노드 i와 노드 i+1 사이의 유효 열전도율
        k_eff = np.zeros(N - 1)
        for i in range(N - 1):
            # 노드 i (상단)와 노드 i+1 (하단) 사이의 유효 열전도율 계산
            # T[i]는 상단 노드, T[i+1]는 하단 노드
            k_eff[i] = self.effective_conductivity(T[i], T[i+1])
        
        # 노드 간 유효 전도 계수 계산: K_eff = k_eff * A / dh
        # K_eff[i]는 노드 i와 노드 i+1 사이의 유효 전도 계수 [W/K]
        K_eff = k_eff * self.A / self.dh
            
        # ---- TDMA 계수 기본 구성 ----------------------------------------------------
        '''
        TDMA 계수 (a, b, c, d) 및 heat source term (S) 초기화
        유효 열전도율 방식: 전도와 대류를 통합적으로 고려한 K_eff 사용
        '''
        a = np.zeros(N); b = np.zeros(N); c = np.zeros(N); d = np.zeros(N)
        S = np.zeros(N)
        
        if heater_node is not None:
            idx = heater_node - 1
            if 0 <= idx < N:
                S[idx] = heater_capacity

        # 최상단 노드 (0) TDMA 계수 별도 계산
        # 노드 0과 노드 1 사이의 유효 전도 계수: K_eff[0]
        a[0] = 0
        b[0] = self.C * self.V/dt + G_use + K_eff[0] + UA[0]
        c[0] = -(K_eff[0] + G_use)
        d[0] = self.C * self.V*T[0]/dt + UA[0]*T_amb + S[0]
        
        # 중간 노드 (1~N-2) TDMA 계수 계산
        for i in range(1, N-1):
            # 노드 i-1과 노드 i 사이의 유효 전도 계수: K_eff[i-1] (위쪽)
            # 노드 i와 노드 i+1 사이의 유효 전도 계수: K_eff[i] (아래쪽)
            K_eff_upper = K_eff[i-1]
            K_eff_lower = K_eff[i]
            
            a[i] = -K_eff_upper
            b[i] = self.C * self.V/dt + G_use + K_eff_upper + K_eff_lower + UA[i]
            c[i] = -(K_eff_lower + G_use)
            d[i] = self.C * self.V*T[i]/dt + UA[i]*T_amb + S[i]
        
        # 최하단 노드 (N-1) TDMA 계수 별도 계산
        # 노드 N-2와 노드 N-1 사이의 유효 전도 계수: K_eff[N-2]
        a[N-1] = -K_eff[N-2]
        b[N-1] = self.C * self.V/dt + G_use + K_eff[N-2] + UA[N-1]
        c[N-1] = 0
        d[N-1] = self.C * self.V*T[N-1]/dt + UA[N-1]*T_amb + S[N-1] + G_use*T_in

        # ---- self 변수화 --------------------------------------------------------------
        self.G_use = G_use
        self.G_loop = G_loop
        self.k_eff = k_eff  # 유효 열전도율 배열 [W/m·K]
        self.K_eff = K_eff  # 유효 전도 계수 배열 [W/K]
        
        # ---- 외부 루프(지정 구간 강제 대류) 반영 ------------------------------------
        if (G_loop > 0.0) and (loop_outlet_node is not None) and (loop_inlet_node is not None):
            out_idx = int(loop_outlet_node) - 1
            in_idx  = int(loop_inlet_node)  - 1
            if 0 <= out_idx < N and 0 <= in_idx < N and out_idx != in_idx:
                # 루프 스트림 유입 온도 (outlet 측 온도 기준)
                T_stream_out = T[out_idx]                           # n 시점 사용(안정적)
                T_loop_in = T_stream_out + Q_loop / max(G_loop, eps)
                # (선택) 비현실적 고온 방지용 소프트 클램프 예시:
                # T_loop_in = min(T_loop_in, T_stream_out + 50.0)

                _add_loop_advection_terms(a, b, c, d, in_idx, out_idx, G_loop, T_loop_in)

        # ---- 선형계 풀이 ------------------------------------------------------------
        T_next = TDMA(a, b, c, d)

        return T_next
    
    def info(self, as_dict: bool = False, precision: int = 3):
        """
        현재 탱크/모델 설정을 요약해서 보여줍니다.

        Parameters
        ----------
        as_dict : bool
            True면 dict로 반환, False면 사람이 읽기 좋은 문자열을 print 후 None 반환
        precision : int
            표시 유효숫자(소수 자리) 제어
        """

        # 파생량 계산
        H      = float(self.H)
        D      = float(self.D)
        N      = int(self.N)
        dz     = float(self.dh)
        A      = float(self.A)
        V_node = float(self.V)
        V_tot  = V_node * N
        C_node = float(self.C * self.V)
        C_tot  = C_node * N
        K_ax   = float(self.K)            # 축방향 전도 등가전달계수 [W/K] (층간)
        UA_arr = np.asarray(self.UA, dtype=float)
        UA_sum = float(UA_arr.sum())
        UA_min = float(UA_arr.min()) if UA_arr.size else np.nan
        UA_max = float(UA_arr.max()) if UA_arr.size else np.nan

        out = {
            "geometry": {
                "H_m": H, "D_m": D, "area_m2": A,
                "layers_N": N, "dz_m": dz,
                "volume_node_m3": V_node, "volume_total_m3": V_tot
            },
            "thermal": {
                "C_node_J_per_K": C_node, "C_total_J_per_K": C_tot,
                "K_axial_W_per_K": K_ax,
                "UA_sum_W_per_K": UA_sum,
                "UA_min_W_per_K": UA_min,
                "UA_max_W_per_K": UA_max
            }
        }

        if as_dict:
            return out

        # pretty print
        p = precision
        def fmt(x): 
            try: 
                return f"{x:.{p}g}" if abs(x) >= 1 else f"{x:.{p}f}"
            except Exception:
                return str(x)

        lines = []
        lines.append("=== StratifiedTankTDMA :: Model Info ===")
        lines.append("[Geometry]")
        lines.append(f"  H = {fmt(H)} m,  D = {fmt(D)} m,  A = {fmt(A)} m²")
        lines.append(f"  N = {N} layers,  dz = {fmt(dz)} m")
        lines.append(f"  V_node = {fmt(V_node)} m³,  V_total = {fmt(V_tot)} m³")
        lines.append("[Thermal]")
        lines.append(f"  C_node = {fmt(C_node)} J/K,  C_total = {fmt(C_tot)} J/K")
        lines.append(f"  K_axial (conduction) = {fmt(K_ax)} W/K")
        lines.append(f"  UA_sum = {fmt(UA_sum)} W/K  " f"(min {fmt(UA_min)}, max {fmt(UA_max)})")
        lines.append("[Mixing]")
        lines.append(f"  C_d_mix = {fmt(getattr(self, 'C_d_mix', None))}")
        print("\n".join(lines))

# %%
