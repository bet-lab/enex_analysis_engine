"""
Stratified tank model using TDMA (Tri-Diagonal Matrix Algorithm).

This module provides a 1D stratified hot-water tank model with vertical discretization.
The model uses effective thermal conductivity to account for both molecular conduction
and natural convection driven by buoyancy forces.
"""

import numpy as np
import math
from . import calc_util as cu
from .constants import c_w, rho_w, k_w, mu_w, g, beta
from .enex_functions import (
    calc_UA_tank_arr,
    TDMA,
    _add_loop_advection_terms
)


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
