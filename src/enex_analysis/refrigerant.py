"""
Refrigerant cycle calculations and optimization.
"""
from typing import Any

import CoolProp.CoolProp as CP
import numpy as np


def calc_ref_state(
    refrigerant: str,
    T_evap: float,
    T_cond: float,
    dT_superheat: float = 0.0,
    dT_subcool: float = 0.0,
    is_active: bool = True
) -> dict[str, float]:
    """Calculate thermodynamic states of a standard vapor compression cycle.

    Parameters
    ----------
    refrigerant : str
        Refrigerant name (e.g., 'R134a', 'R410A').
    T_evap : float
        Evaporation temperature [K].
    T_cond : float
        Condensation temperature [K].
    dT_superheat : float, optional
        Superheat at evaporator outlet [K]. Default is 0.0.
    dT_subcool : float, optional
        Subcool at condenser outlet [K]. Default is 0.0.
    is_active : bool, optional
        If False, returns a dict with np.nan values.

    Returns
    -------
    dict[str, float]
        Dictionary containing state points (P, T, H, S) for the cycle.
    """
    if not is_active:
        return {'COP': np.nan}

    res: dict[str, float] = {}

    # Pressures
    P_evap = float(CP.PropsSI('P', 'T', T_evap, 'Q', 1, refrigerant))
    P_cond = float(CP.PropsSI('P', 'T', T_cond, 'Q', 0, refrigerant))

    # State 1: Compressor inlet
    T1 = T_evap + dT_superheat
    res['P1'] = P_evap
    res['T1'] = T1
    res['H1'] = float(CP.PropsSI('H', 'T', T1, 'P', P_evap, refrigerant))
    res['S1'] = float(CP.PropsSI('S', 'T', T1, 'P', P_evap, refrigerant))

    # State 2: Compressor outlet (Isentropic)
    res['P2'] = P_cond
    res['S2'] = res['S1']
    res['H2'] = float(CP.PropsSI('H', 'P', P_cond, 'S', res['S2'], refrigerant))
    res['T2'] = float(CP.PropsSI('T', 'P', P_cond, 'S', res['S2'], refrigerant))

    # State 3: Condenser outlet
    T3 = T_cond - dT_subcool
    res['P3'] = P_cond
    res['T3'] = T3
    res['H3'] = float(CP.PropsSI('H', 'T', T3, 'P', P_cond, refrigerant))
    res['S3'] = float(CP.PropsSI('S', 'T', T3, 'P', P_cond, refrigerant))

    # State 4: Evaporator inlet (Isenthalpic expansion)
    res['P4'] = P_evap
    res['H4'] = res['H3']
    res['T4'] = float(CP.PropsSI('T', 'P', P_evap, 'H', res['H4'], refrigerant))
    res['S4'] = float(CP.PropsSI('S', 'P', P_evap, 'H', res['H4'], refrigerant))

    # Basic performance
    q_evap = res['H1'] - res['H4']
    w_comp = res['H2'] - res['H1']
    res['COP'] = q_evap / w_comp if w_comp > 0 else np.nan

    return res

def create_lmtd_constraints() -> tuple[Any, Any]:
    """Create LMTD-based constraint functions for cycle optimization.

    Optimization requires that the heat transfer calculated by LMTD matches
    the heat transferred by the refrigerant cycle.

    Returns
    -------
    tuple[Any, Any]
        Tuple of constraint functions (constraint_tank, constraint_hx).
    """
    def constraint_tank(perf: dict[str, Any]) -> float:
        """Condenser constraint: Q_LMTD_cond - Q_ref_cond = 0"""
        if perf is None or 'Q_cond' not in perf or 'Q_cond_LMTD' not in perf:
            return 1e6
        return float(perf['Q_cond_LMTD'] - perf['Q_cond'])

    def constraint_hx(perf: dict[str, Any]) -> float:
        """Evaporator constraint: Q_LMTD_evap - Q_ref_evap = 0"""
        if perf is None or 'Q_evap' not in perf or 'Q_evap_LMTD' not in perf:
            return 1e6
        return float(perf['Q_evap_LMTD'] - perf['Q_evap'])

    return constraint_tank, constraint_hx

def find_ref_loop_optimal_operation(
    simulator_func: Any,
    refrigerant: str,
    load_W: float,
    initial_guess: list[float],
    bounds: list[tuple[float, float]],
    constraint_funcs: list[Any] | None = None
) -> dict[str, Any] | None:
    """Find the optimal operation point for the refrigerant loop.

    Minimizes compressor power while satisfying target load and LMTD constraints.

    Parameters
    ----------
    simulator_func : callable
        Function that takes `[dT_ref_HX, dT_ref_tank]` and returns a perf dict.
    refrigerant : str
        Refrigerant name.
    load_W : float
        Target heat load [W].
    initial_guess : list[float]
        Initial guess for `[dT_evap, dT_cond]`.
    bounds : list[tuple[float, float]]
        Bounds for `[dT_evap, dT_cond]`.
    constraint_funcs : list[callable], optional
        List of constraint functions. Each takes `perf` and returns a value
        that should be 0.

    Returns
    -------
    dict[str, Any] | None
        Optimal performance dictionary, or None if optimization fails.
    """
    from scipy.optimize import minimize

    def objective(x: np.ndarray) -> float:
        perf = simulator_func(x)
        if perf is None or 'W_comp' not in perf:
            return 1e6

        # Add penalty if load is not met
        load_diff = abs(perf.get('Q_cond', 0) - load_W)
        penalty = (load_diff / load_W)**2 * 1e5 if load_W > 0 else 0

        return float(perf['W_comp'] + penalty)

    constraints = []
    if constraint_funcs:
        for cf in constraint_funcs:
            def make_constraint(c_func: Any) -> Any:
                def constraint(x: np.ndarray) -> float:
                    perf = simulator_func(x)
                    return float(c_func(perf))
                return constraint

            constraints.append({'type': 'eq', 'fun': make_constraint(cf)})

    try:
        res = minimize(
            objective,
            initial_guess,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP',
            options={'disp': False, 'ftol': 1e-4, 'maxiter': 50}
        )
        if res.success:
            return simulator_func(res.x)
    except Exception:
        pass

    return None
