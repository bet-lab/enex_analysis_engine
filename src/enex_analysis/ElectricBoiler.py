"""Electric resistance boiler — dynamic hot-water tank model.

System chain:
    Electric Heater → Hot Water Tank → 3-Way Mixing Valve → Service Water

The heater is modelled as a pure-resistance element whose full
electrical input becomes heat (``Q_heat_source = E_heater``).  The
tank energy / mass balance is solved via ``fsolve`` on the shared
``tank_mass_energy_residual`` (same physics as ASHPB).

Exergy topology (3 components):
    1. Heater   — electricity = exergy
    2. Tank     — stored, loss, destruction
    3. Mixing   — destruction
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import calc_util as cu
from .constants import c_w, rho_w
from .enex_functions import (
    build_dhw_usage_ratio,
    calc_exergy_flow,
    calc_mixing_valve,
    calc_simple_tank_UA,
    convert_electricity_to_exergy,
)
from .dynamic_context import (
    ControlState,
    StepContext,
    Subsystem,
    determine_heat_source_on_off,
    determine_tank_refill_flow,
    tank_mass_energy_residual,
)


@dataclass
class ElectricBoiler:
    """Electric resistance boiler with hot-water storage tank.

    Parameters are passed via ``__init__`` (explicit typed
    constructor, same pattern as ``AirSourceHeatPumpBoiler``).
    """

    def __init__(
        self,
        # 1. Heater ----------------------------------------
        heater_capacity: float = 5000.0,

        # 2. Tank geometry / insulation --------------------
        r0: float = 0.2,
        H: float = 0.8,
        x_shell: float = 0.01,
        x_ins: float = 0.10,
        k_shell: float = 25,
        k_ins: float = 0.03,
        h_o: float = 15,

        # 3. Temperature set-points ------------------------
        T_tank_w_upper_bound: float = 65.0,
        T_tank_w_lower_bound: float = 60.0,
        T_mix_w_out: float = 45.0,
        T_sup_w: float = 10.0,
        dV_mix_w_out_max: float = 0.001,

        # 4. Tank water level management -------------------
        tank_always_full: bool = True,
        tank_level_lower_bound: float = 0.5,
        tank_level_upper_bound: float = 1.0,
        dV_tank_w_in_refill: float = 0.001,
        prevent_simultaneous_flow: bool = False,

        # 5. Operating schedule ----------------------------
        on_schedule: list[tuple[float, float]]
            | None = None,

        # 6. Subsystems ------------------------------------
        stc=None,
        uv=None,
    ):
        if on_schedule is None:
            on_schedule = [(0.0, 24.0)]

        # --- 1. Heater ---
        self.heater_capacity: float = heater_capacity

        # --- 2. Tank geometry & thermal ---
        self.tank_physical: dict = {
            'r0': r0, 'H': H,
            'x_shell': x_shell, 'x_ins': x_ins,
            'k_shell': k_shell, 'k_ins': k_ins,
            'h_o': h_o,
        }
        self.UA_tank: float = calc_simple_tank_UA(
            **self.tank_physical,
        )
        self.V_tank_full: float = math.pi * r0 ** 2 * H
        self.C_tank: float = c_w * rho_w * self.V_tank_full

        # --- 3. Temperature set-points ---
        self.T_tank_w_upper_bound: float = T_tank_w_upper_bound
        self.T_tank_w_lower_bound: float = T_tank_w_lower_bound
        self.T_sup_w: float = T_sup_w
        self.T_sup_w_K: float = cu.C2K(T_sup_w)
        self.T_tank_w_in_K: float = cu.C2K(T_sup_w)
        self.T_mix_w_out: float = T_mix_w_out
        self.T_mix_w_out_K: float = cu.C2K(T_mix_w_out)
        self.dV_mix_w_out_max: float = dV_mix_w_out_max

        # --- 4. Tank water level ---
        self.tank_always_full: bool = tank_always_full
        self.tank_level_lower_bound: float = (
            tank_level_lower_bound
        )
        self.tank_level_upper_bound: float = (
            tank_level_upper_bound
        )
        self.dV_tank_w_in_refill: float = dV_tank_w_in_refill
        self.prevent_simultaneous_flow: bool = (
            prevent_simultaneous_flow
        )

        # --- 5. Operating schedule ---
        self.on_schedule: list[tuple[float, float]] = (
            on_schedule
        )

        # --- 6. Subsystems ---
        self._subsystems: dict[str, Subsystem] = {}
        if stc is not None:
            self._subsystems['stc'] = stc
        if uv is not None:
            self._subsystems['uv'] = uv

        # --- Flow-rate sync ---
        self.dV_tank_w_in: float = 0.0
        self.dV_tank_w_out: float = 0.0
        self.dV_mix_sup_w_in: float = 0.0
        self.dV_mix_w_out: float = 0.0

    # =============================================================
    # Steady-state performance
    # =============================================================

    def _calc_state(
        self,
        T_tank_w: float,
        T0: float,
        heater_on: bool,
    ) -> dict:
        """Evaluate heater + tank at a given operating point.

        Parameters
        ----------
        T_tank_w : float
            Tank water temperature [°C].
        T0 : float
            Dead-state / outdoor-air temperature [°C].
        heater_on : bool
            Whether the heater is energised.

        Returns
        -------
        dict
            Performance dictionary.
        """
        T_tank_w_K: float = cu.C2K(T_tank_w)
        T0_K: float = cu.C2K(T0)

        # Electric heater: all electricity → heat
        E_heater: float = (
            self.heater_capacity if heater_on else 0.0
        )

        # Tank heat loss
        Q_tank_loss: float = (
            self.UA_tank * (T_tank_w_K - T0_K)
        )

        # Mixing valve
        dV_mix_w_out_val: float = self.dV_mix_w_out
        if dV_mix_w_out_val == 0:
            T_mix_w_out_val: float = np.nan
        else:
            mix: dict = calc_mixing_valve(
                T_tank_w_K,
                self.T_sup_w_K,
                self.T_mix_w_out_K,
            )
            T_mix_w_out_val = mix['T_mix_w_out']

        den: float = max(
            1e-6, T_tank_w_K - self.T_sup_w_K,
        )
        alp: float = min(
            1.0,
            max(
                0.0,
                (self.T_mix_w_out_K - self.T_sup_w_K)
                / den,
            ),
        )
        dV_tank_w_out: float = alp * dV_mix_w_out_val
        dV_tank_w_in: float = self.dV_tank_w_in
        dV_mix_sup_w_in: float = (
            (1 - alp) * dV_mix_w_out_val
        )

        return {
            'heater_is_on': heater_on,

            # Temperatures [°C]
            'T_tank_w [°C]': T_tank_w,
            'T_sup_w [°C]': self.T_sup_w,
            'T_tank_w_in [°C]': cu.K2C(
                self.T_tank_w_in_K,
            ),
            'T_mix_w_out [°C]': T_mix_w_out_val,
            'T0 [°C]': T0,

            # Volume flow rates [m³/s]
            'dV_mix_w_out [m3/s]': (
                dV_mix_w_out_val
                if dV_mix_w_out_val > 0 else np.nan
            ),
            'dV_tank_w_out [m3/s]': (
                dV_tank_w_out
                if dV_tank_w_out > 0 else np.nan
            ),
            'dV_tank_w_in [m3/s]': (
                dV_tank_w_in
                if dV_tank_w_in > 0 else np.nan
            ),
            'dV_mix_sup_w_in [m3/s]': (
                dV_mix_sup_w_in
                if dV_mix_sup_w_in > 0 else np.nan
            ),

            # Energy rates [W]
            'E_heater [W]': E_heater,
            'Q_tank_loss [W]': Q_tank_loss,
            'E_tot [W]': E_heater,
        }

    # =============================================================
    # Steady-state analysis
    # =============================================================

    def analyze_steady(
        self,
        T_tank_w: float,
        T0: float,
        dV_mix_w_out: float = 0.0,
        return_dict: bool = True,
    ) -> dict | pd.DataFrame:
        """Run a steady-state analysis.

        Parameters
        ----------
        T_tank_w : float
            Tank water temperature [°C].
        T0 : float
            Dead-state temperature [°C].
        dV_mix_w_out : float
            Service water flow rate [m³/s].
        return_dict : bool
            If True return dict; else single-row DataFrame.

        Returns
        -------
        dict | pd.DataFrame
        """
        self.dV_mix_w_out = dV_mix_w_out

        heater_on: bool
        if T_tank_w <= self.T_tank_w_lower_bound:
            heater_on = True
        elif T_tank_w >= self.T_tank_w_upper_bound:
            heater_on = False
        else:
            heater_on = True

        result: dict = self._calc_state(
            T_tank_w, T0, heater_on,
        )
        if return_dict:
            return result
        return pd.DataFrame([result])

    # =============================================================
    # Dynamic simulation helpers
    # =============================================================

    def _determine_heater_state(
        self,
        ctx: StepContext,
        is_on_prev: bool,
    ) -> tuple[bool, dict, float]:
        """Heater on/off + state evaluation.

        Parameters
        ----------
        ctx : StepContext
            Current-step immutable context.
        is_on_prev : bool
            Heater state at previous step.

        Returns
        -------
        tuple[bool, dict, float]
            ``(is_on, result, Q_heat_source)``.
        """
        T_tank_w: float = cu.K2C(ctx.T_tank_w_K)

        is_on: bool = determine_heat_source_on_off(
            T_tank_w_C=T_tank_w,
            T_lower=self.T_tank_w_lower_bound,
            T_upper=self.T_tank_w_upper_bound,
            is_on_prev=is_on_prev,
            hour_of_day=ctx.hour_of_day,
            on_schedule=self.on_schedule,
        )

        # Mixing valve flows
        den: float = max(
            1e-6, ctx.T_tank_w_K - self.T_sup_w_K,
        )
        alp: float = min(
            1.0,
            max(
                0.0,
                (self.T_mix_w_out_K - self.T_sup_w_K)
                / den,
            ),
        )
        self.dV_mix_w_out = ctx.dV_mix_w_out
        self.dV_tank_w_out = alp * ctx.dV_mix_w_out
        self.dV_mix_sup_w_in = (
            (1 - alp) * ctx.dV_mix_w_out
        )

        result: dict = self._calc_state(
            T_tank_w, ctx.T0, is_on,
        )

        # Q_heat_source = E_heater (all electricity → heat)
        Q_heat_source: float = result.get(
            'E_heater [W]', 0.0,
        )

        return is_on, result, Q_heat_source

    def _assemble_core_results(
        self,
        ctx: StepContext,
        ctrl: ControlState,
        T_solved_K: float,
        level_solved: float,
        ier: int,
    ) -> dict:
        """Build result dict at solved state."""
        den: float = max(
            1e-6, T_solved_K - self.T_sup_w_K,
        )
        alp: float = min(
            1.0,
            max(
                0.0,
                (self.T_mix_w_out_K - self.T_sup_w_K)
                / den,
            ),
        )
        dV_tank_w_out: float = alp * ctx.dV_mix_w_out
        dV_tank_w_in: float = (
            dV_tank_w_out
            if ctrl.dV_tank_w_in_ctrl is None
            else ctrl.dV_tank_w_in_ctrl
        )

        self.dV_tank_w_out = dV_tank_w_out
        self.dV_tank_w_in = dV_tank_w_in
        self.dV_mix_w_out = ctx.dV_mix_w_out
        self.dV_mix_sup_w_in = (
            (1 - alp) * ctx.dV_mix_w_out
        )

        T_mix_w_out_val: float = (
            calc_mixing_valve(
                T_solved_K,
                self.T_sup_w_K,
                self.T_mix_w_out_K,
            )['T_mix_w_out']
            if ctx.dV_mix_w_out > 0 else np.nan
        )

        r: dict = {}
        r.update(ctrl.result)
        r.update({
            'heater_is_on': ctrl.is_on,
            'Q_tank_loss [W]': (
                self.UA_tank * (T_solved_K - ctx.T0_K)
            ),
            'T_tank_w [°C]': cu.K2C(T_solved_K),
            'T_mix_w_out [°C]': T_mix_w_out_val,
        })

        if (
            not self.tank_always_full
            or (
                self.tank_always_full
                and self.prevent_simultaneous_flow
            )
        ):
            r['tank_level [-]'] = level_solved

        return r

    # =============================================================
    # Main dynamic simulation
    # =============================================================

    def analyze_dynamic(
        self,
        simulation_period_sec: int,
        dt_s: int,
        T_tank_w_init_C: float,
        dhw_usage_schedule,
        T0_schedule,
        tank_level_init: float = 1.0,
        result_save_csv_path: str | None = None,
    ) -> pd.DataFrame:
        """Run a time-stepping dynamic simulation.

        Parameters
        ----------
        simulation_period_sec : int
            Total simulation duration [s].
        dt_s : int
            Time step size [s].
        T_tank_w_init_C : float
            Initial tank temperature [°C].
        dhw_usage_schedule : array-like
            DHW usage schedule.
        T0_schedule : array-like
            Outdoor temperature per step [°C].
        tank_level_init : float
            Initial fractional tank level (0–1).
        result_save_csv_path : str | None
            Optional CSV output path.

        Returns
        -------
        pd.DataFrame
            Per-timestep result DataFrame.
        """
        from scipy.optimize import fsolve

        time: np.ndarray = np.arange(
            0, simulation_period_sec, dt_s,
        )
        tN: int = len(time)

        T0_schedule = np.array(T0_schedule)
        if len(T0_schedule) != tN:
            raise ValueError(
                f"T0_schedule length ({len(T0_schedule)})"
                f" != time length ({tN})",
            )

        self.time: np.ndarray = time
        self.dt: int = dt_s

        # DHW schedule handling
        self.w_use_frac = build_dhw_usage_ratio(
            dhw_usage_schedule, time,
        )

        T_tank_w_K: float = cu.C2K(T_tank_w_init_C)
        tank_level: float = tank_level_init
        is_refilling: bool = False
        is_on_prev: bool = False
        results_data: list[dict] = []

        for n in tqdm(
            range(tN), desc="ElectricBoiler Simulating",
        ):
            t_s: float = time[n]
            hr: float = t_s * cu.s2h
            hour_of_day: float = (
                (t_s % (24 * cu.h2s)) * cu.s2h
            )

            ctx: StepContext = StepContext(
                n=n,
                current_time_s=t_s,
                current_hour=hr,
                hour_of_day=hour_of_day,
                T0=T0_schedule[n],
                T0_K=cu.C2K(T0_schedule[n]),
                preheat_on=False,
                T_tank_w_K=T_tank_w_K,
                tank_level=tank_level,
                dV_mix_w_out=(
                    self.w_use_frac[n]
                    * self.dV_mix_w_out_max
                ),
            )

            # --- Phase A: control decisions ---
            is_on, result, Q_heat_source = (
                self._determine_heater_state(
                    ctx, is_on_prev,
                )
            )
            is_on_prev = is_on

            dV_tank_w_in_ctrl, is_refilling = (
                determine_tank_refill_flow(
                    dt=dt_s,
                    tank_level=ctx.tank_level,
                    dV_tank_w_out=self.dV_tank_w_out,
                    V_tank_full=self.V_tank_full,
                    tank_always_full=self.tank_always_full,
                    prevent_simultaneous_flow=(
                        self.prevent_simultaneous_flow
                    ),
                    tank_level_lower_bound=(
                        self.tank_level_lower_bound
                    ),
                    tank_level_upper_bound=(
                        self.tank_level_upper_bound
                    ),
                    dV_tank_w_in_refill=(
                        self.dV_tank_w_in_refill
                    ),
                    is_refilling=is_refilling,
                    use_stc=False,
                    mode='',
                    preheat_on=False,
                )
            )

            ctrl: ControlState = ControlState(
                is_on=is_on,
                Q_heat_source=Q_heat_source,
                dV_tank_w_in_ctrl=dV_tank_w_in_ctrl,
                result=result,
            )

            # --- Phase A-2: subsystem step ---
            sub_states: dict[str, dict] = {}
            for name, sub in self._subsystems.items():
                sub_states[name] = sub.step(
                    ctx, ctrl, dt_s,
                    self.T_tank_w_in_K,
                )

            # --- Phase B: implicit solve ---
            sol, _info, ier, _msg = fsolve(
                tank_mass_energy_residual,
                [ctx.T_tank_w_K, ctx.tank_level],
                args=(
                    ctx, ctrl, dt_s,
                    self.T_tank_w_in_K,
                    self.T_sup_w_K,
                    self.T_mix_w_out_K,
                    self.C_tank,
                    self.UA_tank,
                    self.V_tank_full,
                    self._subsystems,
                    sub_states,
                ),
                full_output=True,
            )

            T_tank_w_K = sol[0]
            tank_level = max(0.001, min(1.0, sol[1]))

            # --- Phase C: core + subsystem results ---
            r: dict = self._assemble_core_results(
                ctx, ctrl, T_tank_w_K,
                tank_level, ier,
            )
            for name, sub in self._subsystems.items():
                r.update(sub.assemble_results(
                    ctx, ctrl,
                    sub_states[name], T_tank_w_K,
                ))
            results_data.append(r)

        results_df: pd.DataFrame = pd.DataFrame(
            results_data,
        )
        results_df = self.postprocess_exergy(results_df)
        if result_save_csv_path:
            results_df.to_csv(
                result_save_csv_path, index=False,
            )
        return results_df

    # =============================================================
    # Exergy post-processing (ElectricBoiler-specific)
    # =============================================================

    def postprocess_exergy(
        self, df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute electric-boiler exergy variables.

        Exergy topology (3 components):

        1. Electricity = exergy (heater, UV)
        2. Water exergy (tank inlet/outlet, mixing valve)
        3. Heat loss exergy, tank stored exergy
        4. Component-level exergy destruction
        5. Exergetic efficiency metrics

        Parameters
        ----------
        df : pd.DataFrame
            Result DataFrame from ``analyze_dynamic()``.

        Returns
        -------
        pd.DataFrame
            DataFrame with exergy columns appended.
        """
        df = df.copy()

        T0_K = cu.C2K(df['T0 [°C]'])
        T_tank_K = cu.C2K(df['T_tank_w [°C]'])

        # ── 1. Electricity = exergy ────────────────────────
        df = convert_electricity_to_exergy(df)

        # ── 2. Water exergy (inlet / outlet) ───────────────
        df['X_tank_w_in [W]'] = calc_exergy_flow(
            c_w * rho_w
            * df['dV_tank_w_in [m3/s]'].fillna(0),
            cu.C2K(df['T_tank_w_in [°C]']),
            T0_K,
        )
        df['X_tank_w_out [W]'] = calc_exergy_flow(
            c_w * rho_w
            * df['dV_tank_w_out [m3/s]'].fillna(0),
            T_tank_K,
            T0_K,
        )
        df['X_mix_w_out [W]'] = calc_exergy_flow(
            c_w * rho_w
            * df['dV_mix_w_out [m3/s]'].fillna(0),
            cu.C2K(df['T_mix_w_out [°C]']),
            T0_K,
        )
        df['X_mix_sup_w_in [W]'] = calc_exergy_flow(
            c_w * rho_w
            * df['dV_mix_sup_w_in [m3/s]'].fillna(0),
            cu.C2K(df['T_sup_w [°C]']),
            T0_K,
        )

        # ── 3. Heat loss exergy ────────────────────────────
        df['X_tank_loss [W]'] = (
            df['Q_tank_loss [W]']
            * (1 - T0_K / T_tank_K)
        )

        # ── 4. Tank stored exergy ──────────────────────────
        tank_level = (
            df['tank_level [-]']
            if 'tank_level [-]' in df.columns
            else 1.0
        )
        C_tank_actual = self.C_tank * tank_level
        T_tank_K_prev = T_tank_K.shift(1)
        df['Xst_tank [W]'] = (
            (1 - T0_K / T_tank_K)
            * C_tank_actual
            * (T_tank_K - T_tank_K_prev)
            / self.dt
        )
        df.loc[df.index[0], 'Xst_tank [W]'] = 0.0

        # ── 5. Subsystem exergy (protocol) ─────────────────
        X_sub_tot_add = 0.0
        X_sub_in_tank_add = 0.0
        X_sub_out_tank_add = 0.0

        for _name, sub in self._subsystems.items():
            if hasattr(sub, 'calc_exergy'):
                ex_res = sub.calc_exergy(df, T0_K)
                if ex_res is not None:
                    for col_name, s in (
                        ex_res.columns.items()
                    ):
                        df[col_name] = s
                    X_sub_tot_add = (
                        X_sub_tot_add + ex_res.X_tot_add
                    )
                    X_sub_in_tank_add = (
                        X_sub_in_tank_add
                        + ex_res.X_in_tank_add
                    )
                    X_sub_out_tank_add = (
                        X_sub_out_tank_add
                        + ex_res.X_out_tank_add
                    )

        # ── 6. Total exergy input ──────────────────────────
        X_tot = df['E_heater [W]'].fillna(0)
        if 'X_uv [W]' in df.columns:
            X_tot = X_tot + df['X_uv [W]'].fillna(0)
        X_tot = X_tot + X_sub_tot_add
        df['X_tot [W]'] = X_tot

        # ── 7. Component exergy destruction ────────────────
        # Xc = ΣX_in − ΣX_out ≥ 0 (2nd law)

        # 7a. Mixing valve
        df['Xc_mix [W]'] = (
            df['X_tank_w_out [W]'].fillna(0)
            + df['X_mix_sup_w_in [W]'].fillna(0)
            - df['X_mix_w_out [W]'].fillna(0)
        )

        # 7b. Storage tank
        # Heater exergy = E_heater (no Carnot factor,
        # since electricity is pure exergy directly
        # absorbed by the tank water)
        X_in_tank = (
            df['E_heater [W]'].fillna(0)
            + df['X_tank_w_in [W]'].fillna(0)
        )
        if 'X_uv [W]' in df.columns:
            X_in_tank = (
                X_in_tank
                + df['X_uv [W]'].fillna(0)
            )
        X_in_tank = X_in_tank + X_sub_in_tank_add

        X_out_tank = (
            df['X_tank_loss [W]']
            + df['Xst_tank [W]']
        )
        if 'X_tank_w_out [W]' in df.columns:
            X_out_tank = (
                X_out_tank
                + df['X_tank_w_out [W]'].fillna(0)
            )
        X_out_tank = X_out_tank + X_sub_out_tank_add

        df['Xc_tank [W]'] = X_in_tank - X_out_tank

        # ── 8. Exergetic efficiency ────────────────────────
        df['X_eff_sys [-]'] = (
            df['X_tank_w_out [W]'].fillna(0)
            / df['X_tot [W]'].replace(0, np.nan)
        )

        return df
