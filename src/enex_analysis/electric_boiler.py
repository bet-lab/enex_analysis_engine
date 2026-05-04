"""Electric Boiler Component Model."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import calc_util as cu
from .constants import c_w, rho_w
from .dynamic_context import (
    ControlState,
    StepContext,
    determine_heat_source_on_off,
    determine_tank_refill_flow,
    tank_mass_energy_residual,
)
from .enex_functions import (
    build_dhw_usage_ratio,
    calc_exergy_flow,
    calc_mixing_valve_flows,
    calc_mixing_valve_temp,
)


class ElectricBoiler:
    def __init__(
        self,
        *,
        heater_capacity: float,
        V_tank_full: float,
        T_sup_w_C: float,
        T_tank_w_lower_bound_C: float,
        T_tank_w_upper_bound_C: float,
        T_tank_w_in_C: float,
        T_mix_w_out_C: float,
        dV_mix_w_out_max: float,
        on_schedule: dict | None = None,
        UA_tank: float = 0.0,
        M_tank_empty: float = 0.0,
        C_tank_empty: float = 0.0,
        dV_tank_w_in_refill: float | None = None,
        tank_always_full: bool = True,
        tank_level_lower_bound: float = 0.2,
        tank_level_upper_bound: float = 0.8,
        prevent_simultaneous_flow: bool = True,
        subsystems: dict | None = None,
    ) -> None:
        self.heater_capacity: float = heater_capacity
        self.V_tank_full: float = V_tank_full
        self.T_sup_w: float = T_sup_w_C
        self.T_sup_w_K: float = cu.C2K(T_sup_w_C)
        self.T_tank_w_lower_bound: float = T_tank_w_lower_bound_C
        self.T_tank_w_upper_bound: float = T_tank_w_upper_bound_C
        self.T_tank_w_in_K: float = cu.C2K(T_tank_w_in_C)
        self.T_mix_w_out_K: float = cu.C2K(T_mix_w_out_C)
        self.dV_mix_w_out_max: float = dV_mix_w_out_max
        self.on_schedule: dict | None = on_schedule
        self.UA_tank: float = UA_tank
        self.dV_tank_w_in_refill = dV_tank_w_in_refill
        self.tank_always_full: bool = tank_always_full
        self.tank_level_lower_bound: float = tank_level_lower_bound
        self.tank_level_upper_bound: float = tank_level_upper_bound
        self.prevent_simultaneous_flow: bool = prevent_simultaneous_flow
        self.M_tank: float = M_tank_empty + rho_w * self.V_tank_full
        self.C_tank: float = C_tank_empty + c_w * self.M_tank
        self._subsystems = subsystems if subsystems is not None else {}

        self.dV_tank_w_out: float = 0.0
        self.dV_tank_w_in: float = 0.0
        self.dV_mix_w_out: float = 0.0
        self.dV_mix_sup_w_in: float = 0.0

    @staticmethod
    def _calc_tank_flow_context(
        dV_mix_w_out: float,
        T_tank_w_K: float,
        T_sup_w_K: float,
        T_mix_w_out_K: float,
        dV_tank_w_in_override: float | None = None,
    ) -> dict:
        mix_state = calc_mixing_valve_temp(T_tank_w_K, T_sup_w_K, T_mix_w_out_K)
        flows = calc_mixing_valve_flows(dV_mix_w_out, mix_state["alp"])
        dV_tank_w_out = flows["dV_hot_in"]
        dV_tank_w_in = dV_tank_w_out if dV_tank_w_in_override is None else dV_tank_w_in_override
        return {
            "alp": mix_state["alp"],
            "dV_mix_w_out": dV_mix_w_out,
            "dV_tank_w_out": dV_tank_w_out,
            "dV_tank_w_in": dV_tank_w_in,
            "dV_mix_sup_w_in": flows["dV_cold_in"],
        }

    def _calc_state(
        self,
        T_tank_w: float,
        T0: float,
        heater_on: bool,
        *,
        flow_state: dict,
    ) -> dict:
        T_tank_w_K: float = cu.C2K(T_tank_w)
        T0_K: float = cu.C2K(T0)
        E_heater: float = self.heater_capacity if heater_on else 0.0
        Q_tank_loss: float = self.UA_tank * (T_tank_w_K - T0_K)

        dV_mix_w_out_val = flow_state["dV_mix_w_out"]
        dV_tank_w_out = flow_state["dV_tank_w_out"]
        dV_tank_w_in = flow_state["dV_tank_w_in"]
        dV_mix_sup_w_in = flow_state["dV_mix_sup_w_in"]

        if dV_mix_w_out_val == 0:
            T_mix_w_out_val: float = np.nan
        else:
            T_mix_w_out_val = calc_mixing_valve_temp(T_tank_w_K, self.T_sup_w_K, self.T_mix_w_out_K)["T_mix_w_out"]

        return {
            "heater_is_on": heater_on,
            "T_tank_w [°C]": T_tank_w,
            "T_sup_w [°C]": self.T_sup_w,
            "T_tank_w_in [°C]": cu.K2C(self.T_tank_w_in_K),
            "T_mix_w_out [°C]": T_mix_w_out_val,
            "T0 [°C]": T0,
            "dV_mix_w_out [m3/s]": (dV_mix_w_out_val if dV_mix_w_out_val > 0 else np.nan),
            "dV_tank_w_out [m3/s]": (dV_tank_w_out if dV_tank_w_out > 0 else np.nan),
            "dV_tank_w_in [m3/s]": (dV_tank_w_in if dV_tank_w_in > 0 else np.nan),
            "dV_mix_sup_w_in [m3/s]": (dV_mix_sup_w_in if dV_mix_sup_w_in > 0 else np.nan),
            "E_heater [W]": E_heater,
            "Q_tank_loss [W]": Q_tank_loss,
            "E_tot [W]": E_heater,
        }

    def analyze_steady(
        self,
        T_tank_w: float,
        T0: float,
        dV_mix_w_out: float = 0.0,
        return_dict: bool = True,
    ) -> dict | pd.DataFrame:
        heater_on: bool
        if T_tank_w <= self.T_tank_w_lower_bound:
            heater_on = True
        elif T_tank_w >= self.T_tank_w_upper_bound:
            heater_on = False
        else:
            heater_on = True

        flow_state = self._calc_tank_flow_context(
            dV_mix_w_out=dV_mix_w_out,
            T_tank_w_K=cu.C2K(T_tank_w),
            T_sup_w_K=self.T_sup_w_K,
            T_mix_w_out_K=self.T_mix_w_out_K,
        )
        result: dict = self._calc_state(T_tank_w, T0, heater_on, flow_state=flow_state)
        if return_dict:
            return result
        return pd.DataFrame([result])

    def _determine_heater_state(
        self,
        ctx: StepContext,
        is_on_prev: bool,
    ) -> tuple[bool, dict, float]:
        T_tank_w: float = cu.K2C(ctx.T_tank_w_K)
        is_on: bool = determine_heat_source_on_off(
            T_tank_w_C=T_tank_w,
            T_lower=self.T_tank_w_lower_bound,
            T_upper=self.T_tank_w_upper_bound,
            is_on_prev=is_on_prev,
            hour_of_day=ctx.hour_of_day,
            on_schedule=self.on_schedule.get("winter", []) if self.on_schedule else [(0.0, 24.0)],
        )
        flow_state = self._calc_tank_flow_context(
            dV_mix_w_out=ctx.dV_mix_w_out,
            T_tank_w_K=ctx.T_tank_w_K,
            T_sup_w_K=self.T_sup_w_K,
            T_mix_w_out_K=self.T_mix_w_out_K,
        )
        self.dV_mix_w_out = flow_state["dV_mix_w_out"]
        self.dV_tank_w_out = flow_state["dV_tank_w_out"]
        self.dV_tank_w_in = flow_state["dV_tank_w_in"]
        self.dV_mix_sup_w_in = flow_state["dV_mix_sup_w_in"]

        result: dict = self._calc_state(
            T_tank_w,
            ctx.T0,
            is_on,
            flow_state=flow_state,
        )
        Q_heat_source: float = result.get("E_heater [W]", 0.0)
        return is_on, result, Q_heat_source

    # ------------------------------------------------------------------
    # Subsystem / Scenario Hooks
    # ------------------------------------------------------------------

    def _needs_solar_input(self) -> bool:
        return False

    def _get_activation_flags(self, hour_of_day: float) -> dict[str, bool]:
        return {}

    def _build_residual_fn(
        self,
        ctx: StepContext,
        ctrl: ControlState,
        dt_s: float,
        T_tank_w_in_K_n: float,
        T_sup_w_K_n: float,
        tank_level: float,
        sub_states: dict,
    ) -> Callable[[float], float] | None:
        return None  # Fallback to fsolve(tank_mass_energy_residual)

    def _run_subsystems(
        self,
        ctx: StepContext,
        ctrl: ControlState,
        dt: float,
        T_tank_w_in_K: float,
    ) -> dict[str, dict]:
        return {}

    def _augment_results(
        self,
        r: dict,
        ctx: StepContext,
        ctrl: ControlState,
        sub_states: dict[str, dict],
        T_solved_K: float,
    ) -> dict:
        return r

    def _assemble_core_results(
        self, ctx: StepContext, ctrl: ControlState, T_solved_K: float, level_solved: float, ier: int
    ) -> dict:
        flow_state = self._calc_tank_flow_context(
            dV_mix_w_out=ctx.dV_mix_w_out,
            T_tank_w_K=T_solved_K,
            T_sup_w_K=self.T_sup_w_K,
            T_mix_w_out_K=self.T_mix_w_out_K,
            dV_tank_w_in_override=ctrl.dV_tank_w_in_ctrl,
        )
        self.dV_tank_w_out = flow_state["dV_tank_w_out"]
        self.dV_tank_w_in = flow_state["dV_tank_w_in"]
        self.dV_mix_w_out = flow_state["dV_mix_w_out"]
        self.dV_mix_sup_w_in = flow_state["dV_mix_sup_w_in"]

        T_mix_w_out_val: float = (
            calc_mixing_valve_temp(T_solved_K, self.T_sup_w_K, self.T_mix_w_out_K)["T_mix_w_out"]
            if ctx.dV_mix_w_out > 0
            else np.nan
        )
        r: dict = {}
        r.update(ctrl.result)
        r.update(
            {
                "heater_is_on": ctrl.is_on,
                "Q_tank_loss [W]": (self.UA_tank * (T_solved_K - ctx.T0_K)),
                "T_tank_w [°C]": cu.K2C(T_solved_K),
                "T_mix_w_out [°C]": T_mix_w_out_val,
            }
        )
        if not self.tank_always_full or (self.tank_always_full and self.prevent_simultaneous_flow):
            r["tank_level [-]"] = level_solved
        return r

    def _postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        T0_K = cu.C2K(df["T0 [°C]"])

        # Tank calculations
        T_tank_K = cu.C2K(df["T_tank_w [°C]"])
        T_tank_K_prev = T_tank_K.shift(1).fillna(T_tank_K)
        tank_level = df["tank_level [-]"] if "tank_level [-]" in df.columns else 1.0
        C_tank_actual = self.C_tank * tank_level

        df["X_tank_loss [W]"] = df["Q_tank_loss [W]"] * (1 - T0_K / T_tank_K)
        df["Xst_tank [W]"] = (
            (1 - T0_K / T_tank_K)
            * C_tank_actual
            * (T_tank_K - T_tank_K_prev)
            / getattr(self, "dt", 3600.0)  # Assume 1h if not set
        )
        df.loc[df.index[0], "Xst_tank [W]"] = 0.0

        # Subsystem exergy
        X_sub_tot_add = 0.0
        X_sub_in_tank_add = 0.0
        X_sub_out_tank_add = 0.0

        for _name, sub in self._subsystems.items():
            if hasattr(sub, "calc_exergy"):
                ex_res = sub.calc_exergy(df, T0_K)
                if ex_res is not None:
                    for col_name, s in ex_res.columns.items():
                        df[col_name] = s
                    X_sub_tot_add = X_sub_tot_add + ex_res.X_tot_add
                    X_sub_in_tank_add = X_sub_in_tank_add + ex_res.X_in_tank_add
                    X_sub_out_tank_add = X_sub_out_tank_add + ex_res.X_out_tank_add

        # Flow exergies
        G_mix_out = c_w * rho_w * df["dV_mix_w_out [m3/s]"].fillna(0)
        G_mix_sup_w = c_w * rho_w * df["dV_mix_sup_w_in [m3/s]"].fillna(0)
        G_tank_w_out = c_w * rho_w * df["dV_tank_w_out [m3/s]"].fillna(0)
        G_tank_w_in = c_w * rho_w * df["dV_tank_w_in [m3/s]"].fillna(0)

        df["Q_tank_w_out [W]"] = G_mix_out * (cu.C2K(df["T_mix_w_out [°C]"]) - self.T_sup_w_K)

        df["X_mix_w_out [W]"] = calc_exergy_flow(G_tank_w_out, T_tank_K, T0_K)
        df["X_tank_w_in [W]"] = calc_exergy_flow(G_tank_w_in, self.T_tank_w_in_K, T0_K)
        df["X_mix_sup_w_in [W]"] = calc_exergy_flow(G_mix_sup_w, self.T_sup_w_K, T0_K)
        df["X_tank_w_out [W]"] = calc_exergy_flow(G_mix_out, cu.C2K(df["T_mix_w_out [°C]"]), T0_K)

        # Totals
        X_tot = df["E_heater [W]"].fillna(0)
        if "X_uv [W]" in df.columns:
            X_tot = X_tot + df["X_uv [W]"].fillna(0)
        X_tot = X_tot + X_sub_tot_add
        df["X_tot [W]"] = X_tot

        # Destruction
        df["Xc_mix [W]"] = (
            df["X_tank_w_out [W]"].fillna(0) + df["X_mix_sup_w_in [W]"].fillna(0) - df["X_mix_w_out [W]"].fillna(0)
        )
        X_in_tank = df["E_heater [W]"].fillna(0) + df["X_tank_w_in [W]"].fillna(0)
        if "X_uv [W]" in df.columns:
            X_in_tank = X_in_tank + df["X_uv [W]"].fillna(0)
        X_in_tank = X_in_tank + X_sub_in_tank_add

        X_out_tank = df["X_tank_loss [W]"] + df["Xst_tank [W]"]
        if "X_mix_w_out [W]" in df.columns:
            X_out_tank = X_out_tank + df["X_mix_w_out [W]"].fillna(0)
        X_out_tank = X_out_tank + X_sub_out_tank_add

        df["Xc_tank [W]"] = X_in_tank - X_out_tank

        # Efficiency
        df["X_eff_sys [-]"] = df["X_tank_w_out [W]"].fillna(0) / df["X_tot [W]"].replace(0, np.nan)

        return df

    def postprocess_exergy(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._postprocess(df)

    def analyze_dynamic(
        self,
        simulation_period_sec: int,
        dt_s: int,
        T_tank_w_init_C: float,
        dhw_usage_schedule,
        T0_schedule,
        I_DN_schedule=None,
        I_dH_schedule=None,
        T_sup_w_schedule=None,
        tank_level_init: float = 1.0,
        result_save_csv_path: str | None = None,
    ) -> pd.DataFrame:
        from scipy.optimize import fsolve

        time: np.ndarray = np.arange(0, simulation_period_sec, dt_s)
        tN: int = len(time)
        T0_schedule = np.array(T0_schedule)
        I_DN_schedule = np.array(I_DN_schedule) if I_DN_schedule is not None else np.zeros(tN)
        I_dH_schedule = np.array(I_dH_schedule) if I_dH_schedule is not None else np.zeros(tN)

        self.time: np.ndarray = time
        self.dt: int = dt_s
        self.w_use_frac = build_dhw_usage_ratio(dhw_usage_schedule, time)

        T_tank_w_K: float = cu.C2K(T_tank_w_init_C)
        tank_level: float = tank_level_init
        is_refilling: bool = False
        is_on_prev: bool = False
        results_data: list[dict] = []

        for n in tqdm(range(tN), desc="ElectricBoiler Simulating"):
            t_s: float = time[n]
            hr: float = t_s * cu.s2h
            hour_of_day: float = (t_s % (24 * cu.h2s)) * cu.s2h

            ctx = StepContext(
                n=n,
                current_time_s=t_s,
                current_hour=hr,
                hour_of_day=hour_of_day,
                T0=T0_schedule[n],
                T0_K=cu.C2K(T0_schedule[n]),
                activation_flags=self._get_activation_flags(hour_of_day),
                I_DN=I_DN_schedule[n],
                I_dH=I_dH_schedule[n],
                T_tank_w_K=T_tank_w_K,
                tank_level=tank_level,
                dV_mix_w_out=(self.w_use_frac[n] * self.dV_mix_w_out_max),
            )

            is_on, result, Q_heat_source = self._determine_heater_state(ctx, is_on_prev)
            is_on_prev = is_on

            dV_tank_w_in_ctrl, is_refilling = determine_tank_refill_flow(
                dt=dt_s,
                tank_level=ctx.tank_level,
                dV_tank_w_out=self.dV_tank_w_out,
                V_tank_full=self.V_tank_full,
                tank_always_full=self.tank_always_full,
                prevent_simultaneous_flow=self.prevent_simultaneous_flow,
                tank_level_lower_bound=self.tank_level_lower_bound,
                tank_level_upper_bound=self.tank_level_upper_bound,
                dV_tank_w_in_refill=(self.dV_tank_w_in_refill or 0.0),
                is_refilling=is_refilling,
            )

            ctrl = ControlState(
                is_on=is_on,
                Q_heat_source=Q_heat_source,
                dV_tank_w_in_ctrl=dV_tank_w_in_ctrl,
                result=result,
            )

            sub_states = self._run_subsystems(ctx, ctrl, dt_s, self.T_tank_w_in_K)

            T_override_K: float | None = None
            for state in sub_states.values():
                if state.get("T_tank_w_in_override_K") is not None:
                    T_override_K = state["T_tank_w_in_override_K"]
            eval_T_tank_w_in_K = T_override_K if T_override_K else self.T_tank_w_in_K

            res_fn = self._build_residual_fn(
                ctx, ctrl, dt_s, eval_T_tank_w_in_K, self.T_sup_w_K, tank_level, sub_states
            )

            if res_fn is not None:
                _valid_res_fn = res_fn

                def fn(x):
                    return [_valid_res_fn(x[0]), x[1] - tank_level]

                sol, *_ = fsolve(fn, [ctx.T_tank_w_K, ctx.tank_level])
                ier = 1
            else:
                sol, _info, ier, _msg = fsolve(
                    tank_mass_energy_residual,
                    [ctx.T_tank_w_K, ctx.tank_level],
                    args=(
                        ctx,
                        ctrl,
                        dt_s,
                        eval_T_tank_w_in_K,
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

            r = self._assemble_core_results(ctx, ctrl, T_tank_w_K, tank_level, ier)
            for name, sub in self._subsystems.items():
                if hasattr(sub, "assemble_results"):
                    r.update(sub.assemble_results(ctx, ctrl, sub_states.get(name, {}), T_tank_w_K))
            r = self._augment_results(r, ctx, ctrl, sub_states, T_tank_w_K)
            results_data.append(r)

        results_df = pd.DataFrame(results_data)
        results_df = self.postprocess_exergy(results_df)
        if result_save_csv_path:
            results_df.to_csv(result_save_csv_path, index=False)
        return results_df
