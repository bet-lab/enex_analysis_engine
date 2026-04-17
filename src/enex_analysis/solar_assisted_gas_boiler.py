"""Solar-assisted gas boiler — dynamic hot-water tank model.

System chain (with tank):
    STC → Combustion Chamber → Hot Water Tank → Mixing Valve → Service

The solar thermal collector (STC) is injected as a
``Subsystem`` (from ``subsystems.py``).  The combustion
chamber supplements/replaces solar heat as needed.
The STC's ``mode`` (``tank_circuit`` / ``mains_preheat``)
determines whether it operates on the tank loop or the
mains water supply.

Exergy topology (5 components):
    1. NG chemical exergy, exhaust exergy
    2. STC exergy (via ``calc_exergy()`` protocol)
    3. Tank — stored, loss, destruction
    4. Mixing — destruction
    5. System total — NG + STC pump electricity
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import calc_util as cu
from .constants import c_w, ex_eff_NG, rho_w
from .dynamic_context import (
    ControlState,
    StepContext,
    Subsystem,
    determine_heat_source_on_off,
    determine_tank_refill_flow,
    tank_mass_energy_residual,
)
from .enex_functions import (
    build_dhw_usage_ratio,
    calc_exergy_flow,
    calc_mixing_valve,
    calc_simple_tank_UA,
    convert_electricity_to_exergy,
)


@dataclass
class SolarAssistedGasBoiler:
    """Gas boiler with solar thermal collector assist.

    The STC is passed as the ``stc`` constructor argument
    and registered via the ``Subsystem`` protocol.
    """

    def __init__(
        self,
        # 1. Combustion ------------------------------------
        eta_comb: float = 0.9,
        T_exh: float = 70.0,
        burner_capacity: float = 15000.0,
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
        on_schedule: list[tuple[float, float]] | None = None,
        # 6. STC subsystem (required) ----------------------
        stc=None,
        # 7. UV lamp subsystem -----------------------------
        uv=None,
        # 8. STC placement mode ----------------------------
        stc_mode: str = "tank_circuit",
        # 9. Preheat schedule ------------------------------
        preheat_schedule: list[tuple[float, float]] | None = None,
        # 10. Solar schedules (irradiance) -----------------
        I_DN_schedule=None,
        I_dH_schedule=None,
    ):
        if on_schedule is None:
            on_schedule = [(0.0, 24.0)]
        if preheat_schedule is None:
            preheat_schedule = [(6.0, 18.0)]

        # --- 1. Combustion ---
        if eta_comb <= 0 or eta_comb > 1:
            raise ValueError(
                f"eta_comb must be in (0, 1], got {eta_comb}",
            )
        self.eta_comb: float = eta_comb
        self.T_exh: float = T_exh
        self.T_exh_K: float = cu.C2K(T_exh)
        self.burner_capacity: float = burner_capacity

        # --- 2. Tank ---
        self.tank_physical: dict = {
            "r0": r0,
            "H": H,
            "x_shell": x_shell,
            "x_ins": x_ins,
            "k_shell": k_shell,
            "k_ins": k_ins,
            "h_o": h_o,
        }
        self.UA_tank: float = calc_simple_tank_UA(
            **self.tank_physical,
        )
        self.V_tank_full: float = math.pi * r0**2 * H
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
        self.tank_level_lower_bound: float = tank_level_lower_bound
        self.tank_level_upper_bound: float = tank_level_upper_bound
        self.dV_tank_w_in_refill: float = dV_tank_w_in_refill
        self.prevent_simultaneous_flow: bool = prevent_simultaneous_flow

        # --- 5. Operating schedule ---
        self.on_schedule: list[tuple[float, float]] = on_schedule

        # --- 6/7. STC + UV subsystems ---
        self._subsystems: dict[str, Subsystem] = {}
        self.use_stc: bool = stc is not None
        self.stc_mode: str = stc_mode
        if self.use_stc:
            self._subsystems["stc"] = stc
        if uv is not None:
            self._subsystems["uv"] = uv

        # --- 9. Preheat schedule ---
        self.preheat_schedule: list[tuple[float, float]] = preheat_schedule

        # --- 10. Solar schedules ---
        self.I_DN_schedule = I_DN_schedule
        self.I_dH_schedule = I_dH_schedule

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
        burner_on: bool,
    ) -> dict:
        """Evaluate combustion + tank at a given operating point.

        Parameters
        ----------
        T_tank_w : float
            Tank water temperature [°C].
        T0 : float
            Dead-state temperature [°C].
        burner_on : bool
            Whether the burner is firing.

        Returns
        -------
        dict
            Performance dictionary.
        """
        T_tank_w_K: float = cu.C2K(T_tank_w)
        T0_K: float = cu.C2K(T0)

        Q_tank_loss: float = self.UA_tank * (T_tank_w_K - T0_K)

        # Combustion: fuel → useful heat
        Q_comb_w: float = self.burner_capacity if burner_on else 0.0
        E_NG: float = Q_comb_w / self.eta_comb if burner_on else 0.0
        Q_exh: float = (1 - self.eta_comb) * E_NG if burner_on else 0.0

        # NG effective temperature for exergy
        T_NG_K: float = T0_K / max(
            1e-6,
            1 - ex_eff_NG,
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
            T_mix_w_out_val = mix["T_mix_w_out"]

        den: float = max(
            1e-6,
            T_tank_w_K - self.T_sup_w_K,
        )
        alp: float = min(
            1.0,
            max(
                0.0,
                (self.T_mix_w_out_K - self.T_sup_w_K) / den,
            ),
        )
        dV_tank_w_out: float = alp * dV_mix_w_out_val
        dV_tank_w_in: float = self.dV_tank_w_in
        dV_mix_sup_w_in: float = (1 - alp) * dV_mix_w_out_val

        return {
            "burner_is_on": burner_on,
            "T_tank_w [°C]": T_tank_w,
            "T_sup_w [°C]": self.T_sup_w,
            "T_tank_w_in [°C]": cu.K2C(
                self.T_tank_w_in_K,
            ),
            "T_mix_w_out [°C]": T_mix_w_out_val,
            "T_exh [°C]": self.T_exh,
            "T0 [°C]": T0,
            "dV_mix_w_out [m3/s]": (dV_mix_w_out_val if dV_mix_w_out_val > 0 else np.nan),
            "dV_tank_w_out [m3/s]": (dV_tank_w_out if dV_tank_w_out > 0 else np.nan),
            "dV_tank_w_in [m3/s]": (dV_tank_w_in if dV_tank_w_in > 0 else np.nan),
            "dV_mix_sup_w_in [m3/s]": (dV_mix_sup_w_in if dV_mix_sup_w_in > 0 else np.nan),
            "E_NG [W]": E_NG,
            "Q_comb_w [W]": Q_comb_w,
            "Q_exh [W]": Q_exh,
            "Q_tank_loss [W]": Q_tank_loss,
            "T_NG [K]": T_NG_K,
            "E_tot [W]": E_NG,
        }

    # =============================================================
    # Dynamic simulation helpers
    # =============================================================

    def _determine_burner_state(
        self,
        ctx: StepContext,
        is_on_prev: bool,
    ) -> tuple[bool, dict, float]:
        """Burner on/off + state evaluation.

        Parameters
        ----------
        ctx : StepContext
            Current-step immutable context.
        is_on_prev : bool
            Burner state at previous step.

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
            1e-6,
            ctx.T_tank_w_K - self.T_sup_w_K,
        )
        alp: float = min(
            1.0,
            max(
                0.0,
                (self.T_mix_w_out_K - self.T_sup_w_K) / den,
            ),
        )
        self.dV_mix_w_out = ctx.dV_mix_w_out
        self.dV_tank_w_out = alp * ctx.dV_mix_w_out
        self.dV_mix_sup_w_in = (1 - alp) * ctx.dV_mix_w_out

        result: dict = self._calc_state(
            T_tank_w,
            ctx.T0,
            is_on,
        )

        Q_heat_source: float = result.get(
            "Q_comb_w [W]",
            0.0,
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
            1e-6,
            T_solved_K - self.T_sup_w_K,
        )
        alp: float = min(
            1.0,
            max(
                0.0,
                (self.T_mix_w_out_K - self.T_sup_w_K) / den,
            ),
        )
        dV_tank_w_out: float = alp * ctx.dV_mix_w_out
        dV_tank_w_in: float = dV_tank_w_out if ctrl.dV_tank_w_in_ctrl is None else ctrl.dV_tank_w_in_ctrl

        self.dV_tank_w_out = dV_tank_w_out
        self.dV_tank_w_in = dV_tank_w_in
        self.dV_mix_w_out = ctx.dV_mix_w_out
        self.dV_mix_sup_w_in = (1 - alp) * ctx.dV_mix_w_out

        T_mix_w_out_val: float = (
            calc_mixing_valve(
                T_solved_K,
                self.T_sup_w_K,
                self.T_mix_w_out_K,
            )["T_mix_w_out"]
            if ctx.dV_mix_w_out > 0
            else np.nan
        )

        r: dict = {}
        r.update(ctrl.result)
        r.update(
            {
                "burner_is_on": ctrl.is_on,
                "Q_tank_loss [W]": (self.UA_tank * (T_solved_K - ctx.T0_K)),
                "T_tank_w [°C]": cu.K2C(T_solved_K),
                "T_mix_w_out [°C]": T_mix_w_out_val,
            }
        )

        if not self.tank_always_full or (self.tank_always_full and self.prevent_simultaneous_flow):
            r["tank_level [-]"] = level_solved

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
        I_DN_schedule=None,
        I_dH_schedule=None,
        tank_level_init: float = 1.0,
        result_save_csv_path: str | None = None,
    ) -> pd.DataFrame:
        """Run dynamic simulation with STC + gas boiler.

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
        I_DN_schedule : array-like, optional
            Direct-normal irradiance per step [W/m²].
        I_dH_schedule : array-like, optional
            Diffuse-horizontal irradiance [W/m²].
        tank_level_init : float
            Initial tank level (0–1).
        result_save_csv_path : str | None
            Optional CSV output path.

        Returns
        -------
        pd.DataFrame
        """
        from scipy.optimize import fsolve

        time: np.ndarray = np.arange(
            0,
            simulation_period_sec,
            dt_s,
        )
        tN: int = len(time)

        T0_schedule = np.array(T0_schedule)
        if len(T0_schedule) != tN:
            raise ValueError(
                f"T0_schedule length ({len(T0_schedule)}) != time length ({tN})",
            )

        # Use constructor schedules as fallback
        if I_DN_schedule is None:
            I_DN_schedule = self.I_DN_schedule if self.I_DN_schedule is not None else np.zeros(tN)
        if I_dH_schedule is None:
            I_dH_schedule = self.I_dH_schedule if self.I_dH_schedule is not None else np.zeros(tN)
        I_DN_schedule = np.array(I_DN_schedule)
        I_dH_schedule = np.array(I_dH_schedule)

        self.time: np.ndarray = time
        self.dt: int = dt_s

        self.w_use_frac = build_dhw_usage_ratio(
            dhw_usage_schedule,
            time,
        )

        T_tank_w_K: float = cu.C2K(T_tank_w_init_C)
        tank_level: float = tank_level_init
        is_refilling: bool = False
        is_on_prev: bool = False
        results_data: list[dict] = []

        use_stc: bool = self.use_stc
        stc_mode: str = self.stc_mode

        for n in tqdm(
            range(tN),
            desc="SolarAssistedGasBoiler Simulating",
        ):
            t_s: float = time[n]
            hr: float = t_s * cu.s2h
            hour_of_day: float = (t_s % (24 * cu.h2s)) * cu.s2h

            # Preheat window check
            preheat_on: bool = False
            if self.use_stc:
                for start_h, end_h in self.preheat_schedule:
                    if start_h <= hour_of_day < end_h:
                        preheat_on = True
                        break

            ctx: StepContext = StepContext(
                n=n,
                current_time_s=t_s,
                current_hour=hr,
                hour_of_day=hour_of_day,
                T0=T0_schedule[n],
                T0_K=cu.C2K(T0_schedule[n]),
                preheat_on=preheat_on,
                T_tank_w_K=T_tank_w_K,
                tank_level=tank_level,
                dV_mix_w_out=(self.w_use_frac[n] * self.dV_mix_w_out_max),
                I_DN=float(I_DN_schedule[n]),
                I_dH=float(I_dH_schedule[n]),
            )

            # --- Phase A: control decisions ---
            is_on, result, Q_heat_source = self._determine_burner_state(
                ctx,
                is_on_prev,
            )
            is_on_prev = is_on

            dV_tank_w_in_ctrl, is_refilling = determine_tank_refill_flow(
                dt=dt_s,
                tank_level=ctx.tank_level,
                dV_tank_w_out=self.dV_tank_w_out,
                V_tank_full=self.V_tank_full,
                tank_always_full=self.tank_always_full,
                prevent_simultaneous_flow=(self.prevent_simultaneous_flow),
                tank_level_lower_bound=(self.tank_level_lower_bound),
                tank_level_upper_bound=(self.tank_level_upper_bound),
                dV_tank_w_in_refill=(self.dV_tank_w_in_refill),
                is_refilling=is_refilling,
                use_stc=use_stc,
                mode=stc_mode,
                preheat_on=ctx.preheat_on,
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
                    ctx,
                    ctrl,
                    dt_s,
                    self.T_tank_w_in_K,
                )

            # --- Phase B: implicit solve ---
            sol, _info, ier, _msg = fsolve(
                tank_mass_energy_residual,
                [ctx.T_tank_w_K, ctx.tank_level],
                args=(
                    ctx,
                    ctrl,
                    dt_s,
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
                ctx,
                ctrl,
                T_tank_w_K,
                tank_level,
                ier,
            )
            for name, sub in self._subsystems.items():
                r.update(
                    sub.assemble_results(
                        ctx,
                        ctrl,
                        sub_states[name],
                        T_tank_w_K,
                    )
                )
            results_data.append(r)

        results_df: pd.DataFrame = pd.DataFrame(
            results_data,
        )
        results_df = self.postprocess_exergy(results_df)
        if result_save_csv_path:
            results_df.to_csv(
                result_save_csv_path,
                index=False,
            )
        return results_df

    # =============================================================
    # Exergy post-processing
    # =============================================================

    def postprocess_exergy(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute solar-gas-boiler exergy variables.

        Exergy topology (5 components):

        1. NG chemical exergy, exhaust exergy
        2. STC exergy via ``calc_exergy()`` protocol
        3. Water exergy (tank inlet/outlet, mixing valve)
        4. Heat loss / stored / tank destruction
        5. Exergetic efficiency

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

        T0_K = cu.C2K(df["T0 [°C]"])
        T_tank_K = cu.C2K(df["T_tank_w [°C]"])

        # ── 1. NG exergy ───────────────────────────────────
        df["X_NG [W]"] = ex_eff_NG * df["E_NG [W]"]

        # ── 2. Exhaust exergy ──────────────────────────────
        T_exh_K = cu.C2K(df["T_exh [°C]"])
        df["X_exh [W]"] = df["Q_exh [W]"] * (1 - T0_K / T_exh_K)

        # ── 3. Combustion heat exergy ──────────────────────
        df["X_comb_w [W]"] = df["Q_comb_w [W]"] * (1 - T0_K / T_tank_K)

        # ── 4. Electricity = exergy (UV, STC pump) ────────
        df = convert_electricity_to_exergy(df)

        # ── 5. Water exergy ────────────────────────────────
        df["X_tank_w_in [W]"] = calc_exergy_flow(
            c_w * rho_w * df["dV_tank_w_in [m3/s]"].fillna(0),
            cu.C2K(df["T_tank_w_in [°C]"]),
            T0_K,
        )
        df["X_tank_w_out [W]"] = calc_exergy_flow(
            c_w * rho_w * df["dV_tank_w_out [m3/s]"].fillna(0),
            T_tank_K,
            T0_K,
        )
        df["X_mix_w_out [W]"] = calc_exergy_flow(
            c_w * rho_w * df["dV_mix_w_out [m3/s]"].fillna(0),
            cu.C2K(df["T_mix_w_out [°C]"]),
            T0_K,
        )
        df["X_mix_sup_w_in [W]"] = calc_exergy_flow(
            c_w * rho_w * df["dV_mix_sup_w_in [m3/s]"].fillna(0),
            cu.C2K(df["T_sup_w [°C]"]),
            T0_K,
        )

        # ── 6. Heat loss exergy ────────────────────────────
        df["X_tank_loss [W]"] = df["Q_tank_loss [W]"] * (1 - T0_K / T_tank_K)

        # ── 7. Tank stored exergy ──────────────────────────
        tank_level = df["tank_level [-]"] if "tank_level [-]" in df.columns else 1.0
        C_tank_actual = self.C_tank * tank_level
        T_tank_K_prev = T_tank_K.shift(1)
        df["Xst_tank [W]"] = (1 - T0_K / T_tank_K) * C_tank_actual * (T_tank_K - T_tank_K_prev) / self.dt
        df.loc[df.index[0], "Xst_tank [W]"] = 0.0

        # ── 8. Subsystem exergy (STC protocol) ─────────────
        X_sub_tot_add = 0.0
        X_sub_in_tank_add = 0.0
        X_sub_out_tank_add = 0.0

        for _name, sub in self._subsystems.items():
            if hasattr(sub, "calc_exergy"):
                ex_res = sub.calc_exergy(df, T0_K)
                if ex_res is not None:
                    for col_name, s in ex_res.columns.items():
                        df[col_name] = s
                    X_sub_tot_add = X_sub_tot_add + ex_res.X_tot_add  # type: ignore[operator]
                    X_sub_in_tank_add = (
                        X_sub_in_tank_add + ex_res.X_in_tank_add  # type: ignore[operator]
                    )
                    X_sub_out_tank_add = (
                        X_sub_out_tank_add + ex_res.X_out_tank_add  # type: ignore[operator]
                    )

        # ── 9. Total exergy input ──────────────────────────
        X_tot = df["X_NG [W]"].fillna(0)
        if "X_uv [W]" in df.columns:
            X_tot = X_tot + df["X_uv [W]"].fillna(0)
        X_tot = X_tot + X_sub_tot_add
        df["X_tot [W]"] = X_tot

        # ── 10. Component exergy destruction ───────────────
        # Xc = ΣX_in − ΣX_out ≥ 0 (2nd law)

        # 10a. Combustion chamber
        df["Xc_comb [W]"] = (
            df["X_NG [W]"].fillna(0)
            + df["X_tank_w_in [W]"].fillna(0)
            - df["X_comb_w [W]"].fillna(0)
            - df["X_exh [W]"].fillna(0)
        )

        # 10b. Mixing valve
        df["Xc_mix [W]"] = (
            df["X_tank_w_out [W]"].fillna(0) + df["X_mix_sup_w_in [W]"].fillna(0) - df["X_mix_w_out [W]"].fillna(0)
        )

        # 10c. Storage tank
        X_in_tank = df["X_comb_w [W]"].fillna(0) + df["X_tank_w_in [W]"].fillna(0)
        if "X_uv [W]" in df.columns:
            X_in_tank = X_in_tank + df["X_uv [W]"].fillna(0)
        X_in_tank = X_in_tank + X_sub_in_tank_add

        X_out_tank = df["X_tank_loss [W]"] + df["Xst_tank [W]"]
        if "X_tank_w_out [W]" in df.columns:
            X_out_tank = X_out_tank + df["X_tank_w_out [W]"].fillna(0)
        X_out_tank = X_out_tank + X_sub_out_tank_add

        df["Xc_tank [W]"] = X_in_tank - X_out_tank

        # ── 11. Exergetic efficiency ───────────────────────
        df["X_eff_sys [-]"] = df["X_comb_w [W]"].fillna(0) / df["X_tot [W]"].replace(0, np.nan)

        return df
