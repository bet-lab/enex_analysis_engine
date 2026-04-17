"""EB with SolarThermalCollector — mains_preheat placement."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from . import calc_util as cu
from .constants import c_w, rho_w
from .electric_boiler import ElectricBoiler
from .subsystems import SolarThermalCollector

if TYPE_CHECKING:
    from .dynamic_context import ControlState, StepContext


class EB_STC_preheat(ElectricBoiler):
    def __init__(
        self,
        *,
        stc: SolarThermalCollector,
        **kwargs,
    ) -> None:
        if not isinstance(stc, SolarThermalCollector):
            raise TypeError(f"stc must be a SolarThermalCollector instance, got {type(stc)!r}")
        super().__init__(**kwargs)
        self._stc: SolarThermalCollector = stc
        self.stc = stc

    def _needs_solar_input(self) -> bool:
        return True

    def _get_activation_flags(self, hour_of_day: float) -> dict[str, bool]:
        return {"stc": self._stc.is_preheat_on(hour_of_day)}

    def _run_subsystems(
        self,
        ctx: "StepContext",
        ctrl: "ControlState",
        dt: float,
        T_tank_w_in_K: float,
    ) -> dict[str, dict]:
        dV_feed: float = ctrl.dV_tank_w_in_ctrl if ctrl.dV_tank_w_in_ctrl is not None else ctx.dV_mix_w_out

        stc_active: bool = False
        stc_result: dict = {}

        if ctx.activation_flags.get("stc", False) and dV_feed > 0:
            probe = self._stc.calc_performance(
                I_DN_stc=ctx.I_DN,
                I_dH_stc=ctx.I_dH,
                T_stc_w_in_K=T_tank_w_in_K,
                T0_K=ctx.T0_K,
                dV_stc=dV_feed,
                is_active=True,
            )
            stc_active = probe["T_stc_w_out_K"] > T_tank_w_in_K

            if stc_active:
                stc_result = probe
            else:
                stc_result = self._stc.calc_performance(
                    I_DN_stc=ctx.I_DN,
                    I_dH_stc=ctx.I_dH,
                    T_stc_w_in_K=T_tank_w_in_K,
                    T0_K=ctx.T0_K,
                    dV_stc=dV_feed,
                    is_active=False,
                )
        else:
            stc_result = self._stc.calc_performance(
                I_DN_stc=ctx.I_DN,
                I_dH_stc=ctx.I_dH,
                T_stc_w_in_K=T_tank_w_in_K,
                T0_K=ctx.T0_K,
                dV_stc=max(dV_feed, 1e-6),
                is_active=False,
            )

        E_pump: float = self._stc.E_stc_pump if stc_active else 0.0
        T_override: float | None = stc_result.get("T_stc_pump_w_out_K") if stc_active else None

        return {
            "stc": {
                "stc_active": stc_active,
                "stc_result": stc_result,
                "T_tank_w_in_override_K": T_override,
                "E_subsystem": E_pump,
                "Q_contribution": 0.0,
            }
        }

    def _augment_results(
        self,
        r: dict,
        ctx: "StepContext",
        ctrl: "ControlState",
        sub_states: dict[str, dict],
        T_solved_K: float,
    ) -> dict:
        state = sub_states["stc"]
        stc_active: bool = state["stc_active"]
        stc_result: dict = state["stc_result"]
        E_pump: float = state["E_subsystem"]

        T_stc_w_out_K: float = state["stc_result"].get("T_stc_pump_w_out_K", np.nan)

        r.update(
            {
                "stc_active [-]": stc_active,
                "I_DN_stc [W/m2]": ctx.I_DN,
                "I_dH_stc [W/m2]": ctx.I_dH,
                "I_sol_stc [W/m2]": stc_result.get("I_sol_stc", np.nan),
                "Q_sol_stc [W]": stc_result.get("Q_sol_stc", np.nan),
                "S_sol_stc [W/K]": stc_result.get("S_sol_stc", np.nan),
                "X_sol_stc [W]": stc_result.get("X_sol_stc", np.nan),
                "Q_stc_w_out [W]": stc_result.get("Q_stc_w_out", 0.0),
                "Q_stc_pump_w_out [W]": stc_result.get("Q_stc_pump_w_out", 0.0),
                "Q_stc_w_in [W]": stc_result.get("Q_stc_w_in", 0.0),
                "Q_l_stc [W]": stc_result.get("Q_l_stc", np.nan),
                "dV_stc [m3/s]": (ctrl.dV_tank_w_in_ctrl if ctrl.dV_tank_w_in_ctrl is not None else ctx.dV_mix_w_out),
                "T_stc_w_out [°C]": cu.K2C(T_stc_w_out_K) if not np.isnan(T_stc_w_out_K) else np.nan,
                "T_stc_w_in [°C]": cu.K2C(T_solved_K),
                "T_stc [°C]": cu.K2C(stc_result.get("T_stc_K", np.nan)),
                "E_stc_pump [W]": E_pump,
            }
        )
        return r

    def _postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        from .enex_functions import calc_exergy_flow

        df = super()._postprocess(df)
        if "T_stc_w_in [°C]" not in df.columns or "T_stc_w_out [°C]" not in df.columns:
            return df

        T0_K = cu.C2K(df["T0 [°C]"])
        T_stc_w_in_K = cu.C2K(df["T_stc_w_in [°C]"])
        T_stc_w_out_K = cu.C2K(df["T_stc_w_out [°C]"])
        T_stc_pump_w_out_K = T_stc_w_out_K
        T_stc_K = cu.C2K(df["T_stc [°C]"])

        G_stc = c_w * rho_w * df["dV_stc [m3/s]"].fillna(0)

        df["X_stc_w_in [W]"] = calc_exergy_flow(G_stc, T_stc_w_in_K, T0_K)
        df["X_stc_w_out [W]"] = calc_exergy_flow(G_stc, T_stc_w_out_K, T0_K)
        df["X_stc_pump_w_out [W]"] = calc_exergy_flow(G_stc, T_stc_pump_w_out_K, T0_K)

        E_pump = df["E_stc_pump [W]"].fillna(0)
        df["X_stc_pump [W]"] = E_pump

        df["X_l_stc [W]"] = df["Q_l_stc [W]"].fillna(0) * (1 - T0_K / T_stc_K.replace(0, np.nan))

        is_stc_active = df.get("stc_active [-]", False)
        if "X_sol_stc [W]" in df.columns:
            Xc_raw = (
                df["X_sol_stc [W]"].fillna(0)
                + df["X_stc_w_in [W]"].fillna(0)
                + E_pump
                - df["X_stc_pump_w_out [W]"].fillna(0)
                - df["X_l_stc [W]"].fillna(0)
            )
            df["Xc_stc [W]"] = np.where(is_stc_active, Xc_raw, 0.0)

        df["X_tot [W]"] = df["X_tot [W]"].add(E_pump, fill_value=0)

        return df
