"""EB Scenario: Electric Boiler driven by PV + ESS with Grid/Dump integration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import calc_util as cu
from .electric_boiler import ElectricBoiler
from .subsystems import EnergyStorageSystem, PhotovoltaicSystem

class EB_PV_ESS(ElectricBoiler):
    def __init__(
        self,
        *,
        pv: PhotovoltaicSystem,
        ess: EnergyStorageSystem | None = None,
        eta_inv: float = 0.95,
        T_inv_K: float = 313.15,
        **kwargs,
    ) -> None:
        if not isinstance(pv, PhotovoltaicSystem):
            raise TypeError("pv must be a PhotovoltaicSystem instance")
        super().__init__(**kwargs)
        self._pv = pv
        self._ess = ess if ess is not None else EnergyStorageSystem()
        self._eta_inv = eta_inv
        self._T_inv_K = T_inv_K
        self.pv = pv

    def _needs_solar_input(self) -> bool:
        return True

    def _get_activation_flags(self, hour_of_day: float) -> dict[str, bool]:
        return {}

    def _run_subsystems(self, ctx, ctrl, dt, T_tank_w_in_K) -> dict[str, dict]:
        self._current_dt = dt
        return {}

    def _augment_results(self, base_result: dict, ctx, ctrl, sub_states: dict, T_solved_K: float) -> dict:
        eb_res = super()._augment_results(base_result, ctx, ctrl, sub_states, T_solved_K)
        
        e_heater = float(eb_res.get("E_heater [W]", 0.0))
        E_eb_load = 0.0 if np.isnan(e_heater) else e_heater

        dt = getattr(self, "_current_dt", 3600.0)
        T0_K = ctx.T0_K
        pv_r = self._pv.calc_performance(ctx.I_DN, ctx.I_dH, T0_K)
        E_ctrl_out: float = pv_r["E_ctrl_out"]

        E_dc_req: float = E_eb_load / self._eta_inv if self._eta_inv > 0 else 0.0

        if E_eb_load == 0.0:
            E_dc_to_inv = 0.0
            ess_r = self._ess.charge(E_ctrl_out, dt, T0_K)
            E_dump = max(0.0, E_ctrl_out - ess_r["E_ess_chg"])
            E_grid_import = 0.0
        elif E_ctrl_out >= E_dc_req:
            E_dc_to_inv = E_dc_req
            E_dc_excess = E_ctrl_out - E_dc_req
            ess_r = self._ess.charge(E_dc_excess, dt, T0_K)
            E_dump = max(0.0, E_dc_excess - ess_r["E_ess_chg"])
            E_grid_import = 0.0
        else:
            E_dc_to_inv_from_pv = E_ctrl_out
            E_ess_needed = E_dc_req - E_dc_to_inv_from_pv
            ess_r = self._ess.discharge(E_ess_needed, dt, T0_K)
            E_dc_to_inv = E_dc_to_inv_from_pv + ess_r["E_ess_dis"]
            E_dump = 0.0
            E_inv_out_available = self._eta_inv * E_dc_to_inv
            E_grid_import = max(0.0, E_eb_load - E_inv_out_available)

        E_inv_out = min(self._eta_inv * E_dc_to_inv + E_grid_import, E_eb_load)
        Q_l_inv = (1.0 - self._eta_inv) * E_dc_to_inv
        S_l_inv = Q_l_inv / self._T_inv_K
        X_l_inv = Q_l_inv - S_l_inv * T0_K
        X_c_inv = S_l_inv * T0_K

        pv_cols = {
            "I_sol_pv [W/m2]": pv_r["I_sol_pv"],
            "T_pv [°C]": cu.K2C(pv_r["T_pv_K"]),
            "E_pv_out [W]": pv_r["E_pv_out"],
            "E_ctrl_out [W]": E_ctrl_out,
            "X_sol [W]": pv_r["X_sol"],
            "X_pv_out [W]": pv_r["X_pv_out"],
            "X_ctrl_out [W]": pv_r["X_ctrl_out"],
            "X_c_pv [W]": pv_r["X_c_pv"],
            "X_c_ctrl [W]": pv_r["X_c_ctrl"],
            "X_l_pv [W]": pv_r["X_l_pv"],
            "X_l_ctrl [W]": pv_r["X_l_ctrl"],
        }
        ess_cols = {
            "E_ess_chg [W]": ess_r["E_ess_chg"],
            "E_ess_dis [W]": ess_r["E_ess_dis"],
            "SOC_ess [-]": ess_r["SOC_ess"],
            "X_c_ess [W]": ess_r["X_c_ess"],
            "X_l_ess [W]": ess_r["X_l_ess"],
        }
        route_cols = {
            "E_inv_out [W]": E_inv_out,
            "E_grid_import [W]": E_grid_import,
            "E_dump [W]": E_dump,
            "X_c_inv [W]": X_c_inv,
            "X_l_inv [W]": X_l_inv,
        }
        eb_res.update(pv_cols)
        eb_res.update(ess_cols)
        eb_res.update(route_cols)
        return eb_res

    def _postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = super()._postprocess(df)
        df["X_tot_sys [W]"] = df["X_sol [W]"].fillna(0) + df["E_grid_import [W]"].fillna(0)
        if "X_uv [W]" in df.columns:
            df["X_tot_sys [W]"] += df["X_uv [W]"].fillna(0)
        df["Xc_sys_tot [W]"] = df.filter(regex=r"^Xc_").sum(axis=1)
        df["cop_grid [-]"] = df["Q_tank_w_out [W]"] / df["E_grid_import [W]"].replace(0, np.nan)
        df["ex_eff_sys [-]"] = (df["Xst_tank [W]"] + df["X_tank_w_out [W]"]) / df["X_tot_sys [W]"].replace(0, np.nan)
        return df
