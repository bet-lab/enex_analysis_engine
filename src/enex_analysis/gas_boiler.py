"""Gas boiler model for steady-state and dynamic energy analysis.

Models a direct-supply gas boiler without a storage tank.
The system chain is: Combustion Chamber → Mixing Valve → Service Water.
Full energy, entropy, and exergy balances are computed at each operating point.
"""

import math

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import calc_util as cu
from .constants import (
    c_w,
    ex_eff_NG,
    rho_w,
)
from .enex_functions import (
    build_dhw_usage_ratio,
)


class GasBoiler:
    """Direct-supply gas boiler with energy / entropy / exergy accounting.

    The boiler heats mains water to ``T_comb_setpoint`` and mixes it with
    bypass water through a mixing valve to deliver service water at
    ``T_serv_w``.  No thermal storage tank is used.
    """

    def __init__(
        self,
        eta_comb=0.9,  # Combustion efficiency [-]
        T_serv_w=45.0,  # Service (delivery) water temperature [°C]
        T_sup_w=15.0,  # Mains water supply temperature [°C]
        T_exh=70.0,  # Exhaust gas temperature [°C]
        T_comb_setpoint=60.0,  # Boiler outlet setpoint [°C]
        dV_w_serv_m3s=0.0001,  # Maximum service flow rate [m³/s]
    ):

        self.eta_comb = eta_comb
        self.T_serv_w = T_serv_w
        self.T_sup_w = T_sup_w
        self.T_exh = T_exh
        self.T_comb_setpoint = T_comb_setpoint

        # Kelvin conversions
        self.T_serv_w_K = cu.C2K(self.T_serv_w)
        self.T_sup_w_K = cu.C2K(self.T_sup_w)
        self.T_exh_K = cu.C2K(self.T_exh)
        self.T_comb_setpoint_K = cu.C2K(self.T_comb_setpoint)

        self.dV_w_serv_m3s = dV_w_serv_m3s
        self.Q_comb_load_threshold = 100.0  # Minimum combustion load [W]

    @staticmethod
    def _build_flow_state(
        dV_w_serv: float,
        T_serv_w_K: float,
        T_sup_w_K: float,
        T_comb_setpoint_K: float,
    ) -> dict:
        den = max(1e-6, T_comb_setpoint_K - T_sup_w_K)
        alp = min(1.0, max(0.0, (T_serv_w_K - T_sup_w_K) / den))

        dV_w_sup_comb = alp * dV_w_serv
        dV_w_sup_mix = (1 - alp) * dV_w_serv

        T_serv_w_actual_K = alp * T_comb_setpoint_K + (1 - alp) * T_sup_w_K
        
        return {
            "alp": alp,
            "dV_w_serv": dV_w_serv,
            "dV_w_sup_comb": dV_w_sup_comb,
            "dV_w_sup_mix": dV_w_sup_mix,
            "T_serv_w_actual_K": T_serv_w_actual_K,
        }

    def _calc_on_state(self, Q_comb_load, T0, flow_state: dict):
        """Compute energy / entropy / exergy balance for the ON state.

        Parameters
        ----------
        Q_comb_load : float
            Required combustion heat load [W].
        T0 : float
            Dead-state (ambient) temperature for exergy analysis [°C].
        flow_state : dict
            Flow rates and mixed temperatures resulting from `_build_flow_state`.

        Returns
        -------
        dict
            Result dictionary with energy, entropy, and exergy terms.
        """
        T0_K = cu.C2K(T0)

        alp = flow_state["alp"]
        dV_w_serv = flow_state["dV_w_serv"]
        dV_w_sup_comb = flow_state["dV_w_sup_comb"]
        dV_w_sup_mix = flow_state["dV_w_sup_mix"]
        T_serv_w_actual_K = flow_state["T_serv_w_actual_K"]
        T_serv_w_actual = cu.K2C(T_serv_w_actual_K)
        T_w_comb_out = self.T_comb_setpoint

        # --- Combustion chamber ---
        E_NG = Q_comb_load / self.eta_comb if self.eta_comb > 0 else 0.0
        Q_w_comb_out = (
            c_w * rho_w * dV_w_sup_comb * (self.T_comb_setpoint_K - T0_K)
        )
        Q_exh = (1 - self.eta_comb) * E_NG
        Q_w_sup = c_w * rho_w * dV_w_sup_comb * (self.T_sup_w_K - T0_K)

        # --- Mixing valve ---
        Q_w_sup_mix = c_w * rho_w * dV_w_sup_mix * (self.T_sup_w_K - T0_K)
        Q_w_serv = c_w * rho_w * dV_w_serv * (T_serv_w_actual_K - T0_K)

        # --- Entropy balance ---
        T_NG = T0_K / (1 - ex_eff_NG)  # Effective NG temperature [K]

        S_NG = (1 / T_NG) * E_NG
        S_w_sup = c_w * rho_w * dV_w_sup_comb * math.log(self.T_sup_w_K / T0_K)
        S_w_comb_out = (
            c_w
            * rho_w
            * dV_w_sup_comb
            * math.log(self.T_comb_setpoint_K / T0_K)
        )
        S_exh = (1 / self.T_exh_K) * Q_exh
        S_g_comb = (S_w_comb_out + S_exh) - (S_NG + S_w_sup)

        S_w_sup_mix = (
            c_w * rho_w * dV_w_sup_mix * math.log(self.T_sup_w_K / T0_K)
        )
        S_w_serv = c_w * rho_w * dV_w_serv * math.log(T_serv_w_actual_K / T0_K)
        S_g_mix = S_w_serv - (S_w_comb_out + S_w_sup_mix)

        # --- Exergy balance ---
        X_NG = ex_eff_NG * E_NG
        X_w_sup = (
            c_w
            * rho_w
            * dV_w_sup_comb
            * (
                (self.T_sup_w_K - T0_K)
                - T0_K * math.log(self.T_sup_w_K / T0_K)
            )
        )
        X_w_comb_out = (
            c_w
            * rho_w
            * dV_w_sup_comb
            * (
                (self.T_comb_setpoint_K - T0_K)
                - T0_K * math.log(self.T_comb_setpoint_K / T0_K)
            )
        )
        X_exh = (1 - T0_K / self.T_exh_K) * Q_exh
        X_c_comb = S_g_comb * T0_K

        X_w_sup_mix = (
            c_w
            * rho_w
            * dV_w_sup_mix
            * (
                (self.T_sup_w_K - T0_K)
                - T0_K * math.log(self.T_sup_w_K / T0_K)
            )
        )
        X_w_serv = (
            c_w
            * rho_w
            * dV_w_serv
            * (
                (T_serv_w_actual_K - T0_K)
                - T0_K * math.log(T_serv_w_actual_K / T0_K)
            )
        )
        X_c_mix = S_g_mix * T0_K

        # Total exergy consumption
        X_c_tot = X_c_comb + X_c_mix
        X_eff = X_w_serv / X_NG if X_NG > 0 else 0.0

        # --- Build result dictionary ---
        result = {
            "is_on": True,
            "converged": True,
            "Q_comb_load [W]": Q_comb_load,
            "E_NG [W]": E_NG,
            "Q_w_comb_out [W]": Q_w_comb_out,
            "Q_exh [W]": Q_exh,
            "Q_w_sup [W]": Q_w_sup,
            "Q_w_sup_mix [W]": Q_w_sup_mix,
            "Q_w_serv [W]": Q_w_serv,
            "dV_w_serv [m3/s]": dV_w_serv,
            "dV_w_sup_comb [m3/s]": dV_w_sup_comb,
            "dV_w_sup_mix [m3/s]": dV_w_sup_mix,
            "T0 [°C]": T0,
            "T_serv_w [°C]": T_serv_w_actual,
            "T_sup_w [°C]": self.T_sup_w,
            "T_w_comb_out [°C]": T_w_comb_out,
            "T_exh [°C]": self.T_exh,
            "alp [-]": alp,
            # Entropy
            "S_NG [W/K]": S_NG,
            "S_w_sup [W/K]": S_w_sup,
            "S_w_comb_out [W/K]": S_w_comb_out,
            "S_exh [W/K]": S_exh,
            "S_g_comb [W/K]": S_g_comb,
            "S_w_sup_mix [W/K]": S_w_sup_mix,
            "S_w_serv [W/K]": S_w_serv,
            "S_g_mix [W/K]": S_g_mix,
            # Exergy
            "X_NG [W]": X_NG,
            "X_w_sup [W]": X_w_sup,
            "X_w_comb_out [W]": X_w_comb_out,
            "X_exh [W]": X_exh,
            "X_c_comb [W]": X_c_comb,
            "X_w_sup_mix [W]": X_w_sup_mix,
            "X_w_serv [W]": X_w_serv,
            "X_c_mix [W]": X_c_mix,
            "X_c_tot [W]": X_c_tot,
            "X_eff [-]": X_eff,
        }

        # Balance dictionaries (for print_balance utility)
        result["energy_balance"] = {
            "combustion chamber": {
                "in": {"E_NG": E_NG, "Q_w_sup": Q_w_sup},
                "out": {"Q_w_comb_out": Q_w_comb_out, "Q_exh": Q_exh},
            },
            "mixing valve": {
                "in": {
                    "Q_w_comb_out": Q_w_comb_out,
                    "Q_w_sup_mix": Q_w_sup_mix,
                },
                "out": {"Q_w_serv": Q_w_serv},
            },
        }

        result["entropy_balance"] = {
            "combustion chamber": {
                "in": {"S_NG": S_NG, "S_w_sup": S_w_sup},
                "out": {"S_w_comb_out": S_w_comb_out, "S_exh": S_exh},
                "gen": {"S_g_comb": S_g_comb},
            },
            "mixing valve": {
                "in": {
                    "S_w_comb_out": S_w_comb_out,
                    "S_w_sup_mix": S_w_sup_mix,
                },
                "out": {"S_w_serv": S_w_serv},
                "gen": {"S_g_mix": S_g_mix},
            },
        }

        result["exergy_balance"] = {
            "combustion chamber": {
                "in": {"X_NG": X_NG, "X_w_sup": X_w_sup},
                "out": {"X_w_comb_out": X_w_comb_out, "X_exh": X_exh},
                "con": {"X_c_comb": X_c_comb},
            },
            "mixing valve": {
                "in": {
                    "X_w_comb_out": X_w_comb_out,
                    "X_w_sup_mix": X_w_sup_mix,
                },
                "out": {"X_w_serv": X_w_serv},
                "con": {"X_c_mix": X_c_mix},
            },
        }

        return result

    def _calc_off_state(self, T0):
        """Compute off-state result (zero loads, retain temperature fields).

        Parameters
        ----------
        T0 : float
            Dead-state temperature [°C].

        Returns
        -------
        dict
            Off-state result dictionary.
        """
        flow_state = self._build_flow_state(
            dV_w_serv=0.0,
            T_serv_w_K=self.T_serv_w_K,
            T_sup_w_K=self.T_sup_w_K,
            T_comb_setpoint_K=self.T_comb_setpoint_K,
        )
        result = self._calc_on_state(Q_comb_load=0.0, T0=T0, flow_state=flow_state)

        # Zero out all numeric values except temperatures and mixing ratio
        for key, value in result.items():
            if (
                isinstance(value, (int, float))
                and "T_" not in key
                and "alp" not in key
            ):
                result[key] = 0.0

        result["is_on"] = False
        result["converged"] = True

        result["T0 [°C]"] = T0
        result["T_serv_w [°C]"] = cu.K2C(flow_state["T_serv_w_actual_K"])
        result["T_sup_w [°C]"] = self.T_sup_w
        result["T_w_comb_out [°C]"] = self.T_comb_setpoint
        result["T_exh [°C]"] = self.T_exh

        return result

    def analyze_steady(
        self,
        T0,
        dV_w_serv=None,
        return_dict=True,
    ):
        """Run a steady-state analysis at the given operating point.

        Parameters
        ----------
        T0 : float
            Dead-state (ambient) temperature [°C].
        dV_w_serv : float, optional
            Service water flow rate [m³/s]. Defaults to 0.
        return_dict : bool
            If True return dict; if False return single-row DataFrame.

        Returns
        -------
        dict or pd.DataFrame
            Operating-point result.
        """
        if dV_w_serv is None:
            dV_w_serv = 0.0

        flow_state = self._build_flow_state(
            dV_w_serv=dV_w_serv,
            T_serv_w_K=self.T_serv_w_K,
            T_sup_w_K=self.T_sup_w_K,
            T_comb_setpoint_K=self.T_comb_setpoint_K,
        )

        dV_w_sup_comb = flow_state["dV_w_sup_comb"]
        Q_comb_load = (
            c_w
            * rho_w
            * dV_w_sup_comb
            * (self.T_comb_setpoint_K - self.T_sup_w_K)
        )

        is_on = Q_comb_load > self.Q_comb_load_threshold

        if abs(Q_comb_load) <= self.Q_comb_load_threshold or not is_on:
            result = self._calc_off_state(T0=T0)
        else:
            result = self._calc_on_state(
                Q_comb_load=Q_comb_load, T0=T0, flow_state=flow_state
            )

        if return_dict:
            return result
        else:
            return pd.DataFrame([result])

    def analyze_dynamic(
        self,
        simulation_period_sec,
        dt_s,
        dhw_usage_schedule,
        T0_schedule,
        heater_capacity_const=None,
        heater_capacity_schedule=None,
        result_save_csv_path=None,
    ):
        """Run a time-stepping dynamic simulation.

        Parameters
        ----------
        simulation_period_sec : int
            Total simulation duration [s].
        dt_s : int
            Time step size [s].
        dhw_usage_schedule : list of tuple
            DHW schedule as ``(start_str, end_str, fraction)`` entries.
        T0_schedule : array-like
            Dead-state temperature per time step [°C].
        heater_capacity_const : float, optional
            Fixed heater power [W].
        heater_capacity_schedule : array-like, optional
            Time-varying heater power [W].
        result_save_csv_path : str, optional
            Path to save result CSV.

        Returns
        -------
        pd.DataFrame
            Per-timestep simulation results.
        """
        if simulation_period_sec % dt_s != 0:
            raise ValueError("simulation_period_sec must be divisible by dt_s")
        if self.dV_w_serv_m3s < 0:
            raise ValueError("dV_w_serv_m3s must be greater than 0")
        if dhw_usage_schedule == []:
            raise ValueError("dhw_usage_schedule must be provided")

        time = np.arange(0, simulation_period_sec, dt_s)
        tN = len(time)
        T0_schedule = np.array(T0_schedule)
        if len(T0_schedule) != tN:
            raise ValueError(
                f"T0_schedule length ({len(T0_schedule)}) must match time array length ({tN})"
            )

        results_data = []
        self.time = time
        self.dt = dt_s

        # Build schedule ratio array
        self.w_use_frac = build_dhw_usage_ratio(dhw_usage_schedule, self.time)

        for n in tqdm(range(tN), desc="GasBoiler Simulating"):
            step_results = {}
            T0 = T0_schedule[n]

            # Current service flow
            dV_w_serv = self.w_use_frac[n] * self.dV_w_serv_m3s

            flow_state = self._build_flow_state(
                dV_w_serv=dV_w_serv,
                T_serv_w_K=self.T_serv_w_K,
                T_sup_w_K=self.T_sup_w_K,
                T_comb_setpoint_K=self.T_comb_setpoint_K,
            )
            dV_w_sup_comb = flow_state["dV_w_sup_comb"]

            # Required combustion load
            Q_comb_load = (
                c_w
                * rho_w
                * dV_w_sup_comb
                * (self.T_comb_setpoint_K - self.T_sup_w_K)
            )

            is_on = (self.T_serv_w > self.T_sup_w) and (dV_w_sup_comb > 0)

            result = self._calc_on_state(
                Q_comb_load=Q_comb_load, T0=T0, flow_state=flow_state
            )

            step_results.update(result)
            step_results["is_on"] = is_on
            step_results["time [s]"] = time[n]
            results_data.append(step_results)

        results_df = pd.DataFrame(results_data)
        if result_save_csv_path:
            results_df.to_csv(result_save_csv_path, index=False)

        return results_df
