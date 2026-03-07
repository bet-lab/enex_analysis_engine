"""Air source heat pump boiler — physics-based cycle model.

Resolves a vapour-compression refrigerant cycle coupled to an outdoor-air
evaporator with a VSD fan and a lumped-capacitance hot-water tank.
At each time step the model finds the minimum-power operating point
(compressor + fan) via bounded 1-D optimisation (Brent's method) over
the evaporator approach temperature difference.  The condenser approach
temperature is determined analytically from the target heat load.

Optional subsystems (injected via constructor):
- ``SolarThermalCollector`` — tank-circuit or mains-preheat placement
- (future) ``PVPanel`` — photovoltaic integration

Tank-level management and UV disinfection are built-in features
configured through constructor parameters.
"""

import contextlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from tqdm import tqdm

from . import calc_util as cu
from .constants import c_a, c_w, rho_a, rho_w
from .dynamic_context import (
    ControlState,
    StepContext,
    determine_heat_source_on_off,
    determine_tank_refill_flow,
    tank_mass_energy_residual,
)
from .enex_functions import (
    calc_energy_flow,
    calc_fan_power_from_dV_fan,
    calc_HX_perf_for_target_heat,
    calc_mixing_valve,
    calc_ref_state,
    calc_simple_tank_UA,
)
from .subsystems import SolarThermalCollector


@dataclass
class AirSourceHeatPumpBoiler:
    """Air source heat pump boiler with outdoor-air evaporator.

    The refrigerant cycle is resolved via CoolProp with
    user-specified superheat / subcool margins.  The condenser
    approach temperature is determined analytically
    (``dT_ref_cond = Q_cond_target / UA_cond``), and a bounded
    1-D optimiser (Brent's method) minimises total electrical
    input (``E_cmp + E_ou_fan``) over the evaporator approach.
    """

    def __init__(
        self,
        # 1. Refrigerant / cycle / compressor -----------
        ref: str = "R134a",
        V_disp_cmp: float = 0.0002,
        eta_cmp_isen: float = 0.8,
        dT_superheat: float = 3.0,
        dT_subcool: float = 3.0,
        # 2. Heat exchanger -----------------------------
        UA_cond_design: float = 2000.0,
        UA_evap_design: float = 1000.0,
        # 3. Outdoor unit fan ---------------------------
        dV_ou_fan_a_design: float = 1.5,
        dP_ou_fan_design: float = 90.0,
        A_cross_ou: float = 0.25**2 * np.pi,
        eta_ou_fan_design: float = 0.6,
        # 4. Tank / control / load ----------------------
        T_tank_w_upper_bound: float = 65.0,
        T_tank_w_lower_bound: float = 60.0,
        T_mix_w_out: float = 40.0,
        T_sup_w: float = 15.0,
        hp_capacity: float = 15000.0,
        dV_mix_w_out_max: float = 0.0045,
        # Tank insulation
        r0: float = 0.2,
        H: float = 1.2,
        x_shell: float = 0.005,
        x_ins: float = 0.05,
        k_shell: float = 25,
        k_ins: float = 0.03,
        h_o: float = 15,
        # 5. Tank water level management ----------------
        tank_always_full: bool = True,
        tank_level_lower_bound: float = 0.5,
        tank_level_upper_bound: float = 1.0,
        dV_tank_w_in_refill: float = 0.001,
        prevent_simultaneous_flow: bool = False,
        # 7. HP operating schedule ----------------------
        hp_on_schedule: list[tuple[float, float]] | None = None,
        # 8. Subsystems (class-based injection) ---------
        stc: SolarThermalCollector | None = None,
        uv=None,
        # ASHRAE 90.1-2022 VSD coefficients
        vsd_coeffs_ou: dict | None = None,
    ):
        if hp_on_schedule is None:
            hp_on_schedule = [(0.0, 24.0)]
        if vsd_coeffs_ou is None:
            vsd_coeffs_ou = {
                "c1": 0.0013,
                "c2": 0.1470,
                "c3": 0.9506,
                "c4": -0.0998,
                "c5": 0.0,
            }

        # --- 1. Refrigerant / cycle / compressor ---
        self.ref: str = ref
        self.V_disp_cmp: float = V_disp_cmp
        self.eta_cmp_isen: float = eta_cmp_isen
        self.dT_superheat: float = dT_superheat
        self.dT_subcool: float = dT_subcool
        self.hp_capacity: float = hp_capacity

        # --- 2. Heat exchanger UA ---
        self.UA_cond_design: float = UA_cond_design
        self.UA_evap_design: float = UA_evap_design

        # --- 3. Outdoor unit fan ---
        self.dV_ou_fan_a_design: float = dV_ou_fan_a_design
        self.dP_ou_fan_design: float = dP_ou_fan_design
        self.eta_ou_fan_design: float = eta_ou_fan_design
        self.A_cross_ou: float = A_cross_ou
        self.E_ou_fan_design: float = (
            dV_ou_fan_a_design * dP_ou_fan_design / eta_ou_fan_design
        )
        self.vsd_coeffs_ou: dict = vsd_coeffs_ou
        self.fan_params_ou: dict = {
            "fan_design_flow_rate": dV_ou_fan_a_design,
            "fan_design_power": self.E_ou_fan_design,
        }

        # --- 4. Tank geometry and thermal props ---
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

        self.dV_mix_w_out_max: float = dV_mix_w_out_max
        self.T_tank_w_upper_bound: float = T_tank_w_upper_bound
        self.T_tank_w_lower_bound: float = T_tank_w_lower_bound
        self.T_sup_w: float = T_sup_w
        self.T_sup_w_K: float = cu.C2K(T_sup_w)
        self.T_tank_w_in: float = T_sup_w
        self.T_mix_w_out: float = T_mix_w_out
        self.T_tank_w_in_K: float = cu.C2K(T_sup_w)
        self.T_mix_w_out_K: float = cu.C2K(T_mix_w_out)

        # --- 5. Tank water level management ---
        self.tank_always_full: bool = tank_always_full
        self.tank_level_lower_bound: float = tank_level_lower_bound
        self.tank_level_upper_bound: float = tank_level_upper_bound
        self.dV_tank_w_in_refill: float = dV_tank_w_in_refill
        self.prevent_simultaneous_flow: bool = prevent_simultaneous_flow

        # --- 6. HP operating schedule ---
        self.hp_on_schedule: list[tuple[float, float]] = hp_on_schedule

        # --- 7. Subsystems ---
        self.stc: SolarThermalCollector | None = stc
        self._subsystems: dict[str, Any] = {}
        if stc is not None:
            self._subsystems["stc"] = stc
        if uv is not None:
            self._subsystems["uv"] = uv

        # Flow-rate sync variables
        self.dV_tank_w_in: float = 0.0
        self.dV_tank_w_out: float = 0.0
        self.dV_mix_sup_w_in: float = 0.0
        self.dV_mix_w_out: float = 0.0

    # =============================================================
    # Refrigerant cycle physics (ASHP-specific)
    # =============================================================

    def _calc_state(
        self,
        dT_ref_evap: float,
        T_tank_w: float,
        Q_cond_target: float,
        T0: float,
    ) -> dict | None:
        """Evaluate refrigerant cycle at a given operating point.

        Parameters
        ----------
        dT_ref_evap : float
            Evaporator approach ΔT [K].
        T_tank_w : float
            Tank water temperature [°C].
        Q_cond_target : float
            Target condenser heat rate [W].
        T0 : float
            Dead-state / outdoor-air temperature [°C].

        Returns
        -------
        dict | None
            Cycle performance dictionary; ``None`` if infeasible.
        """
        dT_ref_cond: float = (
            Q_cond_target / self.UA_cond_design if Q_cond_target > 0 else 0.0
        )

        T_tank_w_K: float = cu.C2K(T_tank_w)
        T0_K: float = cu.C2K(T0)

        T_evap_sat_K: float = T0_K - dT_ref_evap
        T_cond_sat_K: float = T_tank_w_K + dT_ref_cond

        is_active: bool = Q_cond_target > 0.0

        cs: dict = calc_ref_state(
            T_evap_K=T_evap_sat_K,
            T_cond_K=T_cond_sat_K,
            refrigerant=self.ref,
            eta_cmp_isen=self.eta_cmp_isen,
            mode="heating",
            dT_superheat=self.dT_superheat,
            dT_subcool=self.dT_subcool,
            is_active=is_active,
        )

        m_dot_ref: float = (
            Q_cond_target
            / (cs["h_ref_cmp_out [J/kg]"] - cs["h_ref_exp_in [J/kg]"])
            if is_active
            else 0.0
        )
        Q_ref_cond: float = (
            m_dot_ref
            * (cs["h_ref_cmp_out [J/kg]"] - cs["h_ref_exp_in [J/kg]"])
            if is_active
            else 0.0
        )
        Q_ref_evap: float = (
            m_dot_ref
            * (cs["h_ref_cmp_in [J/kg]"] - cs["h_ref_exp_out [J/kg]"])
            if is_active
            else 0.0
        )
        E_cmp: float = (
            m_dot_ref
            * (cs["h_ref_cmp_out [J/kg]"] - cs["h_ref_cmp_in [J/kg]"])
            if is_active
            else 0.0
        )
        cmp_rps: float = (
            m_dot_ref / (self.V_disp_cmp * cs["rho_ref_cmp_in [kg/m3]"])
            if is_active
            else 0.0
        )

        HX_perf_ou: dict = calc_HX_perf_for_target_heat(
            Q_ref_target=(Q_ref_evap if is_active else 0.0),
            T_ou_a_in_C=T0,
            T_ref_evap_sat_K=cs["T_ref_evap_sat_K"],
            T_ref_cond_sat_l_K=cs["T_ref_cond_sat_l_K"],
            A_cross=self.A_cross_ou,
            UA_design=self.UA_evap_design,
            dV_fan_design=self.dV_ou_fan_a_design,
            is_active=is_active,
        )

        if HX_perf_ou.get("converged", True) is False:
            return {
                "converged": False,
                "_hx_diag": HX_perf_ou,
            }

        dV_ou_a: float = HX_perf_ou["dV_fan"]
        v_ou_a: float = dV_ou_a / self.A_cross_ou if is_active else 0.0
        T_ou_a_mid: float = HX_perf_ou["T_ou_a_mid"]
        Q_ou_a: float = HX_perf_ou["Q_ou_air"]

        E_ou_fan: float = calc_fan_power_from_dV_fan(
            dV_fan=dV_ou_a,
            fan_params=self.fan_params_ou,
            vsd_coeffs=self.vsd_coeffs_ou,
            is_active=is_active,
        )

        T_ou_a_out: float = (
            T_ou_a_mid + E_ou_fan / (c_a * rho_a * dV_ou_a)
            if is_active
            else T0
        )
        (
            self.eta_ou_fan_design * dV_ou_a / E_ou_fan * 100
            if is_active
            else 0.0
        )


        dV_tank_w_out: float = self.dV_tank_w_out
        dV_tank_w_in: float = self.dV_tank_w_in
        dV_mix_sup_w_in: float = self.dV_mix_sup_w_in
        dV_mix_w_out_val: float = self.dV_mix_w_out

        if dV_mix_w_out_val == 0:
            T_mix_w_out_val: float = np.nan
            T_mix_w_out_val_K: float = np.nan
        else:
            mix: dict = calc_mixing_valve(
                T_tank_w_K,
                self.T_sup_w_K,
                self.T_mix_w_out_K,
            )
            T_mix_w_out_val = mix["T_mix_w_out"]
            T_mix_w_out_val_K = mix["T_mix_w_out_K"]

        Q_tank_w_in: float = calc_energy_flow(
            G=c_w * rho_w * dV_tank_w_in,
            T=self.T_tank_w_in_K,
            T0=T0_K,
        )
        Q_tank_w_out: float = calc_energy_flow(
            G=c_w * rho_w * dV_tank_w_out,
            T=T_tank_w_K,
            T0=T0_K,
        )
        Q_mix_sup_w_in: float = calc_energy_flow(
            G=c_w * rho_w * dV_mix_sup_w_in,
            T=self.T_sup_w_K,
            T0=T0_K,
        )
        Q_mix_w_out: float = calc_energy_flow(
            G=c_w * rho_w * dV_mix_w_out_val,
            T=T_mix_w_out_val_K,
            T0=T0_K,
        )

        result: dict = cs.copy()

        result.update(
            {
                "hp_is_on": (Q_cond_target > 0),
                "converged": True,
                # Temperatures [°C]
                "T_ou_a_in [°C]": T0,
                "T_ou_a_mid [°C]": T_ou_a_mid,
                "T_ou_a_out [°C]": T_ou_a_out,
                "T_tank_w [°C]": T_tank_w,
                "T_sup_w [°C]": self.T_sup_w,
                "T_tank_w_in [°C]": self.T_tank_w_in,
                "T_mix_w_out [°C]": T_mix_w_out_val,
                "T0 [°C]": T0,
                # Volume flow rates [m3/s]
                "dV_ou_a [m3/s]": dV_ou_a,
                "v_ou_a [m/s]": v_ou_a,
                "dV_mix_w_out [m3/s]": (
                    dV_mix_w_out_val if dV_mix_w_out_val > 0 else np.nan
                ),
                "dV_tank_w_out [m3/s]": (
                    dV_tank_w_out if dV_tank_w_out > 0 else np.nan
                ),
                "dV_tank_w_in [m3/s]": (
                    dV_tank_w_in if dV_tank_w_in > 0 else np.nan
                ),
                "dV_mix_sup_w_in [m3/s]": (
                    dV_mix_sup_w_in if dV_mix_sup_w_in > 0 else np.nan
                ),
                "m_dot_ref [kg/s]": m_dot_ref,  # Mass flow rate [kg/s]
                "cmp_rpm [rpm]": cmp_rps * 60,  # Compressor speed [rpm]
                # Energy rates [W]
                "E_ou_fan [W]": E_ou_fan,
                "Q_ref_evap [W]": Q_ref_evap,
                "Q_ou_a [W]": Q_ou_a,
                "E_cmp [W]": E_cmp,
                "Q_ref_cond [W]": Q_ref_cond,
                "Q_tank_w_in [W]": Q_tank_w_in,
                "Q_tank_w_out [W]": Q_tank_w_out,
                "Q_mix_sup_w_in [W]": Q_mix_sup_w_in,
                "Q_mix_w_out [W]": Q_mix_w_out,
                "E_tot [W]": E_cmp + E_ou_fan,
            }
        )

        return result

    def _optimize_operation(
        self,
        T_tank_w: float,
        Q_cond_target: float,
        T0: float,
    ):
        """Find min-power operating point (Brent 1-D).

        Parameters
        ----------
        T_tank_w : float
            Tank water temperature [°C].
        Q_cond_target : float
            Target condenser heat rate [W].
        T0 : float
            Dead-state temperature [°C].

        Returns
        -------
        scipy.optimize.OptimizeResult
        """

        def _objective(dT_ref_evap: float) -> float:
            try:
                perf: dict | None = self._calc_state(
                    dT_ref_evap=dT_ref_evap,
                    T_tank_w=T_tank_w,
                    Q_cond_target=Q_cond_target,
                    T0=T0,
                )
                if perf is None or not isinstance(perf, dict):
                    return 1e6
                E_tot: float = perf.get(
                    "E_tot [W]",
                    np.nan,
                )
                return E_tot if not np.isnan(E_tot) else 1e6
            except Exception:
                return 1e6

        return minimize_scalar(
            _objective,
            bounds=(5.0, 15.0),
            method="bounded",
            options={"maxiter": 200, "xatol": 1e-6},
        )

    # =============================================================
    # Steady-state analysis
    # =============================================================

    def analyze_steady(
        self,
        T_tank_w: float,
        T0: float,
        dV_mix_w_out: float | None = None,
        Q_cond_target: float | None = None,
        return_dict: bool = True,
    ) -> dict | pd.DataFrame:
        """Run a steady-state analysis.

        Exactly one of ``dV_mix_w_out`` or ``Q_cond_target``
        must be provided.

        Parameters
        ----------
        T_tank_w : float
            Tank water temperature [°C].
        T0 : float
            Dead-state temperature [°C].
        dV_mix_w_out : float | None
            Service water flow rate [m3/s].
        Q_cond_target : float | None
            Target condenser heat rate [W].
        return_dict : bool
            If True return dict; else single-row DataFrame.

        Returns
        -------
        dict | pd.DataFrame
        """
        if dV_mix_w_out is None and Q_cond_target is None:
            raise ValueError(
                "Either dV_mix_w_out or Q_cond_target must be provided.",
            )
        if dV_mix_w_out is not None and Q_cond_target is not None:
            raise ValueError(
                "Cannot provide both dV_mix_w_out and Q_cond_target.",
            )

        T_tank_w_K: float = cu.C2K(T_tank_w)
        T0_K: float = cu.C2K(T0)

        if dV_mix_w_out is None:
            dV_mix_w_out = 0.0

        Q_tank_loss: float = self.UA_tank * (T_tank_w_K - T0_K)
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

        self.dV_mix_w_out = dV_mix_w_out
        self.dV_tank_w_out = alp * dV_mix_w_out
        self.dV_mix_sup_w_in = (1 - alp) * dV_mix_w_out

        if Q_cond_target is None:
            Q_tank_w_use: float = (
                c_w
                * rho_w
                * self.dV_tank_w_out
                * (T_tank_w_K - self.T_sup_w_K)
            )
            Q_cond_target = Q_tank_loss + Q_tank_w_use

        if T_tank_w <= self.T_tank_w_lower_bound:
            hp_is_on: bool = True
        elif T_tank_w > self.T_tank_w_upper_bound:
            hp_is_on = False
        else:
            hp_is_on = Q_cond_target > 0

        if Q_cond_target <= 0 or not hp_is_on:
            result: dict | None = self._calc_state(
                dT_ref_evap=5.0,
                T_tank_w=T_tank_w,
                Q_cond_target=0.0,
                T0=T0,
            )
        else:
            opt_result = self._optimize_operation(
                T_tank_w=T_tank_w,
                Q_cond_target=Q_cond_target,
                T0=T0,
            )
            result = None
            with contextlib.suppress(Exception):
                result = self._calc_state(
                    dT_ref_evap=opt_result.x,
                    T_tank_w=T_tank_w,
                    T0=T0,
                    Q_cond_target=Q_cond_target,
                )

            if result is None or not isinstance(
                result,
                dict,
            ):
                try:
                    result = self._calc_state(
                        dT_ref_evap=5.0,
                        T_tank_w=T_tank_w,
                        Q_cond_target=0.0,
                        T0=T0,
                    )
                except Exception:
                    result = {
                        "hp_is_on": False,
                        "converged": False,
                        "Q_ref_cond [W]": 0.0,
                        "Q_ref_evap [W]": 0.0,
                        "E_cmp [W]": 0.0,
                        "E_ou_fan [W]": 0.0,
                        "E_tot [W]": 0.0,
                        "T_tank_w [°C]": T_tank_w,
                        "T0 [°C]": T0,
                    }

            if result is not None and isinstance(
                result,
                dict,
            ) and "opt_result" in locals() and hasattr(opt_result, "success"):
                result["converged"] = opt_result.success
                if not result["converged"]:
                    print("Optimization failed")

        if return_dict:
            return result
        return pd.DataFrame([result])

    # =============================================================
    # Private helpers for analyze_dynamic
    # =============================================================

    def _determine_hp_state(
        self,
        ctx: StepContext,
        hp_is_on_prev: bool,
    ) -> tuple[bool, dict, float]:
        """HP on/off + cycle optimisation for one step.

        Parameters
        ----------
        ctx : StepContext
            Current-step immutable context.
        hp_is_on_prev : bool
            HP state at previous step.

        Returns
        -------
        tuple[bool, dict, float]
            ``(hp_is_on, hp_result, Q_ref_cond)``.
        """
        T_tank_w: float = cu.K2C(ctx.T_tank_w_K)

        hp_is_on: bool = determine_heat_source_on_off(
            T_tank_w_C=T_tank_w,
            T_lower=self.T_tank_w_lower_bound,
            T_upper=self.T_tank_w_upper_bound,
            is_on_prev=hp_is_on_prev,
            hour_of_day=ctx.hour_of_day,
            on_schedule=self.hp_on_schedule,
        )

        Q_cond_target: float = self.hp_capacity if hp_is_on else 0.0

        # Mixing valve flows for _calc_state
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

        if Q_cond_target == 0:
            hp_result = self._calc_state(
                5.0,
                T_tank_w,
                0.0,
                ctx.T0,
            )
        else:
            opt = self._optimize_operation(
                T_tank_w,
                Q_cond_target,
                ctx.T0,
            )
            hp_result = self._calc_state(
                opt.x,
                T_tank_w,
                Q_cond_target,
                ctx.T0,
            )
            if (
                not opt.success
                or hp_result is None
                or hp_result.get("converged") is False
            ):
                self._print_opt_failure(
                    ctx,
                    T_tank_w,
                    Q_cond_target,
                    opt,
                    hp_result,
                )
                raise ValueError(
                    f"Optimization failed at step "
                    f"{ctx.n} "
                    f"(hour={ctx.hour_of_day:.2f}h): "
                    f"T_tank_w={T_tank_w:.1f}°C, "
                    f"T0={ctx.T0:.1f}°C, "
                    f"Q_target={Q_cond_target:.0f}W",
                )

        if hp_result is None:
            hp_result = {}

        return (
            hp_is_on,
            hp_result,
            float(hp_result.get("Q_ref_cond [W]", 0.0)),
        )

    def _print_opt_failure(
        self,
        ctx: StepContext,
        T_tank_w: float,
        Q_cond_target: float,
        opt,
        hp_result: dict | None,
    ) -> None:
        """Print detailed optimisation failure diagnostics."""
        print(f"\n{'=' * 70}")
        print(
            f"[HP OPTIMIZATION FAILED] Step n={ctx.n}, "
            f"hour_of_day={ctx.hour_of_day:.2f}h",
        )
        print(f"{'=' * 70}")
        print("  Operating conditions:")
        print(f"    T_tank_w     = {T_tank_w:.2f} °C")
        print(f"    T0 (outdoor) = {ctx.T0:.2f} °C")
        print(
            f"    Q_cond_target= {Q_cond_target:.1f} W",
        )
        print("  Optimizer result:")
        print(f"    opt.success  = {opt.success}")
        print(f"    opt.x        = {opt.x}")
        dT_cond: float = Q_cond_target / self.UA_cond_design
        print(f"    dT_ref_cond  = {dT_cond:.4f} K")
        print(f"    opt.fun      = {opt.fun:.2f}")
        print(f"    opt.message  = {opt.message}")
        hx_diag: dict = hp_result.get("_hx_diag", {}) if hp_result else {}
        if hx_diag:
            print("  HX bracket failure diagnostics:")
            print(
                f"    Q_ref_target = "
                f"{hx_diag.get('Q_ref_target', np.nan):.1f}"
                f" W",
            )
        print("  Suggested fixes:")
        print("    ↑ dV_ou_fan_a_design")
        print("    ↑ UA_evap_design")
        print("    ↓ hp_capacity")
        print(f"{'=' * 70}\n")

    def _assemble_core_results(
        self,
        ctx: StepContext,
        ctrl: ControlState,
        T_solved_K: float,
        level_solved: float,
        ier: int,
    ) -> dict:
        """Build HP-core result dict at solved state.

        Subsystem results are appended separately by
        each subsystem's ``assemble_results()``.
        """
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
        dV_tank_w_in: float = (
            dV_tank_w_out
            if ctrl.dV_tank_w_in_ctrl is None
            else ctrl.dV_tank_w_in_ctrl
        )

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
                "hp_is_on": ctrl.is_on,
                "Q_tank_loss [W]": (self.UA_tank * (T_solved_K - ctx.T0_K)),
                "T_tank_w [°C]": cu.K2C(T_solved_K),
                "T_mix_w_out [°C]": T_mix_w_out_val,
            }
        )

        if not self.tank_always_full or (
            self.tank_always_full and self.prevent_simultaneous_flow
        ):
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
        T_sup_w_schedule=None,
        tank_level_init: float = 1.0,
        result_save_csv_path: str | None = None,
    ) -> pd.DataFrame:
        """Run a time-stepping dynamic simulation.

        Fully implicit scheme: ``fsolve`` solves for
        ``[T_next, level_next]`` each timestep.

        Parameters
        ----------
        simulation_period_sec : int
            Total simulation duration [s].
        dt_s : int
            Time step size [s].
        T_tank_w_init_C : float
            Initial tank temperature [°C].
        dhw_usage_schedule : np.ndarray
            DHW volumetric flow rate per step [m³/s].
        T0_schedule : array-like
            Outdoor temperature per step [°C].
        I_DN_schedule : array-like | None
            Direct-normal irradiance per step [W/m²].
        I_dH_schedule : array-like | None
            Diffuse-horizontal irradiance [W/m²].
        T_sup_w_schedule : array-like | None
            Mains water supply temperature per step [°C].
            If ``None``, the constructor value ``T_sup_w``
            is used for every step (backward-compatible).
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
            0,
            simulation_period_sec,
            dt_s,
        )
        tN: int = len(time)

        T0_schedule = np.array(T0_schedule)
        if len(T0_schedule) != tN:
            raise ValueError(
                f"T0_schedule length ({len(T0_schedule)})"
                f" != time length ({tN})",
            )
        if I_DN_schedule is not None and len(I_DN_schedule) != tN:
            raise ValueError(
                f"I_DN_schedule length ({len(I_DN_schedule)}) != tN ({tN})",
            )
        if I_dH_schedule is not None and len(I_dH_schedule) != tN:
            raise ValueError(
                f"I_dH_schedule length ({len(I_dH_schedule)}) != tN ({tN})",
            )

        # T_sup_w schedule: fallback to constructor constant
        if T_sup_w_schedule is not None:
            T_sup_w_arr: np.ndarray = np.array(
                T_sup_w_schedule,
                dtype=float,
            )
            if len(T_sup_w_arr) != tN:
                raise ValueError(
                    f"T_sup_w_schedule length "
                    f"({len(T_sup_w_arr)}) != tN ({tN})",
                )
        else:
            T_sup_w_arr = np.full(tN, self.T_sup_w)

        self.time: np.ndarray = time
        self.dt: int = dt_s

        # DHW schedule handling: direct m³/s flow array
        self.dhw_flow_m3s: np.ndarray = np.asarray(
            dhw_usage_schedule,
            dtype=float,
        )
        if len(self.dhw_flow_m3s) != tN:
            raise ValueError(
                f"dhw_usage_schedule length "
                f"({len(self.dhw_flow_m3s)}) != tN ({tN})",
            )

        T_tank_w_K: float = cu.C2K(T_tank_w_init_C)
        tank_level: float = tank_level_init
        is_refilling: bool = False
        hp_is_on_prev: bool = False
        results_data: list[dict] = []

        # STC-related defaults
        stc_sub = self.stc
        use_stc: bool = stc_sub is not None
        mode: str = stc_sub.mode if stc_sub is not None else "tank_circuit"

        for n in tqdm(range(tN), desc="ASHPB Simulating"):
            t_s: float = time[n]
            hr: float = t_s * cu.s2h
            hour_of_day: float = (t_s % (24 * cu.h2s)) * cu.s2h

            # Per-step mains water supply temperature
            T_sup_w_n: float = T_sup_w_arr[n]
            T_sup_w_K_n: float = cu.C2K(T_sup_w_n)
            T_tank_w_in_K_n: float = T_sup_w_K_n

            # Sync self fields for _calc_state compat
            self.T_sup_w = T_sup_w_n
            self.T_sup_w_K = T_sup_w_K_n
            self.T_tank_w_in = T_sup_w_n
            self.T_tank_w_in_K = T_tank_w_in_K_n

            # Preheat window (STC only)
            preheat_on: bool = (
                stc_sub.is_preheat_on(hour_of_day)
                if stc_sub is not None
                else False
            )

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
                dV_mix_w_out=self.dhw_flow_m3s[n],
                I_DN=(I_DN_schedule[n] if use_stc else 0.0),
                I_dH=(I_dH_schedule[n] if use_stc else 0.0),
                T_sup_w_K=T_sup_w_K_n,
            )

            # --- Phase A: control decisions ---
            hp_is_on, hp_result, Q_ref_cond = self._determine_hp_state(
                ctx, hp_is_on_prev
            )
            hp_is_on_prev = hp_is_on

            dV_tank_w_in_ctrl, is_refilling = determine_tank_refill_flow(
                dt=dt_s,
                tank_level=ctx.tank_level,
                dV_tank_w_out=self.dV_tank_w_out,
                V_tank_full=self.V_tank_full,
                tank_always_full=self.tank_always_full,
                prevent_simultaneous_flow=self.prevent_simultaneous_flow,
                tank_level_lower_bound=self.tank_level_lower_bound,
                tank_level_upper_bound=self.tank_level_upper_bound,
                dV_tank_w_in_refill=self.dV_tank_w_in_refill,
                is_refilling=is_refilling,
                use_stc=use_stc,
                mode=mode,
                preheat_on=ctx.preheat_on,
            )

            ctrl: ControlState = ControlState(
                is_on=hp_is_on,
                Q_heat_source=Q_ref_cond,
                dV_tank_w_in_ctrl=dV_tank_w_in_ctrl,
                result=hp_result,
            )

            # --- Phase A-2: subsystem step ---
            sub_states: dict[str, dict] = {}
            for name, sub in self._subsystems.items():
                sub_states[name] = sub.step(
                    ctx,
                    ctrl,
                    dt_s,
                    T_tank_w_in_K_n,
                )

            # --- Phase B: implicit solve ---
            sol, _info, ier, _msg = fsolve(
                tank_mass_energy_residual,
                [ctx.T_tank_w_K, ctx.tank_level],
                args=(
                    ctx,
                    ctrl,
                    dt_s,
                    T_tank_w_in_K_n,
                    T_sup_w_K_n,
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
    # Exergy post-processing (ASHP-specific)
    # =============================================================

    def postprocess_exergy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute ASHP-specific exergy variables.

        Owns the full HP exergy topology:

        1. Refrigerant state-point exergy (CoolProp)
        2. Electricity = exergy (compressor, OU fan, UV)
        3. Air exergy (outdoor unit)
        4. Heat exchanger Carnot exergy (condenser, evaporator)
        5. Water exergy (tank inlet/outlet, mixing valve)
        6. Heat loss exergy, tank stored exergy
        7. Subsystem exergy via ``calc_exergy()`` protocol
        8. Component-level exergy destruction
        9. Exergetic efficiency metrics

        Parameters
        ----------
        df : pd.DataFrame
            Result DataFrame from ``analyze_dynamic()``.

        Returns
        -------
        pd.DataFrame
            DataFrame with exergy columns appended.
        """
        from .enex_functions import (
            calc_exergy_flow,
            calc_refrigerant_exergy,
            convert_electricity_to_exergy,
        )

        df = df.copy()
        T0_K = cu.C2K(df["T0 [°C]"])
        T_tank_K = cu.C2K(df["T_tank_w [°C]"])

        # ── 1. Refrigerant exergy (uses pre-computed h/s from calc_ref_state)
        df = calc_refrigerant_exergy(df, self.ref, T0_K)

        # ── 2. Electricity = exergy ────────────────────────
        df = convert_electricity_to_exergy(df)

        # ── 3. Air exergy (outdoor unit) ───────────────────
        if "dV_ou_a [m3/s]" in df.columns and "T_ou_a_in [°C]" in df.columns:
            G_a = c_a * rho_a * df["dV_ou_a [m3/s]"]
            Tin = cu.C2K(df["T_ou_a_in [°C]"])
            Tmid = cu.C2K(df["T_ou_a_mid [°C]"])
            Tout = (
                cu.C2K(df["T_ou_a_out [°C]"])
                if "T_ou_a_out [°C]" in df.columns
                else Tin
            )
            df["X_a_ou_in [W]"] = calc_exergy_flow(G_a, Tin, T0_K)
            df["X_a_ou_out [W]"] = calc_exergy_flow(G_a, Tout, T0_K)
            df["X_a_ou_mid [W]"] = calc_exergy_flow(G_a, Tmid, T0_K)

        # ── 4. HX exergy (Carnot form) ─────────────────────
        df["X_ref_cond [W]"] = df["Q_ref_cond [W]"] * (
            1 - T0_K / cu.C2K(df["T_ref_cond_sat_v [°C]"])
        )
        df["X_ref_evap [W]"] = df["Q_ref_evap [W]"] * (
            1 - T0_K / cu.C2K(df["T_ref_evap_sat [°C]"])
        )

        # ── 5. Water exergy (inlet / outlet) ───────────────
        df["X_tank_w_in [W]"] = calc_exergy_flow(
            c_w * rho_w * df["dV_tank_w_in [m3/s]"].fillna(0),
            cu.C2K(df["T_tank_w_in [°C]"]),
            T0_K,
        )
        df["X_tank_w_out [W]"] = calc_exergy_flow(
            c_w * rho_w * df["dV_tank_w_out [m3/s]"].fillna(0),
            cu.C2K(df["T_tank_w [°C]"]),
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
        tank_level = (
            df["tank_level [-]"] if "tank_level [-]" in df.columns else 1.0
        )
        C_tank_actual = self.C_tank * tank_level
        T_tank_K_prev = T_tank_K.shift(1)
        df["Xst_tank [W]"] = (
            (1 - T0_K / T_tank_K)
            * C_tank_actual
            * (T_tank_K - T_tank_K_prev)
            / self.dt
        )
        df.loc[df.index[0], "Xst_tank [W]"] = 0.0

        # ── 8. Subsystem exergy (protocol) ─────────────────
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

        # ── 9. Total exergy input (system-level) ──────────
        X_tot = df["E_cmp [W]"] + df["E_ou_fan [W]"]
        if "X_uv [W]" in df.columns:
            X_tot = X_tot + df["X_uv [W]"].fillna(0)
        X_tot = X_tot + X_sub_tot_add
        df["X_tot [W]"] = X_tot

        # ── 10. Component exergy destruction ───────────────
        # Xc = ΣX_in − ΣX_out ≥ 0 (2nd law)
        df["Xc_cmp [W]"] = (
            df["X_cmp [W]"] + df["X_ref_cmp_in [W]"] - df["X_ref_cmp_out [W]"]
        )
        df["Xc_ref_cond [W]"] = (
            df["X_ref_cmp_out [W]"]
            - df["X_ref_exp_in [W]"]
            - df["X_ref_cond [W]"]
        )
        df["Xc_exp [W]"] = df["X_ref_exp_in [W]"] - df["X_ref_exp_out [W]"]
        df["Xc_ref_evap [W]"] = (
            df["X_ref_exp_out [W]"] + df["X_a_ou_in [W]"]
        ) - (df["X_ref_cmp_in [W]"] + df["X_a_ou_mid [W]"])
        df["Xc_ou_fan [W]"] = (
            df["X_ou_fan [W]"] + df["X_a_ou_mid [W]"] - df["X_a_ou_out [W]"]
        )
        df["Xc_mix [W]"] = (
            df["X_tank_w_out [W]"]
            + df["X_mix_sup_w_in [W]"]
            - df["X_mix_w_out [W]"]
        )

        # 10g. Storage tank
        X_in_tank = df["X_ref_cond [W]"] + df["X_tank_w_in [W]"].fillna(0)
        if "X_uv [W]" in df.columns:
            X_in_tank = X_in_tank + df["X_uv [W]"].fillna(0)
        X_in_tank = X_in_tank + X_sub_in_tank_add

        X_out_tank = df[
            "Xst_tank [W]"
        ]  # df['X_tank_loss [W]']를 제외하는 이유는 X_tank_loss 또한 exergy consumption에 포함시기 위함임
        if "X_tank_w_out [W]" in df.columns:
            X_out_tank = X_out_tank + df["X_tank_w_out [W]"].fillna(0)
        X_out_tank = X_out_tank + X_sub_out_tank_add

        df["Xc_tank [W]"] = X_in_tank - X_out_tank

        # ── 11. Exergetic efficiency metrics ───────────────
        df["X_eff_ref [-]"] = df["X_ref_cond [W]"] / df["X_cmp [W]"].replace(
            0, np.nan
        )
        df["X_eff_sys [-]"] = df["X_ref_cond [W]"] / df["X_tot [W]"].replace(
            0, np.nan
        )

        return df
