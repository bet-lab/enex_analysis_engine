"""Attachable subsystems for heat-pump boiler models.

Each subsystem is a self-contained **pure physics engine**: given
physical input state it returns a standardised output dict.
All simulation orchestration (activation logic, result assembly,
exergy calculation) is the responsibility of the scenario class
that uses the subsystem.

Subsystem catalogue
-------------------
- ``SolarThermalCollector`` — flat-plate / evacuated-tube STC
  physics engine (``calc_performance``, ``is_preheat_on``)
- ``PhotovoltaicSystem`` — PV + ESS + inverter chain
- ``UVLamp`` — UV disinfection lamp
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from . import calc_util as cu
from .constants import c_w, k_a, k_D, k_d, rho_w
from .enex_functions import calc_energy_flow

if TYPE_CHECKING:
    import pandas as pd

    from .dynamic_context import ControlState, StepContext, SubsystemExergy


# ------------------------------------------------------------------
# Default subsystem step() return — STC absent/inactive
# ------------------------------------------------------------------

STC_OFF_STEP: dict = {
    "stc_active": False,
    "stc_result": {},
    "T_stc_w_out_K": np.nan,
    "T_stc_pump_w_out_K": np.nan,
    "Q_stc_w_out": 0.0,
    "Q_stc_pump_w_out": 0.0,
    "Q_stc_w_in": 0.0,
    "E_stc_pump": 0.0,
    "Q_contribution": 0.0,
    "E_subsystem": 0.0,
    "T_tank_w_in_override_K": None,
}

# Backward-compatible alias
STC_OFF: dict = STC_OFF_STEP


@dataclass
class SolarThermalCollector:
    """Solar Thermal Collector (STC) — pure physics engine.

    Bundles collector geometry, optical and thermal properties,
    and pump parameters.  This class is a **stateless physics
    calculator**: given physical inputs it returns a performance
    dict.  All simulation orchestration (activation logic,
    result assembly, exergy calculation) is the responsibility
    of the scenario class that uses this engine.

    The public API consists of:

    - :meth:`calc_performance` — single-timestep thermal performance
    - :meth:`is_preheat_on` — schedule check

    Parameters
    ----------
    A_stc : float
        Collector gross area [m²].
    stc_tilt : float
        Tilt from horizontal [°].
    stc_azimuth : float
        Azimuth angle (180 = south) [°].
    A_stc_pipe : float
        Pipe surface area [m²].
    alpha_stc : float
        Absorptivity [–].
    h_o_stc : float
        External convective coefficient [W/(m²·K)].
    h_r_stc : float
        Radiative coefficient [W/(m²·K)].
    k_ins_stc : float
        Insulation conductivity [W/(m·K)].
    x_air_stc : float
        Air gap thickness [m].
    x_ins_stc : float
        Insulation thickness [m].
    preheat_start_hour : float
        Preheat window start hour.
    preheat_end_hour : float
        Preheat window end hour.
    dV_stc_w : float
        Default STC loop flow rate [m³/s].
    E_stc_pump : float
        STC pump rated power [W].
    """

    # Collector
    A_stc: float = 2.0
    stc_tilt: float = 35.0
    stc_azimuth: float = 180.0
    A_stc_pipe: float = 2.0
    alpha_stc: float = 0.95
    h_o_stc: float = 15.0
    h_r_stc: float = 2.0
    k_ins_stc: float = 0.03
    x_air_stc: float = 0.01
    x_ins_stc: float = 0.05

    # Pump / schedule
    preheat_start_hour: float = 6.0
    preheat_end_hour: float = 18.0
    dV_stc_w: float = 0.001
    E_stc_pump: float = 50.0

    # ----------------------------------------------------------
    # Physics helpers
    # ----------------------------------------------------------

    def calc_overall_heat_transfer_coeff(self) -> float:
        """Compute overall U-value from parallel resistances.

        The collector has two heat-loss paths in parallel:

        - **Path 1** (top): radiation gap ‖ air gap → external conv
        - **Path 2** (bottom): insulation → external conv

        Returns
        -------
        float
            Overall heat-loss coefficient [W/(m²·K)].
        """
        R_air = self.x_air_stc / k_a
        R_ins = self.x_ins_stc / self.k_ins_stc
        R_o = 1.0 / self.h_o_stc
        R_r = 1.0 / self.h_r_stc

        # Path 1: radiation ‖ air gap (parallel), then + external
        R1 = (R_r * R_air) / (R_r + R_air) + R_o
        # Path 2: insulation (series) + external
        R2 = R_ins + R_o

        # Two paths in parallel
        return 1.0 / R1 + 1.0 / R2

    def _calc_solar_absorption(
        self,
        I_DN_stc: float,
        I_dH_stc: float,
    ) -> tuple[float, float]:
        """Compute total irradiance and absorbed heat.

        Parameters
        ----------
        I_DN_stc : float
            Direct-normal irradiance [W/m²].
        I_dH_stc : float
            Diffuse-horizontal irradiance [W/m²].

        Returns
        -------
        tuple[float, float]
            ``(I_sol_stc, Q_sol_stc)`` — total irradiance and
            absorbed solar heat rate [W].
        """
        I_sol = I_DN_stc + I_dH_stc
        Q_sol = I_sol * self.A_stc_pipe * self.alpha_stc
        return I_sol, Q_sol

    def _calc_outlet_temperature(
        self,
        U_stc: float,
        G_stc: float,
        Q_sol_stc: float,
        Q_stc_w_in: float,
        T_stc_w_in_K: float,
        T0_K: float,
    ) -> tuple[float, float, float]:
        """Solve for collector outlet and mean plate temps.

        Parameters
        ----------
        U_stc : float
            Overall U-value [W/(m²·K)].
        G_stc : float
            Heat capacity flow rate ``c_w · ρ_w · V̇`` [W/K].
        Q_sol_stc : float
            Absorbed solar heat [W].
        Q_stc_w_in : float
            Inlet energy flow [W].
        T_stc_w_in_K : float
            Inlet water temperature [K].
        T0_K : float
            Dead-state temperature [K].

        Returns
        -------
        tuple[float, float, float]
            ``(T_stc_w_out_K, T_stc_K, ksi_stc)`` — outlet temp,
            mean plate temp, and dimensionless parameter.
        """
        A_U = max(1e-6, self.A_stc_pipe * U_stc)
        ksi = np.exp(-A_U / G_stc)

        T_out_K = ksi * T_stc_w_in_K + (1 - ksi) * T0_K + (1 - ksi) * Q_sol_stc / A_U
        T_stc_K = T0_K + (Q_sol_stc - G_stc * (T_out_K - T_stc_w_in_K)) / A_U
        
        return T_out_K, T_stc_K, ksi

    # ----------------------------------------------------------
    # Main performance calculation
    # ----------------------------------------------------------

    def calc_performance(
        self,
        I_DN_stc: float,
        I_dH_stc: float,
        T_stc_w_in_K: float,
        T0_K: float,
        dV_stc: float | None = None,
        is_active: bool = True,
    ) -> dict:
        """Compute STC thermal performance for one timestep.

        Parameters
        ----------
        I_DN_stc : float
            Direct-normal irradiance [W/m²].
        I_dH_stc : float
            Diffuse-horizontal irradiance [W/m²].
        T_stc_w_in_K : float
            Inlet water temperature [K].
        T0_K : float
            Dead-state temperature [K].
        dV_stc : float | None
            Override flow rate [m³/s]; defaults to
            ``self.dV_stc_w``.
        is_active : bool
            If ``False``, return NaN-filled dict.

        Returns
        -------
        dict
            Performance results including:
            ``I_sol_stc``, ``Q_sol_stc``, ``Q_stc_w_in``,
            ``Q_stc_w_out``, ``Q_stc_pump_w_out``,
            ``ksi_stc``, ``T_stc_w_out_K``,
            ``T_stc_pump_w_out_K``, ``T_stc_w_in_K``,
            ``T_stc_K``, ``Q_l_stc``.
        """
        if dV_stc is None:
            dV_stc = self.dV_stc_w

        if not is_active:
            return {
                "I_sol_stc": np.nan,
                "Q_sol_stc": np.nan,
                "S_sol_stc": np.nan,
                "X_sol_stc": np.nan,
                "Q_stc_w_in": np.nan,
                "Q_stc_w_out": np.nan,
                "Q_stc_pump_w_out": np.nan,
                "ksi_stc": np.nan,
                "T_stc_pump_w_out_K": T_stc_w_in_K,
                "T_stc_w_out_K": T_stc_w_in_K,
                "T_stc_w_in_K": T_stc_w_in_K,
                "T_stc_K": np.nan,
                "Q_l_stc": np.nan,
            }

        U_stc = self.calc_overall_heat_transfer_coeff()
        I_sol_stc, Q_sol_stc = self._calc_solar_absorption(
            I_DN_stc,
            I_dH_stc,
        )
        S_sol_stc = (
            k_D * I_DN_stc**0.9 + k_d * I_dH_stc**0.9
        ) * self.A_stc_pipe
        X_sol_stc = Q_sol_stc - S_sol_stc * T0_K

        G_stc = c_w * rho_w * dV_stc
        Q_stc_w_in = calc_energy_flow(
            G_stc,
            T_stc_w_in_K,
            T0_K,
        )

        T_out_K, T_stc_K, ksi = self._calc_outlet_temperature(
            U_stc,
            G_stc,
            Q_sol_stc,
            Q_stc_w_in,
            T_stc_w_in_K,
            T0_K,
        )

        # Pump heat addition
        T_stc_pump_w_out_K = T_out_K + self.E_stc_pump / G_stc

        # Heat transport rates
        Q_stc_w_out = calc_energy_flow(
            G_stc,
            T_out_K,
            T0_K,
        )
        Q_stc_pump_w_out = calc_energy_flow(
            G_stc,
            T_stc_pump_w_out_K,
            T0_K,
        )

        # Collector heat loss
        Q_l_stc = self.A_stc_pipe * U_stc * (T_stc_K - T0_K)

        return {
            "I_sol_stc": I_sol_stc,
            "Q_sol_stc": Q_sol_stc,
            "S_sol_stc": S_sol_stc,
            "X_sol_stc": X_sol_stc,
            "Q_stc_w_in": Q_stc_w_in,
            "Q_stc_w_out": Q_stc_w_out,
            "Q_stc_pump_w_out": Q_stc_pump_w_out,
            "ksi_stc": ksi,
            "T_stc_pump_w_out_K": T_stc_pump_w_out_K,
            "T_stc_w_out_K": T_out_K,
            "T_stc_w_in_K": T_stc_w_in_K,
            "T_stc_K": T_stc_K,
            "Q_l_stc": Q_l_stc,
        }

    # ----------------------------------------------------------
    # Schedule helper
    # ----------------------------------------------------------

    def is_preheat_on(self, hour_of_day: float) -> bool:
        """Check whether *hour_of_day* falls in the window.

        Parameters
        ----------
        hour_of_day : float
            Hour within the day (0–24).

        Returns
        -------
        bool
        """
        return self.preheat_start_hour <= hour_of_day < self.preheat_end_hour


# ------------------------------------------------------------------
# Photovoltaic System
# ------------------------------------------------------------------


@dataclass
class PhotovoltaicSystem:
    """Photovoltaic System (PV + Controller + ESS + Inverter).

    Computes PV energy generation, stores it in a Battery (ESS) with
    dynamic state-of-charge (SOC) tracking, and calculates entropy/exergy
    balances for each stage based on the ``nomenclature.md`` conventions.

    Implements the ``Subsystem`` protocol.
    """

    # Panel physics
    A_pv: float = 5.0
    alp_pv: float = 0.9
    pv_tilt: float = 35.0
    pv_azimuth: float = 180.0
    h_o: float = 15.0

    # Efficiencies
    eta_pv: float = 0.20
    eta_ctrl: float = 0.95
    eta_ess_chg: float = 0.90
    eta_ess_dis: float = 0.90
    eta_inv: float = 0.95

    # ESS Capacity and Init
    C_ess_max: float = 3.6e6  # e.g., 1 kWh = 3.6e6 Joules
    SOC_init: float = 0.0

    # Temperatures (K)
    T_ctrl_K: float = 308.15
    T_ess_K: float = 313.15
    T_inv_K: float = 313.15

    # Internal dynamic state
    SOC_ess: float = field(init=False)

    def __post_init__(self):
        self.SOC_ess = self.SOC_init

    def step(
        self,
        ctx: StepContext,
        ctrl: ControlState,
        dt: float,
        T_tank_w_in_K: float,
    ) -> dict:
        """Compute PV/ESS state for one simulation timestep."""
        # Note: ctx.I_DN and ctx.I_dH are assumed to be Plane-of-Array (POA)
        # or already corrected for pv_tilt/pv_azimuth by the external context.
        I_sol = ctx.I_DN + ctx.I_dH

        # Ensure small non-zero denominator for logs
        T0_K = max(1e-3, ctx.T0_K)

        # Stage 1: PV Cell
        # T_pv = T0 + I_sol*(alpha - eta)/(2*h_o)
        T_pv_K = T0_K + (I_sol * (self.alp_pv - self.eta_pv)) / (2 * self.h_o)
        E_pv_out = self.A_pv * self.eta_pv * I_sol
        Q_l_pv = 2 * self.A_pv * self.h_o * (T_pv_K - T0_K)

        # Entropy & Exergy for PV
        from .constants import k_D, k_d

        s_DN = k_D * (ctx.I_DN**0.9)
        s_dH = k_d * (ctx.I_dH**0.9)
        s_sol = s_DN + s_dH

        S_sol = self.A_pv * self.alp_pv * s_sol
        S_pv_out = 0.0  # (1/inf) * E_pv_out
        S_l_pv = (1.0 / T_pv_K) * Q_l_pv if T_pv_K > 0 else 0.0
        S_g_pv = S_pv_out + S_l_pv - S_sol

        X_sol = self.A_pv * self.alp_pv * (I_sol - s_sol * T0_K)
        X_pv_out = E_pv_out
        X_l_pv = (1.0 - T0_K / T_pv_K) * Q_l_pv if T_pv_K > 0 else 0.0
        X_c_pv = S_g_pv * T0_K

        # Stage 2: Charge Controller
        E_ctrl_out = self.eta_ctrl * E_pv_out
        Q_l_ctrl = (1.0 - self.eta_ctrl) * E_pv_out
        S_ctrl_out = 0.0
        S_l_ctrl = (1.0 / self.T_ctrl_K) * Q_l_ctrl
        S_g_ctrl = S_ctrl_out + S_l_ctrl - S_pv_out

        X_ctrl_out = E_ctrl_out
        X_l_ctrl = Q_l_ctrl - S_l_ctrl * T0_K
        X_c_ctrl = S_g_ctrl * T0_K

        # Stage 3: ESS (Battery)
        # HP load is provided via ctrl.result
        E_hp_load = ctrl.result.get("E_tot [W]", 0.0)
        # Required ESS discharge to meet HP load
        E_ess_dis_target = E_hp_load / self.eta_inv if self.eta_inv > 0 else 0.0
        
        # Charge ESS with available PV power
        E_ess_chg = E_ctrl_out
        
        # Enforce SOC constraints
        energy_available_to_store = (1.0 - self.SOC_ess) * self.C_ess_max
        max_chg_power = (
            energy_available_to_store / (self.eta_ess_chg * dt)
            if dt > 0
            else 0.0
        )
        E_ess_chg_actual = min(E_ess_chg, max_chg_power)
        
        energy_available_to_discharge = self.SOC_ess * self.C_ess_max
        max_dis_power = (
            energy_available_to_discharge * self.eta_ess_dis / dt
            if dt > 0
            else 0.0
        )
        E_ess_dis = min(E_ess_dis_target, max_dis_power)

        # Excess generation goes unused or dumped for now

        # Update SOC via Euler integration
        self.SOC_ess += (
            (
                E_ess_chg_actual * self.eta_ess_chg
                - E_ess_dis / self.eta_ess_dis
            )
            * dt
            / self.C_ess_max
        )
        self.SOC_ess = max(0.0, min(1.0, self.SOC_ess))

        # We treat the battery output Exergy/Energy based on what we discharge
        E_ess_out = E_ess_dis  # The actual power going to inverter
        Q_l_ess = (1.0 - self.eta_ess_chg) * E_ess_chg_actual + (
            1.0 / self.eta_ess_dis - 1.0
        ) * E_ess_dis

        S_ess_out = 0.0
        S_l_ess = (1.0 / self.T_ess_K) * Q_l_ess
        # Battery entropy generation includes tracking storage over time,
        # but for instant bounds we follow the PV_to_Converter logic:
        S_g_ess = S_ess_out + S_l_ess - S_ctrl_out

        X_ess_out = E_ess_out
        X_l_ess = Q_l_ess - S_l_ess * T0_K
        X_c_ess = S_g_ess * T0_K

        # Stage 4: Inverter
        E_inv_out = self.eta_inv * E_ess_out
        Q_l_inv = (1.0 - self.eta_inv) * E_ess_out
        S_inv_out = 0.0
        S_l_inv = (1.0 / self.T_inv_K) * Q_l_inv
        S_g_inv = S_inv_out + S_l_inv - S_ess_out

        X_inv_out = E_inv_out
        X_l_inv = Q_l_inv - S_l_inv * T0_K
        X_c_inv = S_g_inv * T0_K

        raw_result = {
            "I_sol_pv": I_sol,
            "T_pv_K": T_pv_K,
            "E_pv_out": E_pv_out,
            "Q_l_pv": Q_l_pv,
            "X_sol": X_sol,
            "X_pv_out": X_pv_out,
            "X_l_pv": X_l_pv,
            "X_c_pv": X_c_pv,
            "E_ctrl_out": E_ctrl_out,
            "Q_l_ctrl": Q_l_ctrl,
            "X_ctrl_out": X_ctrl_out,
            "X_l_ctrl": X_l_ctrl,
            "X_c_ctrl": X_c_ctrl,
            "E_ess_chg": E_ess_chg_actual,
            "E_ess_dis": E_ess_dis,
            "E_ess_out": E_ess_out,
            "SOC_ess": self.SOC_ess,
            "Q_l_ess": Q_l_ess,
            "X_ess_out": X_ess_out,
            "X_l_ess": X_l_ess,
            "X_c_ess": X_c_ess,
            "E_inv_out": E_inv_out,
            "Q_l_inv": Q_l_inv,
            "X_inv_out": X_inv_out,
            "X_l_inv": X_l_inv,
            "X_c_inv": X_c_inv,
        }

        return {
            "Q_contribution": 0.0,
            "E_subsystem": 0.0,  # Negative if it provides power to HP but currently disconnected
            "T_tank_w_in_override_K": None,
            "pv_result": raw_result,
        }

    def assemble_results(
        self,
        ctx: StepContext,
        ctrl: ControlState,
        step_state: dict,
        T_solved_K: float,
    ) -> dict:
        """Assemble dictionary for dataframe insertion."""
        pvr = step_state.get("pv_result", {})
        return {
            "I_sol_pv [W/m2]": pvr.get("I_sol_pv", 0.0),
            "T_pv [°C]": cu.K2C(pvr.get("T_pv_K", ctx.T0_K)),
            "E_pv_out [W]": pvr.get("E_pv_out", 0.0),
            "E_ctrl_out [W]": pvr.get("E_ctrl_out", 0.0),
            "E_ess_chg [W]": pvr.get("E_ess_chg", 0.0),
            "E_ess_dis [W]": pvr.get("E_ess_dis", 0.0),
            "SOC_ess [-]": pvr.get("SOC_ess", 0.0),
            "E_inv_out [W]": pvr.get("E_inv_out", 0.0),
            "X_pv_out [W]": pvr.get("X_pv_out", 0.0),
            "X_ctrl_out [W]": pvr.get("X_ctrl_out", 0.0),
            "X_ess_out [W]": pvr.get("X_ess_out", 0.0),
            "X_inv_out [W]": pvr.get("X_inv_out", 0.0),
            "X_c_pv [W]": pvr.get("X_c_pv", 0.0),
            "X_c_ctrl [W]": pvr.get("X_c_ctrl", 0.0),
            "X_c_ess [W]": pvr.get("X_c_ess", 0.0),
            "X_c_inv [W]": pvr.get("X_c_inv", 0.0),
            "Q_l_pv [W]": pvr.get("Q_l_pv", 0.0),
        }

    def calc_exergy(
        self,
        df: pd.DataFrame,
        T0_K: pd.Series,
    ) -> SubsystemExergy | None:
        """PV exergy is computed inline in ``step()``.

        No additional post-processing or tank-boundary
        contributions are needed.  Returns ``None``.
        """
        return None


# ------------------------------------------------------------------
# UV Disinfection Lamp
# ------------------------------------------------------------------


@dataclass
class UVLamp:
    """UV disinfection lamp subsystem.

    The lamp switches on periodically (``num_switching``
    times per ``period_sec``, each for ``exposure_sec``).
    All electrical input is converted to heat inside the
    tank (``Q_contribution = E_uv``).

    Parameters
    ----------
    lamp_watts : float
        Rated lamp power [W].
    exposure_sec : float
        Duration of each on-cycle [s].
    num_switching : int
        Number of on-cycles per period.
    period_sec : float
        Switching period [s] (default 3 h = 10 800 s).
    """

    lamp_watts: float = 0.0
    exposure_sec: float = 0.0
    num_switching: int = 1
    period_sec: float = 3 * cu.h2s

    def step(
        self,
        ctx: StepContext,
        ctrl: ControlState,
        dt: float,
        T_tank_w_in_K: float,
    ) -> dict:
        """Compute UV lamp state for one timestep."""
        from .enex_functions import calc_uv_lamp_power

        E_uv: float = calc_uv_lamp_power(
            ctx.current_time_s,
            self.period_sec,
            self.num_switching,
            self.exposure_sec,
            self.lamp_watts,
        )
        return {
            "Q_contribution": E_uv,
            "E_subsystem": E_uv,
            "T_tank_w_in_override_K": None,
            "E_uv": E_uv,
        }

    def assemble_results(
        self,
        ctx: StepContext,
        ctrl: ControlState,
        step_state: dict,
        T_solved_K: float,
    ) -> dict:
        """Report UV power for DataFrame output."""
        E_uv: float = step_state.get("E_uv", 0.0)
        if E_uv > 0 or self.lamp_watts > 0:
            return {"E_uv [W]": E_uv}
        return {}

    def calc_exergy(
        self,
        df: pd.DataFrame,
        T0_K: pd.Series,
    ) -> SubsystemExergy | None:
        """UV exergy = electricity (handled by E→X conversion).

        No additional post-processing needed.
        Returns ``None``.
        """
        return None
