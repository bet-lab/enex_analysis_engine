"""Attachable subsystems for heat-pump boiler models.

Each subsystem is a self-contained class that bundles its
configuration parameters with the methods that operate on them.
A boiler model receives subsystem instances as optional
constructor arguments, enabling plug-in / plug-out composition.

Subsystem catalogue
-------------------
- ``SolarThermalCollector`` — flat-plate / evacuated-tube STC
- (future) ``PVPanel`` — photovoltaic power generation
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import calc_util as cu
from .constants import c_w, rho_w
from .enex_functions import calc_stc_performance


# ------------------------------------------------------------------
# Default "STC OFF" snapshot — used when STC is absent or inactive
# ------------------------------------------------------------------

STC_OFF: dict = {
    'stc_active': False,
    'stc_result': {},
    'T_stc_w_out_K': np.nan,
    'T_stc_w_final_K': np.nan,
    'Q_stc_w_out': 0.0,
    'Q_stc_w_in': 0.0,
    'E_stc_pump': 0.0,
}


@dataclass
class SolarThermalCollector:
    """Solar Thermal Collector (STC) subsystem.

    Bundles collector geometry, optical and thermal properties,
    pump parameters, and preheat-window settings.  The two
    placement modes (``tank_circuit`` and ``mains_preheat``) are
    handled internally by :meth:`calculate_dynamic`.

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
        STC loop flow rate [m³/s].
    E_stc_pump : float
        STC pump rated power [W].
    stc_placement : str
        ``'tank_circuit'`` or ``'mains_preheat'``.
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

    # Placement
    stc_placement: str = 'tank_circuit'

    # ----------------------------------------------------------
    # Convenience properties
    # ----------------------------------------------------------

    @property
    def is_enabled(self) -> bool:
        """Return ``True`` when the collector area is positive."""
        return self.A_stc > 0

    @property
    def _stc_kwargs(self) -> dict:
        """Keyword dict consumed by ``calc_stc_performance``."""
        return {
            'A_stc_pipe': self.A_stc_pipe,
            'alpha_stc': self.alpha_stc,
            'h_o_stc': self.h_o_stc,
            'h_r_stc': self.h_r_stc,
            'k_ins_stc': self.k_ins_stc,
            'x_air_stc': self.x_air_stc,
            'x_ins_stc': self.x_ins_stc,
            'E_pump': self.E_stc_pump,
        }

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
        return (
            self.preheat_start_hour
            <= hour_of_day
            < self.preheat_end_hour
        )

    # ----------------------------------------------------------
    # Core dynamic calculation
    # ----------------------------------------------------------

    def calculate_dynamic(
        self,
        I_DN: float,
        I_dH: float,
        T_tank_w_K: float,
        T0_K: float,
        preheat_on: bool,
        dV_tank_w_in: float,
        T_tank_w_in_K: float,
    ) -> dict:
        """Compute STC performance for a single timestep.

        Handles both ``tank_circuit`` and ``mains_preheat``
        placement modes, including probing calculations for
        activation decisions.

        Parameters
        ----------
        I_DN : float
            Direct-normal irradiance [W/m²].
        I_dH : float
            Diffuse-horizontal irradiance [W/m²].
        T_tank_w_K : float
            Current tank water temperature [K].
        T0_K : float
            Dead-state temperature [K].
        preheat_on : bool
            Whether the preheat window is active.
        dV_tank_w_in : float
            Current refill flow rate [m³/s].
        T_tank_w_in_K : float
            Mains water inlet temperature [K].

        Returns
        -------
        dict
            Keys: ``stc_active``, ``stc_result``,
            ``T_stc_w_out_K``, ``T_stc_w_final_K``,
            ``Q_stc_w_out``, ``Q_stc_w_in``,
            ``E_stc_pump``.
        """
        stc_active: bool = False
        stc_result: dict = {}
        T_stc_w_out_K: float = np.nan
        T_stc_w_final_K: float = np.nan
        Q_stc_w_out: float = 0.0
        Q_stc_w_in: float = 0.0
        E_stc_pump_val: float = 0.0

        kw: dict = self._stc_kwargs

        if self.stc_placement == 'tank_circuit':
            stc_result = self._calc_tank_circuit(
                I_DN, I_dH, T_tank_w_K, T0_K,
                preheat_on, kw,
            )
            stc_active = stc_result['_active']
            E_stc_pump_val = (
                self.E_stc_pump if stc_active else 0.0
            )
            T_stc_w_out_K = stc_result['T_stc_w_out_K']
            T_stc_w_final_K = stc_result.get(
                'T_stc_w_final_K', T_stc_w_out_K,
            )
            Q_stc_w_out = stc_result.get(
                'Q_stc_w_out', 0.0,
            )
            Q_stc_w_in = stc_result.get(
                'Q_stc_w_in', 0.0,
            )

        elif self.stc_placement == 'mains_preheat':
            stc_result = self._calc_mains_preheat(
                I_DN, I_dH, T_tank_w_in_K, T0_K,
                preheat_on, dV_tank_w_in, kw,
            )
            stc_active = stc_result['_active']
            E_stc_pump_val = (
                self.E_stc_pump if stc_active else 0.0
            )
            T_stc_w_out_K = stc_result['T_stc_w_out_K']
            Q_stc_w_out = stc_result.get(
                'Q_stc_w_out', 0.0,
            )
            Q_stc_w_in = stc_result.get(
                'Q_stc_w_in', 0.0,
            )

        # Strip internal flag before returning
        stc_result.pop('_active', None)

        return {
            'stc_active': stc_active,
            'stc_result': stc_result,
            'T_stc_w_out_K': T_stc_w_out_K,
            'T_stc_w_final_K': T_stc_w_final_K,
            'Q_stc_w_out': Q_stc_w_out,
            'Q_stc_w_in': Q_stc_w_in,
            'E_stc_pump': E_stc_pump_val,
        }

    # ----------------------------------------------------------
    # Result assembly for DataFrame output
    # ----------------------------------------------------------

    def assemble_results(
        self,
        ctx_I_DN: float,
        ctx_I_dH: float,
        ctrl_stc_active: bool,
        ctrl_E_stc_pump: float,
        stc_result: dict,
        T_solved_K: float,
        T_stc_w_out_K: float,
    ) -> dict:
        """Build STC-related entries for the step result dict.

        Parameters
        ----------
        ctx_I_DN, ctx_I_dH : float
            Irradiance values for this step.
        ctrl_stc_active : bool
            Whether STC was active.
        ctrl_E_stc_pump : float
            STC pump power [W].
        stc_result : dict
            Raw STC performance dict.
        T_solved_K : float
            Solved tank temperature [K].
        T_stc_w_out_K : float
            STC water outlet temperature [K].

        Returns
        -------
        dict
            Result entries for STC columns.
        """
        T_stc_w_final_K: float = stc_result.get(
            'T_stc_w_final_K', T_stc_w_out_K,
        )
        return {
            'stc_active [-]': ctrl_stc_active,
            'I_DN_stc [W/m2]': ctx_I_DN,
            'I_dH_stc [W/m2]': ctx_I_dH,
            'I_sol_stc [W/m2]': stc_result.get(
                'I_sol_stc', np.nan,
            ),
            'Q_sol_stc [W]': stc_result.get(
                'Q_sol_stc', np.nan,
            ),
            'Q_stc_w_out [W]': stc_result.get(
                'Q_stc_w_out', 0.0,
            ),
            'Q_stc_w_in [W]': stc_result.get(
                'Q_stc_w_in', 0.0,
            ),
            'Q_l_stc [W]': stc_result.get(
                'Q_l_stc', np.nan,
            ),
            'T_stc_w_out [°C]': (
                cu.K2C(T_stc_w_out_K)
                if not np.isnan(T_stc_w_out_K)
                else np.nan
            ),
            'T_stc_w_in [°C]': cu.K2C(T_solved_K),
            'T_stc [°C]': cu.K2C(
                stc_result.get('T_stc_K', np.nan),
            ),
            'E_stc_pump [W]': ctrl_E_stc_pump,
        }

    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------

    def _calc_tank_circuit(
        self,
        I_DN: float,
        I_dH: float,
        T_tank_w_K: float,
        T0_K: float,
        preheat_on: bool,
        kw: dict,
    ) -> dict:
        """Tank-circuit STC with probing activation."""
        probe: dict = calc_stc_performance(
            I_DN_stc=I_DN,
            I_dH_stc=I_dH,
            T_stc_w_in_K=T_tank_w_K,
            T0_K=T0_K,
            dV_stc=self.dV_stc_w,
            is_active=True,
            **kw,
        )

        active: bool = (
            preheat_on
            and probe['T_stc_w_out_K'] > T_tank_w_K
        )

        if active:
            result: dict = probe
        else:
            result = calc_stc_performance(
                I_DN_stc=I_DN,
                I_dH_stc=I_dH,
                T_stc_w_in_K=T_tank_w_K,
                T0_K=T0_K,
                dV_stc=self.dV_stc_w,
                is_active=False,
                **kw,
            )

        result['_active'] = active
        return result

    def _calc_mains_preheat(
        self,
        I_DN: float,
        I_dH: float,
        T_tank_w_in_K: float,
        T0_K: float,
        preheat_on: bool,
        dV_tank_w_in: float,
        kw: dict,
    ) -> dict:
        """Mains-preheat STC with probing activation."""
        if preheat_on and dV_tank_w_in > 0:
            probe: dict = calc_stc_performance(
                I_DN_stc=I_DN,
                I_dH_stc=I_dH,
                T_stc_w_in_K=T_tank_w_in_K,
                T0_K=T0_K,
                dV_stc=dV_tank_w_in,
                is_active=True,
                **kw,
            )
            if probe['T_stc_w_out_K'] > T_tank_w_in_K:
                probe['_active'] = True
                return probe
            else:
                result: dict = calc_stc_performance(
                    I_DN_stc=I_DN,
                    I_dH_stc=I_dH,
                    T_stc_w_in_K=T_tank_w_in_K,
                    T0_K=T0_K,
                    dV_stc=dV_tank_w_in,
                    is_active=False,
                    **kw,
                )
                result['_active'] = False
                return result
        else:
            result = calc_stc_performance(
                I_DN_stc=I_DN,
                I_dH_stc=I_dH,
                T_stc_w_in_K=T_tank_w_in_K,
                T0_K=T0_K,
                dV_stc=1.0,
                is_active=False,
                **kw,
            )
            result['_active'] = False
            return result
