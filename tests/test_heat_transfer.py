"""Tests for heat_transfer module functions."""

import pytest

from enex_analysis.heat_transfer import (
    calc_h_vertical_plate,
    calc_LMTD_counter_flow,
    calc_LMTD_parallel_flow,
    calc_simple_tank_UA,
    darcy_friction_factor,
)


class TestDarcyFrictionFactor:
    """Darcy friction factor tests."""

    def test_laminar_flow(self):
        """Re < 2300: Hagen-Poiseuille f = 64/Re."""
        f = darcy_friction_factor(100.0, e=0.001, d=0.1)
        assert f == pytest.approx(0.64)

    def test_laminar_returns_64_over_re(self):
        """Exact laminar formula check."""
        re = 500.0
        f = darcy_friction_factor(re, e=0.0, d=0.05)
        assert f == pytest.approx(64.0 / re)

    def test_turbulent_flow_positive(self):
        """Re > 2300 should return a positive friction factor."""
        f = darcy_friction_factor(10000.0, e=0.001, d=0.1)
        assert f > 0


class TestLMTD:
    """Log-mean temperature difference tests."""

    def test_counter_flow_basic(self):
        """Basic counter-flow LMTD should be positive."""
        lmtd = calc_LMTD_counter_flow(
            Th_in=80.0, Th_out=60.0, Tc_in=20.0, Tc_out=50.0
        )
        assert lmtd > 0

    def test_parallel_flow_basic(self):
        """Basic parallel-flow LMTD should be positive."""
        lmtd = calc_LMTD_parallel_flow(
            Th_in=80.0, Th_out=60.0, Tc_in=20.0, Tc_out=40.0
        )
        assert lmtd > 0

    def test_counter_flow_equal_delta_t(self):
        """When ΔT1 == ΔT2, LMTD should equal that value (special case)."""
        lmtd = calc_LMTD_counter_flow(
            Th_in=80.0, Th_out=60.0, Tc_in=40.0, Tc_out=60.0
        )
        assert lmtd == pytest.approx(20.0, abs=0.1)


class TestSimpleTankUA:
    """calc_simple_tank_UA tests."""

    def test_positive_result(self):
        """UA should be positive for typical tank dimensions."""
        ua = calc_simple_tank_UA(
            r0=0.25,
            H=1.5,
            x_ins=0.05,
            k_ins=0.04,
        )
        assert ua > 0

    def test_thicker_insulation_reduces_ua(self):
        """More insulation → lower UA (less heat loss)."""
        ua_thin = calc_simple_tank_UA(
            r0=0.25,
            H=1.5,
            x_ins=0.03,
            k_ins=0.04,
        )
        ua_thick = calc_simple_tank_UA(
            r0=0.25,
            H=1.5,
            x_ins=0.10,
            k_ins=0.04,
        )
        assert ua_thick < ua_thin


class TestVerticalPlateHTC:
    """calc_h_vertical_plate tests."""

    def test_positive_result(self):
        """Should return positive heat transfer coefficient."""
        h = calc_h_vertical_plate(T_s=350.0, T_inf=300.0, L=1.0)
        assert h > 0

    def test_higher_delta_t_increases_h(self):
        """Larger temperature difference → higher natural convection h."""
        h_small = calc_h_vertical_plate(T_s=310.0, T_inf=300.0, L=1.0)
        h_large = calc_h_vertical_plate(T_s=370.0, T_inf=300.0, L=1.0)
        assert h_large > h_small
