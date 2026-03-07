"""Tests for COP calculation functions."""

import pytest

from enex_analysis.cop import (
    calc_ASHP_cooling_COP,
    calc_ASHP_heating_COP,
    calc_GSHP_COP,
)


class TestASHPCoolingCOP:
    """Air source heat pump cooling COP tests."""

    def test_standard_conditions(self):
        """COP should return a positive float under normal conditions."""
        cop = calc_ASHP_cooling_COP(
            T_a_int_out=300.0,  # ~27°C
            T_a_ext_in=308.0,  # ~35°C
            Q_r_int=5000.0,
            Q_r_max=10000.0,
            COP_ref=3.5,
        )
        assert isinstance(cop, float)
        assert cop > 0

    def test_plr_clamp_low(self):
        """Very low PLR should be clamped to 0.2."""
        cop_low = calc_ASHP_cooling_COP(300.0, 308.0, 100.0, 10000.0, 3.5)
        cop_at_02 = calc_ASHP_cooling_COP(300.0, 308.0, 2000.0, 10000.0, 3.5)
        assert cop_low == cop_at_02

    def test_plr_clamp_high(self):
        """PLR > 1.0 should be clamped to 1.0."""
        cop_over = calc_ASHP_cooling_COP(300.0, 308.0, 15000.0, 10000.0, 3.5)
        cop_at_1 = calc_ASHP_cooling_COP(300.0, 308.0, 10000.0, 10000.0, 3.5)
        assert cop_over == cop_at_1


class TestASHPHeatingCOP:
    """Air source heat pump heating COP tests."""

    def test_standard_conditions(self):
        cop = calc_ASHP_heating_COP(T0=280.0, Q_r_int=5000.0, Q_r_max=10000.0)
        assert isinstance(cop, float)
        assert cop > 0

    def test_plr_clamp_low(self):
        cop_low = calc_ASHP_heating_COP(280.0, 100.0, 10000.0)
        cop_at_02 = calc_ASHP_heating_COP(280.0, 2000.0, 10000.0)
        assert cop_low == cop_at_02

    def test_higher_outdoor_temp_increases_cop(self):
        """Warmer outdoor temp should generally yield higher COP."""
        cop_cold = calc_ASHP_heating_COP(263.0, 5000.0, 10000.0)  # -10°C
        cop_warm = calc_ASHP_heating_COP(283.0, 5000.0, 10000.0)  # 10°C
        assert cop_warm > cop_cold


class TestGSHPCOP:
    """Ground source heat pump COP tests."""

    def test_standard_conditions(self):
        cop = calc_GSHP_COP(
            Tg=288.0, T_cond=318.0, T_evap=278.0, theta_hat=0.5
        )
        assert isinstance(cop, float)
        assert cop > 0

    def test_invalid_temperatures(self):
        """T_cond <= T_evap should raise ValueError."""
        with pytest.raises(ValueError, match="T_cond must be greater"):
            calc_GSHP_COP(288.0, 278.0, 318.0, 0.5)

    def test_large_theta_hat(self):
        """With large theta_hat, COP approaches standard Carnot."""
        cop = calc_GSHP_COP(
            Tg=288.0, T_cond=318.0, T_evap=278.0, theta_hat=100.0
        )
        assert isinstance(cop, float)
        assert cop > 0
