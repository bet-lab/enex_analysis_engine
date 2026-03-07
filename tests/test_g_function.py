"""Tests for G-function and air property helpers."""

import pytest

from enex_analysis.g_function import (
    air_dynamic_viscosity,
    air_prandtl_number,
    f,
)


class TestGFunctionHelpers:
    """Tests for f() helper used in G-function calculations."""

    def test_f_at_zero(self):
        """f(0) should be 0."""
        assert f(0.0) == pytest.approx(0.0, abs=1e-10)

    def test_f_positive_input(self):
        """f should return a finite float for positive input."""
        result = f(1.0)
        assert isinstance(result, float)
        assert result > 0


class TestAirProperties:
    """Tests for air viscosity and Prandtl number."""

    def test_viscosity_at_room_temp(self):
        """Air viscosity at ~20°C should be ~1.8e-5 Pa·s."""
        mu = air_dynamic_viscosity(293.15)
        assert 1.5e-5 < mu < 2.5e-5

    def test_viscosity_increases_with_temperature(self):
        """Higher temperature → higher viscosity for gases."""
        mu_cold = air_dynamic_viscosity(273.15)
        mu_hot = air_dynamic_viscosity(373.15)
        assert mu_hot > mu_cold

    def test_prandtl_number(self):
        """Air Prandtl number should be ~0.71."""
        pr = air_prandtl_number(300.0)
        assert pr == pytest.approx(0.71)
