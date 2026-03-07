"""Tests for TDMA solver and advection term utilities."""

import numpy as np

from enex_analysis.tdma import TDMA, _add_loop_advection_terms


class TestTDMA:
    """Tri-diagonal matrix algorithm solver tests."""

    def test_identity_system(self):
        """Identity matrix: solution == RHS."""
        n = 4
        a = np.zeros(n)
        b = np.ones(n)
        c = np.zeros(n)
        d = np.array([1.0, 2.0, 3.0, 4.0])
        result = TDMA(a, b, c, d)
        np.testing.assert_allclose(result, d)

    def test_simple_tridiagonal(self):
        """Known 3x3 tridiagonal system."""
        a = np.array([0.0, -1.0, -1.0])
        b = np.array([2.0, 2.0, 2.0])
        c = np.array([-1.0, -1.0, 0.0])
        d = np.array([1.0, 0.0, 1.0])
        result = TDMA(a, b, c, d)
        # Verify A @ x = d
        A_mat = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
        np.testing.assert_allclose(A_mat @ result, d, atol=1e-10)

    def test_returns_1d_array(self):
        """Result should be a flat 1D array."""
        a = np.zeros(3)
        b = np.ones(3)
        c = np.zeros(3)
        d = np.array([1.0, 2.0, 3.0])
        result = TDMA(a, b, c, d)
        assert result.ndim == 1
        assert len(result) == 3


class TestAddLoopAdvectionTerms:
    """Tests for _add_loop_advection_terms in-place modification."""

    def test_upward_flow_modifies_coefficients(self):
        """Upward flow (in_idx > out_idx) should modify b and c arrays."""
        n = 5
        a = np.zeros(n)
        b = np.ones(n)
        c = np.zeros(n)
        d = np.zeros(n)
        G_loop = 100.0
        T_loop_in = 300.0

        _add_loop_advection_terms(a, b, c, d, in_idx=4, out_idx=1, G_loop=G_loop, T_loop_in=T_loop_in)

        # Inlet node b[4] should be increased
        assert b[4] == 1.0 + G_loop
        # Inlet node d[4] should have G_loop * T_loop_in
        assert d[4] == G_loop * T_loop_in

    def test_downward_flow_modifies_coefficients(self):
        """Downward flow (in_idx < out_idx) should modify a and b arrays."""
        n = 5
        a = np.zeros(n)
        b = np.ones(n)
        c = np.zeros(n)
        d = np.zeros(n)
        G_loop = 50.0
        T_loop_in = 350.0

        _add_loop_advection_terms(a, b, c, d, in_idx=1, out_idx=4, G_loop=G_loop, T_loop_in=T_loop_in)

        assert b[1] == 1.0 + G_loop
        assert d[1] == G_loop * T_loop_in

    def test_invalid_negative_flow_does_nothing(self, capsys):
        """Negative G_loop should print warning and not modify arrays."""
        n = 3
        a = np.zeros(n)
        b = np.ones(n)
        c = np.zeros(n)
        d = np.zeros(n)

        _add_loop_advection_terms(a, b, c, d, in_idx=0, out_idx=2, G_loop=-10.0, T_loop_in=300.0)

        np.testing.assert_array_equal(b, np.ones(n))
        captured = capsys.readouterr()
        assert "Warning" in captured.out
