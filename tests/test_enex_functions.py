
from enex_analysis import enex_functions as ef
from enex_analysis.constants import c_w, rho_w


def test_constants():
    assert c_w == 4186
    assert rho_w == 1000

def test_friction_factor():
    f = ef.darcy_friction_factor(100.0, e=0.001, d=0.1)
    assert f == 0.64  # 64 / 100

def test_heating_cop():
    cop = ef.calc_ASHP_heating_COP(20.0, 45.0, 10000.0)
    assert isinstance(cop, float)
