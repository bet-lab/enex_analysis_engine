
from enex_analysis.air_source_heat_pump_boiler import AirSourceHeatPumpBoiler
from enex_analysis.electric_boiler import ElectricBoiler


def test_electric_boiler_init():
    boiler = ElectricBoiler(heater_capacity=10000)
    assert boiler.heater_capacity == 10000

def test_heat_pump_boiler_init():
    hp = AirSourceHeatPumpBoiler(hp_capacity=10000)
    assert hp.hp_capacity == 10000
