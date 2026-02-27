# Steady-State Analysis Engine

> Module: `enex_analysis.enex_engine`

## Overview

Collection of `@dataclass`-based steady-state energy system models. Each class
computes energy, entropy, and exergy balances for a single operating point via
the `system_update()` method. These are simpler, algebraic models compared to
the dynamic simulation classes (`AirSourceHeatPumpBoiler`, `GasBoiler`, etc.).

## Available Classes

### Domestic Hot Water Systems

| Class | Description |
|---|---|
| `ElectricBoiler` | Electric resistance heater with tank |
| `GasBoiler` | Gas-fired boiler (steady-state version in engine) |
| `HeatPumpBoiler` | Air-source HP boiler (simplified, fan-curve based) |
| `SolarAssistedGasBoiler` | Gas boiler with solar thermal preheat |
| `GroundSourceHeatPumpBoiler` | Ground-source HP boiler (steady-state in engine) |

### Heat Pump Systems (Space Conditioning)

| Class | Description |
|---|---|
| `AirSourceHeatPump_cooling` | ASHP in cooling mode |
| `AirSourceHeatPump_heating` | ASHP in heating mode |
| `GroundSourceHeatPump_cooling` | GSHP in cooling mode |
| `GroundSourceHeatPump_heating` | GSHP in heating mode |

### Other

| Class | Description |
|---|---|
| `ElectricHeater` | Radiant / convective electric heater with panel |
| `Fan` | Fan performance model (curve-fitted) |
| `Pump` | Pump performance model (curve-fitted) |

## Common Interface

All system classes share the same pattern:

```python
@dataclass
class SystemModel:
    # Input fields as dataclass attributes
    field1: float = default_value
    field2: float = default_value

    def __post_init__(self):
        # Derived quantities computed from inputs

    def system_update(self):
        # Compute energy, entropy, exergy balances
```

## Usage

### Electric Boiler Example

```python
from enex_analysis.enex_engine import ElectricBoiler

eb = ElectricBoiler()
eb.T0_C = 5.0       # Outdoor temperature [°C]
eb.T_tank_w_C = 55.0 # Tank water temperature [°C]
eb.dV_use = 0.0001   # Service flow rate [m³/s]
eb.system_update()

# Access results as instance attributes
print(f"Heater power: {eb.E_heater:.1f} W")
print(f"Tank loss:    {eb.Q_l_tank:.1f} W")
```

### Heat Pump Boiler Example

```python
from enex_analysis.enex_engine import HeatPumpBoiler

hpb = HeatPumpBoiler()
hpb.T0_C = 5.0
hpb.T_tank_w_C = 55.0
hpb.dV_use = 0.0001
hpb.system_update()

print(f"COP: {hpb.COP:.2f}")
print(f"Compressor power: {hpb.E_cmp:.1f} W")
```

### Fan / Pump Performance

```python
from enex_analysis.enex_engine import Fan

fan = Fan()
fan.show_graph()   # Plot flow-vs-pressure and flow-vs-efficiency curves
power = fan.get_power(fan.fan1, dV_fan=1.5)
```

## API Reference

All classes expose:

| Method | Description |
|---|---|
| `system_update()` | Compute all energy / entropy / exergy balances |
| `show_graph()` | *(Fan/Pump only)* Plot performance curves |
