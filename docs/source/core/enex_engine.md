# Steady-State Analysis Engine

> Module: `enex_analysis.enex_engine`

## Overview

The `enex_analysis.enex_engine` module originally housed all steady-state and dynamic system models. After a significant refactoring (Phase 3), the major dynamic simulation classes (e.g., `AirSourceHeatPumpBoiler`, `GasBoiler`, `GroundSourceHeatPumpBoiler`, `ElectricBoiler`) have been extracted into their own dedicated Python files within the package to improve maintainability.

Currently, `enex_engine.py` retains:
1. **Steady-state heat pump cycle models** (cooling and heating mode classes for ASHP and GSHP).
2. **Backward compatibility aliases** (e.g., `HeatPumpBoiler = AirSourceHeatPumpBoiler`).

For the physical and mathematical details of the dynamic boiler systems, see the respective documentation in `docs/guides/02_systems`.

## Available Classes in `enex_engine.py`

### Heat Pump Systems (Steady-State Operation)

These are simpler, algebraic steady-state models. Each class computes energy, entropy, and exergy balances for a single operating point via the `system_update()` method.

| Class | Description |
|---|---|
| `AirSourceHeatPump_cooling` | Single-step ASHP refrigerant cycle model in cooling mode |
| `AirSourceHeatPump_heating` | Single-step ASHP refrigerant cycle model in heating mode |
| `GroundSourceHeatPump_cooling` | Single-step GSHP refrigerant cycle model in cooling mode |
| `GroundSourceHeatPump_heating` | Single-step GSHP refrigerant cycle model in heating mode |

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
