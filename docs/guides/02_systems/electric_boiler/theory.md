# Electric Boiler

> Module: `enex_analysis.ElectricBoiler`

## Overview

Electric resistance boiler with hot-water storage tank. The heater is modelled
as a pure-resistance element whose full electrical input becomes heat
(`Q_heat_source = E_heater`). Uses the same fully implicit `fsolve` scheme
on `tank_mass_energy_residual` as `AirSourceHeatPumpBoiler`, sharing the
`dynamic_context` infrastructure.

## System Architecture

```
  Electricity → Electric Heater → Hot Water Tank → 3-Way Mixing Valve → Service Water
                                       ↑
                                   Mains Water
  Optional subsystems:
    SolarThermalCollector → Tank
    UVLamp → Tank
```

## Key Parameters

### Heater

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `heater_capacity` | 5000.0 | W | Rated heater capacity |

### Tank Geometry / Insulation

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `r0` | 0.2 | m | Tank inner radius |
| `H` | 0.8 | m | Tank height |
| `x_shell` | 0.01 | m | Shell thickness |
| `x_ins` | 0.10 | m | Insulation thickness |
| `k_shell` | 25 | W/(m·K) | Shell conductivity |
| `k_ins` | 0.03 | W/(m·K) | Insulation conductivity |
| `h_o` | 15 | W/(m²·K) | External convective coefficient |

### Temperature Set-Points / Load

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `T_tank_w_upper_bound` | 65.0 | °C | Tank upper setpoint |
| `T_tank_w_lower_bound` | 60.0 | °C | Tank lower setpoint |
| `T_mix_w_out` | 45.0 | °C | Service water delivery temperature |
| `T_sup_w` | 10.0 | °C | Mains water supply temperature |
| `dV_mix_w_out_max` | 0.001 | m³/s | Max service flow rate |

### Tank Water Level Management

| Parameter | Default | Description |
|---|---|---|
| `tank_always_full` | `True` | Force tank to stay full (inflow = outflow) |
| `tank_level_lower_bound` | 0.5 | Level trigger for refill start |
| `tank_level_upper_bound` | 1.0 | Level trigger for refill stop |
| `dV_tank_w_in_refill` | 0.001 m³/s | Refill flow rate |
| `prevent_simultaneous_flow` | `False` | Exclusive-flow mode |

### Subsystems

| Parameter | Type | Description |
|---|---|---|
| `stc` | `SolarThermalCollector \| None` | Solar thermal collector |
| `uv` | `UVLamp \| None` | UV disinfection lamp |

## Usage

### Steady-State Analysis

```python
from enex_analysis import ElectricBoiler

eb = ElectricBoiler(heater_capacity=5000.0, T_sup_w=10.0)

result = eb.analyze_steady(
    T_tank_w=55.0,
    T0=5.0,
    dV_mix_w_out=0.0005,
)
print(f"Heater power: {result['E_heater [W]']:.1f} W")
```

### Dynamic Simulation

```python
import numpy as np

dt_s = 60
tN = len(np.arange(0, 86400, dt_s))
T0_schedule = np.full(tN, 5.0)

df = eb.analyze_dynamic(
    simulation_period_sec=86400,
    dt_s=dt_s,
    T_tank_w_init_C=20.0,
    dhw_usage_schedule=[("7:00", "8:00", 1.0), ("19:00", "21:00", 1.0)],
    T0_schedule=T0_schedule,
)
```

### Exergy Post-Processing

Exergy columns are automatically appended after `analyze_dynamic()`.

```python
# Exergy columns already in df:
# X_tot, X_tank_w_in, X_tank_w_out, X_mix_w_out, X_tank_loss,
# Xst_tank, Xc_mix, Xc_tank, X_eff_sys
```

## API Reference

| Method | Description |
|---|---|
| `analyze_steady(T_tank_w, T0, ...)` | Single operating point analysis |
| `analyze_dynamic(...)` | Time-stepping dynamic simulation (fully implicit) |
| `postprocess_exergy(df)` | Add exergy columns (called automatically) |

### Internal Methods

| Method | Description |
|---|---|
| `_calc_state(T_tank_w, T0, heater_on)` | Evaluate heater + tank |
| `_determine_heater_state(ctx, is_on_prev)` | Hysteresis on/off + state evaluation |
| `_assemble_core_results(...)` | Post-solve reporting dict assembly |

## References

- Shares `dynamic_context` infrastructure with all tank-based models
- See also: [dynamic_context guide](dynamic_context.md), [subsystems guide](subsystems.md)
