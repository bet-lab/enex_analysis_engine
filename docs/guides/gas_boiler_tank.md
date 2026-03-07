# Gas Boiler with Tank (GasBoilerTank)

> Module: `enex_analysis.GasBoilerTank`

## Overview

Gas-fired boiler with hot-water storage tank. The combustion model uses a
constant thermal efficiency (`eta_comb`) applied to a fixed burner capacity.
Uses the same `dynamic_context` infrastructure as `AirSourceHeatPumpBoiler`
and `ElectricBoiler`, with a fully implicit `fsolve` scheme on
`tank_mass_energy_residual`.

## System Architecture

```
  Natural Gas → Combustion Chamber → Hot Water Tank → 3-Way Mixing Valve → Service Water
                     ↓                    ↑
                 Exhaust Gas          Mains Water

  Optional subsystems:
    SolarThermalCollector → Tank
    UVLamp → Tank
```

## Key Parameters

### Combustion

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `eta_comb` | 0.9 | — | Combustion efficiency |
| `T_exh` | 70.0 | °C | Exhaust gas temperature |
| `burner_capacity` | 15000.0 | W | Burner rated capacity |

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
| `tank_always_full` | `True` | Force tank to stay full |
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
from enex_analysis import GasBoilerTank

gbt = GasBoilerTank(
    eta_comb=0.9,
    burner_capacity=15000.0,
    T_sup_w=10.0,
)

result = gbt.analyze_steady(
    T_tank_w=55.0,
    T0=5.0,
    dV_mix_w_out=0.0005,
)
print(f"Gas input: {result['E_NG [W]']:.1f} W")
```

### Dynamic Simulation

```python
import numpy as np

dt_s = 60
tN = len(np.arange(0, 86400, dt_s))
T0_schedule = np.full(tN, 5.0)

df = gbt.analyze_dynamic(
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
# Key exergy columns:
# X_NG, X_exh, X_tank_w_in, X_tank_w_out, X_mix_w_out,
# X_tank_loss, Xst_tank, Xc_comb, Xc_tank, Xc_mix, X_eff_sys
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
| `_calc_state(T_tank_w, T0, burner_on)` | Evaluate combustion + tank |
| `_determine_burner_state(ctx, is_on_prev)` | Hysteresis on/off + state evaluation |
| `_assemble_core_results(...)` | Post-solve reporting dict assembly |

## References

- Shares `dynamic_context` infrastructure with all tank-based models
- See also: [dynamic_context guide](dynamic_context.md), [subsystems guide](subsystems.md)
