# Gas Boiler

> Module: `enex_analysis.GasBoiler`

## Overview

Gas-fired boiler model with direct hot-water supply (no storage tank).
Computes energy, entropy, and exergy balances for the combustion chamber,
mixing valve, and service water delivery at each time step.

## Architecture

```
  Natural Gas → Combustion Chamber → Mixing Valve → Service Water
                     ↓                    ↑
                 Exhaust Gas          Mains Water
```

## Key Parameters

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `eta_comb` | 0.9 | — | Combustion efficiency |
| `T_mains_C` | 15.0 | °C | Mains water inlet temperature |
| `T_exh` | 70.0 | °C | Exhaust gas temperature |
| `T_comb_setpoint` | 60.0 | °C | Boiler outlet setpoint temperature |
| `dV_w_serv_m3s` | 0.0001 | m³/s | Maximum service water flow rate |

## Usage

### Steady-State Analysis

```python
from enex_analysis import GasBoiler

gb = GasBoiler(
    eta_comb=0.9,
    T_comb_setpoint=60.0,
    dV_w_serv_m3s=0.0001,
)

result = gb.analyze_steady(
    T0=5.0,
    dV_w_serv=0.00005,   # Actual service flow [m³/s]
)

print(f"Gas consumption: {result['Q_gas [W]']:.1f} W")
```

### Dynamic Simulation

```python
result_df = gb.analyze_dynamic(
    simulation_period_sec=86400,
    dt_s=60,
    schedule_entries=[("7:00", "8:00", 1.0), ("19:00", "21:00", 1.0)],
    T0_schedule=[("0:00", "24:00", 5.0)],
)
```

## API Reference

| Method | Description |
|---|---|
| `analyze_steady(T0, dV_w_serv, ...)` | Single operating point analysis |
| `analyze_dynamic(simulation_period_sec, dt_s, ...)` | Time-stepping dynamic simulation |
