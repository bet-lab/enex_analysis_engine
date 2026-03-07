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
| `T_serv_w` | 45.0 | °C | Service (delivery) water temperature |
| `T_sup_w` | 15.0 | °C | Mains water supply temperature |
| `T_exh` | 70.0 | °C | Exhaust gas temperature |
| `T_comb_setpoint` | 60.0 | °C | Boiler outlet setpoint temperature |
| `dV_w_serv_m3s` | 0.0001 | m³/s | Maximum service water flow rate |

## Usage

### Steady-State Analysis

```python
from enex_analysis import GasBoiler

gb = GasBoiler(
    eta_comb=0.9,
    T_serv_w=45.0,
    T_comb_setpoint=60.0,
    dV_w_serv_m3s=0.0001,
)

result = gb.analyze_steady(
    T0=5.0,
    dV_w_serv=0.00005,
)

print(f"Gas input: {result['E_NG [W]']:.1f} W")
print(f"Exergy efficiency: {result['X_eff [-]']:.3f}")
```

### Dynamic Simulation

```python
import numpy as np

dt_s = 60
tN = len(np.arange(0, 86400, dt_s))
T0_schedule = np.full(tN, 5.0)

df = gb.analyze_dynamic(
    simulation_period_sec=86400,
    dt_s=dt_s,
    dhw_usage_schedule=[("7:00", "8:00", 1.0), ("19:00", "21:00", 1.0)],
    T0_schedule=T0_schedule,
)
```

## API Reference

| Method | Description |
|---|---|
| `analyze_steady(T0, dV_w_serv, ...)` | Single operating point analysis |
| `analyze_dynamic(simulation_period_sec, dt_s, ...)` | Time-stepping dynamic simulation |

### Internal Methods

| Method | Description |
|---|---|
| `_calc_on_state(Q_comb_load, T0, dV_w_serv)` | Compute full energy/entropy/exergy balance for ON state |
| `_calc_off_state(T0)` | Zero-load result dict for OFF state |
