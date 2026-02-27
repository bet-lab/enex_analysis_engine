# Ground Source Heat Pump Boiler (GSHPB)

> Module: `enex_analysis.GroundSourceHeatPumpBoiler`

## Overview

Physics-based ground source heat pump boiler model with borehole heat exchanger,
refrigerant cycle resolution, and UV disinfection support. The evaporator uses
ground-loop water instead of outdoor air, providing stable source temperatures
year-round.

## Architecture

```
  Ground Loop ──→ Evaporator (HX) ──→ Compressor ──→ Condenser (HX) ──→ Tank
       ↑                                                                    │
       └────────────────── Borehole Heat Exchanger ◄────────────────────────┘

  Optional:  UV Disinfection Lamp (periodic switching)
```

## Key Parameters

### Refrigerant / Cycle

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `refrigerant` | `'R410A'` | — | Refrigerant type |
| `V_disp_cmp` | 0.0005 | m³ | Compressor displacement |
| `eta_cmp_isen` | 0.7 | — | Isentropic efficiency |
| `dT_superheat` | 3.0 | K | Superheat |
| `dT_subcool` | 3.0 | K | Subcool |

### Ground Loop / Borehole

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `T_b_f_in` | 15.0 | °C | Borehole fluid inlet temperature |
| `UA_evap_design` | 3000.0 | W/K | Evaporator design UA |

### UV Disinfection

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `uv_lamp_power` | 40.0 | W | UV-C lamp power |
| `num_switching_per_3hour` | 1 | — | UV switching frequency per 3 h |

## Usage

### Steady-State Analysis

```python
from enex_analysis import GroundSourceHeatPumpBoiler

gshp = GroundSourceHeatPumpBoiler(
    refrigerant='R410A',
    V_disp_cmp=0.0005,
)

result = gshp.analyze_steady(
    T_tank_w=55.0,         # Tank water temperature [°C]
    T_b_f_in=15.0,         # Ground loop inlet temperature [°C]
    Q_cond_load=8000,      # Target heat rate [W]
    T0=5.0,                # Dead-state temperature [°C]
)

print(f"COP: {result['cop_sys']:.2f}")
```

### Dynamic Simulation

```python
result_df = gshp.analyze_dynamic(
    simulation_period_sec=86400,
    dt_s=60,
    T_tank_w_init_C=20.0,
    schedule_entries=[("7:00", "8:00", 1.0), ("19:00", "21:00", 1.0)],
    T0_schedule=[("0:00", "24:00", 5.0)],
)
```

## API Reference

| Method | Description |
|---|---|
| `analyze_steady(T_tank_w, T_b_f_in, ...)` | Single operating point analysis |
| `analyze_dynamic(simulation_period_sec, dt_s, ...)` | Dynamic simulation with tank |
