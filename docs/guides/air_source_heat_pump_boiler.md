# Air Source Heat Pump Boiler (ASHPB)

> Module: `enex_analysis.AirSourceHeatPumpBoiler`

## Overview

Physics-based air source heat pump boiler model with refrigerant cycle resolution,
dynamic tank simulation, and optional solar thermal collector (STC) integration.
The model finds the optimal compressor speed and fan airflow at each time step by
minimizing total electrical power (`E_cmp + E_fan`) subject to LMTD heat exchanger
constraints.

## Architecture

```
                ┌─────────────────────┐
  Outdoor Air → │  Evaporator (HX)    │ ← Refrigerant
                └─────────┬───────────┘
                          │
                ┌─────────▼───────────┐
                │  Compressor (VSD)   │  ← Optimization variable: RPM
                └─────────┬───────────┘
                          │
                ┌─────────▼───────────┐
                │  Condenser (HX)     │ → Hot water to Tank
                └─────────┬───────────┘
                          │
                ┌─────────▼───────────┐
                │  Expansion Valve    │
                └─────────────────────┘

  Optional:  Solar Thermal Collector → Tank (preheat or circuit mode)
```

## Key Parameters

### Refrigerant / Cycle / Compressor

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `ref` | `'R134a'` | — | Refrigerant type (CoolProp string) |
| `V_disp_cmp` | 0.0002 | m³ | Compressor displacement volume |
| `eta_cmp_isen` | 0.8 | — | Isentropic efficiency |
| `dT_superheat` | 3.0 | K | Evaporator outlet superheat |
| `dT_subcool` | 3.0 | K | Condenser outlet subcool |

### Heat Exchanger

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `UA_cond_design` | 2000.0 | W/K | Condenser design UA |
| `UA_evap_design` | 1000.0 | W/K | Evaporator design UA |
| `A_cross_ou` | π×0.3² | m² | Outdoor unit cross-section area |

### Fan

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `dV_ou_design` | 2.5 | m³/s | Design airflow rate |
| `dP_fan_design` | 150.0 | Pa | Design static pressure |
| `eta_fan_design` | 0.6 | — | Design fan efficiency |

### Storage Tank

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `r_tank` | 0.2 | m | Tank radius |
| `H_tank` | 1.2 | m | Tank height |
| `T_tank_w_lower` | 50.0 | °C | Tank water lower bound setpoint |
| `T_tank_w_upper` | 65.0 | °C | Tank water upper bound setpoint |

## Usage

### Steady-State Analysis

```python
from enex_analysis import AirSourceHeatPumpBoiler

hp = AirSourceHeatPumpBoiler(
    ref='R134a',
    V_disp_cmp=0.0002,
    UA_cond_design=2000.0,
    UA_evap_design=1000.0,
)

result = hp.analyze_steady(
    T_tank_w=55.0,    # Tank water temperature [°C]
    T0=5.0,           # Outdoor air temperature [°C]
    Q_cond_load=5000,  # Target heat rate [W]
)

print(f"COP: {result['cop_sys']:.2f}")
print(f"Compressor power: {result['E_cmp [W]']:.1f} W")
```

### Dynamic Simulation

```python
result_df = hp.analyze_dynamic(
    simulation_period_sec=86400,    # 24 hours
    dt_s=60,                        # 60 s timestep
    T_tank_w_init_C=20.0,          # Initial tank temperature
    schedule_entries=[
        ("7:00", "8:00",  1.0),     # Morning peak
        ("12:00", "13:00", 0.5),    # Midday
        ("19:00", "21:00", 1.0),    # Evening peak
    ],
    T0_schedule=[("0:00", "24:00", 5.0)],
    result_save_csv_path='result.csv',
)
```

### Exergy Post-Processing

```python
df_ex = hp.postprocess_exergy(result_df)
print(f"Total exergy consumption: {df_ex['Xc_tot [W]'].sum():.0f} W")
```

## API Reference

| Method | Description |
|---|---|
| `analyze_steady(T_tank_w, T0, ...)` | Single operating point analysis |
| `analyze_dynamic(simulation_period_sec, dt_s, ...)` | Time-stepping dynamic simulation |
| `postprocess_exergy(df)` | Add exergy columns to result DataFrame |

## References

- ASHRAE Standard 90.1-2022 (VSD fan power curves)
- CoolProp library for refrigerant properties
