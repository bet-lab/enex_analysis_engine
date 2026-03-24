# Solar-Assisted Gas Boiler (SolarAssistedGasBoiler)

> Module: `enex_analysis.SolarAssistedGasBoiler`

## Overview

Gas boiler with solar thermal collector (STC) assist and hot-water storage tank.
The STC is passed as the `stc` constructor argument and registered via the
`Subsystem` protocol. Uses the same `dynamic_context` infrastructure as the
other tank-based models, with fully implicit `fsolve` solving.

## System Architecture

```
  Solar Radiation → SolarThermalCollector (STC)
                          ↓
  Natural Gas → Combustion Chamber → Hot Water Tank → Mixing Valve → Service Water
                     ↓                    ↑
                 Exhaust Gas          Mains Water

  Optional subsystems:
    SolarThermalCollector (required, injected as stc)
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

### Subsystems

| Parameter | Type | Description |
|---|---|---|
| `stc` | `SolarThermalCollector` | Solar thermal collector (see `subsystems` guide) |
| `uv` | `UVLamp \| None` | UV disinfection lamp |

### Solar Schedules

| Parameter | Type | Description |
|---|---|---|
| `I_DN_schedule` | `array-like \| None` | Direct-normal irradiance per step [W/m²] |
| `I_dH_schedule` | `array-like \| None` | Diffuse-horizontal irradiance per step [W/m²] |

## Usage

### Dynamic Simulation

```python
from enex_analysis import SolarAssistedGasBoiler
from enex_analysis.subsystems import SolarThermalCollector
import numpy as np

stc = SolarThermalCollector(
    A_stc=4.0,
    mode='tank_circuit',
    stc_tilt=35.0,
    stc_azimuth=180.0,
)

sagb = SolarAssistedGasBoiler(
    eta_comb=0.9,
    burner_capacity=15000.0,
    stc=stc,
)

dt_s = 60
tN = len(np.arange(0, 86400, dt_s))
T0_schedule = np.full(tN, 5.0)
I_DN = np.full(tN, 400.0)
I_dH = np.full(tN, 100.0)

df = sagb.analyze_dynamic(
    simulation_period_sec=86400,
    dt_s=dt_s,
    T_tank_w_init_C=20.0,
    dhw_usage_schedule=[("7:00", "8:00", 1.0), ("19:00", "21:00", 1.0)],
    T0_schedule=T0_schedule,
    I_DN_schedule=I_DN,
    I_dH_schedule=I_dH,
)
```

### Exergy Post-Processing

Exergy columns are automatically appended after `analyze_dynamic()`.

```python
# Key exergy columns:
# X_NG, X_exh, X_sol_stc, X_stc_w_in, X_stc_w_out,
# X_tank_w_in, X_tank_w_out, X_mix_w_out,
# X_tank_loss, Xst_tank, Xc_comb, Xc_stc, Xc_tank, Xc_mix, X_eff_sys
```

## API Reference

| Method | Description |
|---|---|
| `analyze_dynamic(...)` | Time-stepping dynamic simulation (fully implicit) |
| `postprocess_exergy(df)` | Add exergy columns (called automatically) |

### Internal Methods

| Method | Description |
|---|---|
| `_calc_state(T_tank_w, T0, burner_on)` | Evaluate combustion + tank |
| `_determine_burner_state(ctx, is_on_prev)` | Burner hysteresis on/off + state evaluation |
| `_assemble_core_results(...)` | Post-solve reporting dict assembly |

## References

- Shares `dynamic_context` infrastructure with all tank-based models
- STC physics: see [subsystems guide](subsystems.md)
- See also: [dynamic_context guide](dynamic_context.md)
