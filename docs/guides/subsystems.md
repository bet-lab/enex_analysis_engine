# Subsystems — Attachable Equipment Modules

> Module: `enex_analysis.subsystems`

## Overview

Provides self-contained subsystem classes that can be plugged into any
boiler model. Each subsystem bundles its configuration parameters with the
methods that operate on them, enabling **plug-in / plug-out composition** via
optional constructor injection. All subsystems implement the `Subsystem` protocol
defined in `dynamic_context`.

```python
# Example: attach STC and UV to an ASHPB
from enex_analysis.subsystems import SolarThermalCollector, UVLamp
from enex_analysis import AirSourceHeatPumpBoiler

stc = SolarThermalCollector(A_stc=4.0, mode='tank_circuit')
uv = UVLamp(lamp_watts=40.0, exposure_sec=300, num_switching=1)
hp = AirSourceHeatPumpBoiler(..., stc=stc, uv=uv)
```

## Architecture

```mermaid
graph TB
  HP["Heat Pump / Boiler Model<br/>(ASHPB / EB / GBT / SAGB)"]
  STC["SolarThermalCollector"]
  PV["PhotovoltaicSystem"]
  UV["UVLamp"]

  HP -->|self._subsystems['stc']| STC
  HP -->|self._subsystems['pv']| PV
  HP -->|self._subsystems['uv']| UV
```

### Extension Pattern

To add a new subsystem:

1. Create a `@dataclass` class in `subsystems.py` implementing the `Subsystem` protocol
2. Add `step()`, `assemble_results()`, and `calc_exergy()` methods
3. Add an optional parameter to the boiler's `__init__`
4. Wire the subsystem into the simulation loop

---

## `SolarThermalCollector`

Flat-plate or evacuated-tube solar thermal collector with two placement modes.

### Parameters

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `A_stc` | 2.0 | m² | Collector gross area |
| `stc_tilt` | 35.0 | ° | Tilt from horizontal |
| `stc_azimuth` | 180.0 | ° | Azimuth (180 = south) |
| `A_stc_pipe` | 2.0 | m² | Pipe surface area |
| `alpha_stc` | 0.95 | – | Absorptivity |
| `h_o_stc` | 15.0 | W/(m²·K) | External convective coeff |
| `h_r_stc` | 2.0 | W/(m²·K) | Radiative coeff |
| `k_ins_stc` | 0.03 | W/(m·K) | Insulation conductivity |
| `x_air_stc` | 0.01 | m | Air gap thickness |
| `x_ins_stc` | 0.05 | m | Insulation thickness |
| `preheat_start_hour` | 6.0 | h | Preheat window start |
| `preheat_end_hour` | 18.0 | h | Preheat window end |
| `dV_stc_w` | 0.001 | m³/s | STC loop flow rate |
| `E_stc_pump` | 50.0 | W | STC pump rated power |
| `mode` | `'tank_circuit'` | – | `'tank_circuit'` or `'mains_preheat'` |

### Placement Modes

| Mode | Description |
|---|---|
| `tank_circuit` | STC heats water circulated from/to the tank |
| `mains_preheat` | STC preheats mains water before it enters the tank |

### Methods

| Method | Description |
|---|---|
| `is_enabled` | Property: `True` when `A_stc > 0` |
| `is_preheat_on(hour)` | Check if hour falls in preheat window |
| `calc_overall_heat_transfer_coeff()` | Compute overall U-value from parallel resistances |
| `calc_performance(...)` | Core STC thermal analysis (one operating point) |
| `step(...)` | *Protocol method:* Compute STC state for one timestep |
| `assemble_results(...)` | *Protocol method:* Build STC result entries for DataFrame |
| `calc_exergy(...)` | *Protocol method:* Compute STC exergy columns and tank-boundary addends |
| `calculate_dynamic(...)` | Backward-compatible core dynamic calculation |

### Usage

```python
stc = SolarThermalCollector(
    A_stc=4.0,
    stc_tilt=35.0,
    stc_azimuth=180.0,
    mode='tank_circuit',
    preheat_start_hour=6,
    preheat_end_hour=18,
)

# Standalone usage (one timestep):
result = stc.calculate_dynamic(
    I_DN=500.0,
    I_dH=100.0,
    T_tank_w_K=333.15,
    T0_K=278.15,
    preheat_on=True,
    dV_tank_w_in=0.001,
    T_tank_w_in_K=288.15,
)

# Plugged into a boiler:
hp = AirSourceHeatPumpBoiler(..., stc=stc)
df = hp.analyze_dynamic(...)
```

---

## `PhotovoltaicSystem`

PV + Controller + ESS (Battery) + DC/AC Inverter subsystem with dynamic
state-of-charge (SOC) tracking and full entropy/exergy accounting.

### Parameters

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `A_pv` | 5.0 | m² | Panel surface area |
| `alp_pv` | 0.9 | – | Surface absorptivity |
| `pv_tilt` | 35.0 | ° | Tilt from horizontal |
| `pv_azimuth` | 180.0 | ° | Azimuth (180 = south) |
| `h_o` | 15.0 | W/(m²·K) | Outdoor heat transfer coefficient |
| `eta_pv` | 0.20 | – | PV panel efficiency |
| `eta_ctrl` | 0.95 | – | Controller efficiency |
| `eta_ess_chg` | 0.90 | – | ESS charge efficiency |
| `eta_ess_dis` | 0.90 | – | ESS discharge efficiency |
| `eta_inv` | 0.95 | – | DC/AC inverter efficiency |
| `C_ess_max` | 3.6e6 | J | ESS capacity (default 1 kWh) |
| `SOC_init` | 0.0 | – | Initial state of charge |
| `T_ctrl_K` | 308.15 | K | Controller operating temperature |
| `T_ess_K` | 313.15 | K | ESS operating temperature |
| `T_inv_K` | 313.15 | K | Inverter operating temperature |

### Stage Model

| Stage | Component | Output Variables | Exergy Destruction |
|---|---|---|---|
| 1 | PV Cell | `E_pv_out`, `X_pv_out` | `X_c_pv` |
| 2 | Controller | `E_ctrl_out`, `X_ctrl_out` | `X_c_ctrl` |
| 3 | ESS (Battery) | `E_ess_out`, `X_ess_out`, `SOC_ess` | `X_c_ess` |
| 4 | DC/AC Inverter | `E_inv_out`, `X_inv_out` | `X_c_inv` |

### Usage

```python
from enex_analysis.subsystems import PhotovoltaicSystem

pv = PhotovoltaicSystem(A_pv=10.0, eta_pv=0.22, C_ess_max=3.6e6)
hp = AirSourceHeatPumpBoiler(..., pv=pv)  # future integration
```

---

## `UVLamp`

UV disinfection lamp that switches on periodically. All electrical input is
converted to heat inside the tank (`Q_contribution = E_uv`).

### Parameters

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `lamp_watts` | 0.0 | W | Rated lamp power |
| `exposure_sec` | 0.0 | s | Duration of each on-cycle |
| `num_switching` | 1 | – | Number of on-cycles per period |
| `period_sec` | 10800 | s | Switching period (default 3 h) |

### Usage

```python
from enex_analysis.subsystems import UVLamp

uv = UVLamp(lamp_watts=40.0, exposure_sec=300, num_switching=1)
hp = AirSourceHeatPumpBoiler(..., uv=uv)
```

---

## Constants

### `STC_OFF_STEP`

Default result dict returned when no STC is attached or when STC is inactive. An alias `STC_OFF` is provided for backward compatibility.

```python
STC_OFF_STEP = {
    'stc_active': False,
    'stc_result': {},
    'T_stc_w_out_K': np.nan,
    'T_stc_pump_w_out_K': np.nan,
    'Q_stc_w_out': 0.0,
    'Q_stc_pump_w_out': 0.0,
    'Q_stc_w_in': 0.0,
    'E_stc_pump': 0.0,
    'Q_contribution': 0.0,
    'E_subsystem': 0.0,
    'T_tank_w_in_override_K': None,
}
```

## References

- Low-level STC physics: `enex_functions.calc_stc_performance()`
- Used by: `AirSourceHeatPumpBoiler`, `ElectricBoiler`, `GasBoilerTank`, `SolarAssistedGasBoiler`
