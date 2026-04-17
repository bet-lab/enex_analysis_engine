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


## Usage Guide & Examples

# Jupyter Notebook Implementation Guide (For Cursor)

This document provides instructions and specifications for generating the `example.ipynb` notebook for this model. Since the actual notebook generation is deferred to Cursor, please follow these guidelines when constructing the `.ipynb` file.

## 1. Objective
Create an interactive Jupyter Notebook (`example.ipynb`) that demonstrates how to initialize, run, and visualize the simulation for this specific system/model using the `enex_analysis_engine`.

## 2. Notebook Structure Requirements

The `.ipynb` file should contain the following sequential sections (as Markdown and Code cells):

### 2.1. Introduction
- **Markdown Cell**: Add a title and a brief description of the model being simulated. 
- Mention the key components and inputs required.

### 2.2. Setup & Imports
- **Code Cell**: Import necessary modules from `src.enex_analysis`.
  - `DynamicContext` from `enex_analysis.dynamic_context`
  - The model class (e.g., `<ModelName>`)
  - Any utility or visualization modules (e.g., `enex_analysis.visualization` or `matplotlib.pyplot`)
  - Boundary conditions (if needed, e.g., `weather.py`, `dhw.py`)

### 2.3. Context Initialization
- **Code Cell**: Initialize the `DynamicContext`.
  - Set the simulation `time_step` (e.g., 60 seconds).
  - Load boundary conditions (Weather, DHW profiles).

### 2.4. Model Instantiation & Parameter Configuration
- **Markdown Cell**: Briefly explain the chosen parameters.
- **Code Cell**: Instantiate the model. Set typical or default parameters based on the corresponding `theory.md` document.

### 2.5. Simulation Loop
- **Code Cell**: Write a loop to run the simulation over a defined duration (e.g., 1 day or 1 week).
  - Example logic:
    ```python
    results = []
    for _ in range(simulation_steps):
        # Update context
        # Run model step
        # Store results
    ```
- Convert the stored results into a `pandas.DataFrame` for easy plotting.

### 2.6. Results & Visualization
- **Markdown Cell**: Explain what the plots will show (e.g., Temperatures over time, Power consumption, COP).
- **Code Cell**: Use `dartwork-mpl` (or standard `matplotlib`) to generate clear, high-quality plots of the simulation results. Ensure axes are labeled correctly with units.

## 3. Cursor Implementation Command
To generate the notebook, you can provide Cursor with this command:
*"Cursor, please read this `example_guide_for_cursor.md` file and the adjacent `theory.md` file. Use them to generate a complete, working `example.ipynb` in this directory based on the guidelines provided."*


## Web Examples

# Solar-Assisted Gas Boiler

## Basic Usage

```python
from enex_analysis import SolarAssistedGasBoiler, print_balance

# Initialize solar-assisted gas boiler
solar_boiler = SolarAssistedGasBoiler()

# Set solar conditions
solar_boiler.I_DN = 500      # Direct normal irradiance [W/m²]
solar_boiler.I_dH = 200      # Diffuse horizontal irradiance [W/m²]
solar_boiler.A_stc = 2       # Solar collector area [m²]
solar_boiler.alpha = 0.95    # Absorptivity

# Set operating conditions
solar_boiler.eta_comb = 0.9
solar_boiler.T_w_comb = 60
solar_boiler.T_w_serv = 45
solar_boiler.T_w_sup = 10
solar_boiler.dV_w_serv = 1.2

# Run calculation
solar_boiler.system_update()

# Access results
print(f"Solar heat gain: {solar_boiler.Q_sol:.2f} W")
print(f"Natural gas input: {solar_boiler.E_NG:.2f} W")
print(f"Collector outlet temperature: {solar_boiler.T_w_stc_out - 273.15:.2f} °C")
print(f"Exergy efficiency: {solar_boiler.X_eff:.4f}")

# Print balances
print("\n=== Solar Collector Exergy Balance ===")
print_balance({"solar thermal panel": solar_boiler.exergy_balance["solar thermal panel"]})

print("\n=== Combustion Chamber Exergy Balance ===")
print_balance({"combustion chamber": solar_boiler.exergy_balance["combustion chamber"]})
```

## Solar Irradiance Study

```python
import numpy as np
import matplotlib.pyplot as plt
from enex_analysis import SolarAssistedGasBoiler

# Study effect of solar irradiance on gas consumption
irradiances = np.linspace(0, 1000, 51)
gas_inputs = []
solar_gains = []

solar_boiler = SolarAssistedGasBoiler()
solar_boiler.A_stc = 2
solar_boiler.T_w_comb = 60
solar_boiler.T_w_serv = 45
solar_boiler.T_w_sup = 10
solar_boiler.dV_w_serv = 1.2

for I in irradiances:
    solar_boiler.I_DN = I * 0.7  # Assume 70% direct
    solar_boiler.I_dH = I * 0.3  # 30% diffuse
    solar_boiler.system_update()
    gas_inputs.append(solar_boiler.E_NG)
    solar_gains.append(solar_boiler.Q_sol)

plt.figure(figsize=(10, 5))
plt.plot(irradiances, gas_inputs, 'r-', linewidth=2, label='Gas Input')
plt.plot(irradiances, solar_gains, 'b-', linewidth=2, label='Solar Gain')
plt.xlabel('Total Solar Irradiance [W/m²]')
plt.ylabel('Power [W]')
plt.grid(True)
plt.legend()
plt.title('Effect of Solar Irradiance on System Performance')
plt.show()
```
