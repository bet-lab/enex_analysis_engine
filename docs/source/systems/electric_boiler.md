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

# Electric Boiler

## Basic Usage

```python
from enex_analysis import ElectricBoiler, print_balance

# Initialize boiler with default parameters
boiler = ElectricBoiler()

# Set operating conditions
boiler.T_w_tank = 60   # Tank water temperature [°C]
boiler.T_w_sup = 10    # Supply water temperature [°C]
boiler.T_w_serv = 45   # Service water temperature [°C]
boiler.T0 = 0          # Reference temperature [°C]
boiler.dV_w_serv = 1.2 # Service water flow rate [L/min]

# Modify tank properties if needed
boiler.r0 = 0.2        # Tank inner radius [m]
boiler.H = 0.8         # Tank height [m]
boiler.x_ins = 0.10    # Insulation thickness [m]
boiler.k_ins = 0.03    # Insulation thermal conductivity [W/mK]

# Run calculation
boiler.system_update()

# Access results
print(f"Electric power input: {boiler.E_heater:.2f} W")
print(f"Tank heat loss: {boiler.Q_l_tank:.2f} W")
print(f"Exergy efficiency: {boiler.X_eff:.4f}")

# Print balances
print("\n=== Energy Balance ===")
print_balance(boiler.energy_balance)

print("\n=== Entropy Balance ===")
print_balance(boiler.entropy_balance)

print("\n=== Exergy Balance ===")
print_balance(boiler.exergy_balance)
```

## Parameter Study

```python
import numpy as np
import matplotlib.pyplot as plt
from enex_analysis import ElectricBoiler

# Study effect of tank temperature on exergy efficiency
tank_temps = np.linspace(50, 70, 21)
exergy_effs = []
power_inputs = []

boiler = ElectricBoiler()
boiler.T_w_sup = 10
boiler.T_w_serv = 45
boiler.dV_w_serv = 1.2

for T_tank in tank_temps:
    boiler.T_w_tank = T_tank
    boiler.system_update()
    exergy_effs.append(boiler.X_eff)
    power_inputs.append(boiler.E_heater)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(tank_temps, exergy_effs, 'b-', linewidth=2)
ax1.set_xlabel('Tank Temperature [°C]')
ax1.set_ylabel('Exergy Efficiency [-]')
ax1.grid(True)
ax1.set_title('Exergy Efficiency vs Tank Temperature')

ax2.plot(tank_temps, power_inputs, 'r-', linewidth=2)
ax2.set_xlabel('Tank Temperature [°C]')
ax2.set_ylabel('Electric Power Input [W]')
ax2.grid(True)
ax2.set_title('Power Input vs Tank Temperature')

plt.tight_layout()
plt.show()
```
