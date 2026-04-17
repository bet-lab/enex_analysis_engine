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

# Gas Boiler with Storage Tank

The `GasBoilerTank` model allows simulating a conventional gas boiler connected to a stratified thermal storage tank. This is commonly used for Domestic Hot Water (DHW) systems or space heating buffering where the boiler does not operate instantaneously to match the load, but rather cycles to maintain the tank temperature.

## Implementation Example

```python
from enex_analysis.gas_boiler_tank import GasBoilerTank

# Define the boiler parameters and tank properties
system = GasBoilerTank(
    LHV_gas=35.8e6,  # J/m3
    boiler_eta=0.9,  # 90% efficiency
    tank_volume=0.3, # 300 Liters
    V_dot_max=15.0 / 60000, # Flow rate
    # Additional required parameters...
)

# In a simulation loop:
# results = system.simulate_step(context, control_params...)
```

The tank physics relies on the `StratifiedTankTDMA` mathematical model to calculate internal temperature nodes dynamically.
