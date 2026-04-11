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

# Gas Boiler

## Basic Usage

```python
from enex_analysis import GasBoiler, print_balance

# Initialize gas boiler
boiler = GasBoiler()

# Set operating conditions
boiler.eta_comb = 0.9      # Combustion efficiency
boiler.T_w_tank = 60       # Tank water temperature [°C]
boiler.T_w_sup = 10        # Supply water temperature [°C]
boiler.T_w_serv = 45       # Service water temperature [°C]
boiler.T_exh = 70          # Exhaust gas temperature [°C]
boiler.dV_w_serv = 1.2     # Service water flow rate [L/min]

# Run calculation
boiler.system_update()

# Access results
print(f"Natural gas energy input: {boiler.E_NG:.2f} W")
print(f"Exhaust heat loss: {boiler.Q_exh:.2f} W")
print(f"Exergy efficiency: {boiler.X_eff:.4f}")
print(f"Total exergy consumed: {boiler.X_c_tot:.2f} W")

# Print exergy balance
print_balance(boiler.exergy_balance, decimal=2)
```

## Efficiency Comparison

```python
from enex_analysis import ElectricBoiler, GasBoiler

# Compare electric and gas boilers
elec_boiler = ElectricBoiler()
gas_boiler = GasBoiler()

# Same operating conditions
for boiler in [elec_boiler, gas_boiler]:
    boiler.T_w_tank = 60
    boiler.T_w_sup = 10
    boiler.T_w_serv = 45
    boiler.dV_w_serv = 1.2
    boiler.system_update()

print("Electric Boiler:")
print(f"  Power input: {elec_boiler.E_heater:.2f} W")
print(f"  Exergy efficiency: {elec_boiler.X_eff:.4f}")

print("\nGas Boiler:")
print(f"  Energy input: {gas_boiler.E_NG:.2f} W")
print(f"  Exergy efficiency: {gas_boiler.X_eff:.4f}")
print(f"  Exergy input: {gas_boiler.X_NG:.2f} W")
```
