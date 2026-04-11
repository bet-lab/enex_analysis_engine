# ASHPB with Solar Thermal Collector Integration
> Modules: `enex_analysis.ashpb_stc_preheat`, `enex_analysis.ashpb_stc_tank`

## Overview
These scenario classes integrate the `SolarThermalCollector` (STC) subsystem into the Air Source Heat Pump Boiler (ASHPB). They demonstrate how a stateless physics engine (the STC) can be wired into different thermal topologies.

## Topologies

### 1. ASHPB_STC_tank
- **Module**: `ashpb_stc_tank.py`
- **Mechanism**: The STC circulates water directly with the primary storage tank. 
- **Control**: The STC acts as an additional heat source to the thermal mass. When the predefined `preheat` schedule is active and the solar generation provides a net positive temperature gain, the STC pump turns on, injecting heat (`Q_stc_w_out`) directly into the tank's energy balance residual solver.

### 2. ASHPB_STC_preheat
- **Module**: `ashpb_stc_preheat.py`
- **Mechanism**: The STC preheats the incoming cold mains water before it enters the principal storage tank or mixing valve.
- **Control**: The cold supply water (`T_sup_w`) passes through the STC first. The heated water is then sent to the mixing valve, reducing the amount of hot water drawn from the heat pump tank. This indirectly saves heat pump compressor energy.

## Usage
Refer to the `example_guide_for_cursor.md` in this directory to generate an interactive `.ipynb` simulation scenario.


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
