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
