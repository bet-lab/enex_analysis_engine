# ASHPB with PV and ESS Integration
> Module: `enex_analysis.ashpb_pv_ess`

## Overview
The `ASHPB_PV_ESS` scenario class inherits from the standalone `AirSourceHeatPumpBoiler` and integrates the `PhotovoltaicSystem` and `EnergyStorageSystem` subsystems to simulate a grid-connected, renewable-powered thermal system.

## Energy Routing Logic
The class orchestrates the interaction between the heat pump load, PV generation, and the battery via the `_run_subsystems` and `_augment_results` override hooks:

1. **Calculate Load and Generation**:
   - The heat pump calculates its total electrical power demand (`E_tot = E_cmp + E_ou_fan + E_iu_fan + ...`).
   - The PV physics engine calculates the instantaneous generation (`E_pv_out`) from solar irradiance.

2. **Power Balance and ESS Dispatch**:
   - **Surplus Power (PV > Load)**: The excess power is sent to the ESS to charge it. If the ESS reaches `SOC_max` or the charge request exceeds the internal chemistry bounds, the remaining power is exported to the grid (`E_grid_export`).
   - **Deficit Power (PV < Load)**: The shortfall is requested from the ESS. If the ESS hits `SOC_min` or the discharge bounds are exceeded, the remaining deficit is imported from the grid (`E_grid_import`).

3. **Data Logging**:
   - Appends all subsystem state variables (`SOC_ess`, `E_grid_import`, `E_grid_export`, PV temperatures, and Exergy flows) to the final results DataFrame.

## Usage
Refer to the `example_guide_for_cursor.md` in this directory to generate an interactive `.ipynb` simulation scenario.
