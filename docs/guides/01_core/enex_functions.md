# Utility Functions Reference

> Module: `enex_analysis.enex_functions`

## Overview

Library of 73+ utility functions organized into functional categories.
These functions are used internally by the simulation classes and can also
be called directly for custom analysis workflows.

**Note:** As part of a major codebase refactoring, most of these utility functions have been decoupled into smaller dedicated backend modules (e.g., `cop.py`, `thermodynamics.py`, `heat_transfer.py`, `weather.py`, `g_function.py`, `refrigerant.py`). They are currently re-exported by `enex_functions.py` for backward compatibility, but new code should preferably import them from their specific component modules.

## Function Categories

### 1. Friction and Flow

| Function | Description |
|---|---|
| `darcy_friction_factor(Re, e_d)` | Darcy friction factor from Reynolds number and relative roughness |
| `calc_h_vertical_plate(T_s, T_inf, L)` | Natural convection coefficient for vertical plate (Churchill & Chu) |

### 2. Polynomial Functions

| Function | Description |
|---|---|
| `linear_function(x, a, b)` | y = a·x + b |
| `quadratic_function(x, a, b, c)` | y = a·x² + b·x + c |
| `cubic_function(x, a, b, c, d)` | y = a·x³ + b·x² + c·x + d |
| `quartic_function(x, a, b, c, d, e)` | y = a·x⁴ + b·x³ + c·x² + d·x + e |

### 3. Air Properties

| Function | Description |
|---|---|
| `air_dynamic_viscosity(T_K)` | Air viscosity via Sutherland's formula [Pa·s] |
| `air_prandtl_number(T_K)` | Air Prandtl number (≈ 0.71) |

### 4. Exergy and Entropy

| Function | Description |
|---|---|
| `generate_entropy_exergy_term(energy, Tsys, T0, fluid)` | Compute S and X terms from energy |
| `calc_energy_flow(G, T, T0)` | Energy flow rate for advection |
| `calc_exergy_flow(G, T, T0)` | Exergy flow rate for material streams (vectorized) |
| `calc_refrigerant_exergy(df, ref, T0_K, P0)` | Refrigerant state-point exergy using CoolProp |
| `convert_electricity_to_exergy(df)` | Copy all `E_*` columns to `X_*` (electricity = 100% exergy) |

### 5. Flow and Mixing

| Function | Description |
|---|---|
| `calc_mixing_valve(T_tank_w_K, T_tank_w_in_K, T_mix_w_out_K)` | 3-way mixing valve output temperature and ratio α |
| `calc_uv_lamp_power(current_time_s, period_sec, num_switching, ...)` | UV lamp instantaneous power (periodic on/off) |
| `calc_Orifice_flow_coefficient(D0, D1)` | Orifice flow coefficient |
| `calc_boussinessq_mixing_flow(T_up, T_lo, A, dz, C_d)` | Buoyancy-driven mixing flow |

### 6. Heat Pump COP Calculations

| Function | Description |
|---|---|
| `calc_ASHP_cooling_COP(T_a_int_out, T_a_ext_in, Q, Q_max, COP_ref)` | ASHP cooling COP (IBPSA 2023) |
| `calc_ASHP_heating_COP(T0, Q, Q_max)` | ASHP heating COP (PLR-based) |
| `calc_GSHP_COP(Tg, T_cond, T_evap, theta_hat)` | GSHP modified Carnot COP |

### 7. Ground Heat Exchanger

| Function | Description |
|---|---|
| `f(x)` | Helper for g-function calculation |
| `chi(s, rb, H, z0)` | Helper for g-function calculation |
| `G_FLS(t, ks, as_, rb, H)` | Finite Line Source g-function (cached) |

### 8. TDMA Solver

| Function | Description |
|---|---|
| `TDMA(a, b, c, d)` | Solve tri-diagonal matrix system |
| `_add_loop_advection_terms(a, b, c, d, ...)` | Add forced convection terms to TDMA coefficients |

### 9. Tank Heat Transfer

| Function | Description |
|---|---|
| `calc_UA_tank_arr(r0, x_shell, x_ins, ...)` | Per-node UA array for cylindrical tank |
| `calc_simple_tank_UA(r0, H, x_shell, x_ins, k_shell, k_ins, h_o)` | Simplified whole-tank UA |
| `update_tank_temperature(T_tank_w_K, Q_gain, UA_tank, T0_K, C_tank, dt)` | Crank-Nicolson lumped-capacitance tank update |

### 10. Heat Exchanger

| Function | Description |
|---|---|
| `calc_LMTD_counter_flow(T_hi, T_ho, T_ci, T_co)` | Counter-flow LMTD [K] |
| `calc_LMTD_parallel_flow(T_hi, T_ho, T_ci, T_co)` | Parallel-flow LMTD [K] |
| `calc_UA_from_dV_fan(dV, dV_design, A, UA)` | Velocity-dependent UA (Dittus-Boelter) |
| `calc_HX_perf_for_target_heat(Q_target, ...)` | Solve fan airflow for target heat rate (bisect) |

### 11. Fan Power

| Function | Description |
|---|---|
| `calc_fan_power_from_dV_fan(dV, params, vsd_coeffs, ...)` | Fan power via ASHRAE 90.1 VSD curve |

### 12. Refrigerant Cycle

| Function | Description |
|---|---|
| `calc_ref_state(T_evap_K, T_cond_K, refrigerant, eta_cmp_isen, T0_K, mode, dT_superheat, dT_subcool, is_active)` | Calculate all 4 cycle state points with superheat/subcool |
| `create_lmtd_constraints()` | Generate LMTD constraint functions for optimization |
| `find_ref_loop_optimal_operation(...)` | Find minimum-power operating point |

### 13. Schedule and Water Use

| Function | Description |
|---|---|
| `build_dhw_usage_ratio(entries, t_array)` | Build time-series ratio array from schedule entries |
| `check_hp_schedule_active(hour, hp_on_schedule)` | Check if current hour is in active HP schedule |
| `calc_total_water_use_from_schedule(schedule, peak, ...)` | Compute total daily water consumption |
| `make_dhw_schedule_from_Annex_42_profile(...)` | Convert IEA Annex 42 flow profile to schedule |
| `calc_cold_water_temp(df, date_str)` | EnergyPlus algorithm for mains water temperature |

### 14. UV Disinfection

| Function | Description |
|---|---|
| `get_uv_params_from_turbidity(ntu)` | UV parameters from turbidity table |
| `calc_uv_exposure_time(radius, power, ...)` | Required UV exposure time (radial model) |

### 15. Solar Thermal Collector

| Function | Description |
|---|---|
| `calc_stc_performance(I_DN_stc, I_dH_stc, T_stc_w_in_K, T0_K, ...)` | STC thermal analysis (kept for backward compatibility; prefer `subsystems.SolarThermalCollector` for new code) |

### 16. KMA Weather Data & Solar Decomposition

| Function | Description |
|---|---|
| `load_kma_solar_csv(csv_path, encoding)` | Load KMA 1-min cumulative GHI CSV → instantaneous W/m² |
| `load_kma_T0_sol_hourly_csv(csv_path, encoding)` | Load KMA hourly T0 + solar CSV (auto-detects columns) |
| `decompose_ghi_to_poa(ghi, lat, lon, tilt, azimuth, ...)` | GHI → DNI+DHI → POA irradiance via `pvlib` |

### 17. Visualization

| Function | Description |
|---|---|
| `print_balance(balance, decimal)` | Print formatted energy/entropy/exergy balance |
| `print_simulation_summary(df, dt, dV_design)` | Print simulation statistics |
| `plot_th_diagram(ax, result, ...)` | T-h diagram for refrigerant cycle |
| `plot_ph_diagram(ax, result, ...)` | P-h diagram for refrigerant cycle |
| `plot_ts_diagram(ax, result, ...)` | T-s diagram for refrigerant cycle |

## Usage Examples

### Exergy Flow Calculation

```python
from enex_analysis.enex_functions import calc_exergy_flow
from enex_analysis.calc_util import C2K
from enex_analysis.constants import c_w, rho_w

G = c_w * rho_w * 0.0001   # Heat capacity flow [W/K]
T = C2K(60.0)               # Stream temperature [K]
T0 = C2K(5.0)               # Dead-state temperature [K]

X_flow = calc_exergy_flow(G, T, T0)
print(f"Exergy flow: {X_flow:.1f} W")
```

### Building a DHW Schedule

```python
from enex_analysis.enex_functions import build_dhw_usage_ratio
import numpy as np

entries = [
    ("7:00",  "8:00",  1.0),   # Morning peak
    ("12:00", "13:00", 0.5),   # Midday
    ("19:00", "21:00", 1.0),   # Evening peak
]
t_array = np.arange(0, 86400, 60)   # 60 s timestep
ratios = build_dhw_usage_ratio(entries, t_array)
```

### Refrigerant Cycle State Points

```python
from enex_analysis.enex_functions import calc_ref_state
from enex_analysis.calc_util import C2K

states = calc_ref_state(
    T_evap_K=C2K(5.0),
    T_cond_K=C2K(55.0),
    refrigerant='R134a',
    eta_cmp_isen=0.8,
    T0_K=C2K(5.0),
    mode='heating',
    dT_superheat=3.0,
    dT_subcool=3.0,
    is_active=True,
)
print(f"h1: {states['h1']:.0f} J/kg, h2: {states['h2']:.0f} J/kg")
```

### Loading KMA Solar Data

```python
from enex_analysis.enex_functions import load_kma_solar_csv, decompose_ghi_to_poa

ghi_df = load_kma_solar_csv('Seoul_25_solar_1min.csv')
poa = decompose_ghi_to_poa(
    ghi=ghi_df['GHI_Wm2'],
    latitude=37.57, longitude=126.97,
    tilt=35.0, azimuth=180.0,
)

I_DN_schedule = poa['poa_direct'].values
I_dH_schedule = poa['poa_diffuse'].values
```

### 3-Way Mixing Valve

```python
from enex_analysis.enex_functions import calc_mixing_valve
from enex_analysis.calc_util import C2K

mix = calc_mixing_valve(
    T_tank_w_K=C2K(65.0),
    T_tank_w_in_K=C2K(15.0),
    T_mix_w_out_K=C2K(40.0),
)
print(f"α = {mix['alp']:.3f}, T_mix = {mix['T_mix_w_out']:.1f} °C")
```

### Refrigerant Exergy Post-Processing

```python
from enex_analysis.enex_functions import calc_refrigerant_exergy
from enex_analysis.calc_util import C2K

# After running a dynamic ASHP simulation:
df = calc_refrigerant_exergy(df, ref='R134a', T0_K=C2K(df['T0 [°C]']))
# Appends x_ref_cmp_in, X_ref_cmp_in, etc.
```
