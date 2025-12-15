# Input/Output Documentation

This document provides detailed specifications for all components, functions, and utilities in the ENEX Analysis Engine.

## Table of Contents

1. [Unit Conversion Utilities](#unit-conversion-utilities)
2. [Component Models](#component-models)
3. [Balance Dictionary Structure](#balance-dictionary-structure)
4. [Utility Functions](#utility-functions)

---

## Unit Conversion Utilities

The `calc_util` module provides comprehensive unit conversion constants and functions.

### Temperature Conversions

```python
from enex_analysis import C2K, K2C

# Celsius to Kelvin
T_K = C2K(25)  # Returns 298.15

# Kelvin to Celsius  
T_C = K2C(298.15)  # Returns 25.0
```

### Time Conversions

Available constants: `d2h`, `d2m`, `d2s`, `h2d`, `m2d`, `s2d`, `h2m`, `h2s`, `m2h`, `s2h`, `m2s`, `s2m`, `y2d`, `d2y`

```python
from enex_analysis.calc_util import h2s, s2h

hours = 2
seconds = hours * h2s  # 7200
```

### Length Conversions

Available constants: `m2cm`, `cm2m`, `m2mm`, `mm2m`, `m2km`, `km2m`, `cm2mm`, `mm2cm`, `in2cm`, `cm2in`, `ft2m`, `m2ft`

### Area Conversions

Available constants: `m22cm2`, `cm22m2`, `m22mm2`, `mm22m2`

### Volume Conversions

Available constants: `m32cm3`, `cm32m3`, `m32L`, `L2m3`

### Mass Conversions

Available constants: `kg2g`, `g2kg`, `kg2mg`, `mg2kg`, `kg2t`, `t2kg`

### Energy Conversions

Available constants: `J2kJ`, `kJ2J`, `J2MJ`, `MJ2J`, `J2GJ`, `GJ2J`, `kWh2J`, `J2kWh`

### Power Conversions

Available constants: `W2kW`, `kW2W`, `W2MW`, `MW2W`

### Pressure Conversions

Available constants: `Pa2kPa`, `kPa2Pa`, `Pa2MPa`, `MPa2Pa`, `Pa2bar`, `bar2Pa`, `atm2Pa`, `Pa2atm`

### Angle Conversions

Available constants: `d2r`, `r2d`

---

## Component Models

### ElectricBoiler

A model for an electric boiler system with hot water tank and mixing valve.

#### Input Parameters

All parameters are set in `__post_init__` and can be modified before calling `system_update()`:

| Parameter | Type | Unit | Default | Description |
|-----------|------|------|---------|-------------|
| `T_w_tank` | float | °C | 60 | Tank water temperature |
| `T_w_sup` | float | °C | 10 | Supply water temperature |
| `T_w_serv` | float | °C | 45 | Service (tap) water temperature |
| `T0` | float | °C | 0 | Reference (environmental) temperature |
| `dV_w_serv` | float | L/min | 1.2 | Service water flow rate |
| `r0` | float | m | 0.2 | Tank inner radius |
| `H` | float | m | 0.8 | Tank height |
| `x_shell` | float | m | 0.01 | Shell thickness |
| `x_ins` | float | m | 0.10 | Insulation thickness |
| `k_shell` | float | W/mK | 25 | Shell thermal conductivity |
| `k_ins` | float | W/mK | 0.03 | Insulation thermal conductivity |
| `h_o` | float | W/m²K | 15 | Overall heat transfer coefficient |

#### Output Attributes

After calling `system_update()`, the following attributes are available:

**Energy-related:**
- `E_heater`: Electric power input [W]
- `Q_w_tank`: Heat transfer from tank water [W]
- `Q_w_sup_tank`: Heat transfer from supply water to tank [W]
- `Q_l_tank`: Heat loss from tank [W]
- `Q_w_serv`: Heat transfer to service water [W]

**Exergy-related:**
- `X_heater`: Exergy of electric input [W]
- `X_w_tank`: Exergy of tank water [W]
- `X_w_serv`: Exergy of service water [W]
- `X_c_tank`: Exergy consumed in tank [W]
- `X_c_mix`: Exergy consumed in mixing valve [W]
- `X_c_tot`: Total exergy consumed [W]
- `X_eff`: Exergy efficiency [-]

**Balance Dictionaries:**
- `energy_balance`: Energy balance for each subsystem
- `entropy_balance`: Entropy balance for each subsystem
- `exergy_balance`: Exergy balance for each subsystem

---

### GasBoiler

A model for a gas boiler system with combustion chamber, hot water tank, and mixing valve.

#### Input Parameters

| Parameter | Type | Unit | Default | Description |
|-----------|------|------|---------|-------------|
| `eta_comb` | float | - | 0.9 | Combustion efficiency |
| `T_w_tank` | float | °C | 60 | Tank water temperature |
| `T_w_sup` | float | °C | 10 | Supply water temperature |
| `T_w_serv` | float | °C | 45 | Service water temperature |
| `T0` | float | °C | 0 | Reference temperature |
| `T_exh` | float | °C | 70 | Exhaust gas temperature |
| `dV_w_serv` | float | L/min | 1.2 | Service water flow rate |
| `r0` | float | m | 0.2 | Tank inner radius |
| `H` | float | m | 0.8 | Tank height |
| `x_shell` | float | m | 0.01 | Shell thickness |
| `x_ins` | float | m | 0.10 | Insulation thickness |
| `k_shell` | float | W/mK | 25 | Shell thermal conductivity |
| `k_ins` | float | W/mK | 0.03 | Insulation thermal conductivity |
| `h_o` | float | W/m²K | 15 | Overall heat transfer coefficient |

#### Output Attributes

**Energy-related:**
- `E_NG`: Natural gas energy input [W]
- `Q_w_comb_out`: Heat transfer from combustion chamber [W]
- `Q_exh`: Heat loss from exhaust gases [W]
- `Q_w_tank`: Heat transfer from tank water [W]
- `Q_l_tank`: Heat loss from tank [W]

**Exergy-related:**
- `X_NG`: Exergy of natural gas input [W] (exergy efficiency = 0.93)
- `X_w_comb_out`: Exergy of water from combustion chamber [W]
- `X_exh`: Exergy of exhaust gases [W]
- `X_c_comb`: Exergy consumed in combustion chamber [W]
- `X_c_tank`: Exergy consumed in tank [W]
- `X_c_mix`: Exergy consumed in mixing valve [W]
- `X_c_tot`: Total exergy consumed [W]
- `X_eff`: Exergy efficiency [-]

---

### HeatPumpBoiler

A model for an air-source heat pump boiler system with external unit, refrigerant loop, hot water tank, and mixing valve.

#### Input Parameters

| Parameter | Type | Unit | Default | Description |
|-----------|------|------|---------|-------------|
| `eta_fan` | float | - | 0.6 | External fan efficiency |
| `COP` | float | - | 2.5 | Coefficient of Performance |
| `dP` | float | Pa | 200 | Pressure difference across fan |
| `T0` | float | °C | 0 | Reference temperature |
| `T_a_ext_out` | float | °C | T0 - 5 | External unit air outlet temperature |
| `T_r_ext` | float | °C | T0 - 10 | External unit refrigerant temperature |
| `T_w_tank` | float | °C | 60 | Tank water temperature |
| `T_r_tank` | float | °C | T_w_tank + 5 | Tank refrigerant temperature |
| `T_w_serv` | float | °C | 45 | Service water temperature |
| `T_w_sup` | float | °C | 10 | Supply water temperature |
| `dV_w_serv` | float | L/min | 1.2 | Service water flow rate |
| `r0` | float | m | 0.2 | Tank inner radius |
| `H` | float | m | 0.8 | Tank height |
| `x_shell` | float | m | 0.01 | Shell thickness |
| `x_ins` | float | m | 0.10 | Insulation thickness |
| `k_shell` | float | W/mK | 25 | Shell thermal conductivity |
| `k_ins` | float | W/mK | 0.03 | Insulation thermal conductivity |
| `h_o` | float | W/m²K | 15 | Overall heat transfer coefficient |

#### Output Attributes

**Energy-related:**
- `E_fan`: External fan power [W]
- `E_cmp`: Compressor power [W]
- `Q_r_tank`: Heat transfer from refrigerant to tank [W]
- `Q_r_ext`: Heat transfer from external unit to refrigerant [W]
- `Q_a_ext_in`: Heat transfer from air entering external unit [W]
- `Q_a_ext_out`: Heat transfer from air leaving external unit [W]

**Exergy-related:**
- `X_fan`: Exergy of fan input [W]
- `X_cmp`: Exergy of compressor input [W]
- `X_r_tank`: Exergy of refrigerant to tank [W]
- `X_r_ext`: Exergy of refrigerant from external unit [W]
- `X_c_ext`: Exergy consumed in external unit [W]
- `X_c_r`: Exergy consumed in refrigerant loop [W]
- `X_c_tank`: Exergy consumed in tank [W]
- `X_c_mix`: Exergy consumed in mixing valve [W]
- `X_c_tot`: Total exergy consumed [W]
- `X_eff`: Exergy efficiency [-]

---

### SolarAssistedGasBoiler

A model for a solar-assisted gas boiler system with solar thermal collector, combustion chamber, and mixing valve.

#### Input Parameters

| Parameter | Type | Unit | Default | Description |
|-----------|------|------|---------|-------------|
| `alpha` | float | - | 0.95 | Absorptivity of collector |
| `eta_comb` | float | - | 0.9 | Combustion efficiency |
| `I_DN` | float | W/m² | 500 | Direct normal solar irradiance |
| `I_dH` | float | W/m² | 200 | Diffuse horizontal solar irradiance |
| `A_stc` | float | m² | 2 | Solar thermal collector area |
| `T0` | float | °C | 0 | Reference temperature |
| `T_w_comb` | float | °C | 60 | Combustion chamber water temperature |
| `T_w_serv` | float | °C | 45 | Service water temperature |
| `T_w_sup` | float | °C | 10 | Supply water temperature |
| `T_exh` | float | °C | 70 | Exhaust gas temperature |
| `dV_w_serv` | float | L/min | 1.2 | Service water flow rate |
| `h_o` | float | W/m²K | 15 | Overall heat transfer coefficient |
| `h_r` | float | W/m²K | 2 | Radiative heat transfer coefficient |
| `k_ins` | float | W/mK | 0.03 | Insulation thermal conductivity |
| `x_air` | float | m | 0.01 | Air layer thickness |
| `x_ins` | float | m | 0.05 | Insulation layer thickness |

#### Output Attributes

**Energy-related:**
- `Q_sol`: Solar heat gain [W]
- `Q_w_stc_out`: Heat transfer from collector outlet [W]
- `Q_l`: Heat loss from collector [W]
- `E_NG`: Natural gas energy input [W]
- `Q_exh`: Heat loss from exhaust gases [W]
- `Q_w_comb`: Heat transfer from combustion chamber [W]

**Exergy-related:**
- `X_sol`: Exergy of solar radiation [W]
- `X_w_stc_out`: Exergy of water from collector [W]
- `X_l`: Exergy loss from collector [W]
- `X_c_stc`: Exergy consumed in collector [W]
- `X_NG`: Exergy of natural gas input [W]
- `X_exh`: Exergy of exhaust gases [W]
- `X_c_comb`: Exergy consumed in combustion chamber [W]
- `X_c_mix`: Exergy consumed in mixing valve [W]
- `X_eff`: Exergy efficiency [-]

---

### GroundSourceHeatPumpBoiler

A model for a ground-source heat pump boiler system with ground heat exchanger, refrigerant loop, hot water tank, and mixing valve.

#### Input Parameters

| Parameter | Type | Unit | Default | Description |
|-----------|------|------|---------|-------------|
| `time` | float | h | 10 | Operating time |
| `T0` | float | °C | 0 | Reference temperature |
| `T_w_tank` | float | °C | 60 | Tank water temperature |
| `T_w_serv` | float | °C | 45 | Service water temperature |
| `T_w_sup` | float | °C | 10 | Supply water temperature |
| `T_g` | float | °C | 11 | Undisturbed ground temperature |
| `T_r_tank` | float | °C | T_w_tank + 5 | Tank refrigerant temperature |
| `dT_r_exch` | float | K | -5 | Temperature difference (refrigerant - fluid after exchange) |
| `dV_w_serv` | float | L/min | 1.2 | Service water flow rate |
| `r0` | float | m | 0.2 | Tank inner radius |
| `H` | float | m | 0.8 | Tank height |
| `x_shell` | float | m | 0.01 | Shell thickness |
| `x_ins` | float | m | 0.10 | Insulation thickness |
| `k_shell` | float | W/mK | 25 | Shell thermal conductivity |
| `k_ins` | float | W/mK | 0.03 | Insulation thermal conductivity |
| `h_o` | float | W/m²K | 15 | Overall heat transfer coefficient |
| `D_b` | float | m | 0 | Borehole depth |
| `H_b` | float | m | 200 | Borehole height |
| `r_b` | float | m | 0.08 | Borehole radius |
| `R_b` | float | mK/W | 0.108 | Effective borehole thermal resistance |
| `dV_f` | float | L/min | 24 | Volumetric flow rate of fluid |
| `k_g` | float | W/mK | 2.0 | Ground thermal conductivity |
| `c_g` | float | J/(kgK) | 800 | Ground specific heat capacity |
| `rho_g` | float | kg/m³ | 2000 | Ground density |
| `E_pmp` | float | W | 200 | Pump power input |

#### Output Attributes

**Energy-related:**
- `E_cmp`: Compressor power [W]
- `E_pmp`: Pump power [W]
- `Q_r_tank`: Heat transfer from refrigerant to tank [W]
- `Q_r_exch`: Heat transfer from refrigerant in heat exchanger [W]
- `Q_bh`: Heat flow rate from borehole to ground per unit length [W/m]
- `Q_l_tank`: Heat loss from tank [W]

**Exergy-related:**
- `X_cmp`: Exergy of compressor input [W]
- `X_pmp`: Exergy of pump input [W]
- `X_r_int`: Exergy of refrigerant to tank [W]
- `X_r_exch`: Exergy of refrigerant in heat exchanger [W]
- `X_f_in`: Exergy of fluid entering heat exchanger [W]
- `X_f_out`: Exergy of fluid leaving heat exchanger [W]
- `X_g`: Exergy from ground [W]
- `X_b`: Exergy at borehole wall [W]
- `X_c_g`: Exergy consumed in ground [W]
- `X_c_GHE`: Exergy consumed in ground heat exchanger [W]
- `X_c_exch`: Exergy consumed in heat exchanger [W]
- `X_c_r`: Exergy consumed in refrigerant loop [W]
- `X_c_tank`: Exergy consumed in tank [W]
- `X_c_mix`: Exergy consumed in mixing valve [W]
- `X_eff`: Exergy efficiency [-]

**Temperature outputs:**
- `T_f`: Fluid temperature in borehole [K]
- `T_f_in`: Fluid inlet temperature [K]
- `T_f_out`: Fluid outlet temperature [K]
- `T_b`: Borehole wall temperature [K]
- `COP`: Calculated COP [-]

---

### AirSourceHeatPump_cooling

A model for an air-source heat pump in cooling mode.

#### Input Parameters

| Parameter | Type | Unit | Default | Description |
|-----------|------|------|---------|-------------|
| `fan_int` | Fan | - | Fan().fan1 | Internal fan object |
| `fan_ext` | Fan | - | Fan().fan3 | External fan object |
| `Q_r_max` | float | W | 9000 | Maximum cooling capacity |
| `T0` | float | °C | 32 | Environmental temperature |
| `T_a_room` | float | °C | 20 | Room air temperature |
| `T_r_int` | float | °C | T_a_room - 10 | Internal unit refrigerant temperature |
| `T_a_int_out` | float | °C | T_a_room - 5 | Internal unit air outlet temperature |
| `T_a_ext_out` | float | °C | T0 + 10 | External unit air outlet temperature |
| `T_r_ext` | float | °C | T0 + 15 | External unit refrigerant temperature |
| `Q_r_int` | float | W | 6000 | Indoor heat load |
| `COP_ref` | float | - | 4 | Reference COP at standard conditions |

#### Output Attributes

**Energy-related:**
- `COP`: Calculated COP [-]
- `E_cmp`: Compressor power [W]
- `E_fan_int`: Internal fan power [W]
- `E_fan_ext`: External fan power [W]
- `Q_r_ext`: Heat transfer from external unit to refrigerant [W]
- `dV_int`: Volumetric flow rate of internal unit [m³/s]
- `dV_ext`: Volumetric flow rate of external unit [m³/s]

**Exergy-related:**
- `X_a_int_in`: Exergy of air entering internal unit [W]
- `X_a_int_out`: Exergy of air leaving internal unit [W]
- `X_a_ext_in`: Exergy of air entering external unit [W]
- `X_a_ext_out`: Exergy of air leaving external unit [W]
- `X_r_int`: Exergy of refrigerant in internal unit [W]
- `X_r_ext`: Exergy of refrigerant in external unit [W]
- `X_c_int`: Exergy consumed in internal unit [W]
- `X_c_r`: Exergy consumed in refrigerant loop [W]
- `X_c_ext`: Exergy consumed in external unit [W]
- `X_c`: Total exergy consumed [W]
- `X_eff`: Exergy efficiency [-]

---

### AirSourceHeatPump_heating

A model for an air-source heat pump in heating mode.

#### Input Parameters

| Parameter | Type | Unit | Default | Description |
|-----------|------|------|---------|-------------|
| `fan_int` | Fan | - | Fan().fan1 | Internal fan object |
| `fan_ext` | Fan | - | Fan().fan3 | External fan object |
| `Q_r_max` | float | W | 9000 | Maximum heating capacity |
| `T0` | float | °C | 0 | Environmental temperature |
| `T_a_room` | float | °C | 20 | Room air temperature |
| `T_r_int` | float | °C | T_a_room + 15 | Internal unit refrigerant temperature |
| `T_a_int_out` | float | °C | T_a_room + 10 | Internal unit air outlet temperature |
| `T_a_ext_out` | float | °C | T0 - 5 | External unit air outlet temperature |
| `T_r_ext` | float | °C | T0 - 10 | External unit refrigerant temperature |
| `Q_r_int` | float | W | 6000 | Indoor heat load |

#### Output Attributes

Same as `AirSourceHeatPump_cooling` (see above).

---

### GroundSourceHeatPump_cooling

A model for a ground-source heat pump in cooling mode.

#### Input Parameters

| Parameter | Type | Unit | Default | Description |
|-----------|------|------|---------|-------------|
| `time` | float | h | 10 | Operating time |
| `D_b` | float | m | 0 | Borehole depth |
| `H_b` | float | m | 200 | Borehole height |
| `r_b` | float | m | 0.08 | Borehole radius |
| `R_b` | float | mK/W | 0.108 | Effective borehole thermal resistance |
| `dV_f` | float | L/min | 24 | Volumetric flow rate of fluid |
| `k_g` | float | W/mK | 2.0 | Ground thermal conductivity |
| `c_g` | float | J/(kgK) | 800 | Ground specific heat capacity |
| `rho_g` | float | kg/m³ | 2000 | Ground density |
| `E_pmp` | float | W | 200 | Pump power input |
| `fan_int` | Fan | - | Fan().fan1 | Internal fan object |
| `dT_r_exch` | float | K | 5 | Temperature difference |
| `T0` | float | °C | 32 | Environmental temperature |
| `T_g` | float | °C | 15 | Initial ground temperature |
| `T_a_room` | float | °C | 20 | Room air temperature |
| `T_r_exch` | float | °C | 25 | Heat exchanger side refrigerant temperature |
| `T_r_int` | float | °C | T_a_room - 10 | Internal unit refrigerant temperature |
| `T_a_int_out` | float | °C | T_a_room - 5 | Internal unit air outlet temperature |
| `Q_r_int` | float | W | 6000 | Cooling load |

#### Output Attributes

Similar to `GroundSourceHeatPumpBoiler` with additional internal unit outputs.

---

### GroundSourceHeatPump_heating

A model for a ground-source heat pump in heating mode.

#### Input Parameters

Same as `GroundSourceHeatPump_cooling` except:
- `dT_r_exch`: Default -5 K
- `T0`: Default 0 °C
- `T_r_exch`: Default 5 °C
- `T_r_int`: Default T_a_room + 15 °C
- `T_a_int_out`: Default T_a_room + 10 °C

#### Output Attributes

Same as `GroundSourceHeatPump_cooling`.

---

### ElectricHeater

A model for an electric heater with transient analysis.

#### Input Parameters

| Parameter | Type | Unit | Default | Description |
|-----------|------|------|---------|-------------|
| `c` | float | J/(kgK) | 500 | Specific heat capacity |
| `rho` | float | kg/m³ | 7800 | Density |
| `k` | float | W/mK | 50 | Thermal conductivity |
| `D` | float | m | 0.005 | Thickness |
| `H` | float | m | 0.8 | Height |
| `W` | float | m | 1.0 | Width |
| `E_heater` | float | W | 1000 | Electricity input |
| `T0` | float | °C | 0 | Reference temperature |
| `T_mr` | float | °C | 15 | Room surface temperature |
| `T_init` | float | °C | 20 | Initial heater temperature |
| `T_a_room` | float | °C | 20 | Indoor air temperature |
| `epsilon_hs` | float | - | 1 | Heater surface emissivity |
| `epsilon_rs` | float | - | 1 | Room surface emissivity |
| `dt` | float | s | 10 | Time step |

#### Output Attributes

**Time series lists** (after convergence):
- `time`: Time array [s]
- `T_hb_list`: Heater body temperature [K]
- `T_hs_list`: Heater surface temperature [K]
- `E_heater_list`: Electric power [W]
- `Q_st_list`: Storage heat [W]
- `Q_cond_list`: Conduction heat [W]
- `Q_conv_list`: Convection heat [W]
- `Q_rad_hs_list`: Radiation from heater surface [W]
- `Q_rad_rs_list`: Radiation from room surface [W]
- `S_*_list`: Entropy-related lists [W/K]
- `X_*_list`: Exergy-related lists [W]

**Final values:**
- `X_eff`: Exergy efficiency [-]

---

### Fan

A model for fan performance with curve fitting.

#### Input Parameters

The `Fan` class has predefined fan data:
- `fan1`: Centrifugal fan with flow rate, pressure, and efficiency data
- `fan2`: Centrifugal fan with flow rate, pressure, and efficiency data
- `fan3`: Axial fan with flow rate and power data

#### Methods

- `get_efficiency(fan, dV_fan)`: Get efficiency at given flow rate
- `get_pressure(fan, dV_fan)`: Get pressure at given flow rate
- `get_power(fan, dV_fan)`: Get power consumption at given flow rate
- `show_graph()`: Display performance curves

---

### Pump

A model for pump performance with curve fitting.

#### Input Parameters

The `Pump` class has predefined pump data:
- `pump1`: Pump with flow rate and efficiency data
- `pump2`: Pump with flow rate and efficiency data

#### Methods

- `get_efficiency(pump, dV_pmp)`: Get efficiency at given flow rate
- `get_power(pump, V_pmp, dP_pmp)`: Get power consumption for given flow rate and pressure difference
- `show_graph()`: Display performance curves

---

## Balance Dictionary Structure

All component models generate three types of balance dictionaries: `energy_balance`, `entropy_balance`, and `exergy_balance`.

### Structure

```python
balance = {
    "subsystem_name": {
        "in": {
            "symbol_1": value_1,  # [W] or [W/K] for entropy
            "symbol_2": value_2,
            ...
        },
        "out": {
            "symbol_3": value_3,
            ...
        },
        "con": {  # Only for exergy balance
            "symbol_4": value_4,  # Exergy consumed/destroyed
            ...
        },
        "gen": {  # Only for entropy balance
            "symbol_5": value_5,  # Entropy generated
            ...
        }
    },
    ...
}
```

### Energy Balance

- **Units**: [W]
- **Categories**: `in`, `out`
- **Example subsystems**: "hot water tank", "mixing valve", "combustion chamber", "external unit", "refrigerant loop"

### Entropy Balance

- **Units**: [W/K]
- **Categories**: `in`, `out`, `gen` (generated)
- **Example subsystems**: Same as energy balance

### Exergy Balance

- **Units**: [W]
- **Categories**: `in`, `out`, `con` (consumed/destroyed)
- **Example subsystems**: Same as energy balance

### Printing Balances

Use the `print_balance()` function to display balances:

```python
from enex_analysis import print_balance

# Print exergy balance with 2 decimal places
print_balance(boiler.exergy_balance, decimal=2)

# Print entropy balance with 3 decimal places
print_balance(boiler.entropy_balance, decimal=3)
```

---

## Utility Functions

### COP Calculation Functions

#### `calculate_ASHP_cooling_COP`

Calculate COP for air-source heat pump in cooling mode.

**Parameters:**
- `T_a_int_out` (float): Indoor air temperature [K]
- `T_a_ext_in` (float): Outdoor air temperature [K]
- `Q_r_int` (float): Indoor heat load [W]
- `Q_r_max` (float): Maximum cooling capacity [W]
- `COP_ref` (float): Reference COP at standard conditions [-]

**Returns:**
- `COP` (float): Calculated COP [-]

**Reference:** [IBPSA 2023](https://publications.ibpsa.org/proceedings/bs/2023/papers/bs2023_1118.pdf)

#### `calculate_ASHP_heating_COP`

Calculate COP for air-source heat pump in heating mode.

**Parameters:**
- `T0` (float): Environmental temperature [K]
- `Q_r_int` (float): Indoor heat load [W]
- `Q_r_max` (float): Maximum heating capacity [W]

**Returns:**
- `COP` (float): Calculated COP [-]

**Reference:** [MDPI Sustainability 2023](https://www.mdpi.com/2071-1050/15/3/1880)

#### `calculate_GSHP_COP`

Calculate Carnot-based COP for ground-source heat pump.

**Parameters:**
- `Tg` (float): Undisturbed ground temperature [K]
- `T_cond` (float): Condenser refrigerant temperature [K]
- `T_evap` (float): Evaporator refrigerant temperature [K]
- `theta_hat` (float): Dimensionless average fluid temperature [-]

**Returns:**
- `COP` (float): Modified Carnot-based COP [-]

**Reference:** [Energy 2019](https://www.sciencedirect.com/science/article/pii/S0360544219304347)

### Heat Transfer Functions

#### `calc_h_vertical_plate`

Calculate natural convection heat transfer coefficient for a vertical plate.

**Parameters:**
- `T_s` (float): Surface temperature [K]
- `T_inf` (float): Fluid temperature [K]
- `L` (float): Characteristic length [m]

**Returns:**
- `h_cp` (float): Heat transfer coefficient [W/m²K]

**Method:** Churchill & Chu correlation

#### `darcy_friction_factor`

Calculate Darcy friction factor for pipe flow.

**Parameters:**
- `Re` (float): Reynolds number [-]
- `e_d` (float): Relative roughness (e/D) [-]

**Returns:**
- `f` (float): Darcy friction factor [-]

**Method:** 
- Laminar (Re < 2300): f = 64/Re
- Turbulent: Swamee-Jain approximation

### Ground Heat Exchanger Functions

#### `G_FLS`

Calculate g-function for finite line source model (ground heat exchanger).

**Parameters:**
- `t` (float): Time [s]
- `ks` (float): Ground thermal conductivity [W/mK]
- `as_` (float): Thermal diffusivity [m²/s]
- `rb` (float): Borehole radius [m]
- `H` (float): Borehole height [m]

**Returns:**
- `g` (float): g-function [mK/W]

**Note:** Uses caching for performance optimization.

---

## Physical Constants

The following physical constants are defined in the module:

| Constant | Value | Unit | Description |
|----------|-------|------|-------------|
| `c_a` | 1005 | J/(kgK) | Specific heat capacity of air |
| `rho_a` | 1.225 | kg/m³ | Density of air |
| `k_a` | 0.0257 | W/(mK) | Thermal conductivity of air |
| `c_w` | 4186 | J/(kgK) | Specific heat capacity of water |
| `rho_w` | 1000 | kg/m³ | Density of water |
| `mu_w` | 0.001 | Pa·s | Dynamic viscosity of water |
| `k_w` | 0.606 | W/(mK) | Thermal conductivity of water |
| `sigma` | 5.67×10⁻⁸ | W/(m²K⁴) | Stefan-Boltzmann constant |
| `k_D` | 0.000462 | - | Direct solar entropy coefficient |
| `k_d` | 0.0014 | - | Diffuse solar entropy coefficient |
| `ex_eff_NG` | 0.93 | - | Exergy efficiency of natural gas |

---

## Notes

1. **Temperature Units**: Most input temperatures are in °C and are automatically converted to Kelvin internally. Output temperatures are in Kelvin unless otherwise specified.

2. **Flow Rates**: Water flow rates are typically input in L/min and converted to m³/s internally.

3. **Time**: Time inputs are typically in hours and converted to seconds internally.

4. **Reference Temperature**: The reference temperature `T0` is used for exergy calculations and should be set to the environmental temperature.

5. **Iterative Solutions**: Some models (e.g., `GroundSourceHeatPumpBoiler`) use iterative numerical methods to solve coupled equations. Convergence tolerance and maximum iterations are built into the models.

6. **Balance Verification**: Energy balances should satisfy conservation (in = out), while entropy and exergy balances include generation/consumption terms.

