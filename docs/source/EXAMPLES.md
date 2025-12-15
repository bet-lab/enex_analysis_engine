# Usage Examples

This document provides comprehensive usage examples for all components in the ENEX Analysis Engine.

## Table of Contents

1. [Electric Boiler](#electric-boiler)
2. [Gas Boiler](#gas-boiler)
3. [Heat Pump Boiler](#heat-pump-boiler)
4. [Solar-Assisted Gas Boiler](#solar-assisted-gas-boiler)
5. [Ground Source Heat Pump Boiler](#ground-source-heat-pump-boiler)
6. [Air Source Heat Pump (Cooling)](#air-source-heat-pump-cooling)
7. [Air Source Heat Pump (Heating)](#air-source-heat-pump-heating)
8. [Ground Source Heat Pump (Cooling)](#ground-source-heat-pump-cooling)
9. [Ground Source Heat Pump (Heating)](#ground-source-heat-pump-heating)
10. [Electric Heater](#electric-heater)
11. [Fan and Pump](#fan-and-pump)
12. [Unit Conversions](#unit-conversions)
13. [Balance Analysis](#balance-analysis)

---

## Electric Boiler

### Basic Usage

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

### Parameter Study

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

---

## Gas Boiler

### Basic Usage

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

### Efficiency Comparison

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

---

## Heat Pump Boiler

### Basic Usage

```python
from enex_analysis import HeatPumpBoiler, print_balance

# Initialize heat pump boiler
hp_boiler = HeatPumpBoiler()

# Set operating conditions
hp_boiler.COP = 2.5           # Coefficient of Performance
hp_boiler.eta_fan = 0.6       # External fan efficiency
hp_boiler.dP = 200            # Pressure difference [Pa]
hp_boiler.T0 = 0              # Reference temperature [°C]
hp_boiler.T_w_tank = 60       # Tank water temperature [°C]
hp_boiler.T_w_serv = 45       # Service water temperature [°C]
hp_boiler.T_w_sup = 10        # Supply water temperature [°C]
hp_boiler.dV_w_serv = 1.2     # Service water flow rate [L/min]

# Run calculation
hp_boiler.system_update()

# Access results
print(f"Compressor power: {hp_boiler.E_cmp:.2f} W")
print(f"External fan power: {hp_boiler.E_fan:.2f} W")
print(f"Total power input: {hp_boiler.E_cmp + hp_boiler.E_fan:.2f} W")
print(f"Exergy efficiency: {hp_boiler.X_eff:.4f}")

# Print exergy balance
print_balance(hp_boiler.exergy_balance)
```

### COP Sensitivity Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from enex_analysis import HeatPumpBoiler

# Study effect of COP on exergy efficiency
cop_values = np.linspace(2.0, 4.0, 21)
exergy_effs = []
total_power = []

hp_boiler = HeatPumpBoiler()
hp_boiler.T_w_tank = 60
hp_boiler.T_w_serv = 45
hp_boiler.T_w_sup = 10
hp_boiler.dV_w_serv = 1.2

for cop in cop_values:
    hp_boiler.COP = cop
    hp_boiler.system_update()
    exergy_effs.append(hp_boiler.X_eff)
    total_power.append(hp_boiler.E_cmp + hp_boiler.E_fan)

plt.figure(figsize=(10, 5))
plt.plot(cop_values, exergy_effs, 'b-', linewidth=2, label='Exergy Efficiency')
plt.xlabel('COP [-]')
plt.ylabel('Exergy Efficiency [-]')
plt.grid(True)
plt.legend()
plt.title('Effect of COP on Exergy Efficiency')
plt.show()
```

---

## Solar-Assisted Gas Boiler

### Basic Usage

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

### Solar Irradiance Study

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

---

## Ground Source Heat Pump Boiler

### Basic Usage

```python
from enex_analysis import GroundSourceHeatPumpBoiler, print_balance

# Initialize ground-source heat pump boiler
gshp_boiler = GroundSourceHeatPumpBoiler()

# Set ground properties
gshp_boiler.k_g = 2.0         # Ground thermal conductivity [W/mK]
gshp_boiler.c_g = 800         # Ground specific heat [J/(kgK)]
gshp_boiler.rho_g = 2000      # Ground density [kg/m³]
gshp_boiler.T_g = 11          # Undisturbed ground temperature [°C]

# Set borehole properties
gshp_boiler.H_b = 200         # Borehole height [m]
gshp_boiler.r_b = 0.08        # Borehole radius [m]
gshp_boiler.R_b = 0.108       # Borehole thermal resistance [mK/W]
gshp_boiler.dV_f = 24         # Fluid flow rate [L/min]
gshp_boiler.E_pmp = 200       # Pump power [W]

# Set operating conditions
gshp_boiler.time = 10         # Operating time [h]
gshp_boiler.T_w_tank = 60
gshp_boiler.T_w_serv = 45
gshp_boiler.T_w_sup = 10
gshp_boiler.dV_w_serv = 1.2

# Run calculation
gshp_boiler.system_update()

# Access results
print(f"Compressor power: {gshp_boiler.E_cmp:.2f} W")
print(f"Pump power: {gshp_boiler.E_pmp:.2f} W")
print(f"COP: {gshp_boiler.COP:.2f}")
print(f"Borehole heat flow: {gshp_boiler.Q_bh:.2f} W/m")
print(f"Fluid inlet temperature: {gshp_boiler.T_f_in - 273.15:.2f} °C")
print(f"Fluid outlet temperature: {gshp_boiler.T_f_out - 273.15:.2f} °C")
print(f"Exergy efficiency: {gshp_boiler.X_eff:.4f}")

# Print exergy balance
print_balance(gshp_boiler.exergy_balance)
```

### Time-Dependent Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from enex_analysis import GroundSourceHeatPumpBoiler

# Study effect of operating time on performance
times = np.array([1, 5, 10, 24, 48, 72, 168])  # hours
cops = []
fluid_temps = []

gshp_boiler = GroundSourceHeatPumpBoiler()
gshp_boiler.T_w_tank = 60
gshp_boiler.T_w_serv = 45
gshp_boiler.T_w_sup = 10
gshp_boiler.dV_w_serv = 1.2
gshp_boiler.k_g = 2.0
gshp_boiler.H_b = 200

for t in times:
    gshp_boiler.time = t
    gshp_boiler.system_update()
    cops.append(gshp_boiler.COP)
    fluid_temps.append(gshp_boiler.T_f_in - 273.15)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(times, cops, 'b-o', linewidth=2, markersize=8)
ax1.set_xlabel('Operating Time [h]')
ax1.set_ylabel('COP [-]')
ax1.grid(True)
ax1.set_title('COP vs Operating Time')

ax2.plot(times, fluid_temps, 'r-o', linewidth=2, markersize=8)
ax2.set_xlabel('Operating Time [h]')
ax2.set_ylabel('Fluid Inlet Temperature [°C]')
ax2.grid(True)
ax2.set_title('Fluid Temperature vs Operating Time')

plt.tight_layout()
plt.show()
```

---

## Air Source Heat Pump (Cooling)

### Basic Usage

```python
from enex_analysis import AirSourceHeatPump_cooling, print_balance

# Initialize air-source heat pump (cooling)
ashp_cool = AirSourceHeatPump_cooling()

# Set operating conditions
ashp_cool.T0 = 32              # Outdoor temperature [°C]
ashp_cool.T_a_room = 20        # Room temperature [°C]
ashp_cool.Q_r_int = 6000       # Cooling load [W]
ashp_cool.Q_r_max = 9000       # Maximum capacity [W]
ashp_cool.COP_ref = 4          # Reference COP

# Run calculation
ashp_cool.system_update()

# Access results
print(f"COP: {ashp_cool.COP:.2f}")
print(f"Compressor power: {ashp_cool.E_cmp:.2f} W")
print(f"Internal fan power: {ashp_cool.E_fan_int:.2f} W")
print(f"External fan power: {ashp_cool.E_fan_ext:.2f} W")
print(f"Total power: {ashp_cool.E_cmp + ashp_cool.E_fan_int + ashp_cool.E_fan_ext:.2f} W")
print(f"Exergy efficiency: {ashp_cool.X_eff:.4f}")

# Print exergy balance
print_balance(ashp_cool.exergy_balance)
```

### Outdoor Temperature Sensitivity

```python
import numpy as np
import matplotlib.pyplot as plt
from enex_analysis import AirSourceHeatPump_cooling

# Study effect of outdoor temperature on COP
outdoor_temps = np.linspace(25, 40, 16)
cops = []
powers = []

ashp_cool = AirSourceHeatPump_cooling()
ashp_cool.T_a_room = 20
ashp_cool.Q_r_int = 6000
ashp_cool.Q_r_max = 9000
ashp_cool.COP_ref = 4

for T_out in outdoor_temps:
    ashp_cool.T0 = T_out
    ashp_cool.system_update()
    cops.append(ashp_cool.COP)
    powers.append(ashp_cool.E_cmp + ashp_cool.E_fan_int + ashp_cool.E_fan_ext)

plt.figure(figsize=(10, 5))
plt.plot(outdoor_temps, cops, 'b-o', linewidth=2, markersize=6, label='COP')
plt.xlabel('Outdoor Temperature [°C]')
plt.ylabel('COP [-]')
plt.grid(True)
plt.legend()
plt.title('Effect of Outdoor Temperature on COP (Cooling Mode)')
plt.show()
```

---

## Air Source Heat Pump (Heating)

### Basic Usage

```python
from enex_analysis import AirSourceHeatPump_heating, print_balance

# Initialize air-source heat pump (heating)
ashp_heat = AirSourceHeatPump_heating()

# Set operating conditions
ashp_heat.T0 = 0               # Outdoor temperature [°C]
ashp_heat.T_a_room = 20        # Room temperature [°C]
ashp_heat.Q_r_int = 6000       # Heating load [W]
ashp_heat.Q_r_max = 9000       # Maximum capacity [W]

# Run calculation
ashp_heat.system_update()

# Access results
print(f"COP: {ashp_heat.COP:.2f}")
print(f"Compressor power: {ashp_heat.E_cmp:.2f} W")
print(f"Internal fan power: {ashp_heat.E_fan_int:.2f} W")
print(f"External fan power: {ashp_heat.E_fan_ext:.2f} W")
print(f"Exergy efficiency: {ashp_heat.X_eff:.4f}")

# Print exergy balance
print_balance(ashp_heat.exergy_balance)
```

### Heating Performance Curve

```python
import numpy as np
import matplotlib.pyplot as plt
from enex_analysis import AirSourceHeatPump_heating

# Study effect of outdoor temperature on heating COP
outdoor_temps = np.linspace(-10, 10, 21)
cops = []
exergy_effs = []

ashp_heat = AirSourceHeatPump_heating()
ashp_heat.T_a_room = 20
ashp_heat.Q_r_int = 6000
ashp_heat.Q_r_max = 9000

for T_out in outdoor_temps:
    ashp_heat.T0 = T_out
    ashp_heat.system_update()
    cops.append(ashp_heat.COP)
    exergy_effs.append(ashp_heat.X_eff)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(outdoor_temps, cops, 'b-o', linewidth=2, markersize=6)
ax1.set_xlabel('Outdoor Temperature [°C]')
ax1.set_ylabel('COP [-]')
ax1.grid(True)
ax1.set_title('COP vs Outdoor Temperature')

ax2.plot(outdoor_temps, exergy_effs, 'r-o', linewidth=2, markersize=6)
ax2.set_xlabel('Outdoor Temperature [°C]')
ax2.set_ylabel('Exergy Efficiency [-]')
ax2.grid(True)
ax2.set_title('Exergy Efficiency vs Outdoor Temperature')

plt.tight_layout()
plt.show()
```

---

## Ground Source Heat Pump (Cooling)

### Basic Usage

```python
from enex_analysis import GroundSourceHeatPump_cooling, print_balance

# Initialize ground-source heat pump (cooling)
gshp_cool = GroundSourceHeatPump_cooling()

# Set ground properties
gshp_cool.k_g = 2.0
gshp_cool.c_g = 800
gshp_cool.rho_g = 2000
gshp_cool.T_g = 15

# Set borehole properties
gshp_cool.H_b = 200
gshp_cool.r_b = 0.08
gshp_cool.R_b = 0.108
gshp_cool.dV_f = 24
gshp_cool.E_pmp = 200

# Set operating conditions
gshp_cool.time = 10
gshp_cool.T0 = 32
gshp_cool.T_a_room = 20
gshp_cool.Q_r_int = 6000

# Run calculation
gshp_cool.system_update()

# Access results
print(f"COP: {gshp_cool.COP:.2f}")
print(f"Compressor power: {gshp_cool.E_cmp:.2f} W")
print(f"Pump power: {gshp_cool.E_pmp:.2f} W")
print(f"Internal fan power: {gshp_cool.E_fan_int:.2f} W")
print(f"Exergy efficiency: {gshp_cool.X_eff:.4f}")

# Print exergy balance
print_balance(gshp_cool.exergy_balance)
```

---

## Ground Source Heat Pump (Heating)

### Basic Usage

```python
from enex_analysis import GroundSourceHeatPump_heating, print_balance

# Initialize ground-source heat pump (heating)
gshp_heat = GroundSourceHeatPump_heating()

# Set ground properties
gshp_heat.k_g = 2.0
gshp_heat.c_g = 800
gshp_heat.rho_g = 2000
gshp_heat.T_g = 15

# Set borehole properties
gshp_heat.H_b = 200
gshp_heat.r_b = 0.08
gshp_heat.R_b = 0.108
gshp_heat.dV_f = 24
gshp_heat.E_pmp = 200

# Set operating conditions
gshp_heat.time = 10
gshp_heat.T0 = 0
gshp_heat.T_a_room = 20
gshp_heat.Q_r_int = 6000

# Run calculation
gshp_heat.system_update()

# Access results
print(f"COP: {gshp_heat.COP:.2f}")
print(f"Compressor power: {gshp_heat.E_cmp:.2f} W")
print(f"Pump power: {gshp_heat.E_pmp:.2f} W")
print(f"Internal fan power: {gshp_heat.E_fan_int:.2f} W")
print(f"Exergy efficiency: {gshp_heat.X_eff:.4f}")

# Print exergy balance
print_balance(gshp_heat.exergy_balance)
```

---

## Electric Heater

### Basic Usage

```python
from enex_analysis import ElectricHeater, print_balance
import matplotlib.pyplot as plt

# Initialize electric heater
heater = ElectricHeater()

# Set heater properties
heater.E_heater = 1000        # Electric power [W]
heater.T0 = 0                 # Reference temperature [°C]
heater.T_a_room = 20          # Room air temperature [°C]
heater.T_mr = 15              # Room surface temperature [°C]
heater.T_init = 20            # Initial heater temperature [°C]
heater.dt = 10                # Time step [s]

# Run transient calculation
heater.system_update()

# Access final results
print(f"Final heater body temperature: {heater.T_hb - 273.15:.2f} °C")
print(f"Final heater surface temperature: {heater.T_hs - 273.15:.2f} °C")
print(f"Exergy efficiency: {heater.X_eff:.4f}")

# Plot transient behavior
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(heater.time, [T - 273.15 for T in heater.T_hb_list], 'b-', label='Body')
plt.plot(heater.time, [T - 273.15 for T in heater.T_hs_list], 'r-', label='Surface')
plt.xlabel('Time [s]')
plt.ylabel('Temperature [°C]')
plt.legend()
plt.grid(True)
plt.title('Temperature Evolution')

plt.subplot(2, 2, 2)
plt.plot(heater.time, heater.Q_conv_list, 'g-', label='Convection')
plt.plot(heater.time, heater.Q_rad_hs_list, 'm-', label='Radiation')
plt.xlabel('Time [s]')
plt.ylabel('Heat Transfer [W]')
plt.legend()
plt.grid(True)
plt.title('Heat Transfer Rates')

plt.subplot(2, 2, 3)
plt.plot(heater.time, heater.X_heater_list, 'k-', label='Input')
plt.plot(heater.time, [x + y for x, y in zip(heater.X_conv_list, heater.X_rad_hs_list)], 
         'b-', label='Output')
plt.xlabel('Time [s]')
plt.ylabel('Exergy [W]')
plt.legend()
plt.grid(True)
plt.title('Exergy Flows')

plt.subplot(2, 2, 4)
plt.plot(heater.time, heater.X_c_hb_list, 'r-', label='Body')
plt.plot(heater.time, heater.X_c_hs_list, 'b-', label='Surface')
plt.xlabel('Time [s]')
plt.ylabel('Exergy Consumed [W]')
plt.legend()
plt.grid(True)
plt.title('Exergy Destruction')

plt.tight_layout()
plt.show()

# Print final balances
print("\n=== Final Energy Balance ===")
print_balance(heater.energy_balance)

print("\n=== Final Exergy Balance ===")
print_balance(heater.exergy_balance)
```

---

## Fan and Pump

### Fan Performance

```python
from enex_analysis import Fan
import numpy as np
import matplotlib.pyplot as plt

# Initialize fan
fan = Fan()

# Get performance at different flow rates
flow_rates = np.linspace(0.5, 3.0, 26)
pressures = []
powers = []
efficiencies = []

for dV in flow_rates:
    try:
        p = fan.get_pressure(fan.fan1, dV)
        eff = fan.get_efficiency(fan.fan1, dV)
        power = fan.get_power(fan.fan1, dV)
        pressures.append(p)
        efficiencies.append(eff)
        powers.append(power)
    except:
        pressures.append(np.nan)
        efficiencies.append(np.nan)
        powers.append(np.nan)

# Plot performance curves
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(flow_rates, pressures, 'b-', linewidth=2)
axes[0].set_xlabel('Flow Rate [m³/s]')
axes[0].set_ylabel('Pressure [Pa]')
axes[0].grid(True)
axes[0].set_title('Fan Pressure vs Flow Rate')

axes[1].plot(flow_rates, powers, 'r-', linewidth=2)
axes[1].set_xlabel('Flow Rate [m³/s]')
axes[1].set_ylabel('Power [W]')
axes[1].grid(True)
axes[1].set_title('Fan Power vs Flow Rate')

axes[2].plot(flow_rates, efficiencies, 'g-', linewidth=2)
axes[2].set_xlabel('Flow Rate [m³/s]')
axes[2].set_ylabel('Efficiency [-]')
axes[2].grid(True)
axes[2].set_title('Fan Efficiency vs Flow Rate')

plt.tight_layout()
plt.show()

# Display performance graph
fan.show_graph()
```

### Pump Performance

```python
from enex_analysis import Pump
import numpy as np
import matplotlib.pyplot as plt

# Initialize pump
pump = Pump()

# Get performance at different flow rates
flow_rates = np.linspace(2, 6, 21)  # m³/h
dP = 100000  # Pressure difference [Pa]
powers = []
efficiencies = []

for V in flow_rates:
    V_m3s = V / 3600  # Convert to m³/s
    eff = pump.get_efficiency(pump.pump1, V_m3s)
    power = pump.get_power(pump.pump1, V_m3s, dP)
    efficiencies.append(eff)
    powers.append(power)

# Plot performance curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(flow_rates, powers, 'b-', linewidth=2)
ax1.set_xlabel('Flow Rate [m³/h]')
ax1.set_ylabel('Power [W]')
ax1.grid(True)
ax1.set_title('Pump Power vs Flow Rate')

ax2.plot(flow_rates, efficiencies, 'g-', linewidth=2)
ax2.set_xlabel('Flow Rate [m³/h]')
ax2.set_ylabel('Efficiency [-]')
ax2.grid(True)
ax2.set_title('Pump Efficiency vs Flow Rate')

plt.tight_layout()
plt.show()

# Display performance graph
pump.show_graph()
```

---

## Unit Conversions

### Temperature Conversions

```python
from enex_analysis import C2K, K2C

# Convert temperatures
T_C = 25
T_K = C2K(T_C)  # 298.15
T_C_back = K2C(T_K)  # 25.0

print(f"{T_C} °C = {T_K} K")
```

### Time Conversions

```python
from enex_analysis.calc_util import h2s, s2h, d2h

# Convert hours to seconds
hours = 2
seconds = hours * h2s  # 7200

# Convert seconds to hours
hours_back = seconds * s2h  # 2.0

# Convert days to hours
days = 1
hours_from_days = days * d2h  # 24

print(f"{hours} hours = {seconds} seconds")
print(f"{days} day = {hours_from_days} hours")
```

### Energy and Power Conversions

```python
from enex_analysis.calc_util import kWh2J, J2kWh, W2kW, kW2W

# Convert kWh to Joules
kwh = 1
joules = kwh * kWh2J  # 3,600,000

# Convert back
kwh_back = joules * J2kWh  # 1.0

# Convert power
watts = 1000
kilowatts = watts * W2kW  # 1.0

print(f"{kwh} kWh = {joules:,} J")
print(f"{watts} W = {kilowatts} kW")
```

---

## Balance Analysis

### Comparing Energy, Entropy, and Exergy Balances

```python
from enex_analysis import ElectricBoiler, print_balance

# Initialize and run boiler
boiler = ElectricBoiler()
boiler.T_w_tank = 60
boiler.T_w_sup = 10
boiler.T_w_serv = 45
boiler.dV_w_serv = 1.2
boiler.system_update()

# Print all balances
print("=" * 60)
print("ENERGY BALANCE")
print("=" * 60)
print_balance(boiler.energy_balance, decimal=2)

print("\n" + "=" * 60)
print("ENTROPY BALANCE")
print("=" * 60)
print_balance(boiler.entropy_balance, decimal=4)

print("\n" + "=" * 60)
print("EXERGY BALANCE")
print("=" * 60)
print_balance(boiler.exergy_balance, decimal=2)

# Verify energy conservation
for subsystem, balance in boiler.energy_balance.items():
    energy_in = sum(balance.get("in", {}).values())
    energy_out = sum(balance.get("out", {}).values())
    print(f"\n{subsystem}:")
    print(f"  Energy in: {energy_in:.2f} W")
    print(f"  Energy out: {energy_out:.2f} W")
    print(f"  Difference: {energy_in - energy_out:.2f} W")
```

### Exergy Destruction Analysis

```python
from enex_analysis import GasBoiler

# Initialize and run gas boiler
boiler = GasBoiler()
boiler.T_w_tank = 60
boiler.T_w_sup = 10
boiler.T_w_serv = 45
boiler.dV_w_serv = 1.2
boiler.system_update()

# Analyze exergy destruction in each subsystem
print("Exergy Destruction Analysis:")
print("-" * 60)

for subsystem, balance in boiler.exergy_balance.items():
    exergy_in = sum(balance.get("in", {}).values())
    exergy_out = sum(balance.get("out", {}).values())
    exergy_consumed = sum(balance.get("con", {}).values())
    
    print(f"\n{subsystem}:")
    print(f"  Exergy in: {exergy_in:.2f} W")
    print(f"  Exergy out: {exergy_out:.2f} W")
    print(f"  Exergy destroyed: {exergy_consumed:.2f} W")
    print(f"  Destruction ratio: {exergy_consumed/exergy_in*100:.2f}%")

print(f"\nTotal exergy destroyed: {boiler.X_c_tot:.2f} W")
print(f"System exergy efficiency: {boiler.X_eff:.4f}")
```

---

## Advanced Examples

### System Comparison

```python
from enex_analysis import ElectricBoiler, GasBoiler, HeatPumpBoiler
import pandas as pd

# Compare three boiler types
systems = {
    "Electric": ElectricBoiler(),
    "Gas": GasBoiler(),
    "Heat Pump": HeatPumpBoiler()
}

# Set common operating conditions
for name, system in systems.items():
    if hasattr(system, 'T_w_tank'):
        system.T_w_tank = 60
        system.T_w_sup = 10
        system.T_w_serv = 45
        system.dV_w_serv = 1.2
    if hasattr(system, 'COP'):
        system.COP = 2.5
    system.system_update()

# Create comparison table
comparison = []
for name, system in systems.items():
    if hasattr(system, 'E_heater'):
        power_input = system.E_heater
    elif hasattr(system, 'E_NG'):
        power_input = system.E_NG
    elif hasattr(system, 'E_cmp'):
        power_input = system.E_cmp + system.E_fan
    else:
        power_input = 0
    
    comparison.append({
        "System": name,
        "Power Input [W]": power_input,
        "Exergy Efficiency": system.X_eff,
        "Exergy Destroyed [W]": system.X_c_tot
    })

df = pd.DataFrame(comparison)
print(df.to_string(index=False))
```

### Parameter Optimization

```python
from enex_analysis import ElectricBoiler
from scipy.optimize import minimize_scalar
import numpy as np

def exergy_efficiency(tank_temp):
    """Calculate exergy efficiency for given tank temperature"""
    boiler = ElectricBoiler()
    boiler.T_w_tank = tank_temp
    boiler.T_w_sup = 10
    boiler.T_w_serv = 45
    boiler.dV_w_serv = 1.2
    boiler.system_update()
    return -boiler.X_eff  # Negative for minimization

# Find optimal tank temperature
result = minimize_scalar(exergy_efficiency, bounds=(50, 70), method='bounded')
optimal_temp = result.x
optimal_eff = -result.fun

print(f"Optimal tank temperature: {optimal_temp:.2f} °C")
print(f"Maximum exergy efficiency: {optimal_eff:.4f}")

# Verify
boiler = ElectricBoiler()
boiler.T_w_tank = optimal_temp
boiler.T_w_sup = 10
boiler.T_w_serv = 45
boiler.dV_w_serv = 1.2
boiler.system_update()
print(f"Power input at optimal: {boiler.E_heater:.2f} W")
```

---

## Tips and Best Practices

1. **Always call `system_update()`** after modifying parameters and before accessing results.

2. **Check input ranges**: Some parameters have physical limits (e.g., temperatures must be positive, flow rates must be positive).

3. **Unit consistency**: Be aware of input units (typically °C for temperatures, L/min for flow rates) and output units (typically K for temperatures, W for power).

4. **Iterative models**: Ground-source heat pump models use iterative solvers. If convergence issues occur, check input parameters.

5. **Balance verification**: Energy balances should satisfy conservation (in ≈ out), while entropy and exergy balances include generation/consumption terms.

6. **Performance**: For parameter studies, consider caching results or using vectorized operations where possible.

7. **Visualization**: Use `dartwork-mpl` for publication-quality plots as shown in the examples.

