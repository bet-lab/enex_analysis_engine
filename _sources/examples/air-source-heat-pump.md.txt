# Air Source Heat Pump

## Cooling Mode

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

## Heating Mode

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
