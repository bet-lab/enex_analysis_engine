# Ground Source Heat Pump Boiler

## Basic Usage

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

## Time-Dependent Analysis

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
