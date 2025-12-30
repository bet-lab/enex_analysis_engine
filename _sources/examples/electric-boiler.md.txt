# Electric Boiler

## Basic Usage

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

## Parameter Study

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
