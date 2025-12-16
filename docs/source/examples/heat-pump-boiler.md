# Heat Pump Boiler

## Basic Usage

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

## COP Sensitivity Analysis

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
