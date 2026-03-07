# Advanced Examples

## System Comparison

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

## Parameter Optimization

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
