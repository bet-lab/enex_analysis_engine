# Gas Boiler

## Basic Usage

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

## Efficiency Comparison

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
