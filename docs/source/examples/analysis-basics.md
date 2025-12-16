# Analysis Basics

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
