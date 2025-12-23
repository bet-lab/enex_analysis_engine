# Solar-Assisted Gas Boiler

## Basic Usage

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

## Solar Irradiance Study

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
