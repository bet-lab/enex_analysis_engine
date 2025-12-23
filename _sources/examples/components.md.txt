# Components

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
