# Ground Source Heat Pump

## Cooling Mode

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

## Heating Mode

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
