# Gas Boiler with Storage Tank

The `GasBoilerTank` model allows simulating a conventional gas boiler connected to a stratified thermal storage tank. This is commonly used for Domestic Hot Water (DHW) systems or space heating buffering where the boiler does not operate instantaneously to match the load, but rather cycles to maintain the tank temperature.

## Implementation Example

```python
from enex_analysis.gas_boiler_tank import GasBoilerTank

# Define the boiler parameters and tank properties
system = GasBoilerTank(
    LHV_gas=35.8e6,  # J/m3
    boiler_eta=0.9,  # 90% efficiency
    tank_volume=0.3, # 300 Liters
    V_dot_max=15.0 / 60000, # Flow rate
    # Additional required parameters...
)

# In a simulation loop:
# results = system.simulate_step(context, control_params...)
```

The tank physics relies on the `StratifiedTankTDMA` mathematical model to calculate internal temperature nodes dynamically.
