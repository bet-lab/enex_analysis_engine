# Ground Source Heat Pump Refrigerant Cycle

For high-fidelity simulations, the `GroundSourceHeatPump_heating_RefCycle` and `GroundSourceHeatPump_cooling_RefCycle` models perform detailed internal thermodynamic cycle calculations instead of relying on a simple empirical COP curve.

## Advanced Cycle Simulation

These models solve the actual Log Mean Temperature Difference (LMTD) equations for the evaporator and condenser, combined with the compressor's isentropic efficiency equations and the expansion valve physics (enthalpy matching).

```python
from enex_analysis.ground_source_heat_pump_ref_cycle import GroundSourceHeatPump_heating_RefCycle

system = GroundSourceHeatPump_heating_RefCycle(
    # Specify the refrigerant
    refrigerant="R410A",
    
    # Advanced geometric and performance parameters
    UA_cond=5000,
    UA_evap=4000,
    # ...
)
```

This ensures that the exergy destruction rate can be accurately pinpointed to individual components inside the heat pump (e.g., compressor, expansion valve, condenser, evaporator) rather than treating the entire heat pump as a black box.
