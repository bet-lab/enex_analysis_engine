# ASHPB Hybrid Systems (PV/ESS/STC)

Energy-Exergy Analysis Engine provides advanced models for Air Source Heat Pump Boilers (ASHPB) coupled with renewable energy subsystems like Photovoltaics (PV), Energy Storage Systems (ESS), and Solar Thermal Collectors (STC).

## PV and ESS Integration
The `ASHPB_PV_ESS` class models a heat pump system powered by a grid-tied DC-coupled hybrid inverter with PV and Battery integration. The DC routing logic automatically prioritizes solar power for the compressor load, stores excess energy in the ESS, and covers shortfalls using the battery and grid.

```python
from enex_analysis.subsystems import PhotovoltaicSystem, EnergyStorageSystem
from enex_analysis.ashpb_pv_ess import ASHPB_PV_ESS

pv = PhotovoltaicSystem(
    area=10.0,
    eta_ref=0.2,
    # Add other required parameters
)
ess = EnergyStorageSystem(
    capacity_J=10 * 3600000, # 10 kWh
)

system = ASHPB_PV_ESS(
    pv=pv,
    ess=ess,
    # Configure heat pump parameters...
)
```

## STC Preheat Integration
The `ASHPB_STC_preheat` class incorporates a Solar Thermal Collector to preheat domestic hot water or space heating return water before it enters the heat pump, significantly improving the effective COP by reducing the temperature lift required by the compressor.

## STC Tank Integration
The `ASHPB_STC_tank` class models a system where a Solar Thermal Collector directly heats a stratified thermal storage tank. The heat pump acts as a supplementary heat source when the solar energy stored in the tank is insufficient to meet the demand.
