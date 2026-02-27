# Stratified Tank Model (TDMA)

> Module: `enex_analysis.Tank_stratification_model`

## Overview

1D vertically-stratified hot-water tank model solved with the Tri-Diagonal Matrix
Algorithm (TDMA). The model uses an effective thermal conductivity approach that
integrates molecular conduction and buoyancy-driven natural convection (Rayleigh–Nusselt)
into a single coefficient for each node pair.

## Physics

### Effective Conductivity

For each adjacent node pair, the effective conductivity is:

```
k_eff = k_molecular × Nu(Ra)
```

where the Nusselt number depends on stratification stability:

| Condition | Ra Range | Nu Correlation |
|---|---|---|
| Stable (upper warmer) | any | 1.0 + 0.1·(Ra/Ra_crit)^0.25 |
| Unstable, conduction | Ra < 10³ | 1.0 |
| Unstable, laminar | 10³ ≤ Ra < 10⁷ | 0.2·Ra^0.25 |
| Unstable, turbulent | Ra ≥ 10⁷ | 0.1·Ra^0.33 |

### Energy Balance Per Node

Each node solves a semi-implicit energy balance including:
- Storage (thermal capacitance)
- Effective axial conduction (K_eff between adjacent nodes)
- Advection from draw-off flow (inlet at bottom, outlet at top)
- Heat loss to ambient (per-node UA)
- Optional point heater
- Optional external loop (directed advection)

## Parameters

| Parameter | Unit | Description |
|---|---|---|
| `H` | m | Tank height |
| `N` | — | Number of vertical layers |
| `r0` | m | Inner radius |
| `x_shell` | m | Shell thickness |
| `x_ins` | m | Insulation thickness |
| `k_shell` | W/(m·K) | Shell conductivity |
| `k_ins` | W/(m·K) | Insulation conductivity |
| `h_w` | W/(m²·K) | Internal convective coefficient |
| `h_o` | W/(m²·K) | External convective coefficient |
| `C_d_mix` | — | Mixing discharge coefficient (legacy) |

## Usage

### Initialization

```python
from enex_analysis import StratifiedTankTDMA
import numpy as np

tank = StratifiedTankTDMA(
    H=1.2, N=10, r0=0.2,
    x_shell=0.005, x_ins=0.05,
    k_shell=25.0, k_ins=0.03,
    h_w=500.0, h_o=15.0,
    C_d_mix=0.1,
)

tank.info()   # Print geometry and thermal summary
```

### Single Time Step

```python
from enex_analysis.calc_util import C2K

T = np.full(10, C2K(60.0))   # Uniform 60 °C initial state
T_in = C2K(15.0)              # Mains water temperature
T_amb = C2K(20.0)             # Ambient temperature

T_next = tank.update_tank_temp(
    T=T, dt=60.0,
    T_in=T_in,
    dV_use=0.0001,            # Draw-off flow [m³/s]
    T_amb=T_amb, T0=T_amb,
    heater_node=2,            # 1-based heater position
    heater_capacity=3000.0,   # [W]
)
```

### With External Loop (e.g., Heat Pump Condenser)

```python
T_next = tank.update_tank_temp(
    T=T, dt=60.0,
    T_in=T_in, dV_use=0.0,
    T_amb=T_amb, T0=T_amb,
    loop_outlet_node=8,       # 1-based: water exits at node 8
    loop_inlet_node=2,        # 1-based: heated water returns at node 2
    dV_loop=0.0002,           # Loop flow rate [m³/s]
    Q_loop=5000.0,            # Heat input from external source [W]
)
```

## API Reference

| Method | Description |
|---|---|
| `effective_conductivity(T_upper, T_lower)` | Compute k_eff between two nodes [W/(m·K)] |
| `update_tank_temp(T, dt, T_in, dV_use, ...)` | Advance temperatures by one time step |
| `info(as_dict=False, precision=3)` | Print/return model summary |

## References

- Incropera & DeWitt, *Fundamentals of Heat and Mass Transfer*, 7th ed.
- Bejan, *Convection Heat Transfer*, 4th ed.
