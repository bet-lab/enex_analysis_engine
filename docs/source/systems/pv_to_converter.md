# PV-to-Converter System

> Module: `enex_analysis.PV_to_Converter`

## Overview

Four-stage photovoltaic power conversion model with full energy, entropy, and
exergy accounting at each conversion stage:

```
  Solar Radiation → PV Cell → Controller → Battery → DC/AC Converter → AC Output
```

Each stage introduces efficiency losses and irreversibility (entropy generation),
which are tracked to quantify exergy destruction across the entire chain.

## Stage Model

| Stage | Component | Key Variables | Typical η |
|---|---|---|---|
| 0 | Solar Input | `I_sol`, `X_sol` | — |
| 1 | PV Cell | `E_pv0`, `X_pv0`, `X_c_pv` | 17–25 % |
| 2 | Controller | `E_pv1`, `X_c_ctrl` | 98–99.5 % |
| 3 | Battery | `E_pv2`, `X_c_batt` | 90–98 % |
| 4 | DC/AC Inverter | `E_pv3`, `X_c_DC_AC` | 95–99 % |

## Parameters

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `A_pv` | 5.0 | m² | Panel surface area |
| `alp_pv` | 0.9 | — | Surface absorptivity |
| `I_DN` | 500.0 | W/m² | Direct normal irradiance |
| `I_dH` | 150.0 | W/m² | Diffuse horizontal irradiance |
| `h_o` | 15.0 | W/(m²·K) | Outdoor heat transfer coefficient |
| `eta_pv` | 0.20 | — | PV panel efficiency |
| `eta_ctrl` | 0.95 | — | Controller efficiency |
| `eta_batt` | 0.90 | — | Battery efficiency |
| `eta_DC_AC` | 0.95 | — | DC/AC inverter efficiency |
| `T0_C` | 20 | °C | Ambient temperature |

## Usage

```python
from enex_analysis import PV_to_Converter

pv = PV_to_Converter()
pv.I_DN = 800.0      # Override irradiance
pv.I_dH = 200.0
pv.system_update()    # Run all four stages

print(f"PV output:     {pv.E_pv0:.1f} W")
print(f"AC output:     {pv.E_pv3:.1f} W")
print(f"Solar exergy:  {pv.X_sol:.1f} W")
print(f"PV exergy destruction: {pv.X_c_pv:.1f} W")
print(f"Total AC exergy:       {pv.X_pv3:.1f} W")
```

## API Reference

| Method | Description |
|---|---|
| `system_update()` | Compute energy / entropy / exergy for all 4 stages |
