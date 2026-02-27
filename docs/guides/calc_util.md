# Unit Conversion Utilities

> Module: `enex_analysis.calc_util`

This module provides temperature conversion functions and a comprehensive set of
unit conversion multipliers following the `<from>2<to>` naming convention.

## Temperature Conversion Functions

| Function | Conversion | Example |
|---|---|---|
| `K2C(K)` | Kelvin → Celsius | `K2C(293.15)` → `20.0` |
| `C2K(C)` | Celsius → Kelvin | `C2K(20.0)` → `293.15` |
| `F2C(F)` | Fahrenheit → Celsius | `F2C(68.0)` → `20.0` |
| `C2F(C)` | Celsius → Fahrenheit | `C2F(20.0)` → `68.0` |

## Conversion Constants Catalog

### Time

| Constant | Value | Meaning |
|---|---|---|
| `d2h`, `d2m`, `d2s` | 24, 1440, 86400 | Day → hours, minutes, seconds |
| `h2d`, `m2d`, `s2d` | 1/24, ... | Hours/min/sec → days |
| `h2m`, `h2s` | 60, 3600 | Hour → minutes, seconds |
| `m2h`, `s2h` | 1/60, 1/3600 | Minutes/sec → hours |
| `m2s`, `s2m` | 60, 1/60 | Minutes ↔ seconds |
| `y2d`, `d2y` | 365, 1/365 | Year ↔ days |

### Length

| Constant | Value | Meaning |
|---|---|---|
| `m2cm`, `cm2m` | 100, 0.01 | Meters ↔ centimeters |
| `m2mm`, `mm2m` | 1000, 0.001 | Meters ↔ millimeters |
| `m2km`, `km2m` | 0.001, 1000 | Meters ↔ kilometers |
| `in2cm`, `cm2in` | 2.54, ... | Inches ↔ centimeters |
| `ft2m`, `m2ft` | 0.3048, ... | Feet ↔ meters |

### Area

| Constant | Value | Meaning |
|---|---|---|
| `m22cm2`, `cm22m2` | 1e4, 1e-4 | m² ↔ cm² |
| `m22mm2`, `mm22m2` | 1e6, 1e-6 | m² ↔ mm² |

### Volume

| Constant | Value | Meaning |
|---|---|---|
| `m32L`, `L2m3` | 1000, 0.001 | m³ ↔ liters |
| `m32cm3`, `cm32m3` | 1e6, 1e-6 | m³ ↔ cm³ |

### Mass

| Constant | Value | Meaning |
|---|---|---|
| `kg2g`, `g2kg` | 1e3, 1e-3 | kg ↔ grams |
| `kg2t`, `t2kg` | 1e-3, 1e3 | kg ↔ metric tons |

### Energy

| Constant | Value | Meaning |
|---|---|---|
| `J2kJ`, `kJ2J` | 1e-3, 1e3 | Joules ↔ kilojoules |
| `J2MJ`, `MJ2J` | 1e-6, 1e6 | Joules ↔ megajoules |
| `kWh2J`, `J2kWh` | 3.6e6, ... | kWh ↔ Joules |

### Power

| Constant | Value | Meaning |
|---|---|---|
| `W2kW`, `kW2W` | 1e-3, 1e3 | Watts ↔ kilowatts |
| `W2MW`, `MW2W` | 1e-6, 1e6 | Watts ↔ megawatts |

### Pressure

| Constant | Value | Meaning |
|---|---|---|
| `Pa2kPa`, `kPa2Pa` | 1e-3, 1e3 | Pa ↔ kPa |
| `Pa2bar`, `bar2Pa` | 1e-5, 1e5 | Pa ↔ bar |
| `atm2Pa`, `Pa2atm` | 101325, ... | atm ↔ Pa |

### Angle

| Constant | Value | Meaning |
|---|---|---|
| `d2r`, `r2d` | π/180, 180/π | Degrees ↔ radians |

## Usage

```python
from enex_analysis.calc_util import C2K, h2s, L2m3

# Convert 60 °C to Kelvin
T_K = C2K(60.0)    # 333.15 K

# Convert 2 hours to seconds
t_sec = 2 * h2s    # 7200 s

# Convert 200 L/min to m³/s
dV = 200 * L2m3 / 60   # 0.00333 m³/s
```
