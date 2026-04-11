# Physical Constants Reference

> Module: `enex_analysis.constants`

This module defines all physical constants used across the `enex_analysis` package.

## Constants Catalog

### Air Properties

| Constant | Symbol | Value | Unit | Description |
|---|---|---|---|---|
| `c_a` | c_a | 1005 | J/(kg·K) | Specific heat capacity of air |
| `rho_a` | ρ_a | 1.225 | kg/m³ | Density of air |
| `k_a` | k_a | 0.0257 | W/(m·K) | Thermal conductivity of air |

### Water Properties

| Constant | Symbol | Value | Unit | Description |
|---|---|---|---|---|
| `c_w` | c_w | 4186 | J/(kg·K) | Specific heat capacity of water |
| `rho_w` | ρ_w | 1000 | kg/m³ | Density of water |
| `mu_w` | μ_w | 0.001 | Pa·s | Dynamic viscosity of water |
| `k_w` | k_w | 0.606 | W/(m·K) | Thermal conductivity of water |
| `beta` | β | 2.07×10⁻⁴ | 1/K | Volumetric expansion coefficient (at ~20 °C) |

### Physical Constants

| Constant | Symbol | Value | Unit | Description |
|---|---|---|---|---|
| `g` | g | 9.81 | m/s² | Gravitational acceleration |
| `sigma` | σ | 5.67×10⁻⁸ | W/(m²·K⁴) | Stefan-Boltzmann constant |

### Solar Entropy Coefficients

| Constant | Symbol | Value | Unit | Description | Reference |
|---|---|---|---|---|---|
| `k_D` | k_D | 0.000462 | — | Direct solar entropy coefficient | Petela (2003) |
| `k_d` | k_d | 0.0014 | — | Diffuse solar entropy coefficient | Petela (2003) |

### Natural Gas Properties

| Constant | Symbol | Value | Unit | Description | Reference |
|---|---|---|---|---|---|
| `ex_eff_NG` | — | 0.93 | — | Chemical-exergy-to-HHV ratio of LNG | Shukuya (2013) |

### Mathematical Constants

| Constant | Symbol | Value | Description |
|---|---|---|---|
| `SP` | √π | 1.7725 | Square root of π |

## Usage

```python
from enex_analysis.constants import c_w, rho_w, g

# Calculate water thermal capacitance for a 100 L tank
V_tank = 0.1   # m³
C_tank = c_w * rho_w * V_tank   # [J/K]
print(f"Tank capacitance: {C_tank:.0f} J/K")
# → Tank capacitance: 418600 J/K
```
