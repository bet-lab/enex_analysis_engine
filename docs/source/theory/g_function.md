# Borehole and Coil Thermal Resistance (G-function)

> Module: `enex_analysis.g_function`

## Overview

The `g_function` module acts as a bridge between steady-state equivalent thermal resistances and dynamic thermal responses using finite line sources. In this engine architecture, the module supports both vertical geothermal boreholes and horizontal submerged coils targeting rivers or surface water bodies.

It leverages libraries like `pygfunction` to evaluate multiple borehole configurations and provides robust correlation-based solvers, such as the **Churchill-Bernstein** equation, for cross-flow advective environments.

## Feature 1: Borehole Thermal Resistance

For ground source heat pump systems, the transient response of the soil to heat injection/extraction is determined by the spatial superposition of the G-functions.

### `calc_borehole_thermal_resistance`

Evaluate the thermal resistance inside the borehole using the multipole or equivalent classical lines.

```python
from enex_analysis.g_function import calc_borehole_thermal_resistance

Rb = calc_borehole_thermal_resistance(
    r_b=0.075,
    r_out=0.016, 
    r_in=0.013, 
    D=0.05, 
    k_p=0.4, 
    k_g=2.0, 
    h_f=1000.0, 
    J=1
)
```
*Where `J` represents the multipole order (commonly `J=1` covers basic operations).*

## Feature 2: Submerged Coil Thermal Resistance

For water source heat pump systems deployed in rivers, the long-term heat accumulation in the "ground" is nullified because rivers behave as advective infinite heat capacity sinks. However, forced convective resistance over the coil pipe exterior remains crucial. 

### `calc_submerged_coil_thermal_resistance`

This method computes the resistance of a submerged coil subjected to external cross-flow (e.g. river stream) and internal forced flow (refrigerant or brine).

#### Churchill-Bernstein Heat Transfer
The outer convective heat transfer coefficient $h_{ext}$ is mapped by evaluating the Reynolds ($Re_D$) and Prandtl ($Pr$) numbers, and applying the **Churchill-Bernstein** correlation for a cylinder in cross-flow:

$$
  \overline{\text{Nu}}_D = 0.3 + \frac{0.62\,\text{Re}_D^{1/2}\,\text{Pr}^{1/3}}{\left[ 1 + (0.4/\text{Pr})^{2/3} \right]^{1/4}} \left[ 1 + \left( \frac{\text{Re}_D}{282000} \right)^{5/8} \right]^{4/5}
$$

Then $h_{ext} = \frac{\overline{\text{Nu}}_D \times k_f}{D_{out}}$. The internal convection uses the standard internal friction/Nusselt dependencies.

#### High Conductivity Approximation
The function seamlessly integrates with `pygfunction` by substituting the infinite unconstrained river boundary with an extremely massive pseudo-soil conductivity ($k \approx \infty$). This nullifies external soil resistances $R_g \rightarrow 0$ across the engine while perfectly isolating the precise 1D resistance $R_b$.

```python
from enex_analysis.g_function import calc_submerged_coil_thermal_resistance

Rb_submerged = calc_submerged_coil_thermal_resistance(
    v_river=0.5,            # River velocity [m/s]
    T_river_C=15.0,         # Water temp [°C]
    r_out=0.016,
    r_in=0.013,
    L_pipe=150.0,
    m_dot_brine=0.5         # Brine mass flow [kg/s]
)
```
