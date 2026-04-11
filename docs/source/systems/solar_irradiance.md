# Solar Irradiance Models & Anomalies

> This guide explains the fundamental solar irradiance metrics used in the engine and addresses common mathematical anomalies encountered during simulation.

## 1. Fundamental Solar Irradiance Components

The engine relies on three primary solar irradiance metrics, all of which measure power per unit area ($\mathrm{W/m^2}$).

- **GHI (Global Horizontal Irradiance)**: The total amount of shortwave radiation received from above by a surface horizontal to the ground. This is the primary measured value available from the Korea Meteorological Administration (KMA).
- **DNI ($I_{DN}$, Direct Normal Irradiance)**: The amount of solar radiation received per unit area by a surface that is always held **perpendicular (normal)** to the incoming rays from the sun. It only includes radiation directly from the sun, not scattered by the atmosphere.
- **DHI ($I_{dH}$, Diffuse Horizontal Irradiance)**: The amount of solar radiation received per unit area by a surface (not subject to shade or shadow) that does not arrive on a direct path from the sun, but has been scattered by the atmosphere.

> **Note**: $I_{DN}$ and $I_{dH}$ are raw sky components. They are **not** the final irradiance on a specific tilted solar panel (Plane of Array, POA irradiance). To calculate POA irradiance, the engine uses these components along with the panel's tilt and azimuth.

## 2. Calculation Methodology

When only GHI is available (e.g., KMA data), the engine uses empirical models to divide GHI into its direct (DNI) and diffuse (DHI) components. 
We utilize the **Erbs Model** via `pvlib.irradiance.erbs` to perform this separation:

1. Calculate the solar zenith angle ($Z$) for the specific location and time.
2. Estimate the diffuse fraction ($DHI / GHI$) based on the clearness index.
3. Extract DHI, then calculate DNI using the geometric relationship:
   $$ I_{DN} = \frac{GHI - I_{dH}}{\cos(Z)} $$

## 3. Mathematical Anomaly: DNI > 3000 W/m²

### The Phenomenon
During simulations, you may observe $I_{DN}$ values momentarily spiking to mathematically huge numbers, such as > 3,000 W/m² or even higher. 

### Physical Reality
This is **physically impossible**. The absolute maximum possible DNI on Earth's surface cannot exceed the Extraterrestrial Solar Irradiance (the Solar Constant), which is approximately **1,361 W/m²**. At the surface, due to atmospheric attenuation, clear-sky DNI typically reaches a maximum of around 1,000-1,100 W/m² depending on longitude, latitude, and weather.

### Root Cause
This error is a well-known mathematical artifact of the geometric equation:
$$ I_{DN} = \frac{GHI - I_{dH}}{\cos(Z)} $$

At **sunrise and sunset**, the sun is very low on the horizon, meaning the solar zenith angle ($Z$) approaches 90°. 
As $Z \to 90^\circ$, $\cos(Z) \to 0$. 

If there is even a slight temporal discrepancy between the measured GHI (often an hourly average) and the instantaneously calculated solar zenith position, the numerator $(GHI - I_{dH})$ might be a non-zero positive number while the denominator $\cos(Z)$ is infinitesimally small. Dividing a small remainder by a near-zero value causes the DNI calculation to blow up to mathematically infinite or wildly unrealistic values.

### Mitigation
To prevent this anomaly from corrupting downstream models (like Solar Thermal Collectors or PV arrays), the most robust and widely used approach in the building energy simulation industry is **Elevation Cut-off (Zenith Check)**.

When the sun is very low on the horizon (e.g., Zenith angle > 85°), we force DNI to 0 and assume all measured GHI is entirely diffuse (DHI = GHI). At this angle, direct solar radiation is virtually negligible due to high atmospheric mass and horizon shading, rendering the mathematical division-by-zero artifacts irrelevant.

```python
import numpy as np
import pvlib

zenith_arr = np.asarray(solar_position['zenith'])
ghi_arr = np.asarray(df_slice['ghi'])

# 1. Separate GHI into DNI and DHI components
dni_dhi = pvlib.irradiance.erbs(ghi_arr, zenith_arr, np.asarray(times.dayofyear))
dni_arr = np.nan_to_num(dni_dhi['dni'], nan=0.0)
dhi_arr = np.nan_to_num(dni_dhi['dhi'], nan=0.0)

# 2. Apply Elevation Cut-off mask for Sun at horizon (Zenith > 85 deg)
low_sun_mask = zenith_arr > 85.0
dni_arr[low_sun_mask] = 0.0
dhi_arr[low_sun_mask] = ghi_arr[low_sun_mask]
```
