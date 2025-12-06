"""
Physical constants used in energy, entropy, and exergy analysis.

This module contains all physical constants including material properties,
thermodynamic constants, and system-specific parameters.
"""

import numpy as np

# Air properties
c_a = 1005  # Specific heat capacity of air [J/kgK]
rho_a = 1.225  # Density of air [kg/m³]
k_a = 0.0257  # Thermal conductivity of air [W/mK]

# Water properties
c_w = 4186  # Water specific heat [J/kgK]
rho_w = 1000  # Density of water [kg/m³]
mu_w = 0.001  # Water dynamic viscosity [Pa.s]
k_w = 0.606  # Water thermal conductivity [W/mK]

# Thermodynamic constants
sigma = 5.67 * 10**-8  # Stefan-Boltzmann constant [W/m²K⁴]

# Solar entropy coefficients
# Reference: https://www.notion.so/betlab/Scattering-of-photon-particles-coming-from-the-sun-and-their-energy-entropy-exergy-b781821ae9a24227bbf1a943ba9df51a?pvs=4#1ea6947d125d80ddb0a5caec50031ae3
k_D = 0.000462  # Direct solar entropy coefficient [-]
k_d = 0.0014  # Diffuse solar entropy coefficient [-]

# Natural gas exergy efficiency
# Reference: Shukuya - Exergy theory and applications in the built environment, 2013
# The ratio of chemical exergy to higher heating value of liquefied natural gas (LNG) is 0.93.
ex_eff_NG = 0.93  # Exergy efficiency of natural gas [-]

# Mathematical constants
SP = np.sqrt(np.pi)  # Square root of pi

