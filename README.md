# ENEX Analysis Engine

Exergy-focused thermodynamic modeling for building energy systems

## Overview

**ENEX Analysis Engine** is a Python library designed for thermodynamic analysis of building energy systems. With a strong emphasis on second-law (exergy) analysis, it provides component models for various heating systems including electric boilers, gas boilers, heat pumps, and more. Each component consistently calculates energy, entropy, and exergy balances, enabling comprehensive thermodynamic evaluation.

This library is designed for educational, research, and prototyping purposes, making it particularly suitable for building energy system research where exergy analysis is crucial. Whether you're studying system efficiency, comparing different technologies, or developing new models, ENEX Analysis Engine provides a solid foundation for thermodynamic analysis.

## Key Features

### Component Models

The library offers a comprehensive set of component models for building energy systems:

- **Electric Boiler** (`ElectricBoiler`): Hot water system using electric heaters
- **Gas Boiler** (`GasBoiler`): Hot water system using natural gas combustion
- **Air-Source Heat Pump Boiler** (`HeatPumpBoiler`): Hot water system using air-source heat pumps
- **Solar-Assisted Gas Boiler** (`SolarAssistedGasBoiler`): Combined system with solar collectors and gas boiler
- **Ground-Source Heat Pump Boiler** (`GroundSourceHeatPumpBoiler`): Hot water system using ground-source heat pumps
- **Air-Source Heat Pumps** (`AirSourceHeatPump_cooling`, `AirSourceHeatPump_heating`): Air-source heat pumps for cooling and heating applications
- **Ground-Source Heat Pumps** (`GroundSourceHeatPump_cooling`, `GroundSourceHeatPump_heating`): Ground-source heat pumps for cooling and heating applications
- **Electric Heater** (`ElectricHeater`): Dynamic heat transfer analysis for electric heaters
- **Auxiliary Devices**: Fan (`Fan`) and Pump (`Pump`) models

### Thermodynamic Analysis

Each component automatically calculates three types of balances:

1. **Energy Balance**: Energy conservation according to the first law of thermodynamics
2. **Entropy Balance**: Entropy generation and transfer analysis
3. **Exergy Balance**: Exergy destruction analysis according to the second law of thermodynamics

These balances provide a complete picture of system performance from both first-law and second-law perspectives, enabling you to identify inefficiencies and optimization opportunities.

### Utility Functions

The library includes a rich set of utility functions to support your analysis:

- **Unit conversion functions**: Temperature, time, length, energy, power, and more
- **COP (Coefficient of Performance) calculation functions**: For various heat pump systems
- **g-function calculations**: For ground-source heat exchange systems
- **Natural convection heat transfer coefficient calculations**: For thermal analysis
- **Balance result output and visualization**: Easy-to-use functions for displaying results

## Installation

### Requirements

- Python >= 3.10
- `uv` package manager (recommended) or `pip`

### Installation Methods

#### Option A: Local Development Environment (Recommended for Contributors)

This method is recommended if you plan to contribute to the project or need the latest development version.

```bash
# 1) Install uv (Windows PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# For Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) Clone the repository
git clone https://github.com/BET-lab/enex_analysis_engine.git
cd enex_analysis_engine

# 3) Create virtual environment and sync dependencies
uv sync
```

#### Option B: Package Installation

```bash
# Install package (when available on PyPI)
pip install enex-analysis
```

## Quick Start

### Basic Usage Example

Here's a simple example to get you started with an electric boiler system:

```python
from enex_analysis import ElectricBoiler

# Create an electric boiler system
boiler = ElectricBoiler()

# Set system parameters
boiler.T_w_tank = 60  # Tank hot water temperature [°C]
boiler.T_w_sup = 10   # Supply water temperature [°C]
boiler.T_w_serv = 45  # Service water temperature [°C]
boiler.dV_w_serv = 1.2  # Service water flow rate [L/min]

# Run system calculation
boiler.system_update()

# Check results
print(f"Electric heater input power: {boiler.E_heater:.2f} W")
print(f"Exergy efficiency: {boiler.X_eff:.3f}")

# Print exergy balance
from enex_analysis import print_balance
print_balance(boiler.exergy_balance)
```

### Example Output

```
HOT WATER TANK EXERGY BALANCE: =====================

IN ENTRIES:
E_heater: 5234.56 [W]
X_w_sup_tank: 123.45 [W]

OUT ENTRIES:
X_w_tank: 4567.89 [W]
X_l_tank: 234.56 [W]

CONSUMED ENTRIES:
X_c_tank: 555.56 [W]
```

## Project Structure

```
enex_analysis_engine/
├── src/
│   └── enex_analysis/
│       ├── __init__.py          # Package initialization
│       ├── calc_util.py          # Unit conversion utilities
│       ├── enex_engine.py        # Steady-state system models
│       └── enex_engine_dynamic.py # Dynamic system models
├── pyproject.toml                # Project configuration
├── uv.lock                       # Dependency lock file
└── README.md                     # This file
```

### Key Modules

- **`calc_util.py`**: Unit conversion constants and helper functions
  - Temperature conversion: `C2K()`, `K2C()`
  - Time conversion: `h2s`, `s2h`, etc.
  - Energy/power conversion: `J2kWh`, `W2kW`, etc.

- **`enex_engine.py`**: Steady-state system models
  - All boiler and heat pump component classes
  - COP calculation functions
  - Helper functions (heat transfer coefficients, g-functions, etc.)

- **`enex_engine_dynamic.py`**: Dynamic system models
  - Time-dependent simulation models that account for transient behavior

## Core Concepts

### What is Exergy?

Exergy represents the maximum useful work that can be obtained from a system as it reaches equilibrium with its surrounding environment. Exergy analysis is used to evaluate the "quality" of energy and analyze system efficiency from a second-law thermodynamics perspective.

Unlike energy, which is conserved, exergy is destroyed in real processes due to irreversibilities. Exergy destruction (consumption) is a measure of irreversibility and is calculated using the following relationship:

$$X_c = T_0 \cdot S_g$$

where:
- $X_c$: Exergy destruction [W]
- $T_0$: Reference temperature [K]
- $S_g$: Entropy generation rate [W/K]

### Exergy Efficiency

Exergy efficiency is defined as the ratio of useful exergy output to exergy input:

$$\eta_{ex} = \frac{X_{out}}{X_{in}}$$

This value ranges between 0 and 1, with values closer to 1 indicating a more thermodynamically efficient system. Unlike energy efficiency, exergy efficiency accounts for the quality of energy, making it a more meaningful metric for comparing different energy conversion technologies.

## Documentation

For detailed information, please refer to the following documentation:

- **[IO_DOCS.md](IO_DOCS.md)**: Detailed input/output interface documentation for all components
- **[EXAMPLES.md](EXAMPLES.md)**: Comprehensive usage examples and tutorials

English versions of these documents are also available in the `docs/` folder.

## Dependencies

The library relies on the following key packages:

- `numpy`: Numerical computations
- `scipy`: Scientific computing (optimization, integration, etc.)
- `matplotlib`: Visualization
- `dartwork-mpl`: Plot styling (https://github.com/dartwork-repo/dartwork-mpl)
- `pandas`: Data processing
- `dataclasses`: Data class support (built-in with Python 3.7+)

For the complete list of dependencies and their versions, please refer to `pyproject.toml`.

## License

[License information to be added]

## Contributing

Contributions are welcome! We encourage you to:

- Report bugs by opening an issue
- Suggest new features or improvements
- Submit pull requests with your contributions

Please feel free to reach out if you have any questions or ideas for improving the library.

## References

This library is based on the following research and theoretical foundations:

- Shukuya, M. (2013). *Exergy theory and applications in the built environment*. Springer.
- Ground-source heat pump COP calculation: [Research paper](https://www.sciencedirect.com/science/article/pii/S0360544219304347)
- Air-source heat pump COP calculation: [IBPSA paper](https://publications.ibpsa.org/proceedings/bs/2023/papers/bs2023_1118.pdf)

## Contact

Project Maintainer: Habin Jo (habinjo0608@gmail.com)
