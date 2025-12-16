# Energy-Exergy Analysis Engine

**Comprehensive thermodynamic modeling for diverse energy systems through unified energy-exergy analysis**

A Python library that enables simultaneous energy (first-law) and exergy (second-law) analysis of various energy conversion systems. Built for researchers, engineers, and educators who need to understand not just how much energy flows through a system, but also the quality and potential of that energy.

---

## Why Energy-Exergy Analysis?

Traditional energy analysis tells you *how much* energy is used, but not *how well* it's used. Exergy analysis reveals the true thermodynamic efficiency by accounting for energy quality and identifying where irreversibilities occur. Together, energy-exergy analysis provides:

- **Complete thermodynamic picture**: Understand both quantity (energy) and quality (exergy) of energy flows
- **Inefficiency identification**: Pinpoint where and why energy is being destroyed
- **Technology comparison**: Fair comparison between different energy conversion technologies
- **Optimization guidance**: Identify the most promising areas for system improvement

This library makes energy-exergy analysis accessible by providing ready-to-use models for common energy systems, with automatic calculation of energy, entropy, and exergy balances.

---

## Core Capabilities

### Unified Balance Calculations

Every component model automatically calculates three balances:

1. **Energy Balance** (First Law of Thermodynamics): Identifies energy flows and losses


 - **Energy conservation:**

$$
\sum \dot{E}_{in} = \sum \dot{E}_{out} + \dot{E}_{loss} + \frac{dE_{system}}{dt}
$$

   - **Steady state:**

$$
\sum \dot{E}_{in} = \sum \dot{E}_{out} + \dot{E}_{loss}
$$


2. **Entropy Balance** (Second Law of Thermodynamics): Quantifies irreversibilities


- **Entropy transfer and generation:**

$$
\sum \dot{S}_{in} + \dot{S}_{gen} = \sum \dot{S}_{out} + \frac{dS_{system}}{dt}
$$

   - **Steady state:**

$$
\sum \dot{S}_{in} + \dot{S}_{gen} = \sum \dot{S}_{out}
$$


3. **Exergy Balance** (Both First and Second Law of Thermodynamics): Reveals thermodynamic inefficiencies


- **General form:**

$$
\sum \dot{X}_{in} = \sum \dot{X}_{out} + \dot{X}_{destroyed} + \frac{dX_{system}}{dt}
$$

   - **Steady state:**

$$
\sum \dot{X}_{in} = \sum \dot{X}_{out} + \dot{X}_{destroyed}
$$

   - **Exergy destruction:**

$$
\dot{X}_{destroyed} = T_0 \cdot \dot{S}_{gen}
$$


These balances are calculated consistently across all components, enabling system-level analysis and comparison.

### Analysis Features

- **Automatic balance calculation**: Set parameters, run `system_update()`, get all balances
- **Subsystem-level analysis**: Detailed balances for each subsystem (e.g., combustion chamber, heat exchanger, mixing valve)
- **Performance metrics**: Exergy efficiency, COP, and other key performance indicators
- **Visualization support**: Built-in functions for displaying balance results
- **Unit conversion utilities**: Comprehensive unit conversion functions for all common physical quantities

---

## Energy Systems Supported

The library provides models for a wide range of energy conversion systems, organized by application:

### Domestic Hot Water (DHW) Systems

Complete models for residential and commercial hot water production:

- **`ElectricBoiler`**: Electric resistance heating system with hot water storage tank
- **`GasBoiler`**: Natural gas combustion boiler with hot water storage
- **`HeatPumpBoiler`**: Air-source heat pump for hot water production
- **`SolarAssistedGasBoiler`**: Hybrid system combining solar thermal collectors with gas backup
- **`GroundSourceHeatPumpBoiler`**: Ground-source heat pump for hot water production

### Heat Pump Systems

Standalone heat pump models for space conditioning:

- **`AirSourceHeatPump_cooling`**: Air-source heat pump in cooling mode
- **`AirSourceHeatPump_heating`**: Air-source heat pump in heating mode
- **`GroundSourceHeatPump_cooling`**: Ground-source heat pump in cooling mode
- **`GroundSourceHeatPump_heating`**: Ground-source heat pump in heating mode

### Dynamic System Models

Time-dependent analysis for transient behavior:

- **`ElectricHeater`**: Dynamic heat transfer analysis for electric heating elements
- **`ElectricBoiler_Dynamic`**: Time-dependent hot water tank simulation

### Auxiliary Components

Supporting components for complete system modeling:

- **`Fan`**: Air handling fan with performance curves
- **`Pump`**: Fluid circulation pump with efficiency curves

---

## Quick Start

### Example 1: Electric Boiler System

```python
from enex_analysis_engine import ElectricBoiler, print_balance

# Initialize system
boiler = ElectricBoiler()

# Set operating conditions
boiler.T_w_tank = 60   # Tank temperature [°C]
boiler.T_w_sup = 10    # Supply water temperature [°C]
boiler.T_w_serv = 45   # Service water temperature [°C]
boiler.dV_w_serv = 1.2 # Flow rate [L/min]
boiler.T0 = 0          # Reference temperature [°C]

# Run analysis
boiler.system_update()

# View results
print(f"Energy input: {boiler.E_heater:.2f} W")
print(f"Exergy efficiency: {boiler.X_eff:.4f}")

# Print detailed balances
print("\n=== Energy Balance ===")
print_balance(boiler.energy_balance)

print("\n=== Exergy Balance ===")
print_balance(boiler.exergy_balance)
```

### Example 2: Gas Boiler System

```python
from enex_analysis_engine import GasBoiler, print_balance

# Initialize gas boiler
boiler = GasBoiler()

# Set parameters
boiler.T_w_tank = 60
boiler.T_w_sup = 10
boiler.T_w_serv = 45
boiler.dV_w_serv = 1.2
boiler.T0 = 0

# Run analysis
boiler.system_update()

# Compare energy vs exergy efficiency
print(f"Energy efficiency: {boiler.E_eff:.4f}")
print(f"Exergy efficiency: {boiler.X_eff:.4f}")
print(f"Natural gas exergy: {boiler.X_NG:.2f} W")
print(f"Exergy destruction: {boiler.X_c_tot:.2f} W")
```

### Example 3: Air-Source Heat Pump (Heating)

```python
from enex_analysis_engine import AirSourceHeatPump_heating

# Initialize heat pump
hp = AirSourceHeatPump_heating()

# Set conditions
hp.T0 = 0          # Outdoor temperature [°C]
hp.Q_r_int = 5000  # Heating load [W]
hp.Q_r_max = 8000  # Maximum capacity [W]

# Run analysis
hp.system_update()

# View performance
print(f"COP: {hp.COP:.2f}")
print(f"Compressor power: {hp.E_cmp:.2f} W")
print(f"Exergy efficiency: {hp.X_eff:.4f}")
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

---

## Installation

### Requirements

- Python >= 3.10
- `uv` package manager

### Installation Methods

This project uses `uv` for package management. To get started:

```bash
# 1) Install uv
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) Clone repository
git clone https://github.com/bet-lab/enex_analysis_engine.git
cd enex_analysis_engine

# 3) Install dependencies
uv sync
```

---

## Documentation

Comprehensive documentation is available:

- **[📚 Online Documentation](https://bet-lab.github.io/enex_analysis_engine/)**: Full API reference and user guide (Sphinx-generated)
- **[IO_DOCS.md](IO_DOCS.md)**: Complete input/output interface documentation for all components
- **[EXAMPLES.md](EXAMPLES.md)**: Detailed usage examples and tutorials for each component

The online documentation includes:
- Installation guide
- User guide with examples
- Complete API reference for all classes and functions
- Automatic documentation from docstrings

English versions are also available in the `docs/` folder.

---

## Project Structure

```
enex_analysis_engine/
├── src/
│   └── enex_analysis/
│       ├── __init__.py              # Package initialization
│       ├── calc_util.py              # Unit conversion utilities
│       ├── enex_engine.py            # Steady-state system models
│       └── enex_engine_dynamic.py    # Dynamic system models
├── docs/                             # English documentation
├── pyproject.toml                    # Project configuration
├── uv.lock                           # Dependency lock file
├── IO_DOCS.md                        # I/O documentation (Korean)
├── EXAMPLES.md                       # Usage examples (Korean)
└── README.md                         # This file
```

### Key Modules

- **`calc_util.py`**: Unit conversion constants and helper functions
  - Temperature: `C2K()`, `K2C()`
  - Time: `h2s`, `s2h`, `d2h`, etc.
  - Energy/Power: `J2kWh`, `W2kW`, etc.
  - Length, Area, Volume, Mass, Pressure, Angle conversions

- **`enex_engine.py`**: Steady-state system models
  - All component classes (boilers, heat pumps, auxiliaries)
  - COP calculation functions for heat pumps
  - Heat transfer coefficient calculations
  - g-function calculations for ground-source systems
  - Balance calculation utilities

- **`enex_engine_dynamic.py`**: Dynamic system models
  - Time-dependent simulations
  - Transient heat transfer analysis
  - Dynamic tank temperature calculations

---

## Dependencies

Core dependencies:

- `numpy`: Numerical computations
- `scipy`: Scientific computing (optimization, integration, special functions)
- `matplotlib`: Visualization
- `dartwork-mpl`: Plot styling (https://github.com/dartwork-repo/dartwork-mpl)
- `pandas`: Data processing
- `dataclasses`: Data class support (built-in)

For complete dependency list and versions, see `pyproject.toml`.

---
