# Energy-Exergy Analysis Engine

**Comprehensive thermodynamic modeling for diverse energy systems through unified energy-exergy analysis**

A Python library that enables simultaneous energy (first-law) and exergy (second-law) analysis of various energy conversion systems. Built for researchers, engineers, and educators who need to understand not just how much energy flows through a system, but also the quality and potential of that energy.

---

## Why Energy-Exergy Analysis?

Traditional energy analysis tells you _how much_ energy is used, but not _how well_ it's used. Exergy analysis reveals the true thermodynamic efficiency by accounting for energy quality and identifying where irreversibilities occur. Together, energy-exergy analysis provides:

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
2. **Entropy Balance** (Second Law of Thermodynamics): Quantifies irreversibilities
3. **Exergy Balance** (Both First and Second Law of Thermodynamics): Reveals thermodynamic inefficiencies

These balances are calculated consistently across all components, enabling system-level analysis and comparison.

### Analysis Features

- **Automatic balance calculation**: Set parameters, run `system_update()`, get all balances
- **Steady-State & Dynamic Simulation**: Advanced models support `analyze_steady()` and `analyze_dynamic()` with schedules
- **Optimal Operation Search**: Built-in optimization algorithms to find optimal operating conditions
- **Subsystem-level analysis**: Detailed balances for each subsystem (e.g., combustion chamber, heat exchanger, mixing valve)
- **Performance metrics**: Exergy efficiency, COP, and other key performance indicators
- **Visualization support**: Built-in functions for displaying balance results
- **Unit conversion utilities**: Comprehensive unit conversion functions for all common physical quantities

---

## Energy Systems Supported

The library provides models for a wide range of energy conversion systems, organized by application:

### Domestic Hot Water (DHW) Systems

- **`ElectricBoiler`**: Electric resistance heating system with hot water storage tank
- **`GasBoiler`**: Advanced natural gas combustion boiler (supports dynamic schedules)
- **`AirSourceHeatPumpBoiler`**: Air-source heat pump for hot water production (features CoolProp integration and optimal operation search)
- **`SolarAssistedGasBoiler`**: Hybrid system combining solar thermal collectors with gas backup
- **`GroundSourceHeatPumpBoiler`**: Ground-source heat pump for hot water production (features g-function borehole model)

### Heat Pump Systems

- **`AirSourceHeatPump_cooling`**: Air-source heat pump in cooling mode
- **`AirSourceHeatPump_heating`**: Air-source heat pump in heating mode
- **`GroundSourceHeatPump_cooling`**: Ground-source heat pump in cooling mode
- **`GroundSourceHeatPump_heating`**: Ground-source heat pump in heating mode

### Renewable Energy Systems

- **`PV_to_Converter`**: Photovoltaic system with charge controller, battery, and DC/AC converter

### Dynamic System Models

- **`ElectricHeater`**: Dynamic heat transfer analysis for electric heating elements
- **`StratifiedTankTDMA`**: 1D Stratified hot-water tank model using TDMA and effective thermal conductivity

### Auxiliary Components

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

For more examples, see the [Documentation](https://bet-lab.github.io/enex_analysis_engine/).

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
- **[EXAMPLES.md](EXAMPLES.md)**: Additional usage examples and tutorials (Korean)

The online documentation includes:

- **Getting Started**: Installation and quick start guides
- **User Guides**: Detailed guides on using the library features
- **Examples**:
  - [Electric Boiler](docs/source/examples/electric-boiler.md)
  - [Gas Boiler](docs/source/examples/gas-boiler.md)
  - [Heat Pump Systems](docs/source/examples/air-source-heat-pump.md)
  - And more...
- **API Reference**: Complete documentation for all classes and functions

---

## Project Structure

```
enex_analysis_engine/
├── src/
│   └── enex_analysis/
│       ├── __init__.py           # Package initialization
│       ├── AirSourceHeatPumpBoiler.py
│       ├── balance_helpers.py    # Balance calculation utilities
│       ├── calc_util.py          # Unit conversion utilities
│       ├── constants.py          # Physical constants
│       ├── enex_engine.py        # Core system models
│       ├── enex_functions.py     # Shared functions for calc modules
│       ├── GasBoiler.py          # Advanced Gas Boiler model
│       ├── GroundSourceHeatPumpBoiler.py
│       ├── PV_to_Converter.py    # PV system model
│       └── Tank_stratification_model.py # 1D Stratified Tank model
├── docs/                         # Sphinx documentation
├── pyproject.toml                # Project configuration
├── uv.lock                       # Dependency lock file
├── IO_DOCS.md                    # I/O documentation (Korean)
├── EXAMPLES.md                   # Usage examples (Korean)
└── README.md                     # This file
```

### Key Modules

- **`enex_engine.py`**: Contains the core energy system classes (Boilers, Heat Pumps, etc.).
- **`AirSourceHeatPumpBoiler.py`**, **`GasBoiler.py`**, **`GroundSourceHeatPumpBoiler.py`**: Advanced dynamic models for building energy systems.
- **`Tank_stratification_model.py`**, **`PV_to_Converter.py`**: Specialized physical models for thermal storage and renewable generation.
- **`calc_util.py`**: Unit conversion constants (e.g., `C2K`, `h2s`, `W2kW`).
- **`enex_functions.py`**: Shared helper functions used across different engines.
- **`balance_helpers.py`**: Utilities for calculating and formatting thermodynamic balances.

---
