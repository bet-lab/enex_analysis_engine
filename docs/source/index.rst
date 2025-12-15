.. Energy-Exergy Analysis Engine documentation master file

Welcome to Energy-Exergy Analysis Engine's documentation!
==========================================================

**Comprehensive thermodynamic modeling for diverse energy systems through unified energy-exergy analysis**

This Python library enables simultaneous energy (first-law) and exergy (second-law) analysis of various energy conversion systems. Built for researchers, engineers, and educators who need to understand not just how much energy flows through a system, but also the quality and potential of that energy.

Why Energy-Exergy Analysis?
----------------------------

Traditional energy analysis tells you *how much* energy is used, but not *how well* it's used. Exergy analysis reveals the true thermodynamic efficiency by accounting for energy quality and identifying where irreversibilities occur. Together, energy-exergy analysis provides:

* **Complete thermodynamic picture**: Understand both quantity (energy) and quality (exergy) of energy flows
* **Inefficiency identification**: Pinpoint where and why energy is being destroyed
* **Technology comparison**: Fair comparison between different energy conversion technologies
* **Optimization guidance**: Identify the most promising areas for system improvement

This library makes energy-exergy analysis accessible by providing ready-to-use models for common energy systems, with automatic calculation of energy, entropy, and exergy balances.

Core Capabilities
------------------

Unified Balance Calculations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every component model automatically calculates three balances:

1. **Energy Balance** (First Law of Thermodynamics): Identifies energy flows and losses

   .. math::
      \sum \dot{E}_{in} = \sum \dot{E}_{out} + \dot{E}_{loss}

2. **Entropy Balance** (Second Law of Thermodynamics): Quantifies irreversibilities

   .. math::
      \sum \dot{S}_{in} + \dot{S}_{gen} = \sum \dot{S}_{out}

3. **Exergy Balance** (Both First and Second Law): Reveals thermodynamic inefficiencies

   .. math::
      \sum \dot{X}_{in} = \sum \dot{X}_{out} + \dot{X}_{destroyed}

   where exergy destruction is related to entropy generation:

   .. math::
      \dot{X}_{destroyed} = T_0 \cdot \dot{S}_{gen}

These balances are calculated consistently across all components, enabling system-level analysis and comparison.

Analysis Features
^^^^^^^^^^^^^^^^^

* **Automatic balance calculation**: Set parameters, run ``system_update()``, get all balances
* **Subsystem-level analysis**: Detailed balances for each subsystem (e.g., combustion chamber, heat exchanger, mixing valve)
* **Performance metrics**: Exergy efficiency, COP, and other key performance indicators
* **Visualization support**: Built-in functions for displaying balance results
* **Unit conversion utilities**: Comprehensive unit conversion functions for all common physical quantities

Quick Start
-----------

Here's a simple example to get you started:

.. code-block:: python

   import enex_analysis_engine as enex

   # Create an electric boiler instance
   EB = enex.ElectricBoiler()
   
   # Set the reference temperature (in Celsius)
   EB.T0 = 10
   EB.T_w_tank = 60
   EB.T_w_sup = 10
   EB.T_w_serv = 45
   EB.dV_w_serv = 1.2  # L/min
   
   # Update the system calculations
   EB.system_update()
   
   # Print the exergy balance
   enex.print_balance(EB.exergy_balance)
   
   # Access results
   print(f"Electric power input: {EB.E_heater:.2f} W")
   print(f"Exergy efficiency: {EB.X_eff:.4f}")

Energy Systems Supported
-------------------------

The library provides models for a wide range of energy conversion systems:

Domestic Hot Water (DHW) Systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :class:`ElectricBoiler` - Electric resistance heating system with hot water storage tank
* :class:`GasBoiler` - Natural gas combustion boiler with hot water storage
* :class:`HeatPumpBoiler` - Air-source heat pump for hot water production
* :class:`SolarAssistedGasBoiler` - Hybrid system combining solar thermal collectors with gas backup
* :class:`GroundSourceHeatPumpBoiler` - Ground-source heat pump for hot water production

Heat Pump Systems
^^^^^^^^^^^^^^^^^

* :class:`AirSourceHeatPump_cooling` - Air-source heat pump in cooling mode
* :class:`AirSourceHeatPump_heating` - Air-source heat pump in heating mode
* :class:`GroundSourceHeatPump_cooling` - Ground-source heat pump in cooling mode
* :class:`GroundSourceHeatPump_heating` - Ground-source heat pump in heating mode

Dynamic System Models
^^^^^^^^^^^^^^^^^^^^^

* :class:`ElectricHeater` - Dynamic heat transfer analysis for electric heating elements

Auxiliary Components
^^^^^^^^^^^^^^^^^^^

* :class:`Fan` - Air handling fan with performance curves
* :class:`Pump` - Fluid circulation pump with efficiency curves

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   user_guide
   examples
   api
   theory

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
