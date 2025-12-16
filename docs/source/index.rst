.. Energy-Exergy Analysis Engine documentation master file

=====================================
Energy-Exergy Analysis Engine
=====================================

.. rst-class:: lead

   **Comprehensive thermodynamic modeling for diverse energy systems through unified energy-exergy analysis**

This Python library enables simultaneous energy (first-law) and exergy (second-law) analysis of various energy conversion systems. Built for researchers, engineers, and educators who need to understand not just how much energy flows through a system, but also the quality and potential of that energy.

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: 🚀 Getting Started
        :link: getting-started/index
        :link-type: doc

        New to the library? Start here for installation and quick start guides.

    .. grid-item-card:: 📚 User Guides
        :link: guides/index
        :link-type: doc

        Comprehensive guides and tutorials for using the library.

    .. grid-item-card:: 💡 Examples
        :link: examples/index
        :link-type: doc

        Practical examples demonstrating real-world applications.

    .. grid-item-card:: 🔧 API Reference
        :link: api/index
        :link-type: doc

        Complete API documentation for all classes and functions.

Why Energy-Exergy Analysis?
============================

Traditional energy analysis tells you *how much* energy is used, but not *how well* it's used. Exergy analysis reveals the true thermodynamic efficiency by accounting for energy quality and identifying where irreversibilities occur. Together, energy-exergy analysis provides:

✓ **Complete thermodynamic picture**: Understand both quantity (energy) and quality (exergy) of energy flows

✓ **Inefficiency identification**: Pinpoint where and why energy is being destroyed

✓ **Technology comparison**: Fair comparison between different energy conversion technologies

✓ **Optimization guidance**: Identify the most promising areas for system improvement

Core Capabilities
==================

Unified Balance Calculations
-----------------------------

Every component model automatically calculates three balances:

1. **Energy Balance** (First Law of Thermodynamics): Identifies energy flows and losses

   .. math::
      \\sum \\dot{E}_{in} = \\sum \\dot{E}_{out} + \\dot{E}_{loss}

2. **Entropy Balance** (Second Law of Thermodynamics): Quantifies irreversibilities

   .. math::
      \\sum \\dot{S}_{in} + \\dot{S}_{gen} = \\sum \\dot{S}_{out}

3. **Exergy Balance** (Both First and Second Law): Reveals thermodynamic inefficiencies

   .. math::
      \\sum \\dot{X}_{in} = \\sum \\dot{X}_{out} + \\dot{X}_{destroyed}

   where exergy destruction is related to entropy generation:

   .. math::
      \\dot{X}_{destroyed} = T_0 \\cdot \\dot{S}_{gen}

Quick Start Example
===================

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
=========================

The library provides models for a wide range of energy conversion systems:

**Domestic Hot Water (DHW) Systems**

* :class:`~enex_analysis_engine.enex_engine.ElectricBoiler` - Electric resistance heating
* :class:`~enex_analysis_engine.enex_engine.GasBoiler` - Natural gas combustion boiler
* :class:`~enex_analysis_engine.enex_engine.HeatPumpBoiler` - Air-source heat pump
* :class:`~enex_analysis_engine.enex_engine.SolarAssistedGasBoiler` - Hybrid solar+gas system
* :class:`~enex_analysis_engine.enex_engine.GroundSourceHeatPumpBoiler` - Ground-source heat pump

**Heat Pump Systems**

* Air-source and ground-source heat pumps in both heating and cooling modes

**Dynamic System Models**

* :class:`~enex_analysis_engine.enex_engine.ElectricHeater` - Transient heat transfer analysis

**Auxiliary Components**

* :class:`~enex_analysis_engine.enex_engine.Fan` - Air handling fans with performance curves
* :class:`~enex_analysis_engine.enex_engine.Pump` - Fluid circulation pumps

.. toctree::
   :maxdepth: 2
   :caption: Documentation
   :hidden:

   getting-started/index
   guides/index
   examples/index
   api/index
   theory/index

.. toctree::
   :maxdepth: 1
   :caption: Project Links
   :hidden:

   GitHub Repository <https://github.com/bet-lab/enex_analysis_engine>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
