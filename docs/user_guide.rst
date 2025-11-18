User Guide
===========

Getting Started
---------------

Importing the Module
^^^^^^^^^^^^^^^^^^^^

To use the energy system analysis module, import it as follows:

.. code-block:: python

   import en_system_ex_analysis as enex

Example: Creating an Electric Boiler System
---------------------------------------------

Here's a simple example of how to create and use an electric boiler system:

.. code-block:: python

   # Create an electric boiler instance
   EB = enex.ElectricBoiler()
   
   # Set the reference temperature (in Celsius)
   EB.T0 = 10
   
   # Update the system calculations
   EB.system_update()
   
   # Print the exergy balance
   enex.print_balance(EB.exergy_balance)

Available Systems
-----------------

The package provides the following energy system models:

* **ElectricBoiler**: Electric boiler system with energy, entropy, and exergy analysis
* **GasBoiler**: Gas boiler system
* **HeatPumpBoiler**: Heat pump boiler system
* **SolarAssistedGasBoiler**: Solar-assisted gas boiler system
* **GroundSourceHeatPumpBoiler**: Ground source heat pump boiler system
* **AirSourceHeatPump_cooling**: Air source heat pump (cooling mode)
* **AirSourceHeatPump_heating**: Air source heat pump (heating mode)
* **GroundSourceHeatPump_cooling**: Ground source heat pump (cooling mode)
* **GroundSourceHeatPump_heating**: Ground source heat pump (heating mode)
* **ElectricHeater**: Electric heater system
* **Fan**: Fan system
* **Pump**: Pump system

Each system provides:

* Energy balance calculations
* Entropy balance calculations
* Exergy balance calculations
* System update methods for recalculating balances

Utility Functions
-----------------

The package also includes utility functions for unit conversions:

* Temperature conversions (Kelvin ↔ Celsius)
* Time conversions (day, hour, minute, second)
* Length conversions (meter, centimeter, millimeter, etc.)
* Area and volume conversions
* Mass conversions
* Energy conversions (Joule, kilojoule, megajoule, kilowatt-hour)
* Power conversions (Watt, kilowatt, megawatt)
* Pressure conversions (Pascal, kilopascal, megapascal, bar, atmosphere)
* Angle conversions (degree ↔ radian)

See the :doc:`api` documentation for detailed information about each class and function.

