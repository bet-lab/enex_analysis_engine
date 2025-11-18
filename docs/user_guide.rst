User Guide
===========

This guide provides comprehensive instructions for using the en-system-ex-analysis package.

Getting Started
---------------

Importing the Module
^^^^^^^^^^^^^^^^^^^^

To use the energy system analysis module, import it as follows:

.. code-block:: python

   import en_system_ex_analysis as enex

Basic Usage Pattern
^^^^^^^^^^^^^^^^^^^

All system models follow a consistent usage pattern:

1. **Initialize** the system class
2. **Set parameters** (temperatures, flow rates, etc.)
3. **Call** ``system_update()`` to perform calculations
4. **Access results** from the object attributes
5. **Print balances** using ``print_balance()`` function

Example: Creating an Electric Boiler System
---------------------------------------------

Here's a complete example of how to create and use an electric boiler system:

.. code-block:: python

   import en_system_ex_analysis as enex

   # Create an electric boiler instance
   EB = enex.ElectricBoiler()
   
   # Set the reference temperature (in Celsius)
   EB.T0 = 10
   EB.T_w_tank = 60   # Tank water temperature [°C]
   EB.T_w_sup = 10    # Supply water temperature [°C]
   EB.T_w_serv = 45   # Service water temperature [°C]
   EB.dV_w_serv = 1.2 # Service water flow rate [L/min]
   
   # Modify tank properties if needed
   EB.r0 = 0.2        # Tank inner radius [m]
   EB.H = 0.8         # Tank height [m]
   EB.x_ins = 0.10    # Insulation thickness [m]
   EB.k_ins = 0.03    # Insulation thermal conductivity [W/mK]
   
   # Update the system calculations
   EB.system_update()
   
   # Access results
   print(f"Electric power input: {EB.E_heater:.2f} W")
   print(f"Tank heat loss: {EB.Q_l_tank:.2f} W")
   print(f"Exergy efficiency: {EB.X_eff:.4f}")
   
   # Print the exergy balance
   enex.print_balance(EB.exergy_balance)

Available Systems
-----------------

The package provides the following energy system models:

Domestic Hot Water Systems
^^^^^^^^^^^^^^^^^^^^^^^^^^

* **ElectricBoiler**: Electric boiler system with energy, entropy, and exergy analysis
* **GasBoiler**: Gas boiler system with combustion chamber analysis
* **HeatPumpBoiler**: Air-source heat pump boiler system
* **SolarAssistedGasBoiler**: Solar-assisted gas boiler system
* **GroundSourceHeatPumpBoiler**: Ground source heat pump boiler system

Heat Pump Systems
^^^^^^^^^^^^^^^^^

* **AirSourceHeatPump_cooling**: Air source heat pump (cooling mode)
* **AirSourceHeatPump_heating**: Air source heat pump (heating mode)
* **GroundSourceHeatPump_cooling**: Ground source heat pump (cooling mode)
* **GroundSourceHeatPump_heating**: Ground source heat pump (heating mode)

Dynamic Systems
^^^^^^^^^^^^^^^

* **ElectricHeater**: Electric heater system with transient analysis

Auxiliary Components
^^^^^^^^^^^^^^^^^^^^

* **Fan**: Fan system with performance curves
* **Pump**: Pump system with efficiency curves

Each system provides:

* Energy balance calculations
* Entropy balance calculations
* Exergy balance calculations
* System update methods for recalculating balances

Parameter Setting Guide
------------------------

Temperature Parameters
^^^^^^^^^^^^^^^^^^^^^^

All temperature inputs are in **Celsius [°C]**. The system automatically converts them to Kelvin internally.

.. code-block:: python

   system.T0 = 0          # Reference temperature [°C]
   system.T_w_tank = 60   # Tank water temperature [°C]
   system.T_w_sup = 10    # Supply water temperature [°C]

Flow Rate Parameters
^^^^^^^^^^^^^^^^^^^^

Flow rates are typically specified in **L/min** for water systems and **m³/s** for air systems.

.. code-block:: python

   system.dV_w_serv = 1.2  # Service water flow rate [L/min]
   system.dV_a_ext = 2.5   # External air flow rate [m³/s]

Geometric Parameters
^^^^^^^^^^^^^^^^^^^^

Geometric parameters use SI units:

* Length: meters [m]
* Area: square meters [m²]
* Volume: cubic meters [m³]

.. code-block:: python

   system.r0 = 0.2    # Tank inner radius [m]
   system.H = 0.8     # Tank height [m]
   system.A_stc = 2   # Solar collector area [m²]

Thermal Properties
^^^^^^^^^^^^^^^^^^

Thermal properties use standard SI units:

* Thermal conductivity: W/(m·K)
* Specific heat capacity: J/(kg·K)
* Density: kg/m³

.. code-block:: python

   system.k_ins = 0.03    # Insulation thermal conductivity [W/mK]
   system.c_w = 4186     # Water specific heat [J/kgK]
   system.rho_w = 1000   # Water density [kg/m³]

Understanding Results
---------------------

Energy Balance
^^^^^^^^^^^^^

The energy balance shows energy flows into and out of each subsystem:

.. code-block:: python

   print_balance(system.energy_balance)

Energy is conserved, so:

.. math::
   \sum \dot{E}_{in} = \sum \dot{E}_{out} + \dot{E}_{loss}

Entropy Balance
^^^^^^^^^^^^^^^

The entropy balance includes entropy generation due to irreversibilities:

.. code-block:: python

   print_balance(system.entropy_balance)

Entropy balance equation:

.. math::
   \sum \dot{S}_{in} + \dot{S}_{gen} = \sum \dot{S}_{out}

Exergy Balance
^^^^^^^^^^^^^^

The exergy balance shows exergy destruction (irreversibilities):

.. code-block:: python

   print_balance(system.exergy_balance)

Exergy balance equation:

.. math::
   \sum \dot{X}_{in} = \sum \dot{X}_{out} + \dot{X}_{destroyed}

Exergy destruction is related to entropy generation:

.. math::
   \dot{X}_{destroyed} = T_0 \cdot \dot{S}_{gen}

Key Performance Indicators
^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Exergy Efficiency** (``X_eff``): Ratio of useful exergy output to exergy input
* **COP** (for heat pumps): Coefficient of Performance
* **Energy Efficiency** (``E_eff``): Ratio of useful energy output to energy input

Utility Functions
-----------------

The package includes utility functions for unit conversions:

Temperature Conversions
^^^^^^^^^^^^^^^^^^^^^^^

* ``C2K(C)``: Convert Celsius to Kelvin
* ``K2C(K)``: Convert Kelvin to Celsius

.. code-block:: python

   from en_system_ex_analysis import C2K, K2C
   
   T_K = C2K(25)  # 298.15
   T_C = K2C(298.15)  # 25.0

Time Conversions
^^^^^^^^^^^^^^^^

Constants for time conversions:

* ``h2s = 3600``: Hours to seconds
* ``s2h = 1/3600``: Seconds to hours
* ``d2h = 24``: Days to hours
* ``m2s = 60``: Minutes to seconds

.. code-block:: python

   from en_system_ex_analysis.calc_util import h2s, d2h
   
   hours = 2
   seconds = hours * h2s  # 7200
   days = 1
   hours_from_days = days * d2h  # 24

Length Conversions
^^^^^^^^^^^^^^^^^^

* ``m2cm = 100``: Meters to centimeters
* ``cm2m = 1/100``: Centimeters to meters
* ``m2mm = 1e3``: Meters to millimeters
* ``mm2m = 1e-3``: Millimeters to meters

Energy/Power Conversions
^^^^^^^^^^^^^^^^^^^^^^^^^

* ``J2kWh = 1/3.6e6``: Joules to kilowatt-hours
* ``kWh2J = 3.6e6``: Kilowatt-hours to Joules
* ``W2kW = 1e-3``: Watts to kilowatts
* ``kW2W = 1e3``: Kilowatts to Watts

Volume Conversions
^^^^^^^^^^^^^^^^^^^

* ``m32L = 1e3``: Cubic meters to liters
* ``L2m3 = 1e-3``: Liters to cubic meters

Pressure Conversions
^^^^^^^^^^^^^^^^^^^^

* ``Pa2kPa = 1e-3``: Pascals to kilopascals
* ``kPa2Pa = 1e3``: Kilopascals to Pascals
* ``Pa2bar = 1e-5``: Pascals to bars
* ``bar2Pa = 1e5``: Bars to Pascals

Best Practices
--------------

1. **Always call ``system_update()``** after modifying parameters and before accessing results.

2. **Check input ranges**: Some parameters have physical limits (e.g., temperatures must be positive, flow rates must be positive).

3. **Unit consistency**: Be aware of input units (typically °C for temperatures, L/min for flow rates) and output units (typically K for temperatures, W for power).

4. **Iterative models**: Ground-source heat pump models use iterative solvers. If convergence issues occur, check input parameters.

5. **Balance verification**: Energy balances should satisfy conservation (in ≈ out), while entropy and exergy balances include generation/consumption terms.

6. **Performance**: For parameter studies, consider caching results or using vectorized operations where possible.

7. **Visualization**: Use ``dartwork-mpl`` for publication-quality plots.

See the :doc:`api` documentation for detailed information about each class and function, and :doc:`examples` for more comprehensive usage examples.
