API Reference
==============

This page contains the complete API reference for the en-system-ex-analysis package, organized by category.

Utility Module
--------------

.. automodule:: en_system_ex_analysis.calc_util
   :members:
   :undoc-members:
   :show-inheritance:

The utility module provides unit conversion functions and constants for temperature, time, length, energy, power, pressure, and other physical quantities.

Temperature Conversion Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: en_system_ex_analysis.calc_util.C2K
.. autofunction:: en_system_ex_analysis.calc_util.K2C

Unit Conversion Constants
^^^^^^^^^^^^^^^^^^^^^^^^^

The module includes numerous conversion constants:

* **Time**: ``h2s``, ``s2h``, ``d2h``, ``h2d``, ``m2s``, ``s2m``
* **Length**: ``m2cm``, ``cm2m``, ``m2mm``, ``mm2m``, ``m2km``, ``km2m``
* **Area**: ``m22cm2``, ``cm22m2``, ``m22mm2``, ``mm22m2``
* **Volume**: ``m32cm3``, ``cm32m3``, ``m32L``, ``L2m3``
* **Mass**: ``kg2g``, ``g2kg``, ``kg2mg``, ``mg2kg``, ``kg2t``, ``t2kg``
* **Energy**: ``J2kJ``, ``kJ2J``, ``J2MJ``, ``MJ2J``, ``J2GJ``, ``GJ2J``, ``kWh2J``, ``J2kWh``
* **Power**: ``W2kW``, ``kW2W``, ``W2MW``, ``MW2W``
* **Pressure**: ``Pa2kPa``, ``kPa2Pa``, ``Pa2MPa``, ``MPa2Pa``, ``Pa2bar``, ``bar2Pa``, ``atm2Pa``, ``Pa2atm``
* **Angle**: ``d2r``, ``r2d``

Energy System Integration Module
--------------------------------

.. automodule:: en_system_ex_analysis.En_system_intergrated
   :members:
   :undoc-members:
   :show-inheritance:

Domestic Hot Water (DHW) Systems
---------------------------------

ElectricBoiler
^^^^^^^^^^^^^^

.. autoclass:: en_system_ex_analysis.En_system_intergrated.ElectricBoiler
   :members:
   :undoc-members:
   :show-inheritance:

An electric resistance heating system with hot water storage tank. Models energy, entropy, and exergy balances for the tank and mixing valve subsystems.

**Input Parameters:**

* **Temperature [°C]**:
  * ``T_w_tank`` (float, default: 60): Tank water temperature
  * ``T_w_sup`` (float, default: 10): Supply water temperature
  * ``T_w_serv`` (float, default: 45): Service (tap) water temperature
  * ``T0`` (float, default: 0): Reference (environmental) temperature

* **Flow Rate**:
  * ``dV_w_serv`` (float, default: 0.00002): Service water flow rate [m³/s]

* **Tank Geometry [m]**:
  * ``r0`` (float, default: 0.2): Tank inner radius
  * ``H`` (float, default: 0.8): Tank height

* **Tank Thermal Properties**:
  * ``x_shell`` (float, default: 0.01): Shell thickness [m]
  * ``x_ins`` (float, default: 0.10): Insulation thickness [m]
  * ``k_shell`` (float, default: 25): Shell thermal conductivity [W/mK]
  * ``k_ins`` (float, default: 0.03): Insulation thermal conductivity [W/mK]
  * ``h_o`` (float, default: 15): Overall heat transfer coefficient [W/m²K]

**Output Attributes:**

After calling ``system_update()``:

* **Energy**: ``E_heater`` [W], ``Q_w_tank`` [W], ``Q_l_tank`` [W], ``Q_w_serv`` [W]
* **Exergy**: ``X_heater`` [W], ``X_w_tank`` [W], ``X_w_serv`` [W], ``X_c_tank`` [W], ``X_c_mix`` [W], ``X_c_tot`` [W], ``X_eff`` [-]
* **Balances**: ``energy_balance``, ``entropy_balance``, ``exergy_balance``

GasBoiler
^^^^^^^^^

.. autoclass:: en_system_ex_analysis.En_system_intergrated.GasBoiler
   :members:
   :undoc-members:
   :show-inheritance:

A natural gas combustion boiler system with hot water storage tank. Includes combustion chamber, tank, and mixing valve subsystems.

**Input Parameters:**

* **Efficiency**:
  * ``eta_comb`` (float, default: 0.9): Combustion efficiency [-]

* **Temperature [°C]**:
  * ``T_w_tank`` (float, default: 60): Tank water temperature
  * ``T_w_sup`` (float, default: 10): Supply water temperature
  * ``T_w_serv`` (float, default: 45): Service water temperature
  * ``T0`` (float, default: 0): Reference temperature
  * ``T_exh`` (float, default: 70): Exhaust gas temperature

* **Other parameters**: Same as ``ElectricBoiler``

**Output Attributes:**

* **Energy**: ``E_NG`` [W], ``Q_exh`` [W], ``Q_w_comb_out`` [W], ``Q_w_tank`` [W], ``Q_l_tank`` [W]
* **Exergy**: ``X_NG`` [W] (exergy efficiency = 0.93), ``X_exh`` [W], ``X_c_comb`` [W], ``X_c_tank`` [W], ``X_c_mix`` [W], ``X_c_tot`` [W], ``X_eff`` [-]

HeatPumpBoiler
^^^^^^^^^^^^^^

.. autoclass:: en_system_ex_analysis.En_system_intergrated.HeatPumpBoiler
   :members:
   :undoc-members:
   :show-inheritance:

An air-source heat pump boiler system with external unit, refrigerant loop, hot water tank, and mixing valve.

**Input Parameters:**

* **Performance**:
  * ``eta_fan`` (float, default: 0.6): External fan efficiency [-]
  * ``COP`` (float, default: 2.5): Coefficient of Performance [-]

* **Pressure**:
  * ``dP`` (float, default: 200): Pressure difference across fan [Pa]

* **Temperature [°C]**:
  * ``T0`` (float, default: 0): Reference temperature
  * ``T_a_ext_out`` (float, default: -5): External unit air outlet temperature
  * ``T_r_ext`` (float, default: -10): External unit refrigerant temperature
  * ``T_w_tank`` (float, default: 60): Tank water temperature
  * ``T_r_tank`` (float, default: 65): Tank refrigerant temperature
  * ``T_w_serv`` (float, default: 45): Service water temperature
  * ``T_w_sup`` (float, default: 10): Supply water temperature

**Output Attributes:**

* **Energy**: ``E_cmp`` [W], ``E_fan`` [W], ``Q_r_tank`` [W], ``Q_r_ext`` [W], ``Q_a_ext_in`` [W], ``Q_a_ext_out`` [W], ``dV_a_ext`` [m³/s]
* **Exergy**: ``X_cmp`` [W], ``X_fan`` [W], ``X_r_tank`` [W], ``X_r_ext`` [W], ``X_c_ext`` [W], ``X_c_r`` [W]

SolarAssistedGasBoiler
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: en_system_ex_analysis.En_system_intergrated.SolarAssistedGasBoiler
   :members:
   :undoc-members:
   :show-inheritance:

A hybrid system combining solar thermal collectors with gas backup boiler.

**Input Parameters:**

* **Solar Parameters**:
  * ``alpha`` (float, default: 0.95): Collector absorptivity [-]
  * ``I_DN`` (float, default: 500): Direct normal irradiance [W/m²]
  * ``I_dH`` (float, default: 200): Diffuse horizontal irradiance [W/m²]
  * ``A_stc`` (float, default: 2): Solar collector area [m²]

* **Thermal Properties**:
  * ``h_r`` (float, default: 2): Air layer radiation heat transfer coefficient [W/m²K]
  * ``x_air`` (float, default: 0.01): Air layer thickness [m]
  * ``x_ins`` (float, default: 0.05): Insulation thickness [m]
  * ``k_ins`` (float, default: 0.03): Insulation thermal conductivity [W/mK]

* **Temperature [°C]**:
  * ``T_w_comb`` (float, default: 60): Combustion chamber water temperature
  * Other temperatures same as ``GasBoiler``

**Output Attributes:**

* **Solar**: ``Q_sol`` [W], ``T_w_stc_out`` [K], ``T_stc`` [K], ``Q_l`` [W]
* **Exergy**: ``X_sol`` [W], ``X_c_stc`` [W]

GroundSourceHeatPumpBoiler
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: en_system_ex_analysis.En_system_intergrated.GroundSourceHeatPumpBoiler
   :members:
   :undoc-members:
   :show-inheritance:

A ground-source heat pump boiler system with borehole heat exchanger.

**Input Parameters:**

* **Time**:
  * ``time`` (float, default: 10): Operating time [h]

* **Borehole Properties**:
  * ``D_b`` (float, default: 0): Borehole depth [m]
  * ``H_b`` (float, default: 200): Borehole height [m]
  * ``r_b`` (float, default: 0.08): Borehole radius [m]
  * ``R_b`` (float, default: 0.108): Effective borehole thermal resistance [mK/W]

* **Fluid Properties**:
  * ``dV_f`` (float, default: 24): Volumetric flow rate [L/min]
  * ``E_pmp`` (float, default: 200): Pump power input [W]

* **Ground Properties**:
  * ``k_g`` (float, default: 2.0): Ground thermal conductivity [W/mK]
  * ``c_g`` (float, default: 800): Ground specific heat capacity [J/(kgK)]
  * ``rho_g`` (float, default: 2000): Ground density [kg/m³]
  * ``T_g`` (float, default: 11): Ground temperature [°C]

**Output Attributes:**

* **Borehole**: ``Q_bh`` [W/m], ``g_i`` [mK/W], ``T_b`` [K], ``T_f`` [K], ``T_f_in`` [K], ``T_f_out`` [K]
* **Refrigerant**: ``COP`` [-], ``T_r_exch`` [K]
* **Exergy**: ``X_g`` [W], ``X_b`` [W], ``X_f_in`` [W], ``X_f_out`` [W], ``X_pmp`` [W], ``X_c_g`` [W], ``X_c_GHE`` [W], ``X_c_exch`` [W]

Heat Pump Systems
------------------

AirSourceHeatPump_cooling
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: en_system_ex_analysis.En_system_intergrated.AirSourceHeatPump_cooling
   :members:
   :undoc-members:
   :show-inheritance:

Air-source heat pump in cooling mode for space conditioning.

**Input Parameters:**

* **Fan Objects**:
  * ``fan_int`` (Fan, default: ``Fan().fan1``): Internal unit fan
  * ``fan_ext`` (Fan, default: ``Fan().fan3``): External unit fan

* **Performance**:
  * ``Q_r_max`` (float, default: 9000): Maximum cooling capacity [W]
  * ``COP_ref`` (float, default: 4): Reference COP [-]

* **Temperature [°C]**:
  * ``T0`` (float, default: 32): Environmental temperature
  * ``T_a_room`` (float, default: 20): Room air temperature
  * ``T_r_int`` (float, default: 10): Internal unit refrigerant temperature
  * ``T_a_int_out`` (float, default: 15): Internal unit air outlet temperature
  * ``T_a_ext_out`` (float, default: 42): External unit air outlet temperature
  * ``T_r_ext`` (float, default: 47): External unit refrigerant temperature

* **Load**:
  * ``Q_r_int`` (float, default: 6000): Cooling load [W]

**Output Attributes:**

* **Energy**: ``COP`` [-], ``E_cmp`` [W], ``E_fan_int`` [W], ``E_fan_ext`` [W], ``Q_r_ext`` [W], ``dV_int`` [m³/s], ``dV_ext`` [m³/s]
* **Exergy**: ``X_r_int`` [W], ``X_r_ext`` [W], ``X_a_int_in`` [W], ``X_a_int_out`` [W], ``X_c_int`` [W], ``X_c_r`` [W], ``X_c_ext`` [W], ``X_eff`` [-]

AirSourceHeatPump_heating
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: en_system_ex_analysis.En_system_intergrated.AirSourceHeatPump_heating
   :members:
   :undoc-members:
   :show-inheritance:

Air-source heat pump in heating mode for space conditioning.

**Input Parameters:**

Similar to ``AirSourceHeatPump_cooling`` but with different default temperatures:

* ``T0`` (float, default: 0): Environmental temperature [°C]
* ``T_r_int`` (float, default: 35): Internal unit refrigerant temperature [°C]
* ``T_a_int_out`` (float, default: 30): Internal unit air outlet temperature [°C]
* ``T_a_ext_out`` (float, default: -5): External unit air outlet temperature [°C]
* ``T_r_ext`` (float, default: -10): External unit refrigerant temperature [°C]

**Output Attributes:**

Similar to cooling mode but with reversed heat transfer direction.

GroundSourceHeatPump_cooling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: en_system_ex_analysis.En_system_intergrated.GroundSourceHeatPump_cooling
   :members:
   :undoc-members:
   :show-inheritance:

Ground-source heat pump in cooling mode.

**Input Parameters:**

* **Temperature [°C]**:
  * ``T0`` (float, default: 32): Environmental temperature
  * ``T_g`` (float, default: 15): Ground temperature
  * ``T_a_room`` (float, default: 20): Room air temperature
  * ``T_r_exch`` (float, default: 25): Heat exchanger refrigerant temperature
  * ``dT_r_exch`` (float, default: 5): Heat exchanger temperature difference [K]

* **Other parameters**: Similar to ``GroundSourceHeatPumpBoiler``

**Output Attributes:**

Similar to ``GroundSourceHeatPumpBoiler`` but adjusted for cooling mode.

GroundSourceHeatPump_heating
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: en_system_ex_analysis.En_system_intergrated.GroundSourceHeatPump_heating
   :members:
   :undoc-members:
   :show-inheritance:

Ground-source heat pump in heating mode.

**Input Parameters:**

Similar to ``GroundSourceHeatPump_cooling`` but with different default temperatures:

* ``T0`` (float, default: 0): Environmental temperature [°C]
* ``T_r_exch`` (float, default: 5): Heat exchanger refrigerant temperature [°C]
* ``dT_r_exch`` (float, default: -5): Heat exchanger temperature difference [K]

**Output Attributes:**

Similar to cooling mode but with reversed heat transfer direction.

Dynamic Systems
---------------

ElectricHeater
^^^^^^^^^^^^^^

.. autoclass:: en_system_ex_analysis.En_system_intergrated.ElectricHeater
   :members:
   :undoc-members:
   :show-inheritance:

Dynamic heat transfer analysis for electric heating elements with transient behavior.

**Input Parameters:**

* **Material Properties**:
  * ``c`` (float, default: 500): Specific heat capacity [J/(kgK)]
  * ``rho`` (float, default: 7800): Density [kg/m³]
  * ``k`` (float, default: 50): Thermal conductivity [W/mK]

* **Geometry [m]**:
  * ``D`` (float, default: 0.005): Thickness
  * ``H`` (float, default: 0.8): Height
  * ``W`` (float, default: 1.0): Width

* **Electrical Input**:
  * ``E_heater`` (float, default: 1000): Heater power input [W]

* **Temperature [°C]**:
  * ``T0`` (float, default: 0): Reference temperature
  * ``T_mr`` (float, default: 15): Room surface temperature
  * ``T_init`` (float, default: 20): Initial heater temperature
  * ``T_a_room`` (float, default: 20): Room air temperature

* **Radiation Properties**:
  * ``epsilon_hs`` (float, default: 1): Heater surface emissivity [-]
  * ``epsilon_rs`` (float, default: 1): Room surface emissivity [-]

* **Time**:
  * ``dt`` (float, default: 10): Time step [s]

**Output Attributes:**

* **Time History (lists)**: ``time``, ``T_hb_list``, ``T_hs_list``, ``E_heater_list``, ``Q_st_list``, ``Q_cond_list``, ``Q_conv_list``, ``Q_rad_hs_list``, ``Q_rad_rs_list``
* **Entropy/Exergy History**: ``S_*_list``, ``X_*_list``
* **Final Values**: ``X_eff`` [-]

Auxiliary Components
---------------------

Fan
^^^

.. autoclass:: en_system_ex_analysis.En_system_intergrated.Fan
   :members:
   :undoc-members:
   :show-inheritance:

Air handling fan with performance curves. Provides three fan datasets: ``fan1``, ``fan2``, ``fan3``.

**Methods:**

* ``get_efficiency(fan, dV_fan)``: Get fan efficiency at given flow rate
* ``get_pressure(fan, dV_fan)``: Get fan pressure at given flow rate
* ``get_power(fan, dV_fan)``: Get fan power consumption at given flow rate
* ``show_graph()``: Display performance curves

Pump
^^^^

.. autoclass:: en_system_ex_analysis.En_system_intergrated.Pump
   :members:
   :undoc-members:
   :show-inheritance:

Fluid circulation pump with efficiency curves. Provides two pump datasets: ``pump1``, ``pump2``.

**Methods:**

* ``get_efficiency(pump, dV_pmp)``: Get pump efficiency at given flow rate
* ``get_power(pump, V_pmp, dP_pmp)``: Get pump power consumption at given flow rate and pressure difference
* ``show_graph()``: Display performance curves

Utility Functions
-----------------

Heat Transfer Functions
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: en_system_ex_analysis.En_system_intergrated.darcy_friction_factor

Calculates the Darcy friction factor for given Reynolds number and relative roughness.

**Parameters:**
* ``Re`` (float): Reynolds number [-]
* ``e_d`` (float): Relative roughness (e/D) [-]

**Returns:**
* ``float``: Darcy friction factor [-]

.. autofunction:: en_system_ex_analysis.En_system_intergrated.calc_h_vertical_plate

Calculates natural convection heat transfer coefficient for a vertical plate using Churchill & Chu correlation.

**Parameters:**
* ``T_s`` (float): Surface temperature [K]
* ``T_inf`` (float): Fluid temperature [K]
* ``L`` (float): Characteristic length [m]

**Returns:**
* ``float``: Heat transfer coefficient [W/m²K]

COP Calculation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: en_system_ex_analysis.En_system_intergrated.calculate_ASHP_cooling_COP

Calculates COP for air-source heat pump in cooling mode based on outdoor/indoor temperatures and part-load ratio.

**Parameters:**
* ``T_a_int_out`` (float): Indoor air outlet temperature [K]
* ``T_a_ext_in`` (float): Outdoor air inlet temperature [K]
* ``Q_r_int`` (float): Cooling load [W]
* ``Q_r_max`` (float): Maximum capacity [W]
* ``COP_ref`` (float): Reference COP [-]

**Returns:**
* ``float``: Calculated COP [-]

.. autofunction:: en_system_ex_analysis.En_system_intergrated.calculate_ASHP_heating_COP

Calculates COP for air-source heat pump in heating mode based on outdoor temperature and part-load ratio.

**Parameters:**
* ``T0`` (float): Outdoor temperature [K]
* ``Q_r_int`` (float): Heating load [W]
* ``Q_r_max`` (float): Maximum capacity [W]

**Returns:**
* ``float``: Calculated COP [-]

.. autofunction:: en_system_ex_analysis.En_system_intergrated.calculate_GSHP_COP

Calculates modified Carnot-based COP for ground-source heat pump.

**Parameters:**
* ``Tg`` (float): Ground temperature [K]
* ``T_cond`` (float): Condenser refrigerant temperature [K]
* ``T_evap`` (float): Evaporator refrigerant temperature [K]
* ``theta_hat`` (float): Dimensionless mean fluid temperature [-]

**Returns:**
* ``float``: Calculated COP [-]

Balance Functions
^^^^^^^^^^^^^^^^^

.. autofunction:: en_system_ex_analysis.En_system_intergrated.print_balance

Prints energy, entropy, or exergy balance in a formatted way.

**Parameters:**
* ``balance`` (dict): Balance dictionary
* ``decimal`` (int, default: 2): Number of decimal places

**Example Output:**

.. code-block:: text

   HOT WATER TANK EXERGY BALANCE: =====================

   IN ENTRIES:
   E_heater: 5234.56 [W]
   X_w_sup_tank: 123.45 [W]

   OUT ENTRIES:
   X_w_tank: 4567.89 [W]
   X_l_tank: 234.56 [W]

   CONSUMED ENTRIES:
   X_c_tank: 555.56 [W]

Polynomial Functions
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: en_system_ex_analysis.En_system_intergrated.linear_function
.. autofunction:: en_system_ex_analysis.En_system_intergrated.quadratic_wunction
.. autofunction:: en_system_ex_analysis.En_system_intergrated.cubic_wunction
.. autofunction:: en_system_ex_analysis.En_system_intergrated.quartic_wunction

Polynomial functions for curve fitting and interpolation.
