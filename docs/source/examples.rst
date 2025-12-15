Examples
========

This page provides comprehensive usage examples for all components in the Energy-Exergy Analysis Engine package.

Electric Boiler
---------------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from enex_analysis_engine import ElectricBoiler, print_balance

   # Initialize boiler with default parameters
   boiler = ElectricBoiler()

   # Set operating conditions
   boiler.T_w_tank = 60   # Tank water temperature [°C]
   boiler.T_w_sup = 10    # Supply water temperature [°C]
   boiler.T_w_serv = 45   # Service water temperature [°C]
   boiler.T0 = 0          # Reference temperature [°C]
   boiler.dV_w_serv = 1.2 # Service water flow rate [L/min]

   # Modify tank properties if needed
   boiler.r0 = 0.2        # Tank inner radius [m]
   boiler.H = 0.8         # Tank height [m]
   boiler.x_ins = 0.10    # Insulation thickness [m]
   boiler.k_ins = 0.03    # Insulation thermal conductivity [W/mK]

   # Run calculation
   boiler.system_update()

   # Access results
   print(f"Electric power input: {boiler.E_heater:.2f} W")
   print(f"Tank heat loss: {boiler.Q_l_tank:.2f} W")
   print(f"Exergy efficiency: {boiler.X_eff:.4f}")

   # Print balances
   print("\n=== Energy Balance ===")
   print_balance(boiler.energy_balance)

   print("\n=== Entropy Balance ===")
   print_balance(boiler.entropy_balance)

   print("\n=== Exergy Balance ===")
   print_balance(boiler.exergy_balance)

Parameter Study
^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from enex_analysis_engine import ElectricBoiler

   # Study effect of tank temperature on exergy efficiency
   tank_temps = np.linspace(50, 70, 21)
   exergy_effs = []
   power_inputs = []

   boiler = ElectricBoiler()
   boiler.T_w_sup = 10
   boiler.T_w_serv = 45
   boiler.dV_w_serv = 1.2

   for T_tank in tank_temps:
       boiler.T_w_tank = T_tank
       boiler.system_update()
       exergy_effs.append(boiler.X_eff)
       power_inputs.append(boiler.E_heater)

   # Plot results
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

   ax1.plot(tank_temps, exergy_effs, 'b-', linewidth=2)
   ax1.set_xlabel('Tank Temperature [°C]')
   ax1.set_ylabel('Exergy Efficiency [-]')
   ax1.grid(True)
   ax1.set_title('Exergy Efficiency vs Tank Temperature')

   ax2.plot(tank_temps, power_inputs, 'r-', linewidth=2)
   ax2.set_xlabel('Tank Temperature [°C]')
   ax2.set_ylabel('Electric Power Input [W]')
   ax2.grid(True)
   ax2.set_title('Power Input vs Tank Temperature')

   plt.tight_layout()
   plt.show()

Gas Boiler
----------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from enex_analysis_engine import GasBoiler, print_balance

   # Initialize gas boiler
   boiler = GasBoiler()

   # Set operating conditions
   boiler.eta_comb = 0.9      # Combustion efficiency
   boiler.T_w_tank = 60       # Tank water temperature [°C]
   boiler.T_w_sup = 10        # Supply water temperature [°C]
   boiler.T_w_serv = 45       # Service water temperature [°C]
   boiler.T_exh = 70          # Exhaust gas temperature [°C]
   boiler.dV_w_serv = 1.2     # Service water flow rate [L/min]

   # Run calculation
   boiler.system_update()

   # Access results
   print(f"Natural gas energy input: {boiler.E_NG:.2f} W")
   print(f"Exhaust heat loss: {boiler.Q_exh:.2f} W")
   print(f"Exergy efficiency: {boiler.X_eff:.4f}")
   print(f"Total exergy consumed: {boiler.X_c_tot:.2f} W")

   # Print exergy balance
   print_balance(boiler.exergy_balance, decimal=2)

Efficiency Comparison
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from enex_analysis_engine import ElectricBoiler, GasBoiler

   # Compare electric and gas boilers
   elec_boiler = ElectricBoiler()
   gas_boiler = GasBoiler()

   # Same operating conditions
   for boiler in [elec_boiler, gas_boiler]:
       boiler.T_w_tank = 60
       boiler.T_w_sup = 10
       boiler.T_w_serv = 45
       boiler.dV_w_serv = 1.2
       boiler.system_update()

   print("Electric Boiler:")
   print(f"  Power input: {elec_boiler.E_heater:.2f} W")
   print(f"  Exergy efficiency: {elec_boiler.X_eff:.4f}")

   print("\nGas Boiler:")
   print(f"  Energy input: {gas_boiler.E_NG:.2f} W")
   print(f"  Exergy efficiency: {gas_boiler.X_eff:.4f}")
   print(f"  Exergy input: {gas_boiler.X_NG:.2f} W")

Heat Pump Boiler
----------------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from enex_analysis_engine import HeatPumpBoiler, print_balance

   # Initialize heat pump boiler
   hp_boiler = HeatPumpBoiler()

   # Set operating conditions
   hp_boiler.COP = 2.5           # Coefficient of Performance
   hp_boiler.eta_fan = 0.6       # External fan efficiency
   hp_boiler.dP = 200            # Pressure difference [Pa]
   hp_boiler.T0 = 0              # Reference temperature [°C]
   hp_boiler.T_w_tank = 60       # Tank water temperature [°C]
   hp_boiler.T_w_serv = 45       # Service water temperature [°C]
   hp_boiler.T_w_sup = 10        # Supply water temperature [°C]
   hp_boiler.dV_w_serv = 1.2     # Service water flow rate [L/min]

   # Run calculation
   hp_boiler.system_update()

   # Access results
   print(f"Compressor power: {hp_boiler.E_cmp:.2f} W")
   print(f"External fan power: {hp_boiler.E_fan:.2f} W")
   print(f"Total power input: {hp_boiler.E_cmp + hp_boiler.E_fan:.2f} W")
   print(f"Exergy efficiency: {hp_boiler.X_eff:.4f}")

   # Print exergy balance
   print_balance(hp_boiler.exergy_balance)

Solar-Assisted Gas Boiler
--------------------------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from enex_analysis_engine import SolarAssistedGasBoiler, print_balance

   # Initialize solar-assisted gas boiler
   solar_boiler = SolarAssistedGasBoiler()

   # Set solar conditions
   solar_boiler.I_DN = 500      # Direct normal irradiance [W/m²]
   solar_boiler.I_dH = 200      # Diffuse horizontal irradiance [W/m²]
   solar_boiler.A_stc = 2       # Solar collector area [m²]
   solar_boiler.alpha = 0.95    # Absorptivity

   # Set operating conditions
   solar_boiler.eta_comb = 0.9
   solar_boiler.T_w_comb = 60
   solar_boiler.T_w_serv = 45
   solar_boiler.T_w_sup = 10
   solar_boiler.dV_w_serv = 1.2

   # Run calculation
   solar_boiler.system_update()

   # Access results
   print(f"Solar heat gain: {solar_boiler.Q_sol:.2f} W")
   print(f"Natural gas input: {solar_boiler.E_NG:.2f} W")
   print(f"Collector outlet temperature: {solar_boiler.T_w_stc_out - 273.15:.2f} °C")
   print(f"Exergy efficiency: {solar_boiler.X_eff:.4f}")

   # Print balances
   print("\n=== Solar Collector Exergy Balance ===")
   print_balance({"solar thermal panel": solar_boiler.exergy_balance["solar thermal panel"]})

   print("\n=== Combustion Chamber Exergy Balance ===")
   print_balance({"combustion chamber": solar_boiler.exergy_balance["combustion chamber"]})

Ground Source Heat Pump Boiler
-------------------------------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from enex_analysis_engine import GroundSourceHeatPumpBoiler, print_balance

   # Initialize ground-source heat pump boiler
   gshp_boiler = GroundSourceHeatPumpBoiler()

   # Set ground properties
   gshp_boiler.k_g = 2.0         # Ground thermal conductivity [W/mK]
   gshp_boiler.c_g = 800         # Ground specific heat [J/(kgK)]
   gshp_boiler.rho_g = 2000      # Ground density [kg/m³]
   gshp_boiler.T_g = 11          # Undisturbed ground temperature [°C]

   # Set borehole properties
   gshp_boiler.H_b = 200         # Borehole height [m]
   gshp_boiler.r_b = 0.08        # Borehole radius [m]
   gshp_boiler.R_b = 0.108       # Borehole thermal resistance [mK/W]
   gshp_boiler.dV_f = 24         # Fluid flow rate [L/min]
   gshp_boiler.E_pmp = 200       # Pump power [W]

   # Set operating conditions
   gshp_boiler.time = 10         # Operating time [h]
   gshp_boiler.T_w_tank = 60
   gshp_boiler.T_w_serv = 45
   gshp_boiler.T_w_sup = 10
   gshp_boiler.dV_w_serv = 1.2

   # Run calculation
   gshp_boiler.system_update()

   # Access results
   print(f"Compressor power: {gshp_boiler.E_cmp:.2f} W")
   print(f"Pump power: {gshp_boiler.E_pmp:.2f} W")
   print(f"COP: {gshp_boiler.COP:.2f}")
   print(f"Borehole heat flow: {gshp_boiler.Q_bh:.2f} W/m")
   print(f"Fluid inlet temperature: {gshp_boiler.T_f_in - 273.15:.2f} °C")
   print(f"Fluid outlet temperature: {gshp_boiler.T_f_out - 273.15:.2f} °C")
   print(f"Exergy efficiency: {gshp_boiler.X_eff:.4f}")

   # Print exergy balance
   print_balance(gshp_boiler.exergy_balance)

Air Source Heat Pump (Cooling)
-------------------------------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from enex_analysis_engine import AirSourceHeatPump_cooling, print_balance

   # Initialize air-source heat pump (cooling)
   ashp_cool = AirSourceHeatPump_cooling()

   # Set operating conditions
   ashp_cool.T0 = 32              # Outdoor temperature [°C]
   ashp_cool.T_a_room = 20        # Room temperature [°C]
   ashp_cool.Q_r_int = 6000       # Cooling load [W]
   ashp_cool.Q_r_max = 9000       # Maximum capacity [W]
   ashp_cool.COP_ref = 4          # Reference COP

   # Run calculation
   ashp_cool.system_update()

   # Access results
   print(f"COP: {ashp_cool.COP:.2f}")
   print(f"Compressor power: {ashp_cool.E_cmp:.2f} W")
   print(f"Internal fan power: {ashp_cool.E_fan_int:.2f} W")
   print(f"External fan power: {ashp_cool.E_fan_ext:.2f} W")
   print(f"Total power: {ashp_cool.E_cmp + ashp_cool.E_fan_int + ashp_cool.E_fan_ext:.2f} W")
   print(f"Exergy efficiency: {ashp_cool.X_eff:.4f}")

   # Print exergy balance
   print_balance(ashp_cool.exergy_balance)

Air Source Heat Pump (Heating)
-------------------------------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from enex_analysis_engine import AirSourceHeatPump_heating, print_balance

   # Initialize air-source heat pump (heating)
   ashp_heat = AirSourceHeatPump_heating()

   # Set operating conditions
   ashp_heat.T0 = 0               # Outdoor temperature [°C]
   ashp_heat.T_a_room = 20        # Room temperature [°C]
   ashp_heat.Q_r_int = 6000       # Heating load [W]
   ashp_heat.Q_r_max = 9000       # Maximum capacity [W]

   # Run calculation
   ashp_heat.system_update()

   # Access results
   print(f"COP: {ashp_heat.COP:.2f}")
   print(f"Compressor power: {ashp_heat.E_cmp:.2f} W")
   print(f"Internal fan power: {ashp_heat.E_fan_int:.2f} W")
   print(f"External fan power: {ashp_heat.E_fan_ext:.2f} W")
   print(f"Exergy efficiency: {ashp_heat.X_eff:.4f}")

   # Print exergy balance
   print_balance(ashp_heat.exergy_balance)

Electric Heater
---------------

Basic Usage with Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from enex_analysis_engine import ElectricHeater, print_balance
   import matplotlib.pyplot as plt

   # Initialize electric heater
   heater = ElectricHeater()

   # Set heater properties
   heater.E_heater = 1000        # Electric power [W]
   heater.T0 = 0                 # Reference temperature [°C]
   heater.T_a_room = 20          # Room air temperature [°C]
   heater.T_mr = 15              # Room surface temperature [°C]
   heater.T_init = 20            # Initial heater temperature [°C]
   heater.dt = 10                # Time step [s]

   # Run transient calculation
   heater.system_update()

   # Access final results
   print(f"Final heater body temperature: {heater.T_hb - 273.15:.2f} °C")
   print(f"Final heater surface temperature: {heater.T_hs - 273.15:.2f} °C")
   print(f"Exergy efficiency: {heater.X_eff:.4f}")

   # Plot transient behavior
   plt.figure(figsize=(12, 8))

   plt.subplot(2, 2, 1)
   plt.plot(heater.time, [T - 273.15 for T in heater.T_hb_list], 'b-', label='Body')
   plt.plot(heater.time, [T - 273.15 for T in heater.T_hs_list], 'r-', label='Surface')
   plt.xlabel('Time [s]')
   plt.ylabel('Temperature [°C]')
   plt.legend()
   plt.grid(True)
   plt.title('Temperature Evolution')

   plt.subplot(2, 2, 2)
   plt.plot(heater.time, heater.Q_conv_list, 'g-', label='Convection')
   plt.plot(heater.time, heater.Q_rad_hs_list, 'm-', label='Radiation')
   plt.xlabel('Time [s]')
   plt.ylabel('Heat Transfer [W]')
   plt.legend()
   plt.grid(True)
   plt.title('Heat Transfer Rates')

   plt.subplot(2, 2, 3)
   plt.plot(heater.time, heater.X_heater_list, 'k-', label='Input')
   plt.plot(heater.time, [x + y for x, y in zip(heater.X_conv_list, heater.X_rad_hs_list)], 
            'b-', label='Output')
   plt.xlabel('Time [s]')
   plt.ylabel('Exergy [W]')
   plt.legend()
   plt.grid(True)
   plt.title('Exergy Flows')

   plt.subplot(2, 2, 4)
   plt.plot(heater.time, heater.X_c_hb_list, 'r-', label='Body')
   plt.plot(heater.time, heater.X_c_hs_list, 'b-', label='Surface')
   plt.xlabel('Time [s]')
   plt.ylabel('Exergy Consumed [W]')
   plt.legend()
   plt.grid(True)
   plt.title('Exergy Destruction')

   plt.tight_layout()
   plt.show()

   # Print final balances
   print("\n=== Final Energy Balance ===")
   print_balance(heater.energy_balance)

   print("\n=== Final Exergy Balance ===")
   print_balance(heater.exergy_balance)

Fan and Pump
------------

Fan Performance
^^^^^^^^^^^^^^^

.. code-block:: python

   from enex_analysis_engine import Fan
   import numpy as np
   import matplotlib.pyplot as plt

   # Initialize fan
   fan = Fan()

   # Get performance at different flow rates
   flow_rates = np.linspace(0.5, 3.0, 26)
   pressures = []
   powers = []
   efficiencies = []

   for dV in flow_rates:
       try:
           p = fan.get_pressure(fan.fan1, dV)
           eff = fan.get_efficiency(fan.fan1, dV)
           power = fan.get_power(fan.fan1, dV)
           pressures.append(p)
           efficiencies.append(eff)
           powers.append(power)
       except:
           pressures.append(np.nan)
           efficiencies.append(np.nan)
           powers.append(np.nan)

   # Plot performance curves
   fig, axes = plt.subplots(1, 3, figsize=(15, 5))

   axes[0].plot(flow_rates, pressures, 'b-', linewidth=2)
   axes[0].set_xlabel('Flow Rate [m³/s]')
   axes[0].set_ylabel('Pressure [Pa]')
   axes[0].grid(True)
   axes[0].set_title('Fan Pressure vs Flow Rate')

   axes[1].plot(flow_rates, powers, 'r-', linewidth=2)
   axes[1].set_xlabel('Flow Rate [m³/s]')
   axes[1].set_ylabel('Power [W]')
   axes[1].grid(True)
   axes[1].set_title('Fan Power vs Flow Rate')

   axes[2].plot(flow_rates, efficiencies, 'g-', linewidth=2)
   axes[2].set_xlabel('Flow Rate [m³/s]')
   axes[2].set_ylabel('Efficiency [-]')
   axes[2].grid(True)
   axes[2].set_title('Fan Efficiency vs Flow Rate')

   plt.tight_layout()
   plt.show()

   # Display performance graph
   fan.show_graph()

Pump Performance
^^^^^^^^^^^^^^^^

.. code-block:: python

   from enex_analysis_engine import Pump
   import numpy as np
   import matplotlib.pyplot as plt

   # Initialize pump
   pump = Pump()

   # Get performance at different flow rates
   flow_rates = np.linspace(2, 6, 21)  # m³/h
   dP = 100000  # Pressure difference [Pa]
   powers = []
   efficiencies = []

   for V in flow_rates:
       V_m3s = V / 3600  # Convert to m³/s
       eff = pump.get_efficiency(pump.pump1, V_m3s)
       power = pump.get_power(pump.pump1, V_m3s, dP)
       efficiencies.append(eff)
       powers.append(power)

   # Plot performance curves
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

   ax1.plot(flow_rates, powers, 'b-', linewidth=2)
   ax1.set_xlabel('Flow Rate [m³/h]')
   ax1.set_ylabel('Power [W]')
   ax1.grid(True)
   ax1.set_title('Pump Power vs Flow Rate')

   ax2.plot(flow_rates, efficiencies, 'g-', linewidth=2)
   ax2.set_xlabel('Flow Rate [m³/h]')
   ax2.set_ylabel('Efficiency [-]')
   ax2.grid(True)
   ax2.set_title('Pump Efficiency vs Flow Rate')

   plt.tight_layout()
   plt.show()

   # Display performance graph
   pump.show_graph()

Balance Analysis
----------------

Comparing Energy, Entropy, and Exergy Balances
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from enex_analysis_engine import ElectricBoiler, print_balance

   # Initialize and run boiler
   boiler = ElectricBoiler()
   boiler.T_w_tank = 60
   boiler.T_w_sup = 10
   boiler.T_w_serv = 45
   boiler.dV_w_serv = 1.2
   boiler.system_update()

   # Print all balances
   print("=" * 60)
   print("ENERGY BALANCE")
   print("=" * 60)
   print_balance(boiler.energy_balance, decimal=2)

   print("\n" + "=" * 60)
   print("ENTROPY BALANCE")
   print("=" * 60)
   print_balance(boiler.entropy_balance, decimal=4)

   print("\n" + "=" * 60)
   print("EXERGY BALANCE")
   print("=" * 60)
   print_balance(boiler.exergy_balance, decimal=2)

   # Verify energy conservation
   for subsystem, balance in boiler.energy_balance.items():
       energy_in = sum(balance.get("in", {}).values())
       energy_out = sum(balance.get("out", {}).values())
       print(f"\n{subsystem}:")
       print(f"  Energy in: {energy_in:.2f} W")
       print(f"  Energy out: {energy_out:.2f} W")
       print(f"  Difference: {energy_in - energy_out:.2f} W")

System Comparison
-----------------

Comparing Different Boiler Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from enex_analysis_engine import ElectricBoiler, GasBoiler, HeatPumpBoiler
   import pandas as pd

   # Compare three boiler types
   systems = {
       "Electric": ElectricBoiler(),
       "Gas": GasBoiler(),
       "Heat Pump": HeatPumpBoiler()
   }

   # Set common operating conditions
   for name, system in systems.items():
       if hasattr(system, 'T_w_tank'):
           system.T_w_tank = 60
           system.T_w_sup = 10
           system.T_w_serv = 45
           system.dV_w_serv = 1.2
       if hasattr(system, 'COP'):
           system.COP = 2.5
       system.system_update()

   # Create comparison table
   comparison = []
   for name, system in systems.items():
       if hasattr(system, 'E_heater'):
           power_input = system.E_heater
       elif hasattr(system, 'E_NG'):
           power_input = system.E_NG
       elif hasattr(system, 'E_cmp'):
           power_input = system.E_cmp + system.E_fan
       else:
           power_input = 0
       
       comparison.append({
           "System": name,
           "Power Input [W]": power_input,
           "Exergy Efficiency": system.X_eff,
           "Exergy Destroyed [W]": system.X_c_tot
       })

   df = pd.DataFrame(comparison)
   print(df.to_string(index=False))

