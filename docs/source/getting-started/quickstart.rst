Quick Start
===========

This guide will help you get up and running quickly with the Energy-Exergy Analysis Engine.

First Example: Electric Boiler
-------------------------------

Let's create a simple electric res boiler system and analyze its performance:

.. code-block:: python

   import enex_analysis_engine as enex

   # Create an electric boiler instance
   EB = enex.ElectricBoiler()
   
   # Set the reference temperature (in Celsius)
   EB.T0 = 10
   EB.T_w_tank = 60   # Tank water temperature [°C]
   EB.T_w_sup = 10    # Supply water temperature [°C]
   EB.T_w_serv = 45   # Service water temperature [°C]
   EB.dV_w_serv = 1.2 # Service water flow rate [L/min]
   
   # Update the system calculations
   EB.system_update()
   
   # Print the exergy balance
   enex.print_balance(EB.exergy_balance)
   
   # Access results
   print(f"Electric power input: {EB.E_heater:.2f} W")
   print(f"Exergy efficiency: {EB.X_eff:.4f}")

Basic Workflow
--------------

All system models follow this consistent pattern:

1. **Import** the package
2. **Initialize** the system class
3. **Set parameters** (temperatures, flow rates, etc.)
4. **Call** ``system_update()`` to perform calculations
5. **Access results** from object attributes
6. **Print balances** using ``print_balance()``

Next Steps
----------

* Read the :doc:`../guides/user-guide` for comprehensive documentation
* Explore :doc:`../examples/index` for more complex scenarios
* Check the :doc:`../api/index` for detailed API references
* Learn about the :doc:`../theory/index` behind the calculations
