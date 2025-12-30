Configuration
=============

This page explains how to configure and customize the Energy-Exergy Analysis Engine for your specific needs.

System Parameters
-----------------

Reference Temperature
^^^^^^^^^^^^^^^^^^^^^

The reference temperature (``T0``) is crucial for exergy calculations. It represents the dead state temperature:

.. code-block:: python

   system.T0 = 10  # Reference temperature in °C

.. note::
   The reference temperature should represent the ambient conditions. Typical values are 0°C, 10°C, or 25°C.

Default Values
--------------

Each system class comes with sensible default values for physical properties:

* Water density: 1000 kg/m³
* Water specific heat: 4186 J/(kg·K)
* Air density: 1.2 kg/m³
* Insulation thermal conductivity: 0.03 W/(m·K)

You can override these defaults by setting the attributes directly:

.. code-block:: python

   EB = enex.ElectricBoiler()
   EB.k_ins = 0.04  # Custom insulation conductivity
   EB.rho_w = 995   # Custom water density

Advanced Configuration
----------------------

For more complex scenarios, you can:

* Modify subsystem parameters
* Adjust calculation tolerances iterative solvers
* Customize output formats

See the :doc:`../guides/user-guide` for detailed examples.
