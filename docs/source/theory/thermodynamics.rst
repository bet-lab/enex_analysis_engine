Thermodynamics Fundamentals
============================

This page covers the fundamental thermodynamic principles used in energy-exergy analysis.

Laws of Thermodynamics
-----------------------

First Law: Energy Conservation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The energy balance is based on the conservation of energy principle:

.. math::
   \\sum \\dot{E}_{in} = \\sum \\dot{E}_{out} + \\dot{E}_{loss} + \\frac{dE_{system}}{dt}

For steady-state systems:

.. math::
   \\sum \\dot{E}_{in} = \\sum \\dot{E}_{out} + \\dot{E}_{loss}

Energy analysis tells us *how much* energy flows through a system, but not *how well* it's used.

Second Law: Entropy Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The entropy balance accounts for entropy generation due to irreversibilities:

.. math::
   \\sum \\dot{S}_{in} + \\dot{S}_{gen} = \\sum \\dot{S}_{out} + \\frac{dS_{system}}{dt}

For steady-state systems:

.. math::
   \\sum \\dot{S}_{in} + \\dot{S}_{gen} = \\sum \\dot{S}_{out}

Entropy generation is always positive for real processes and represents the irreversibility of the process.

Heat Transfer Correlations
---------------------------

Natural Convection
^^^^^^^^^^^^^^^^^^

For vertical plates, the Churchill & Chu correlation is used:

.. math::
   \\text{Ra}_L = \\frac{g \\beta \\Delta T L^3}{\\nu^2} \\text{Pr}

.. math::
   \\text{Nu}_L = \\left(0.825 + \\frac{0.387 \\text{Ra}_L^{1/6}}{[1 + (0.492/\\text{Pr})^{9/16}]^{8/27}}\\right)^2

.. math::
   h = \\frac{\\text{Nu}_L \\cdot k_{air}}{L}

where:

* :math:`\\text{Ra}_L` is Rayleigh number
* :math:`\\text{Nu}_L` is Nusselt number
* :math:`\\text{Pr}` is Prandtl number
* :math:`g` is gravitational acceleration
* :math:`\\beta` is thermal expansion coefficient
* :math:`\\nu` is kinematic viscosity
* :math:`L` is characteristic length

References
----------

* Churchill, S. W., & Chu, H. H. S. (1975). Correlating equations for laminar and turbulent free convection from a vertical plate. *International Journal of Heat and Mass Transfer*, 18(11), 1323-1329.
