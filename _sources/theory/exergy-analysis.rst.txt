Exergy Analysis Theory
======================

Detailed explanation of exergy analysis principles and their application to energy systems.

What is Exergy?
---------------

Exergy (also called available energy or availability) represents the maximum useful work that can be obtained from a system as it comes into equilibrium with its environment.

Exergy Balance
---------------

The exergy balance combines both first and second law principles:

.. math::
   \\sum \\dot{X}_{in} = \\sum \\dot{X}_{out} + \\dot{X}_{destroyed} + \\frac{dX_{system}}{dt}

For steady-state systems:

.. math::
   \\sum \\dot{X}_{in} = \\sum \\dot{X}_{out} + \\dot{X}_{destroyed}

Exergy destruction is related to entropy generation:

.. math::
   \\dot{X}_{destroyed} = T_0 \\cdot \\dot{S}_{gen}

where :math:`T_0` is the reference (environmental) temperature.

Exergy Efficiency
-----------------

The exergy efficiency is defined as:

.. math::
   \\eta_{ex} = \\frac{\\dot{X}_{useful}}{\\dot{X}_{input}} = 1 - \\frac{\\dot{X}_{destroyed}}{\\dot{X}_{input}}

This provides a true measure of thermodynamic efficiency, accounting for both energy quantity and quality.

Physical Exergy
---------------

Heat Transfer Exergy
^^^^^^^^^^^^^^^^^^^^

For a heat transfer process, the exergy associated with heat transfer at temperature :math:`T` is:

.. math::
   \\dot{X}_Q = \\dot{Q} \\left(1 - \\frac{T_0}{T}\\right)

Mass Flow Exergy
^^^^^^^^^^^^^^^^

For a mass flow, the exergy includes thermal, mechanical, and chemical components. For many systems, thermal exergy dominates:

.. math::
   \\dot{X}_{mass} = \\dot{m} \\left[h - h_0 - T_0(s - s_0)\\right]

where :math:`h` is specific enthalpy and :math:`s` is specific entropy.

For ideal liquids (water) or ideal gases (air) with constant specific heat:

.. math::
   \\dot{X}_{flow} = G \\left[(T - T_0) - T_0 \\ln\\frac{T}{T_0}\\right]

where :math:`G = c \\cdot \\rho \\cdot \\dot{V}` is the heat capacity flow rate [W/K].

System-Specific Exergy Analysis
--------------------------------

Electric Boiler
^^^^^^^^^^^^^^^

The electric boiler converts electrical energy directly to thermal energy. The energy balance for the tank is:

.. math::
   E_{heater} + \\dot{Q}_{w,sup,tank} = \\dot{Q}_{w,tank} + \\dot{Q}_{l,tank}

The exergy efficiency accounts for the quality of energy:

.. math::
   \\eta_{ex} = \\frac{\\dot{X}_{w,serv}}{E_{heater}}

Gas Boiler
^^^^^^^^^^

For natural gas combustion, the chemical exergy is related to the heating value:

.. math::
   X_{NG} = \\eta_{ex,NG} \\cdot E_{NG}

where :math:`\\eta_{ex,NG} = 0.93` for liquefied natural gas (LNG) based on Shukuya (2013).

Air Source Heat Pump Boiler (ASHPB)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ASHPB system consists of seven exergy-accountable components:
compressor, condenser, expansion valve, evaporator, outdoor unit fan,
storage tank, and mixing valve.

**System topology:**

.. code-block:: text

   [outdoor air] ──→ [Evaporator + Fan (OU)] ──→ [cooled air]
                          ↑ ref state 4        ↓ ref state 1
                     [Expansion Valve]     [Compressor] ← E_cmp
                          ↑ ref state 3        ↓ ref state 2
                     [Condenser (in tank)] ──→ Q_ref_cond
                               │
                        [Storage Tank] ← X_uv, X_stc, X_tank_w_in
                          ↓ X_tank_w_out
                        [Mixing Valve] ← X_mix_sup_w_in
                          ↓ X_mix_w_out
                        [User (DHW)]

**Component-level exergy balances** (general form: :math:`X_c = \\sum X_{in} - \\sum X_{out} \\geq 0`):

.. list-table:: ASHPB exergy consumption per component
   :header-rows: 1
   :widths: 20 50 30

   * - Component
     - Exergy Balance
     - Notes
   * - Compressor
     - :math:`X_{c,cmp} = X_{cmp} + X_{ref,cmp\_in} - X_{ref,cmp\_out}`
     - :math:`X_{cmp} = E_{cmp}` (electricity = pure exergy)
   * - Condenser
     - :math:`X_{c,cond} = X_{ref,cmp\_out} - X_{ref,exp\_in} - X_{ref,cond}`
     - :math:`X_{ref,cond} = Q_{ref,cond}(1 - T_0/T_{cond,sat})`
   * - Expansion Valve
     - :math:`X_{c,exp} = X_{ref,exp\_in} - X_{ref,exp\_out}`
     - Isenthalpic throttling
   * - Evaporator (HX)
     - :math:`X_{c,evap} = (X_{ref,exp\_out} + X_{a,ou\_in}) - (X_{ref,cmp\_in} + X_{a,ou\_mid})`
     - Air exits at :math:`T_{ou,a,mid}` (before fan)
   * - OU Fan
     - :math:`X_{c,fan} = X_{ou\_fan} + X_{a,ou\_mid} - X_{a,ou\_out}`
     - Fan heat raises air from :math:`T_{mid}` to :math:`T_{out}`
   * - Storage Tank
     - :math:`X_{c,tank} = (X_{ref,cond} + X_{tank,w\_in} + X_{uv} + X_{stc}) - (X_{tank,w\_out} + X_{tank,loss} + X_{st,tank})`
     - :math:`X_{st,tank}` = stored exergy accumulation
   * - Mixing Valve
     - :math:`X_{c,mix} = X_{tank,w\_out} + X_{mix,sup\_w\_in} - X_{mix,w\_out}`
     - Irreversibility from mixing hot and cold streams

**Refrigerant state-point exergy** is evaluated using CoolProp at each state point:

.. math::
   X_{ref,i} = \\dot{m}_{ref} \\left[(h_i - h_0) - T_0 (s_i - s_0)\\right]

where :math:`h_0, s_0` are evaluated at :math:`(T_0, P_0)`.

**Total system exergy input:**

.. math::
   X_{tot} = E_{cmp} + E_{ou,fan} + E_{uv} + E_{stc,pump}

where UV lamp and STC pump terms are included only when those subsystems are active.

**Exergy efficiency metrics:**

.. math::
   \\eta_{X,ref} = \\frac{X_{ref,cond}}{X_{cmp}}, \\qquad
   \\eta_{X,sys} = \\frac{X_{ref,cond}}{X_{tot}}

**Tank stored exergy** (dynamic accumulation term):

.. math::
   X_{st,tank} = \\left(1 - \\frac{T_0}{T_{tank}}\\right) \\cdot C_{tank} \\cdot \\frac{\\Delta T_{tank}}{\\Delta t}

Heat Pump COP
"""""""""""""

The Coefficient of Performance (COP) for a heat pump is:

.. math::
   \\text{COP} = \\frac{\\dot{Q}_{useful}}{E_{input}}

For air-source heat pumps, the COP depends on outdoor temperature and part-load ratio.

For ground-source heat pumps, a modified Carnot-based COP is used:

.. math::
   \\text{COP} = \\frac{1}{1 - \\frac{T_g}{T_{cond}} + \\frac{\\Delta T \\cdot \\hat{\\theta}}{T_{cond}}}

Solar Thermal Collectors
^^^^^^^^^^^^^^^^^^^^^^^^

For solar thermal collectors, the entropy of solar radiation is calculated as:

.. math::
   S_{sol} = k_D \\cdot I_{DN}^{0.9} + k_d \\cdot I_{dH}^{0.9}

where:

* :math:`k_D = 0.000462` is the direct solar entropy coefficient
* :math:`k_d = 0.0014` is the diffuse solar entropy coefficient
* :math:`I_{DN}` is direct normal irradiance [W/m²]
* :math:`I_{dH}` is diffuse horizontal irradiance [W/m²]

The exergy of solar radiation is:

.. math::
   X_{sol} = Q_{sol} - S_{sol} \\cdot T_0

References
----------

* Shukuya, M. (2013). *Exergy theory and applications in the built environment*. Springer.
* IBPSA (2023). Performance modeling of air-source heat pumps. *Building Simulation Conference*.
* MDPI (2023). Empirical COP correlations for air-source heat pumps. *Sustainability*, 15(3), 1880.
