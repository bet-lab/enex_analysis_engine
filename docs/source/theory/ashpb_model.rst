Air-Source Heat Pump Boiler (ASHPB) Model
=========================================

The Air-Source Heat Pump Boiler (ASHPB) model calculates the dynamic thermodynamic state of the vapor-compression cycle coupled to an outdoor-air evaporator and a lumped-capacitance domestic hot water tank. 

This model analytically integrates refrigerant thermodynamic cycle evaluations using the :math:`\varepsilon`-NTU method. By resolving the energy balance equations across the evaporator, condenser, and thermal storage tank, it determines the physics-based instantaneous minimum electrical power point.

System Modeling Overview
------------------------

The system modeling involves solving the energy balance between the internal refrigerant loop constraints, the external boundary conditions (outdoor air :math:`T_0` and mains water :math:`T_\text{w,sup}`), and the user hot water demand.

Refrigerant Cycle Thermodynamic States
--------------------------------------

The core of the refrigerant cycle relies on resolving the saturation temperatures at the evaporator and the condenser. 

The saturation temperatures are bounded by the thermal conditions at the heat exchangers:

.. math::
   T_\text{ref,evap,sat} = T_0 - \Delta T_\text{ref,evap}

.. math::
   T_\text{ref,cond,sat} = T_\text{w,tank} + \Delta T_\text{ref,cond}

Here, :math:`\Delta T_\text{ref,evap}` represents the approach temperature difference at the evaporator, which serves as the independent variable for our optimization algorithm. For the condenser, :math:`\Delta T_\text{ref,cond}` is determined directly from the heating demand and the heat exchanger's overall heat transfer coefficient:

.. math::
   \Delta T_\text{ref,cond} = \frac{Q_\text{ref,cond}}{UA_\text{cond}}

With user-defined superheat and subcool temperature specifications, the compressor inlet (suction) and expansion valve inlet states are fixed:

.. math::
   T_\text{ref,cmp,in} = T_\text{ref,evap,sat} + \Delta T_\text{superheat}

.. math::
   T_\text{ref,exp,in} = T_\text{ref,cond,sat} - \Delta T_\text{subcool}

From the above conditions, the internal states are solved sequentially using the CoolProp library:

1. Determine the evaporation pressure :math:`P_\text{evap}` and condensation pressure :math:`P_\text{cond}` from saturation temperatures.
2. Determine specific enthalpy :math:`h_\text{ref,cmp,in}` and entropy :math:`s_\text{ref,cmp,in}` at the compressor inlet.
3. Compute isentropic compressor discharge enthalpy :math:`h_\text{2,isen}` based on :math:`P_\text{cond}`.
4. Scale to the real compressor discharge enthalpy :math:`h_\text{ref,cmp,out}` using isentropic efficiency :math:`\eta_\text{cmp,isen}`:

.. math::
   h_\text{ref,cmp,out} = h_\text{ref,cmp,in} + \frac{h_\text{2,isen} - h_\text{ref,cmp,in}}{\eta_\text{cmp,isen}}

5. From the subcooled expansion valve inlet state :math:`h_\text{ref,exp,in}`, the flow undergoes an isenthalpic expansion, producing :math:`h_\text{ref,exp,out} = h_\text{ref,exp,in}` at :math:`P_\text{evap}`.

With these enthalpies, the mass flow rate of the refrigerant :math:`\dot{m}_\text{ref}` required to meet the condenser target heat transfer rate is:

.. math::
   \dot{m}_\text{ref} = \frac{Q_\text{ref,cond}}{h_\text{ref,cmp,out} - h_\text{ref,exp,in}}

The evaporator heat absorption :math:`Q_\text{ref,evap}` and compressor power :math:`E_\text{cmp}` are:

.. math::
   Q_\text{ref,evap} = \dot{m}_\text{ref} (h_\text{ref,cmp,in} - h_\text{ref,exp,out})

.. math::
   E_\text{cmp} = \dot{m}_\text{ref} (h_\text{ref,cmp,out} - h_\text{ref,cmp,in})

The compressor speed (RPM) :math:`N_\text{cmp}` is calculated mechanically without empirical maps:

.. math::
   N_\text{cmp} = \frac{\dot{m}_\text{ref}}{V_\text{disp,cmp} \cdot \rho_\text{ref,cmp,in}} \times 60

Optimal Operation Trajectory
----------------------------

The exact operation point is found dynamically at each step by minimizing the total electrical input :math:`E_\text{tot}` relative to the parameter :math:`\Delta T_\text{ref,evap}`. As :math:`\Delta T_\text{ref,evap}` increases, compression work jumps heavily while required fan flow falls, generating an obvious minimum power state.

Brent's 1-D optimization method evaluates:

.. math::
   \min_{\Delta T_\text{ref,evap}} \quad E_\text{tot} = E_\text{cmp} + E_\text{ou,fan}

Subject to the physical constraint:

.. math::
   1.0 \leq \Delta T_\text{ref,evap} \leq 20.0

Heat Exchanger Energy Balances
------------------------------

For the fin-and-tube evaporator exposed to ambient air, the air-side absorbed thermal energy is:

.. math::
   Q_\text{ou,a} = c_\text{a} \rho_\text{a} \dot{V}_\text{ou,fan,a} (T_\text{ou,a,in} - T_\text{ou,a,mid})

According to typical Colburn *j*-factor relations, we scale the overall heat transfer coefficient for variant flow rates via the Reynolds dependency :math:`j \propto \text{Re}^{-0.29}` dictating convective coefficients :math:`h \propto \dot{V}^{0.71}`:

.. math::
   UA_\text{evap} = UA_\text{evap,design} \left(\frac{\dot{V}_\text{ou,fan,a}}{\dot{V}_\text{ou,fan,a,design}}\right)^{0.71}

We apply the :math:`\varepsilon`-NTU method for resolving the matching temperature bounds:

.. math::
   \varepsilon = 1 - \exp\left(-\frac{UA_\text{evap}}{c_\text{a} \rho_\text{a} \dot{V}_\text{ou,fan,a}}\right)

.. math::
   T_\text{ou,a,mid} = T_\text{ou,a,in} - \varepsilon \left( T_\text{ou,a,in} - T_\text{ref,evap,sat} \right)

The final discharging air temperature rises slightly accounting for the thermal dissipation of the fan motor:

.. math::
   T_\text{ou,a,out} = T_\text{ou,a,mid} + \frac{E_\text{ou,fan}}{c_\text{a} \rho_\text{a} \dot{V}_\text{ou,fan,a}}

Outdoor Unit Fan Power Calculation
----------------------------------

Based on the ASHRAE Standard 90.1 Variable Speed Drive criteria, fan operational power is defined systematically:

.. math::
   E_\text{ou,fan} = E_\text{ou,fan,design} \times \text{PLR}

Where the Part Load Ratio (PLR) comes from polynomial relations bounded to flow ratios :math:`f_\text{flow}`:

.. math::
   f_\text{flow} = \frac{\dot{V}_\text{ou,fan,a}}{\dot{V}_\text{ou,fan,a,design}}

.. math::
   \text{PLR} = c_1 + c_2 f_\text{flow} + c_3 f_\text{flow}^2 + c_4 f_\text{flow}^3

The nominal design fan power assumes baseline pressure drop and efficiency:

.. math::
   E_\text{ou,fan,design} = \frac{\dot{V}_\text{ou,fan,a,design} \times \Delta P_\text{ou,fan,design}}{\eta_\text{ou,fan,design}}

Tank Energy Balance 
-------------------

The stratification is simplified into a fully-mixed lumped capacitance model evaluating temporal derivatives:

.. math::
   C_\text{tank} \frac{\mathrm{d}T_\text{w,tank}}{\mathrm{d}t} = Q_\text{ref,cond} - Q_\text{tank,loss} - Q_\text{tank,w,out} + Q_\text{tank,w,in}

With individual contributions formulated against a static reference thermal envelope defined roughly at ambient :math:`T_0`:

.. math::
   Q_\text{tank,loss} = UA_\text{tank} (T_\text{w,tank} - T_0)

.. math::
   Q_\text{tank,w,out} = c_\text{w} \rho_\text{w} \dot{V}_\text{tank,w,out} (T_\text{w,tank} - T_0)

.. math::
   Q_\text{tank,w,in} = c_\text{w} \rho_\text{w} \dot{V}_\text{tank,w,in} (T_\text{w,sup} - T_0)

Mixing Valve and Network Control
--------------------------------

A standard mixing valve reduces the highly elevated tank hot water temperature down to human operable values (:math:`T_\text{mix,w,out}`) by bypassing direct cold water supplies into the outgoing terminal:

.. math::
   \alpha = \frac{T_\text{mix,w,out} - T_\text{w,sup}}{T_\text{w,tank} - T_\text{w,sup}}

Where :math:`\alpha` determines physical fractional division:

.. math::
   \dot{V}_\text{tank,w,out} = \alpha \dot{V}_\text{mix,w,out}

.. math::
   \dot{V}_\text{mix,sup,w,in} = (1 - \alpha) \dot{V}_\text{mix,w,out}

System COP Metric
-----------------

Coefficient of Performance acts as the primary evaluation parameter unifying the thermodynamic efficacy of generated heat offset against total external parasitic and mechanical electrical input:

.. math::
   \mathrm{COP}_\text{sys} = \frac{Q_\text{ref,cond}}{E_\text{cmp} + E_\text{ou,fan}}
