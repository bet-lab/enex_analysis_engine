# Symbol Naming Convention

> **Version**: 1.4 — Canonical reference for all variable names in the `enex_analysis` package.

## 1. Core Structure

All variable names follow a standardized order separated by underscores (`_`):

```
[Physical Quantity]_[Medium or Subsystem]_[State or Component]_[Direction]
```

### 1.1 Core Principles

1. **Ref-First**: In the refrigerant cycle, `ref` is the primary subsystem. Components (`cmp`, `exp`) and states (`cond`, `evap`) appear *after* `ref`.
2. **Subsystem Priority**: For non-refrigerant systems, the subsystem precedes the medium (e.g., `ou` before `a`).
3. **Hierarchy Rule**: Medium symbols (`_w`, `_a`) MUST follow the entire multi-level subsystem hierarchy (e.g., `dV_ou_fan_a`).

### 1.2 Examples

| Variable | Physical Qty | Subsystem | State | Direction | Meaning |
|---|---|---|---|---|---|
| `Q_tank_w_in` | Heat Rate | `tank` | `w` (Water) | `in` | Heat entering the tank via water |
| `T_ref_cond_sat` | Temp | `ref` | `cond` | `sat` | Refrigerant saturation temp at condensing side |
| `T_ou_fan_a_out` | Temp | `ou` → `fan` | `a` (Air) | `out` | Air outlet temp from outdoor unit fan |
| `T_ref_cmp_in` | Temp | `ref` | `cmp` | `in` | Compressor inlet refrigerant temperature |

---

## 2. Component Symbols

### 2.1 Physical Quantity Prefixes

| Prefix | Quantity | Unit |
|---|---|---|
| `T` | Temperature | °C or K |
| `P` | Pressure | Pa |
| `Q` | Heat energy rate | W |
| `E` | Electrical power rate | W |
| `X` | Exergy flow rate | W |
| `Xc` | Exergy consumption (destruction) | W |
| `Xst` | Exergy storage (accumulation) | W |
| `S` | Entropy rate | W/K |
| `dV` | Volumetric flow rate | m³/s |
| `m_dot` | Mass flow rate | kg/s |
| `h` | Specific enthalpy | J/kg |
| `s` | Specific entropy | J/(kg·K) |
| `x` | Specific exergy | J/kg |
| `I` | Solar irradiance | W/m² |
| `v` | Velocity | m/s |
| `G` | Heat capacity flow | W/K |
| `UA` | Overall heat transfer coefficient | W/K |
| `C` | Heat capacity | J/K |
| `A` | Area | m² |

### 2.2 Medium and Subsystem Symbols

- **Medium**: `w` (water), `a` (air), `ref` (refrigerant), `sol` (solar radiation)
- **Subsystem**: `cmp` (compressor), `cond` (condenser), `evap` (evaporator), `exp` (expansion valve), `ou` (outdoor unit), `tank` (thermal storage), `stc` (solar thermal collector), `mix` (mixing valve), `pump` (pump), `bhe` (borehole heat exchanger)

### 2.3 Subsystem Hierarchy

For nested components: `[Subsystem]_[Component]` (higher → lower).

| Variable | Meaning |
|---|---|
| `E_ou_fan` | Fan power within the outdoor unit |
| `dV_ou_fan_a` | Air flow through the outdoor unit fan |
| `T_stc_pump_w_out` | Water outlet temp from the STC pump |

---

## 3. Refrigerant Cycle State Points

Traditional numerical state points are mapped to descriptive names:

### Primary States

| Legacy | Descriptive | Position |
|---|---|---|
| State 1 | `cmp_in` | Compressor inlet / Evaporator outlet |
| State 2 | `cmp_out` | Compressor outlet / Condenser inlet |
| State 3 | `exp_in` | Expansion valve inlet / Condenser outlet |
| State 4 | `exp_out` | Expansion valve outlet / Evaporator inlet |

### Saturation States

| Legacy | Descriptive | Meaning |
|---|---|---|
| State 1* | `evap_sat` | Saturated vapor at evaporator pressure |
| State 2* | `cond_sat_v` | Saturated vapor at condenser pressure |
| State 3* | `cond_sat_l` | Saturated liquid at condenser pressure |

---

## 4. Boundary Flow Naming (Upstream Rule)

When a flow exists between two subsystems, name it after the **upstream (source)** subsystem:

- Use `Q_tank_w_out` (not `Q_mix_w_in`)
- Use `T_ref_cmp_out` (not `T_ref_cond_in`)
- For external boundaries: use `_in` (e.g., `T_tank_w_in` for mains water)

---

## 5. Implementation Rules

1. **Units in Keys**: Result dict / DataFrame columns use `[unit]` notation: `'T_ref_cmp_in [°C]'`
2. **Aggregates**: System-wide totals omit the subsystem: `E_tot`, `X_tot`, `Xc_tot`
3. **Dimensionless**: `cop_ref`, `cop_sys`, `eta_cmp`, `ksi_stc`
4. **Design Values**: `_design` suffix (e.g., `dV_ou_design`)
5. **Losses**: `_loss` suffix (e.g., `Q_tank_loss`)
6. **Saturation**: `_sat` suffix (e.g., `T_ref_evap_sat`)
7. **Target/Setpoint Values**: `_target` suffix (e.g., `Q_cond_target`). Replaces legacy `_load` suffix.
8. **Heated/Modified Inputs**: `_heated` suffix for values modified by an upstream process (e.g., `T_tank_w_in_heated` = mains water preheated by STC).
9. **Flow Qualifiers (Modifier Before Medium)**: Qualifiers like `sup` (supply/supplemental), `serv` (service), or `heated` that describe the *type* of medium are placed **before** the medium symbol `w`, following the subsystem hierarchy (e.g., `dV_mix_sup_w_in` = supplemental water entering the mixing valve, not `dV_mix_w_in_sup`).

## 6. Heat Exchanger Variable Naming

| Category | Variable | Meaning |
|---|---|---|
| Condenser | `Q_cond_ref` | Heat rejected by refrigerant side |
| | `Q_cond_w` | Heat gained by water side |
| Evaporator | `Q_evap_ref` | Heat absorbed by refrigerant side |
| | `Q_ou_a` | Heat rejected by air side |
