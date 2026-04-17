"""Energy and Exergy Analysis Engine package init."""

from .ashpb_pv_ess import ASHPB_PV_ESS
from .ashpb_stc_preheat import ASHPB_STC_preheat
from .ashpb_stc_tank import ASHPB_STC_tank
from .gshpb_pv_ess import GSHPB_PV_ESS
from .gshpb_stc_preheat import GSHPB_STC_preheat
from .gshpb_stc_tank import GSHPB_STC_tank
from .air_source_heat_pump_boiler import (
    AirSourceHeatPumpBoiler,
)
from .calc_util import (
    C2F,
    C2K,
    F2C,
    GJ2J,
    J2GJ,
    J2MJ,
    K2C,
    MJ2J,
    MW2W,
    W2GW,
    W2MW,
    J2kJ,
    J2kWh,
    L2m3,
    MPa2Pa,
    Pa2atm,
    Pa2bar,
    Pa2kPa,
    Pa2MPa,
    W2kW,
    atm2Pa,
    bar2Pa,
    cm2in,
    cm2m,
    cm2mm,
    cm22m2,
    cm32m3,
    d2h,
    d2m,
    d2r,
    d2s,
    d2y,
    ft2m,
    g2kg,
    h2d,
    h2m,
    h2s,
    in2cm,
    kg2g,
    kg2mg,
    kg2t,
    kJ2J,
    km2m,
    kPa2Pa,
    kW2W,
    kWh2J,
    m2cm,
    m2d,
    m2ft,
    m2h,
    m2km,
    m2mm,
    m2s,
    m22cm2,
    m22mm2,
    m32cm3,
    m32L,
    mg2kg,
    mm2cm,
    mm2m,
    mm22m2,
    r2d,
    s2d,
    s2h,
    s2m,
    t2kg,
    y2d,
)
from .components.electric_heater import (
    ElectricHeater,
)
from .components.fan import (
    Fan,
)
from .components.pump import (
    Pump,
)
from .constants import (
    P0_PA,
    SP,
    T0_K,
)
from .dhw import (
    build_dhw_usage_ratio,
    calc_cold_water_temp,
    calc_total_water_use_from_schedule,
    make_dhw_schedule_from_Annex_42_profile,
)
from .dynamic_context import (
    ControlState,
    StepContext,
    Subsystem,
    SubsystemExergy,
    check_hp_schedule_active,
    determine_heat_source_on_off,
    determine_tank_refill_flow,
    tank_mass_energy_residual,
)
from .electric_boiler import (
    ElectricBoiler,
)
from .enex_engine import (
    AirSourceHeatPump_cooling,
    AirSourceHeatPump_heating,
)
from .enex_functions import (
    G_FLS,
    TDMA,
    air_dynamic_viscosity,
    air_prandtl_number,
    calc_ASHP_cooling_COP,
    calc_ASHP_heating_COP,
    calc_boussinessq_mixing_flow,
    calc_fan_power_from_dV_fan,
    calc_GSHP_COP,
    calc_HX_perf_for_target_heat,
    calc_mixing_valve,
    calc_Orifice_flow_coefficient,
    calc_stc_performance,
    calc_UA_from_dV_fan,
    calc_uv_exposure_time,
    calc_uv_lamp_power,
    chi,
    cubic_function,
    f,
    get_uv_params_from_turbidity,
    linear_function,
    print_balance,
    quadratic_function,
    quartic_function,
    update_tank_temperature,
)
from .gas_boiler import (
    GasBoiler,
)
from .gas_boiler_tank import (
    GasBoilerTank,
)
from .ground_source_heat_pump_boiler import (
    GroundSourceHeatPumpBoiler,
)
from .ground_source_heat_pump import GroundSourceHeatPump
from .ground_source_heat_pump_ref_cycle import (
    GroundSourceHeatPump_RefCycle,
)
from .heat_transfer import (
    TRIDIAG_MATRIX_ALGORITHM,
    calc_h_vertical_plate,
    calc_LMTD_counter_flow,
    calc_LMTD_parallel_flow,
    calc_simple_tank_UA,
    calc_UA_tank_arr,
    darcy_friction_factor,
)
from .refrigerant import (
    calc_ref_state,
    create_lmtd_constraints,
    find_ref_loop_optimal_operation,
)
from .solar_assisted_gas_boiler import (
    SolarAssistedGasBoiler,
)
from .subsystems import (
    PhotovoltaicSystem,
    SolarThermalCollector,
    UVLamp,
)
from .tank_stratification_model import (
    StratifiedTankTDMA,
)
from .thermodynamics import (
    calc_energy_flow,
    calc_exergy_flow,
    calc_refrigerant_exergy,
    convert_electricity_to_exergy,
    generate_entropy_exergy_term,
)
from .visualization import (
    plot_ph_diagram,
    plot_th_diagram,
    plot_ts_diagram,
    print_simulation_summary,
)
from .weather import (
    decompose_ghi_to_poa,
    load_kma_solar_csv,
    load_kma_T0_sol_hourly_csv,
)
