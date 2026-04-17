import math
from dataclasses import dataclass
from enex_analysis.air_source_heat_pump_ref_cycle import AirSourceHeatPump_cooling, AirSourceHeatPump_heating

def simulate_ashp_step(args):
    """
    Worker function for parallel ASHP simulation.
    args: (step, q_load, t0_val)
    """
    step, q_load, t0_val = args
    
    # Identify mode and instantiate model
    if abs(q_load) < 100:
        q_load = 0
        mode = "Off"
        model = AirSourceHeatPump_cooling() # Default to cooling for Off mode
        model.Q_iu = 0
    elif q_load > 0:
        mode = "Cooling"
        model = AirSourceHeatPump_cooling()
        model.Q_iu = q_load
    else:
        mode = "Heating"
        model = AirSourceHeatPump_heating()
        model.Q_iu = -q_load # Input Q_iu is positive for heating model
        
    model.T0 = t0_val
    
    # Solve 2D optimization
    try:
        model.system_update()
    except Exception as e:
        # In case of numerical failure, return a safe dummy state
        return {"Hour": step + 1, "Mode": mode, "Error": str(e), "E_tot": 0}

    # Extract all scalar variables from the model instance
    res = {k: v for k, v in vars(model).items() if isinstance(v, (int, float, str, bool))}
    
    # Ensure key cumulative metrics are present
    res.update({
        "Hour": step + 1,
        "Mode": mode,
        "X_Eff_percent": getattr(model, "X_eff", 0) * 100,
        "E_tot": getattr(model, "E_tot", 0),
        "X_in_tot": getattr(model, "X_in_tot", 0),
        "X_out_tot": getattr(model, "X_out_tot", 0)
    })
    
    return res
