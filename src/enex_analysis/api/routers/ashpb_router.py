import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd

from enex_analysis.air_source_heat_pump_boiler import AirSourceHeatPumpBoiler

router = APIRouter()

class ASHPBSimulationRequest(BaseModel):
    simulation_period_hours: int = 24
    dt_s: int = 60
    T_tank_init_C: float = 45.0
    T0_C: float = 5.0
    dhw_usage_LPM: float = 0.0

@router.post("/simulate/ashpb")
def simulate_ashpb(req: ASHPBSimulationRequest):
    ashpb = AirSourceHeatPumpBoiler()
    
    simulation_period_sec = int(req.simulation_period_hours * 3600)
    tN = int(simulation_period_sec / req.dt_s)
    
    # Generate static schedules based on inputs
    T0_schedule = np.full(tN, req.T0_C)
    dhw_m3s = (req.dhw_usage_LPM / 60.0) / 1000.0  # LPM to m3/s
    dhw_usage_schedule = np.full(tN, dhw_m3s)
    
    I_DN_schedule = np.zeros(tN)
    I_dH_schedule = np.zeros(tN)
    
    df = ashpb.analyze_dynamic(
        simulation_period_sec=simulation_period_sec,
        dt_s=req.dt_s,
        T_tank_w_init_C=req.T_tank_init_C,
        dhw_usage_schedule=dhw_usage_schedule,
        T0_schedule=T0_schedule,
        I_DN_schedule=I_DN_schedule,
        I_dH_schedule=I_dH_schedule,
        result_save_csv_path=None
    )
    
    # Filter columns to reduce payload size
    columns_to_keep = [
        "T_tank_w [°C]", 
        "Q_ref_cond [W]", 
        "E_tot [W]", 
        "T_mix_w_out [°C]"
    ]
    cols = [c for c in columns_to_keep if c in df.columns]
    
    df_out = df[cols].copy()
    df_out["time_h"] = np.arange(tN) * req.dt_s / 3600.0
    
    # Convert NaNs to None for JSON compliance
    df_out = df_out.replace({np.nan: None})
    
    return {"results": df_out.to_dict(orient="records")}
