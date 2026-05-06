import io
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

matplotlib.use("Agg")
import dartwork_mpl as dm
import matplotlib.pyplot as plt

from enex_analysis.air_source_heat_pump_boiler import AirSourceHeatPumpBoiler
from enex_analysis.visualization import plot_ph_diagram, plot_th_diagram

router = APIRouter()
_annual_df = None
ANNUAL_DATA_PATH = Path("/home/habin/Codes/enex_engine/simulation/ashpb_benchmark_and_annual/data/annual_results.csv")

router = APIRouter()


class ASHPBSimulationRequest(BaseModel):
    season_profile: str = "Constant"  # "Constant", "Mild", "Summer", "Winter"
    simulation_period_hours: int = 24
    dt_s: int = 180
    T_tank_init_C: float = 45.0
    T0_C: float = 5.0
    dhw_usage_LPM: float = 0.0
    hp_capacity: float = 15000.0


@router.get("/weather-files")
def get_weather_files():
    return ["Constant", "Mild", "Summer", "Winter"]


def load_season_data(season: str, tN: int, dt_s: int):
    # Base paths
    data_dir = Path("/home/habin/Codes/enex_engine/00 data")

    # 1. Load DHW Data
    dhw_m3s_1day = np.zeros(1440)
    dhw_csv_path = data_dir / "Annex_42_DHW_1min_200L.csv"
    if dhw_csv_path.exists():
        df_dhw = pd.read_csv(dhw_csv_path)
        # Using L/h from Annex 42
        if "Jan" in df_dhw.columns:
            dhw_m3s_1day = df_dhw.loc[0:1439, "Jan"].values / 3600.0 / 1000.0
        else:
            dhw_m3s_1day = df_dhw.iloc[0:1440, 0].values / 3600.0 / 1000.0

    # 2. Load Weather
    seasons_map = {"Mild": "Seoul_251101_T0.csv", "Summer": "Seoul_250801_T0.csv", "Winter": "Seoul_250101_T0.csv"}

    t0_1day_raw = np.full(1440, 5.0)
    if season in seasons_map:
        file_path = data_dir / seasons_map[season]
        if file_path.exists():
            t0_df = pd.read_csv(file_path, encoding="cp949")
            loc_cols = [c for c in t0_df.columns if "지점" in c]
            if loc_cols:
                # filter for Seoul
                t0_df = t0_df[t0_df[loc_cols[0]] == 108]
            temp_cols = [c for c in t0_df.columns if "기온" in c or "temp" in c.lower() or "t0" in c.lower()]
            if temp_cols:
                t0_raw = t0_df[temp_cols[0]].values
                t0_1day_raw = (
                    t0_raw[:1440]
                    if len(t0_raw) >= 1440
                    else np.pad(t0_raw, (0, max(0, 1440 - len(t0_raw))), mode="edge")
                )

    # Resize arrays matching the requested simulation dt_s
    steps_per_day = int(24 * 3600 / dt_s)
    t_raw = np.linspace(0, 1, len(t0_1day_raw))
    t_target = np.linspace(0, 1, steps_per_day)

    t0_1day = np.interp(t_target, t_raw, t0_1day_raw)

    dhw_raw = np.linspace(0, 1, len(dhw_m3s_1day))
    dhw_1day = np.interp(t_target, dhw_raw, dhw_m3s_1day)

    # Tile out to length
    days = int(np.ceil(tN / steps_per_day))
    t0_full = np.tile(t0_1day, days)[:tN]
    dhw_full = np.tile(dhw_1day, days)[:tN]

    return t0_full, dhw_full


@router.post("/simulate/ashpb")
def simulate_ashpb(req: ASHPBSimulationRequest):
    ashpb = AirSourceHeatPumpBoiler(hp_capacity=req.hp_capacity)

    simulation_period_sec = int(req.simulation_period_hours * 3600)
    tN = int(simulation_period_sec / req.dt_s)

    if req.season_profile == "Constant":
        T0_schedule = np.full(tN, req.T0_C)
        dhw_m3s = (req.dhw_usage_LPM / 60.0) / 1000.0
        dhw_usage_schedule = np.full(tN, dhw_m3s)
    else:
        T0_schedule, dhw_usage_schedule = load_season_data(req.season_profile, tN, req.dt_s)

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
        result_save_csv_path=None,
    )

    columns_to_keep = ["T_tank_w [°C]", "E_cmp [W]", "E_ou_fan [W]", "cop_sys [-]", "E_tot [W]"]
    cols = [c for c in columns_to_keep if c in df.columns]

    df_out = df[cols].copy()
    df_out["time_h"] = np.arange(len(df_out)) * req.dt_s / 3600.0

    # To properly compute cumulative energy
    if "E_tot [W]" in df_out.columns:
        df_out["cumulative_energy [kWh]"] = (df_out["E_tot [W]"].fillna(0) * req.dt_s / 3600 / 1000).cumsum()

    # Convert NaNs to None
    df_out = df_out.replace({np.nan: None})
    return {"results": df_out.to_dict(orient="records")}


@router.get("/annual-data", operation_id="get_annual_data")
def get_annual_data():
    global _annual_df
    if not ANNUAL_DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Annual data not found")

    if _annual_df is None:
        _annual_df = pd.read_csv(ANNUAL_DATA_PATH)

    cols_to_keep = ["time_h", "T_tank_w [°C]", "E_tot [W]", "E_cmp [W]", "cop_sys [-]", "hp_is_on"]
    kept = [c for c in cols_to_keep if c in _annual_df.columns]

    # Downsample by 15 (every 15 mins if dt=60s)
    df_down = _annual_df[kept].iloc[::15].copy()
    if "E_tot [W]" in df_down.columns:
        df_down["cumulative_energy [kWh]"] = (df_down["E_tot [W]"].fillna(0) * 60 * 15 / 3600 / 1000).cumsum()

    # Provide original index so frontend can request cyclic diagrams referencing it
    df_down["original_idx"] = df_down.index

    df_down = df_down.replace({np.nan: None})
    return {"results": df_down.to_dict(orient="records")}


class CycleDiagramRequest(BaseModel):
    index: int


@router.post("/cycle-diagram", operation_id="generate_cycle_diagram")
def generate_cycle_diagram(req: CycleDiagramRequest):
    global _annual_df
    if not ANNUAL_DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Annual data not found")

    if _annual_df is None:
        _annual_df = pd.read_csv(ANNUAL_DATA_PATH)

    if req.index < 0 or req.index >= len(_annual_df):
        raise HTTPException(status_code=400, detail="Index out of bounds")

    row = _annual_df.iloc[req.index]
    result_dict = row.to_dict()

    refrigerant = "R134a"  # Model default
    T_tank = row.get("T_tank_w [°C]", 50.0)
    T0 = row.get("T_ou_a_in [°C]", 5.0)

    dm.style.use("scientific")
    plt.rcParams["lines.linewidth"] = 0.5

    FIG_W = dm.cm2in(16.0)
    FIG_H = dm.cm2in(6.0)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(wspace=0.3)

    try:
        plot_ph_diagram(axes[0], result_dict, refrigerant)
        plot_th_diagram(axes[1], result_dict, refrigerant, T_tank=T_tank, T0=T0)
        dm.simple_layout(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        svg_str = buf.read().decode("utf-8")

        return {"svg": svg_str}
    except Exception as e:
        plt.close(fig)
        raise HTTPException(status_code=500, detail=f"Failed to generate diagram: {e}")
