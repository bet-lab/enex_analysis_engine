import os
import json
import requests
import pandas as pd
from pathlib import Path
from .. import calc_util as cu

CACHE_DIR = Path("00_data/.api_cache")

def get_kma_api_key() -> str:
    # 1. 환경변수 확인
    key = os.getenv("KMA_API_KEY")
    if key:
        return key
    
    # 2. 파일 확인
    key_path = Path("KALIS_report/API/API_key_기상청02_지상(종관ASOS)일자료.txt")
    if key_path.exists():
        with open(key_path, "r", encoding="utf-8") as f:
            return f.read().strip()
            
    raise ValueError("KMA_API_KEY environment variable or key file not found.")

def get_kma_weather_data(start_date: str, end_date: str, stn_id: int = 108) -> pd.DataFrame:
    """
    Fetch KMA ASOS Hourly data (Temperature and Solar Irradiance).
    Uses local cache to avoid redundant API calls.
    Returns standard dataframe with 'datetime', 'T0_K', and 'ghi' (W/m2).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"kma_{stn_id}_{start_date}_{end_date}.csv"
    
    if cache_path.exists():
        print(f"Loading KMA weather data from cache: {cache_path}")
        df = pd.read_csv(cache_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        return df

    api_key = get_kma_api_key()
    url = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
    
    all_items = []
    page_no = 1
    num_of_rows = 999
    
    # KMA API parameter format: YYYYMMDD
    start_dt = pd.to_datetime(start_date).strftime("%Y%m%d")
    end_dt = pd.to_datetime(end_date).strftime("%Y%m%d")
    
    while True:
        params = {
            "serviceKey": api_key,
            "pageNo": page_no,
            "numOfRows": num_of_rows,
            "dataType": "JSON",
            "dataCd": "ASOS",
            "dateCd": "HR",
            "stnIds": str(stn_id),
            "endDt": end_dt,
            "endHh": "23",
            "startHh": "00",
            "startDt": start_dt,
        }
        
        # requests.get encodes parameters, but KMA often has issues with already encoded keys
        import urllib.parse
        encoded_key = urllib.parse.unquote(api_key)
        params["serviceKey"] = encoded_key
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise ConnectionError(f"KMA API Error: {response.text}")
            
        try:
            data = response.json()
        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode KMA API response: {response.text}")
            
        header = data.get("response", {}).get("header", {})
        if header.get("resultCode") != "00":
            raise ValueError(f"KMA API Error: {header.get('resultMsg')} ({header.get('resultCode')})")
            
        body = data.get("response", {}).get("body", {})
        items = body.get("items", {}).get("item", [])
        
        if not items:
            break
            
        all_items.extend(items)
        
        total_count = body.get("totalCount", 0)
        if len(all_items) >= total_count:
            break
            
        page_no += 1

    if not all_items:
        raise ValueError(f"No KMA data found for {start_date} to {end_date}.")

    df = pd.DataFrame(all_items)
    df["datetime"] = pd.to_datetime(df["tm"])
    df.set_index("datetime", inplace=True)
    
    # tz_localize for Asia/Seoul to prevent UTC shifting during pvlib algorithms
    df.index = df.index.tz_localize("Asia/Seoul")
    
    # Process Temperature
    df["ta"] = pd.to_numeric(df["ta"], errors="coerce")
    df["T0_K"] = cu.C2K(df["ta"].fillna(method='ffill').fillna(method='bfill'))
    
    # Process Global Horizontal Irradiance (icsr in KMA ASOS is MJ/m2)
    # Some older datasets might use 'ss' for sunshine duration but icsr is standard
    # Irradiance is cumulative per hour in MJ/m2, we convert to average W/m2 over that hour
    if "icsr" in df.columns:
        df["icsr"] = pd.to_numeric(df["icsr"], errors="coerce").fillna(0)
        df["ghi"] = df["icsr"] * cu.MJ2J * cu.s2h
    else:
        df["ghi"] = 0.0
        
    df.loc[df["ghi"] < 0, "ghi"] = 0
    
    final_df = df[["T0_K", "ghi"]]
    final_df.to_csv(cache_path)
    
    return final_df
