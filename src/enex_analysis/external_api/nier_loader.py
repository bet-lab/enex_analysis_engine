import json
import os
from pathlib import Path

import pandas as pd
import requests

from .. import calc_util as cu

CACHE_DIR = Path("00_data/.api_cache")

def get_nier_api_key() -> str:
    key = os.getenv("NIER_API_KEY")
    if key:
        return key

    # 하드코딩된 API 키
    return "ec462489f5aadd290d119226177c6a4706e94cf25cd5a79a2848c49d2cf9fd66"

def get_nier_water_temp(start_date: str, end_date: str, station_code: str = "S04005") -> pd.DataFrame:
    """
    Fetch NIER Water Quality data (Water Temperature).
    Uses local cache to avoid redundant API calls.
    Returns standard dataframe with 'datetime' and 'Ts_C' (°C) / 'Ts_K' (K).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"nier_{station_code}_{start_date}_{end_date}.csv"

    if cache_path.exists():
        print(f"Loading NIER water data from cache: {cache_path}")
        df = pd.read_csv(cache_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        # Guard: reject cache if Ts_C is entirely NaN (corrupt cache)
        ts_all_nan: bool = "Ts_C" in df.columns and bool(df["Ts_C"].isnull().all())
        if ts_all_nan:
            print(
                f"[WARN] Cache '{cache_path}' has all-NaN Ts_C — "
                "invalidating cache and re-fetching from API."
            )
            cache_path.unlink()
        else:
            return df

    api_key = get_nier_api_key()
    url = "https://apis.data.go.kr/1480523/WaterQualityService/getRealTimeWaterQualityList"

    all_items = []
    page_no = 1
    num_of_rows = 999

    start_dt = pd.to_datetime(start_date).strftime("%Y%m%d000000")
    end_dt = pd.to_datetime(end_date).strftime("%Y%m%d235959")

    while True:
        params = {
            "serviceKey": api_key,
            "pageNo": page_no,
            "numOfRows": num_of_rows,
            "resultType": "JSON",
            "siteId": station_code,
            "startDate": start_dt,
            "endDate": end_dt,
        }

        import urllib.parse
        encoded_key = urllib.parse.unquote(api_key)
        params["serviceKey"] = encoded_key

        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise ConnectionError(f"NIER API Error: {response.text}")

        try:
            data = response.json()
        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode NIER API response: {response.text}")

        root = data.get("getRealTimeWaterQualityList", {})
        header = root.get("header", {})
        if header.get("code") != "00":
            raise ValueError(f"NIER API Error: {header.get('message')} ({header.get('code')})")

        items = root.get("item", [])

        if not items:
            break

        if isinstance(items, dict):
            items = [items]

        all_items.extend(items)

        total_count = root.get("totalCount", 0)
        if len(all_items) >= total_count:
            break

        page_no += 1

    if not all_items:
        raise ValueError(f"No NIER data found for {start_date} to {end_date}.")

    df = pd.DataFrame(all_items)
    # Different APIs use different time keys; NIER often uses msrDate and msrTime
    if "msrDate" in df.columns and "msrTime" in df.columns:
        df["datetime"] = pd.to_datetime(df["msrDate"].astype(str) + " " + df["msrTime"].astype(str).str.zfill(2) + ":00:00")
    elif "msrDate" in df.columns:
        df["datetime"] = pd.to_datetime(df["msrDate"].astype(str))
    elif "MSR_DATE" in df.columns and "MSR_TIME" in df.columns:
        df["datetime"] = pd.to_datetime(df["MSR_DATE"].astype(str) + " " + df["MSR_TIME"].astype(str).str.zfill(2) + ":00:00")
    elif "MSR_DATE" in df.columns:
        df["datetime"] = pd.to_datetime(df["MSR_DATE"].astype(str))
    elif "msr_dt" in df.columns:
        df["datetime"] = pd.to_datetime(df["msr_dt"], format="%Y%m%d%H%M")
    else:
        # Fallback to general lookup
        time_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        df["datetime"] = pd.to_datetime(df[time_cols[0]])

    df.set_index("datetime", inplace=True)
    df.index = df.index.tz_localize("Asia/Seoul")  # type: ignore[union-attr]

    # Water Temp Column matching logic
    temp_col = None
    if "M38" in df.columns: # Measured value (Water Temp)
        temp_col = "M38"
    elif "m38" in df.columns: # Measured value (Water Temp)
        temp_col = "m38"
    elif "ITEM_TEMP" in df.columns:
        temp_col = "ITEM_TEMP"
    else:
        raise ValueError("Could not dynamically find water temperature column in NIER API response")

    df["Ts_C"] = pd.to_numeric(df[temp_col], errors="coerce")
    # Interpolate for missing measurements
    df["Ts_C"] = df["Ts_C"].ffill().bfill()
    df["Ts_K"] = cu.C2K(df["Ts_C"])

    final_df = df[["Ts_C", "Ts_K"]]

    # Guard: do not cache if Ts_C is entirely NaN (API returned no valid measurements)
    ts_c_series: pd.Series = pd.Series(df["Ts_C"])
    nan_ratio: float = float(ts_c_series.isnull().mean())
    if nan_ratio == 1.0:
        raise ValueError(
            f"NIER API returned data for station '{station_code}' but all Ts_C values "
            f"are NaN — check API column mapping (found column: '{temp_col}'). "
            "Cache not saved."
        )
    if nan_ratio > 0.5:
        print(
            f"[WARN] NIER Ts_C has {nan_ratio:.1%} NaN for station '{station_code}'. "
            "Consider checking data quality."
        )
    final_df.to_csv(cache_path)

    return final_df  # type: ignore[return-value]
