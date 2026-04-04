# -*- coding: utf-8 -*-
"""
主程式版：救護車派遣 Simulation Driver（v2）
------------------------------------------------
這版已依照你最新提供的資料修正：
1. fleet 改直接讀「各分隊原始救護車配置.xlsx」
2. travel time 改直接讀「GoogleMapsAPI_凌晨三點行車時間.xlsx」中的「分隊到網格中心」工作表
3. 總車數改為 51 台（43 台基礎 + 8 台高級）
4. 時間合併改成「依流程順序自動處理跨日」

你負責的部分：
1. 讀檔
2. 前處理
3. 控制小樣本 / 全量跑法
4. while 迴圈逐案模擬
5. 輸出結果

你同學負責的功能函數請放在 sim_functions.py，
並至少提供以下四個函數：
- initialize_fleet_state(fleet_config_df)
- release_finished_units(fleet_state, current_time)
- select_nearest_available(case_row, fleet_state, travel_lookup)
- mark_unit_busy(fleet_state, ambulance_id, mission_end_time)
"""

from pathlib import Path
import re
import pandas as pd
import numpy as np

# ====== 你同學的功能函數模組 ======
try:
    from sim_functions import (
        initialize_fleet_state,
        release_finished_units,
        select_nearest_available,
        mark_unit_busy,
    )
except ImportError as e:
    raise ImportError(
        "找不到 sim_functions.py。這是你同學要提供的功能函數模組。\n"
        "請確認主程式和 sim_functions.py 放在同一個資料夾。"
    ) from e


# =========================================================
# A. 路徑與參數設定
# =========================================================
# 告訴程式要讀哪幾個檔案
# 控制你這次是跑小樣本還是全量
# 控制你要不要只看有效案件、只看到場案件、只看單一分隊救護

BASE_DIR = Path(__file__).resolve().parent

CASE_FILE = BASE_DIR / "案件_252505_不含精確地址.xlsx"
GEO_FILE = BASE_DIR / "分隊、網格中心、醫院的地址與經緯度.xlsx"
FLEET_CONFIG_FILE = BASE_DIR / "各分隊原始救護車配置.xlsx"
TRAVEL_TIME_FILE = BASE_DIR / "GoogleMapsAPI_凌晨三點行車時間.xlsx"

OUTPUT_DIR = BASE_DIR / "simulation_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---- 模擬模式 ----
RUN_SMALL_SAMPLE = False       # True: 先跑小樣本驗證；False: 跑全資料
ONLY_VALID_CASES = True       # 只保留有效案件
ONLY_ARRIVED_SCENE = True     # 只保留有到案件地點的案件
ONLY_SINGLE_DISPATCH = True   # 先只跑「單一分隊救護」
LIMIT_CASES = None            # 可填 50 / 100；None 代表不另外限制筆數

# 小樣本時間窗
SAMPLE_START = "2022-01-01 08:00:00"
SAMPLE_END   = "2022-01-01 10:00:00"

EXPECTED_TOTAL_UNITS = 51


# =========================================================
# B. 小工具（這些可放主程式）
# =========================================================
def standardize_station_name(raw_name: str) -> str:
    """
    把不同檔案中的分隊名稱統一
    """
    if pd.isna(raw_name):
        return np.nan

    s = str(raw_name)
    s = re.sub(r"（.*?）", "", s)
    s = s.replace("臺北市政府消防局", "")
    s = re.sub(r"第[一二三四五六七八九十]+救災救護大隊", "", s)
    s = re.sub(r"第[一二三四五六七八九十]+大隊", "", s)
    s = s.replace("分隊", "")
    s = s.strip()

    manual_map = {
        "舊莊消防": "舊莊",
        "中崙消防": "中崙",
        "松江消防": "松江",
        "關渡消防": "關渡",
        "士林中隊劍潭": "劍潭",
        "暨信義": "信義",
        "陽明": "陽明山",
    }
    return manual_map.get(s, s)


def _combine_date_and_time(date_norm: pd.Series, time_series: pd.Series) -> pd.Series:
    """
    date(YYYY-mm-dd 00:00:00) + HH:MM:SS -> datetime
    """
    td = pd.to_timedelta(time_series.astype(str).where(time_series.notna(), None))
    return date_norm + td


def build_event_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    依照救護流程順序合併完整 datetime，並自動處理跨日。
    流程順序：受理 -> 派遣 -> 出勤 -> 到達 -> 離開 -> 到院 -> 離院 -> 返隊
    """
    date_norm = pd.to_datetime(df["案發時間"]).dt.normalize()

    accepted_dt = _combine_date_and_time(date_norm, df["受理時間"])
    dispatch_dt = _combine_date_and_time(date_norm, df["派遣時間"])
    depart_dt = _combine_date_and_time(date_norm, df["出勤時間"])
    arrive_dt = _combine_date_and_time(date_norm, df["到達時間"])
    leave_scene_dt = _combine_date_and_time(date_norm, df["離開時間"])
    arrive_hosp_dt = _combine_date_and_time(date_norm, df["到院時間"])
    leave_hosp_dt = _combine_date_and_time(date_norm, df["離院時間"])
    return_dt = _combine_date_and_time(date_norm, df["返隊時間"])

    sequence = [
        ("accepted_dt", accepted_dt),
        ("dispatch_dt", dispatch_dt),
        ("depart_dt", depart_dt),
        ("arrive_dt", arrive_dt),
        ("leave_scene_dt", leave_scene_dt),
        ("arrive_hosp_dt", arrive_hosp_dt),
        ("leave_hosp_dt", leave_hosp_dt),
        ("return_dt", return_dt),
    ]

    out = pd.DataFrame(index=df.index)
    out["accepted_dt"] = accepted_dt
    last_dt = accepted_dt.copy()

    for name, dt in sequence[1:]:
        mask_rollover = dt.notna() & last_dt.notna() & (dt < last_dt)
        dt = dt.copy()
        dt.loc[mask_rollover] = dt.loc[mask_rollover] + pd.Timedelta(days=1)
        out[name] = dt
        last_dt = dt.where(dt.notna(), last_dt)

    return out


def build_travel_lookup_from_matrix(travel_matrix_df: pd.DataFrame) -> dict:
    """
    直接把『分隊到網格中心』矩陣表轉成 lookup。

    原始格式：
    第一欄 = 分隊
    後面每一欄 = 某個 grid_id
    儲存格數值 = travel minutes

    把 Google Maps 的矩陣表轉成字典
    讓後面可以快速查：(station_name, grid_id) -> travel_minutes
    """
    travel_matrix_df = travel_matrix_df.copy()
    travel_matrix_df = travel_matrix_df.rename(columns={travel_matrix_df.columns[0]: "station_name"})
    travel_matrix_df["station_name"] = travel_matrix_df["station_name"].apply(standardize_station_name)

    lookup = {}
    grid_cols = [c for c in travel_matrix_df.columns if c != "station_name"]

    for _, row in travel_matrix_df.iterrows():
        station = row["station_name"]
        for grid_col in grid_cols:
            val = row[grid_col]
            if pd.isna(val):
                continue
            lookup[(station, int(grid_col))] = float(val)

    return lookup


# =========================================================
# C. 讀取與前處理資料
# =========================================================
def load_geo_data():
    """
    讀三張地理表：分隊、醫院、網格
    （這版雖然醫院暫時沒用到，但先讀進來，之後第二版很可能會用到）
    """
    station_geo = pd.read_excel(GEO_FILE, sheet_name="分隊")
    grid_geo = pd.read_excel(GEO_FILE, sheet_name="網格（900）")
    hospital_geo = pd.read_excel(GEO_FILE, sheet_name="醫院")

    station_geo["station_name"] = station_geo["消防分隊"].apply(standardize_station_name)
    grid_geo = grid_geo.rename(columns={
        "網格編號": "grid_id",
        "網格中心緯度": "grid_lat",
        "網格中心經度": "grid_lng",
    })
    hospital_geo = hospital_geo.rename(columns={
        "醫院": "hospital_name",
        "緯度": "hospital_lat",
        "經度": "hospital_lng",
    })
    return station_geo, grid_geo, hospital_geo



def load_case_data():
    """
    這是案件前處理核心。

    它在做的事是：讀案件欄位、合成完整 datetime、把中文欄位改成主程式好用的英文欄位名、做基本過濾
                算歷史 response time、算模擬要用的 service time、排序、切小樣本
    """
    usecols = [
        "案號", "案件地點網格編號", "案發時間", "有效/取消案件", "出車方式",
        "分隊", "車牌號碼", "高級救護隊", "處理情況", "醫院",
        "受理時間", "派遣時間", "出勤時間", "到達時間",
        "離開時間", "到院時間", "離院時間", "返隊時間"
    ]

    df = pd.read_excel(CASE_FILE, sheet_name="case_information_complete252505", usecols=usecols)

    # --- 合併完整 datetime（含跨日處理）---
    event_dt = build_event_datetimes(df)
    df = pd.concat([df, event_dt], axis=1)

    # --- 改英文 / 主程式用欄位名 ---
    df = df.rename(columns={
        "案號": "case_id",
        "案件地點網格編號": "grid_id",
        "有效/取消案件": "case_status",
        "出車方式": "dispatch_mode",
        "分隊": "historical_station",
        "車牌號碼": "historical_plate",
        "高級救護隊": "historical_is_als",
        "處理情況": "handling_type",
        "醫院": "hospital_name",
    })

    df["historical_station"] = df["historical_station"].apply(standardize_station_name)

    # --- 基本過濾 ---
    if ONLY_VALID_CASES:
        df = df[df["case_status"] == "有效案件"].copy()

    if ONLY_ARRIVED_SCENE:
        df = df[df["handling_type"] != "未到案件地點"].copy()

    if ONLY_SINGLE_DISPATCH:
        df = df[df["dispatch_mode"] == "單一分隊救護"].copy()

    df = df[df["grid_id"].notna()].copy()
    df["grid_id"] = df["grid_id"].astype(int)

    # --- 歷史指標（用來比較模擬結果是否合理） ---
    df["historical_response_min"] = (df["arrive_dt"] - df["accepted_dt"]).dt.total_seconds() / 60.0
    df["historical_busy_after_arrival_min"] = (df["return_dt"] - df["arrive_dt"]).dt.total_seconds() / 60.0
    df["historical_busy_after_depart_min"] = (df["return_dt"] - df["depart_dt"]).dt.total_seconds() / 60.0

    # 模擬中的 service time：優先用到達現場後到返隊；若 arrive 缺值再退而求其次
    df["service_minutes_for_sim"] = df["historical_busy_after_arrival_min"]
    mask_missing = df["service_minutes_for_sim"].isna()
    df.loc[mask_missing, "service_minutes_for_sim"] = df.loc[mask_missing, "historical_busy_after_depart_min"]

    # 保留合理數值
    df = df[df["accepted_dt"].notna()].copy()
    df = df[df["service_minutes_for_sim"].notna()].copy()
    df = df[df["historical_response_min"].notna()].copy()
    df = df[df["historical_response_min"] >= 0].copy()
    df = df[df["service_minutes_for_sim"] >= 0].copy()

    # 排序
    df = df.sort_values("accepted_dt").reset_index(drop=True)

    # 小樣本時間窗
    if RUN_SMALL_SAMPLE:
        start_dt = pd.Timestamp(SAMPLE_START)
        end_dt = pd.Timestamp(SAMPLE_END)
        df = df[(df["accepted_dt"] >= start_dt) & (df["accepted_dt"] < end_dt)].copy()

    if LIMIT_CASES is not None:
        df = df.head(LIMIT_CASES).copy()

    return df



def load_fleet_config():
    """
    直接讀各分隊原始救護車配置.xlsx
    原始欄位：
    - 分隊名稱
    - 分隊類型
    - 可派遣高級救護車數
    - 可派遣基礎救護車數

    轉成主程式想要的欄位：
    - station_name
    - vehicle_count
    - als_count
    """
    if not FLEET_CONFIG_FILE.exists():
        raise FileNotFoundError(f"找不到 {FLEET_CONFIG_FILE.name}")

    raw = pd.read_excel(FLEET_CONFIG_FILE, sheet_name=0)
    required_cols = {"分隊名稱", "可派遣高級救護車數", "可派遣基礎救護車數"}
    missing_cols = required_cols - set(raw.columns)
    if missing_cols:
        raise ValueError(f"{FLEET_CONFIG_FILE.name} 缺少欄位：{missing_cols}")

    fleet_config_df = raw.copy()
    fleet_config_df["station_name"] = fleet_config_df["分隊名稱"].apply(standardize_station_name)
    fleet_config_df["als_count"] = fleet_config_df["可派遣高級救護車數"].fillna(0).astype(int)
    fleet_config_df["bls_count"] = fleet_config_df["可派遣基礎救護車數"].fillna(0).astype(int)
    fleet_config_df["vehicle_count"] = fleet_config_df["als_count"] + fleet_config_df["bls_count"]

    fleet_config_df = fleet_config_df[["station_name", "vehicle_count", "als_count"]].copy()

    total_units = int(fleet_config_df["vehicle_count"].sum())
    if total_units != EXPECTED_TOTAL_UNITS:
        print(f"Warning: current total fleet size = {total_units}, expected = {EXPECTED_TOTAL_UNITS}.")

    return fleet_config_df



def load_travel_time_matrix():
    """
    直接讀 GoogleMapsAPI_凌晨三點行車時間.xlsx 的『分隊到網格中心』工作表。
    """
    if not TRAVEL_TIME_FILE.exists():
        raise FileNotFoundError(f"找不到 {TRAVEL_TIME_FILE.name}")

    travel_matrix_df = pd.read_excel(TRAVEL_TIME_FILE, sheet_name="分隊到網格中心")
    if travel_matrix_df.empty:
        raise ValueError("『分隊到網格中心』工作表是空的")

    return travel_matrix_df


# =========================================================
# D. 主模擬流程（這段就是你負責的 main）
# =========================================================
def run_simulation(case_df, fleet_state, travel_lookup):
    """
    主程式最核心的 while 迴圈：
    一筆案件一筆案件讀進來，更新車況、選車、派車、記錄結果
    """
    results = []

    i = 0
    while i < len(case_df):
        case_row = case_df.iloc[i]
        current_time = case_row["accepted_dt"]

        # 1) 先更新哪些車已經恢復 Available
        fleet_state = release_finished_units(fleet_state, current_time)

        # 2) 根據「最近且有空的車」規則選車
        dispatch_result = select_nearest_available(
            case_row=case_row,
            fleet_state=fleet_state,
            travel_lookup=travel_lookup,
        )

        # 3) if / else：有派到 or 沒派到
        if dispatch_result is None:
            results.append({
                "case_id": case_row["case_id"],
                "accepted_dt": case_row["accepted_dt"],
                "grid_id": case_row["grid_id"],
                "assigned_ambulance": None,
                "assigned_station": None,
                "dispatch_dt_sim": pd.NaT,
                "arrival_dt_sim": pd.NaT,
                "travel_min_sim": np.nan,
                "wait_min_sim": np.nan,
                "response_min_sim": np.nan,
                "mission_end_dt_sim": pd.NaT,
                "historical_response_min": case_row["historical_response_min"],
                "status": "NO_UNIT_OR_NO_TRAVEL_TIME",
            })
            i += 1
            continue

        # 4) 模擬任務結束時間：沿用歷史資料中的 service time
        mission_end_dt = dispatch_result["arrival_dt_sim"] + pd.Timedelta(
            minutes=float(case_row["service_minutes_for_sim"])
        )

        # 5) 將該車標記成忙碌
        fleet_state = mark_unit_busy(
            fleet_state=fleet_state,
            ambulance_id=dispatch_result["ambulance_id"],
            mission_end_time=mission_end_dt,
        )

        # 6) 紀錄結果
        results.append({
            "case_id": case_row["case_id"],
            "accepted_dt": case_row["accepted_dt"],
            "grid_id": case_row["grid_id"],
            "assigned_ambulance": dispatch_result["ambulance_id"],
            "assigned_station": dispatch_result["station_name"],
            "dispatch_dt_sim": dispatch_result["dispatch_dt_sim"],
            "arrival_dt_sim": dispatch_result["arrival_dt_sim"],
            "travel_min_sim": dispatch_result["travel_min_sim"],
            "wait_min_sim": dispatch_result["wait_min_sim"],
            "response_min_sim": dispatch_result["response_min_sim"],
            "mission_end_dt_sim": mission_end_dt,
            "historical_response_min": case_row["historical_response_min"],
            "status": "OK",
        })

        i += 1

    result_df = pd.DataFrame(results)
    return result_df, fleet_state


# =========================================================
# E. 輸出與摘要
# =========================================================
def summarize_result(result_df: pd.DataFrame):
    ok_df = result_df[result_df["status"] == "OK"].copy()

    if len(ok_df) == 0:
        return pd.DataFrame([{
            "n_total": len(result_df),
            "n_ok": 0,
            "sim_mean_response": np.nan,
            "sim_median_response": np.nan,
            "sim_p90_response": np.nan,
            "hist_mean_response": np.nan,
            "hist_median_response": np.nan,
        }])

    summary = pd.DataFrame([{
        "n_total": len(result_df),
        "n_ok": len(ok_df),
        "sim_mean_response": ok_df["response_min_sim"].mean(),
        "sim_median_response": ok_df["response_min_sim"].median(),
        "sim_p90_response": ok_df["response_min_sim"].quantile(0.90),
        "hist_mean_response": ok_df["historical_response_min"].mean(),
        "hist_median_response": ok_df["historical_response_min"].median(),
    }])
    return summary



def save_outputs(result_df, summary_df):
    detail_path = OUTPUT_DIR / "simulation_detail.csv"
    summary_path = OUTPUT_DIR / "simulation_summary.csv"

    result_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"Detailed results saved to: {detail_path}")
    print(f"Summary results saved to: {summary_path}")


# =========================================================
# F. main
# =========================================================
def main():
    print("=== Loading data ===")
    station_geo, grid_geo, hospital_geo = load_geo_data()
    case_df = load_case_data()
    fleet_config_df = load_fleet_config()
    travel_matrix_df = load_travel_time_matrix()

    print(f"Number of cases before simulation: {len(case_df):,}")
    print(f"Number of stations (geo file): {station_geo['station_name'].nunique()}")
    print(f"Number of grids (geo file): {grid_geo['grid_id'].nunique()}")
    print(f"Total fleet size in fleet_config: {fleet_config_df['vehicle_count'].sum()}")

    print("=== Building lookup tables / initializing fleet state ===")
    travel_lookup = build_travel_lookup_from_matrix(travel_matrix_df)
    fleet_state = initialize_fleet_state(fleet_config_df)

    print("=== Starting simulation ===")
    result_df, final_fleet_state = run_simulation(
        case_df=case_df,
        fleet_state=fleet_state,
        travel_lookup=travel_lookup,
    )

    print("=== Generating summary ===")
    summary_df = summarize_result(result_df)
    save_outputs(result_df, summary_df)

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
