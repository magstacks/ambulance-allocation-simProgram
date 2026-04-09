# -*- coding: utf-8 -*-
"""
主程式版：救護車派遣 Simulation Driver（v3：納入取消案件＋未到場案件）
-----------------------------------------------------------------------
這版目標：
1. 納入「取消案件」
2. 納入「未到案件地點」案件
3. 仍然先只跑「單一分隊救護」
4. 雙軌 / ALS 規則暫時不處理

這版的重要想法：
- 有到場案件：維持原本做法，用 Google 去程時間 + 歷史到場後忙碌時間
- 未到場案件：仍會派一台車，但不計算 response time；車輛忙碌時間改用歷史的
  dispatch->return（若缺則退而求其次用 depart->return）
- 取消案件：先不另外做特殊派遣規則，只要時間欄位足以估 busy time，就照歷史時間資料建模
"""

from pathlib import Path
import re
import pandas as pd
import numpy as np
import time

try:
    from sim_functions import (
        initialize_fleet_state,
        release_finished_units,
        select_nearest_available,
        mark_unit_busy,
    )
except ImportError as e:
    raise ImportError(
        "找不到 sim_functions.py。請確認主程式和 sim_functions.py 放在同一個資料夾。"
    ) from e


# =========================================================
# A. 路徑與參數設定
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

CASE_FILE = BASE_DIR / "案件_252505_不含精確地址.xlsx"
GEO_FILE = BASE_DIR / "分隊、網格中心、醫院的地址與經緯度.xlsx"
FLEET_CONFIG_FILE = BASE_DIR / "各分隊原始救護車配置.xlsx"
TRAVEL_TIME_FILE = BASE_DIR / "GoogleMapsAPI_凌晨三點行車時間.xlsx"

OUTPUT_DIR = BASE_DIR / "simulation_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 模擬模式
RUN_SMALL_SAMPLE = False     # True: 先跑小樣本驗證；False: 跑全資料
ONLY_VALID_CASES = False      # v3：納入取消案件，所以預設 False
ONLY_ARRIVED_SCENE = False    # v3：納入未到場案件，所以預設 False
ONLY_SINGLE_DISPATCH = True   # 先仍只跑「單一分隊救護」
LIMIT_CASES = None            # 可填 50 / 100；None 代表不另外限制筆數
PROGRESS_EVERY = 1000

# 小樣本時間窗
SAMPLE_START = "2022-01-01 08:00:00"
SAMPLE_END   = "2022-03-01 22:00:00"

EXPECTED_TOTAL_UNITS = 51


# =========================================================
# B. 小工具
# =========================================================
def standardize_station_name(raw_name: str) -> str:
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
    td = pd.to_timedelta(time_series.astype(str).where(time_series.notna(), None))
    return date_norm + td


def build_event_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    依流程順序合併完整 datetime，並自動處理跨日。
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

    out = pd.DataFrame(index=df.index)
    out["accepted_dt"] = accepted_dt
    last_dt = accepted_dt.copy()

    small_back_tol = pd.Timedelta(minutes=2)   # 小誤差容忍，例如 2 分鐘內
    rollover_tol = pd.Timedelta(hours=6)       # 倒退超過 6 小時，才視為跨日

    for name, dt in sequence[1:]:
        dt = dt.copy()

        valid_mask = dt.notna() & last_dt.notna()
        backward_diff = last_dt - dt   # 若 dt 比 last_dt 小，這個值會是正的

        # 1) 小幅倒退：當成秒數被截掉，直接補成 last_dt
        mask_small_back = valid_mask & (dt < last_dt) & (backward_diff <= small_back_tol)
        dt.loc[mask_small_back] = last_dt.loc[mask_small_back]

        # 2) 大幅倒退：視為真的跨日
        mask_rollover = valid_mask & (dt < last_dt) & (backward_diff > rollover_tol)
        dt.loc[mask_rollover] = dt.loc[mask_rollover] + pd.Timedelta(days=1)

        # 3) 中間倒退幅度：先保留，之後可另外抓出來檢查
        out[name] = dt
        last_dt = dt.where(dt.notna(), last_dt)

    return out


def build_travel_lookup_from_matrix(travel_matrix_df: pd.DataFrame) -> dict:
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
    這版前處理重點：
    1. 不再預設排除取消案件 / 未到場案件
    2. 依任務情境分類成：
       - NO_SCENE         : 有出勤但沒有到現場
       - SCENE_NO_HOSP    : 有到現場但沒有到醫院
       - SCENE_TO_HOSP    : 有送醫完整流程
    3. 不同情境使用不同 busy time 規則
    """
    usecols = [
        "案號", "案件地點網格編號", "案發時間", "有效/取消案件", "出車方式",
        "分隊", "車牌號碼", "高級救護隊", "處理情況", "醫院",
        "受理時間", "派遣時間", "出勤時間", "到達時間",
        "離開時間", "到院時間", "離院時間", "返隊時間"
    ]

    df = pd.read_excel(CASE_FILE, sheet_name="case_information_complete252505", usecols=usecols)

    event_dt = build_event_datetimes(df)
    df = pd.concat([df, event_dt], axis=1)

    # 以下為v5新增
    # 受理時間修正版本：若受理時間晚於派遣時間，模擬時改用派遣時間
    df["accepted_dt_raw"] = df["accepted_dt"]
    df["accepted_dt_sim"] = df["accepted_dt"]
    df["accepted_dt_fixed_from_dispatch"] = False

    mask_bad_accepted = (
        df["accepted_dt"].notna()
        & df["dispatch_dt"].notna()
        & (df["accepted_dt"] > df["dispatch_dt"])
    )

    df.loc[mask_bad_accepted, "accepted_dt_sim"] = df.loc[mask_bad_accepted, "dispatch_dt"]
    df.loc[mask_bad_accepted, "accepted_dt_fixed_from_dispatch"] = True

    # 以上為v5新增

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

    # 過濾
    if ONLY_VALID_CASES:
        df = df[df["case_status"] == "有效案件"].copy()

    if ONLY_ARRIVED_SCENE:
        df = df[df["handling_type"] != "未到案件地點"].copy()

    if ONLY_SINGLE_DISPATCH:
        df = df[df["dispatch_mode"] == "單一分隊救護"].copy()

    # grid_id 還是 simulation 必要欄位
    df = df[df["grid_id"].notna()].copy()
    df["grid_id"] = df["grid_id"].astype(int)

    # 依時間欄位判斷任務情境
    df["has_scene_arrival"] = df["arrive_dt"].notna()
    df["has_hospital_arrival"] = df["arrive_hosp_dt"].notna()

    df["task_scenario"] = np.where(
        ~df["has_scene_arrival"],
        "NO_SCENE",
        np.where(~df["has_hospital_arrival"], "SCENE_NO_HOSP", "SCENE_TO_HOSP")
    )

    # 歷史時間指標
    # 只有有到場案件才有反應時間意義
    # df["historical_response_min"] = (df["arrive_dt"] - df["accepted_dt"]).dt.total_seconds() / 60.0
    df["historical_response_min"] = (df["arrive_dt"] - df["accepted_dt_sim"]).dt.total_seconds() / 60.0

    # 各種 busy time 候選值
    df["busy_dispatch_to_return_min"] = (df["return_dt"] - df["dispatch_dt"]).dt.total_seconds() / 60.0
    df["busy_depart_to_return_min"] = (df["return_dt"] - df["depart_dt"]).dt.total_seconds() / 60.0
    df["busy_arrive_to_return_min"] = (df["return_dt"] - df["arrive_dt"]).dt.total_seconds() / 60.0

    # v3 模擬要用的 service/busy time
    df["service_minutes_for_sim"] = np.nan

    # 情境 1：未到場
    '''
    service_minutes_for_sim：
    先算「返隊時間 - 派遣時間」
    如果這個缺值，再改算「返隊時間 - 出勤時間」
    '''
    mask_no_scene = df["task_scenario"] == "NO_SCENE"
    df.loc[mask_no_scene, "service_minutes_for_sim"] = df.loc[mask_no_scene, "busy_dispatch_to_return_min"]
    mask_no_scene_missing = mask_no_scene & df["service_minutes_for_sim"].isna()
    df.loc[mask_no_scene_missing, "service_minutes_for_sim"] = df.loc[mask_no_scene_missing, "busy_depart_to_return_min"]

    # 情境 2/3：有到場
    '''
    service_minutes_for_sim：
    先算「返隊時間 - 到達時間」
    如果這個缺值，再改算「返隊時間 - 出勤時間」
    '''
    mask_arrived = ~mask_no_scene
    df.loc[mask_arrived, "service_minutes_for_sim"] = df.loc[mask_arrived, "busy_arrive_to_return_min"]
    mask_arrived_missing = mask_arrived & df["service_minutes_for_sim"].isna()
    df.loc[mask_arrived_missing, "service_minutes_for_sim"] = df.loc[mask_arrived_missing, "busy_depart_to_return_min"]

    # 保留可模擬資料
    df = df[df["accepted_dt"].notna()].copy()
    df = df[df["service_minutes_for_sim"].notna()].copy()
    df = df[df["service_minutes_for_sim"] >= 0].copy()

    # 有到場的案件才要求 historical_response_min 合理
    mask_bad_arrived_response = (
        df["has_scene_arrival"]
        & (
            df["historical_response_min"].isna()
            | (df["historical_response_min"] < 0)
        )
    )
    df = df[~mask_bad_arrived_response].copy()

    # df = df.sort_values("accepted_dt").reset_index(drop=True)
    df = df.sort_values("accepted_dt_sim").reset_index(drop=True)

    if RUN_SMALL_SAMPLE:
        start_dt = pd.Timestamp(SAMPLE_START)
        end_dt = pd.Timestamp(SAMPLE_END)
        # df = df[(df["accepted_dt"] >= start_dt) & (df["accepted_dt"] < end_dt)].copy()
        df = df[(df["accepted_dt_sim"] >= start_dt) & (df["accepted_dt_sim"] < end_dt)].copy()

    if LIMIT_CASES is not None:
        df = df.head(LIMIT_CASES).copy()
    
    return df



def load_fleet_config():
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
    if not TRAVEL_TIME_FILE.exists():
        raise FileNotFoundError(f"找不到 {TRAVEL_TIME_FILE.name}")

    travel_matrix_df = pd.read_excel(TRAVEL_TIME_FILE, sheet_name="分隊到網格中心")
    if travel_matrix_df.empty:
        raise ValueError("『分隊到網格中心』工作表是空的")

    return travel_matrix_df


# =========================================================
# D. 主模擬流程
# =========================================================
def run_simulation(case_df, fleet_state, travel_lookup):
    results = []

    total_cases = len(case_df)
    start_clock = time.time()

    i = 0
    while i < total_cases:
        case_row = case_df.iloc[i]
        # current_time = case_row["accepted_dt"]
        current_time = case_row["accepted_dt_sim"]
        scenario = case_row["task_scenario"]

        # 每隔幾筆顯示一次進度
        if i == 0 or ((i + 1) % PROGRESS_EVERY == 0):
            elapsed_min = (time.time() - start_clock) / 60
            print(
                f"[Progress] {i + 1:,}/{total_cases:,} "
                f"({(i + 1) / total_cases:.1%}) | "
                f"elapsed = {elapsed_min:.2f} min | "
                f"current accepted_dt = {current_time}",
                flush=True
            )

        # 1) 先更新哪些車已經恢復 Available
        fleet_state = release_finished_units(fleet_state, current_time)

        # 2) 建立給選車函數用的案件資料（把 accepted_dt 換成 accepted_dt_sim）
        case_row_for_sim = case_row.copy()
        case_row_for_sim["accepted_dt"] = case_row["accepted_dt_sim"]

        # 3) 先沿用「最近且有空的車」選車規則
        dispatch_result = select_nearest_available(
            case_row=case_row_for_sim,
            fleet_state=fleet_state,
            travel_lookup=travel_lookup,
        )

        if dispatch_result is None:
            results.append({
                "case_id": case_row["case_id"],
                "case_status": case_row["case_status"],
                "dispatch_mode": case_row["dispatch_mode"],
                "handling_type": case_row["handling_type"],
                "task_scenario": scenario,
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
                "accepted_dt_raw": case_row["accepted_dt_raw"],
                "accepted_dt_sim": case_row["accepted_dt_sim"],
                "accepted_dt_fixed_from_dispatch": case_row["accepted_dt_fixed_from_dispatch"],
            })
            i += 1
            continue

        # 情境分流：
        # A. 未到場案件：不計 arrival / response，用歷史 dispatch->return 類型 busy time
        if scenario == "NO_SCENE":
            mission_end_dt = dispatch_result["dispatch_dt_sim"] + pd.Timedelta(
                minutes=float(case_row["service_minutes_for_sim"])
            )
            arrival_dt_out = pd.NaT
            response_min_out = np.nan
            status_out = "OK_NO_SCENE"
        # B. 有到場案件：維持原本做法
        else:
            mission_end_dt = dispatch_result["arrival_dt_sim"] + pd.Timedelta(
                minutes=float(case_row["service_minutes_for_sim"])
            )
            arrival_dt_out = dispatch_result["arrival_dt_sim"]
            response_min_out = dispatch_result["response_min_sim"]
            status_out = "OK_ARRIVED"

        fleet_state = mark_unit_busy(
            fleet_state=fleet_state,
            ambulance_id=dispatch_result["ambulance_id"],
            mission_end_time=mission_end_dt,
        )

        results.append({
            "case_id": case_row["case_id"],
            "case_status": case_row["case_status"],
            "dispatch_mode": case_row["dispatch_mode"],
            "handling_type": case_row["handling_type"],
            "task_scenario": scenario,
            "accepted_dt": case_row["accepted_dt"],
            "grid_id": case_row["grid_id"],
            "assigned_ambulance": dispatch_result["ambulance_id"],
            "assigned_station": dispatch_result["station_name"],
            "dispatch_dt_sim": dispatch_result["dispatch_dt_sim"],
            "arrival_dt_sim": arrival_dt_out,
            "travel_min_sim": dispatch_result["travel_min_sim"],
            "wait_min_sim": dispatch_result["wait_min_sim"],
            "response_min_sim": response_min_out,
            "mission_end_dt_sim": mission_end_dt,
            "historical_response_min": case_row["historical_response_min"],
            "status": status_out,
            "accepted_dt_raw": case_row["accepted_dt_raw"],
            "accepted_dt_sim": case_row["accepted_dt_sim"],
            "accepted_dt_fixed_from_dispatch": case_row["accepted_dt_fixed_from_dispatch"],
        })

        i += 1

    total_elapsed_min = (time.time() - start_clock) / 60
    print(
        f"[Progress] done: {total_cases:,}/{total_cases:,} "
        f"(100.0%) | elapsed = {total_elapsed_min:.2f} min",
        flush=True
    )

    result_df = pd.DataFrame(results)
    return result_df, fleet_state
  


# =========================================================
# E. 輸出與摘要
# =========================================================
def summarize_result(result_df: pd.DataFrame):
    ok_df = result_df[result_df["status"].isin(["OK_ARRIVED", "OK_NO_SCENE"])].copy()
    arrived_ok_df = ok_df[ok_df["response_min_sim"].notna()].copy()

    return pd.DataFrame([{
        "n_total": len(result_df),
        "n_ok": len(ok_df),
        "n_ok_arrived": len(arrived_ok_df),
        "n_ok_no_scene": int((ok_df["task_scenario"] == "NO_SCENE").sum()) if len(ok_df) > 0 else 0,
        "n_cancel_cases": int((result_df["case_status"] == "取消案件").sum()) if len(result_df) > 0 else 0,
        "sim_mean_response": arrived_ok_df["response_min_sim"].mean() if len(arrived_ok_df) > 0 else np.nan,
        "sim_median_response": arrived_ok_df["response_min_sim"].median() if len(arrived_ok_df) > 0 else np.nan,
        "sim_p90_response": arrived_ok_df["response_min_sim"].quantile(0.90) if len(arrived_ok_df) > 0 else np.nan,
        "hist_mean_response": arrived_ok_df["historical_response_min"].mean() if len(arrived_ok_df) > 0 else np.nan,
        "hist_median_response": arrived_ok_df["historical_response_min"].median() if len(arrived_ok_df) > 0 else np.nan,
    }])



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
    print("Cases by scenario before simulation:")
    print(case_df["task_scenario"].value_counts(dropna=False).to_string())
    print("Cases by valid/cancel before simulation:")
    print(case_df["case_status"].value_counts(dropna=False).to_string())

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
