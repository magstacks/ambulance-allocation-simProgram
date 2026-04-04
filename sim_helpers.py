import pandas as pd


def initialize_fleet_state(fleet_config_df: pd.DataFrame) -> pd.DataFrame:
    """
    把分隊配置表展開成每台車一列的初始狀態。

    必要輸入欄位：
    - station_name
    - vehicle_count
    - als_count

    回傳欄位：
    - ambulance_id
    - station_name
    - current_station
    - home_station
    - is_als
    - status
    - available_at
    """
    required_cols = {"station_name", "vehicle_count", "als_count"}
    missing = required_cols - set(fleet_config_df.columns)
    if missing:
        raise ValueError(f"fleet_config_df 缺少必要欄位: {sorted(missing)}")

    df = fleet_config_df.copy()

    df["station_name"] = df["station_name"].astype(str).str.strip()
    df["vehicle_count"] = pd.to_numeric(df["vehicle_count"], errors="coerce")
    df["als_count"] = pd.to_numeric(df["als_count"], errors="coerce")

    if df[["vehicle_count", "als_count"]].isna().any().any():
        raise ValueError("vehicle_count / als_count 有無法轉成數字的值")

    df["vehicle_count"] = df["vehicle_count"].astype(int)
    df["als_count"] = df["als_count"].astype(int)

    if (df["vehicle_count"] < 0).any() or (df["als_count"] < 0).any():
        raise ValueError("vehicle_count / als_count 不能是負數")

    if (df["als_count"] > df["vehicle_count"]).any():
        bad_rows = df.loc[df["als_count"] > df["vehicle_count"], "station_name"].tolist()
        raise ValueError(f"下列分隊 als_count > vehicle_count: {bad_rows}")

    rows = []
    available_init = pd.Timestamp("1900-01-01 00:00:00")

    for _, row in df.iterrows():
        station_name = row["station_name"]
        vehicle_count = row["vehicle_count"]
        als_count = row["als_count"]

        for i in range(1, vehicle_count + 1):
            rows.append(
                {
                    "ambulance_id": f"{station_name}_{i:02d}",
                    "station_name": station_name,
                    "current_station": station_name,
                    "home_station": station_name,
                    "is_als": i <= als_count,
                    "status": "Available",
                    "available_at": available_init,
                }
            )

    fleet_state = pd.DataFrame(rows)

    if fleet_state["ambulance_id"].duplicated().any():
        dup_ids = fleet_state.loc[fleet_state["ambulance_id"].duplicated(), "ambulance_id"].tolist()
        raise ValueError(f"ambulance_id 不可重複，重複值: {dup_ids}")

    return fleet_state


def release_finished_units(fleet_state: pd.DataFrame, current_time) -> pd.DataFrame:
    """
    把已經完成任務、且 available_at <= current_time 的車改回可用。
    第一版先假設車結束任務後已回 home_station。
    """
    required_cols = {"status", "available_at", "current_station", "home_station"}
    missing = required_cols - set(fleet_state.columns)
    if missing:
        raise ValueError(f"fleet_state 缺少必要欄位: {sorted(missing)}")

    current_time = pd.Timestamp(current_time)
    out = fleet_state.copy()

    out["available_at"] = pd.to_datetime(out["available_at"], errors="coerce")

    mask = (out["status"] == "On duty") & (out["available_at"] <= current_time)

    out.loc[mask, "status"] = "Available"
    out.loc[mask, "current_station"] = out.loc[mask, "home_station"]

    return out


def select_nearest_available(case_row, fleet_state: pd.DataFrame, travel_lookup: dict):
    """
    依最近且可用的車來選車。

    case_row 至少要有：
    - accepted_dt
    - grid_id

    travel_lookup:
    key = (station_name, grid_id)
    value = travel_minutes

    回傳：
    - 若沒車可派，回傳 None
    - 若成功，回傳 dict:
        ambulance_id
        station_name
        dispatch_dt_sim
        arrival_dt_sim
        travel_min_sim
        wait_min_sim
        response_min_sim
    """
    required_cols = {"ambulance_id", "current_station", "status"}
    missing = required_cols - set(fleet_state.columns)
    if missing:
        raise ValueError(f"fleet_state 缺少必要欄位: {sorted(missing)}")

    if isinstance(case_row, dict):
        accepted_dt = case_row.get("accepted_dt")
        grid_id = case_row.get("grid_id")
    else:
        accepted_dt = case_row["accepted_dt"]
        grid_id = case_row["grid_id"]

    accepted_dt = pd.Timestamp(accepted_dt)

    candidates = fleet_state.loc[fleet_state["status"] == "Available"].copy()
    if candidates.empty:
        return None

    candidate_records = []

    for _, row in candidates.iterrows():
        station = row["current_station"]

        # 優先直接查
        travel_min = travel_lookup.get((station, grid_id))

        # 若查不到，試幾種常見型別組合
        if travel_min is None:
            travel_min = travel_lookup.get((str(station), grid_id))
        if travel_min is None:
            travel_min = travel_lookup.get((station, str(grid_id)))
        if travel_min is None:
            travel_min = travel_lookup.get((str(station), str(grid_id)))

        if travel_min is None:
            continue

        try:
            travel_min = float(travel_min)
        except Exception:
            continue

        candidate_records.append(
            {
                "ambulance_id": row["ambulance_id"],
                "station_name": station,
                "travel_min_sim": travel_min,
            }
        )

    if not candidate_records:
        return None

    candidate_df = pd.DataFrame(candidate_records)

    # travel_min 最短優先；若剛好相同，用 ambulance_id 做穩定排序
    candidate_df = candidate_df.sort_values(
        by=["travel_min_sim", "ambulance_id"],
        ascending=[True, True]
    ).reset_index(drop=True)

    best = candidate_df.iloc[0]
    arrival_dt_sim = accepted_dt + pd.to_timedelta(best["travel_min_sim"], unit="m")

    return {
        "ambulance_id": best["ambulance_id"],
        "station_name": best["station_name"],
        "dispatch_dt_sim": accepted_dt,
        "arrival_dt_sim": arrival_dt_sim,
        "travel_min_sim": best["travel_min_sim"],
        "wait_min_sim": 0,
        "response_min_sim": best["travel_min_sim"],
    }


def mark_unit_busy(fleet_state: pd.DataFrame, ambulance_id: str, mission_end_time) -> pd.DataFrame:
    """
    把指定 ambulance_id 的車改成 On duty，並更新 available_at。
    """
    required_cols = {"ambulance_id", "status", "available_at"}
    missing = required_cols - set(fleet_state.columns)
    if missing:
        raise ValueError(f"fleet_state 缺少必要欄位: {sorted(missing)}")

    out = fleet_state.copy()
    mission_end_time = pd.Timestamp(mission_end_time)

    mask = out["ambulance_id"] == ambulance_id
    if mask.sum() == 0:
        raise ValueError(f"找不到 ambulance_id: {ambulance_id}")
    if mask.sum() > 1:
        raise ValueError(f"ambulance_id 重複，無法唯一更新: {ambulance_id}")

    out.loc[mask, "status"] = "On duty"
    out.loc[mask, "available_at"] = mission_end_time

    return out
