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


def _get_case_value(case_row, key, default=None):
    """
    同時支援 dict / pandas Series 取值。
    """
    if isinstance(case_row, dict):
        return case_row.get(key, default)

    if hasattr(case_row, "index") and key in case_row.index:
        return case_row[key]

    return default


def _lookup_travel_minutes(station_name, grid_id, travel_lookup: dict):
    """
    盡量容忍 station_name / grid_id 型別不同的 lookup。
    """
    travel_min = travel_lookup.get((station_name, grid_id))

    if travel_min is None:
        travel_min = travel_lookup.get((str(station_name), grid_id))
    if travel_min is None:
        travel_min = travel_lookup.get((station_name, str(grid_id)))
    if travel_min is None:
        travel_min = travel_lookup.get((str(station_name), str(grid_id)))

    if travel_min is None:
        return None

    try:
        return float(travel_min)
    except Exception:
        return None


def select_dual_dispatch_units(case_row, fleet_state: pd.DataFrame, travel_lookup: dict):
    """
    針對雙軌案件，尋找最早滿足「最近 ALS 分隊可供 1 台 ALS，且總共可派 2 台車」的派遣時間並完成選車。
    """
    required_cols = {
        "ambulance_id",
        "current_station",
        "home_station",
        "is_als",
        "status",
        "available_at",
    }
    missing = required_cols - set(fleet_state.columns)
    if missing:
        raise ValueError(f"fleet_state 缺少必要欄位: {sorted(missing)}")

    def _get_case_value(row, key, default=None):
        if isinstance(row, dict):
            return row.get(key, default)
        if hasattr(row, "index") and key in row.index:
            return row[key]
        return default

    def _lookup_travel_minutes(station_name, grid_id, lookup: dict):
        travel_min = lookup.get((station_name, grid_id))
        if travel_min is None:
            travel_min = lookup.get((str(station_name), grid_id))
        if travel_min is None:
            travel_min = lookup.get((station_name, str(grid_id)))
        if travel_min is None:
            travel_min = lookup.get((str(station_name), str(grid_id)))

        if travel_min is None:
            return None

        try:
            return float(travel_min)
        except Exception:
            return None

    accepted_dt = pd.Timestamp(_get_case_value(case_row, "accepted_dt"))
    grid_id = _get_case_value(case_row, "grid_id")
    required_units = int(_get_case_value(case_row, "required_units", 2))
    require_als = int(_get_case_value(case_row, "require_als", 1))

    if required_units != 2:
        raise ValueError("這版 select_dual_dispatch_units() 只支援 required_units = 2")
    if require_als != 1:
        raise ValueError("這版 select_dual_dispatch_units() 只支援 require_als = 1")

    fs = fleet_state.copy()
    fs["available_at"] = pd.to_datetime(fs["available_at"], errors="coerce")

    def build_candidate_df(dispatch_time: pd.Timestamp) -> pd.DataFrame:
        """
        在 dispatch_time 這一刻，把已可派的車整理成 candidate dataframe。
        - status == 'Available'：直接可派，位置用 current_station
        - status == 'On duty' 但 available_at <= dispatch_time：
          視為已回 home_station，可派
        """
        rows = []

        for _, row in fs.iterrows():
            status = row["status"]
            available_at = row["available_at"]

            is_effectively_available = False
            station_for_dispatch = None

            if status == "Available":
                is_effectively_available = True
                station_for_dispatch = row["current_station"]
            elif pd.notna(available_at) and available_at <= dispatch_time:
                is_effectively_available = True
                station_for_dispatch = row["home_station"]

            if not is_effectively_available:
                continue

            travel_min = _lookup_travel_minutes(station_for_dispatch, grid_id, travel_lookup)
            if travel_min is None:
                continue

            rows.append(
                {
                    "ambulance_id": row["ambulance_id"],
                    "station_name": station_for_dispatch,
                    "is_als": bool(row["is_als"]),
                    "travel_min_sim": float(travel_min),
                }
            )

        if not rows:
            return pd.DataFrame(columns=["ambulance_id", "station_name", "is_als", "travel_min_sim"])

        out = pd.DataFrame(rows)
        out = out.sort_values(
            by=["travel_min_sim", "ambulance_id"],
            ascending=[True, True]
        ).reset_index(drop=True)
        return out

    def pick_units_by_final_rule(candidate_df: pd.DataFrame) -> pd.DataFrame:
        """
        依你們最後定案規則選車：

        1. 先找最近的 ALS 分隊
        2. 若該分隊有 2 台 ALS 可用 -> 派 2 台 ALS（限同一分隊）
        3. 若該分隊只有 1 台 ALS 可用 -> 派 1 台 ALS + 全體最近 1 台 BLS
        4. 若找不到符合這個規則的組合 -> 回空 DataFrame
        """
        if candidate_df.empty:
            return pd.DataFrame(columns=candidate_df.columns)

        als_df = candidate_df[candidate_df["is_als"]].copy()
        if als_df.empty:
            return pd.DataFrame(columns=candidate_df.columns)

        # 先找「最近的 ALS 分隊」
        # 因為 candidate_df 已按 travel_min / ambulance_id 排序，所以第一個 ALS 所在分隊就是最近 ALS 分隊
        nearest_als_row = als_df.iloc[0]
        nearest_als_station = nearest_als_row["station_name"]

        # 該 ALS 分隊底下所有可用 ALS
        als_same_station = als_df[als_df["station_name"] == nearest_als_station].copy()
        als_same_station = als_same_station.sort_values(
            by=["travel_min_sim", "ambulance_id"],
            ascending=[True, True]
        ).reset_index(drop=True)

        # 規則 1：最近 ALS 分隊若能出 2 台 ALS，就派 2 ALS（都來自同一 ALS 分隊）
        if len(als_same_station) >= 2:
            selected = als_same_station.head(2).copy()
            return selected.reset_index(drop=True)

        # 規則 2：最近 ALS 分隊只有 1 台 ALS -> 補全體最近 BLS 1 台
        if len(als_same_station) == 1:
            chosen_als = als_same_station.head(1).copy()

            bls_df = candidate_df[~candidate_df["is_als"]].copy()
            bls_df = bls_df.sort_values(
                by=["travel_min_sim", "ambulance_id"],
                ascending=[True, True]
            ).reset_index(drop=True)

            if len(bls_df) >= 1:
                chosen_bls = bls_df.head(1).copy()
                selected = pd.concat([chosen_als, chosen_bls], ignore_index=True)
                selected = selected.sort_values(
                    by=["travel_min_sim", "ambulance_id"],
                    ascending=[True, True]
                ).reset_index(drop=True)
                return selected

            # 沒有 BLS，就不符合你們最後定案規則
            return pd.DataFrame(columns=candidate_df.columns)

        return pd.DataFrame(columns=candidate_df.columns)

    # 先檢查 accepted_dt 當下
    candidate_now = build_candidate_df(accepted_dt)
    selected_now = pick_units_by_final_rule(candidate_now)

    if len(selected_now) == required_units:
        dispatch_dt = accepted_dt
        wait_min = 0.0
        selected_units = []

        for _, row in selected_now.iterrows():
            arrival_dt = dispatch_dt + pd.to_timedelta(row["travel_min_sim"], unit="m")
            selected_units.append(
                {
                    "ambulance_id": row["ambulance_id"],
                    "station_name": row["station_name"],
                    "is_als": bool(row["is_als"]),
                    "dispatch_dt_sim": dispatch_dt,
                    "arrival_dt_sim": arrival_dt,
                    "travel_min_sim": float(row["travel_min_sim"]),
                    "wait_min_sim": wait_min,
                    "response_min_sim": wait_min + float(row["travel_min_sim"]),
                }
            )

        return {
            "dispatch_dt_sim": dispatch_dt,
            "selected_units": selected_units,
        }

    # 當下不行，就往後找最早可行時間點
    future_times = (
        fs["available_at"]
        .dropna()
        .loc[lambda s: s > accepted_dt]
        .sort_values()
        .unique()
    )

    for future_time in future_times:
        dispatch_dt = pd.Timestamp(future_time)
        candidate_future = build_candidate_df(dispatch_dt)
        selected_future = pick_units_by_final_rule(candidate_future)

        if len(selected_future) != required_units:
            continue

        wait_min = (dispatch_dt - accepted_dt).total_seconds() / 60.0
        selected_units = []

        for _, row in selected_future.iterrows():
            arrival_dt = dispatch_dt + pd.to_timedelta(row["travel_min_sim"], unit="m")
            selected_units.append(
                {
                    "ambulance_id": row["ambulance_id"],
                    "station_name": row["station_name"],
                    "is_als": bool(row["is_als"]),
                    "dispatch_dt_sim": dispatch_dt,
                    "arrival_dt_sim": arrival_dt,
                    "travel_min_sim": float(row["travel_min_sim"]),
                    "wait_min_sim": wait_min,
                    "response_min_sim": wait_min + float(row["travel_min_sim"]),
                }
            )

        return {
            "dispatch_dt_sim": dispatch_dt,
            "selected_units": selected_units,
        }

    # 若整個 fleet_state 都找不到符合最終規則的時點，才回 None
    return None


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
