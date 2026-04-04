至少一定要有的四個函數

1. `initialize_fleet_state(fleet_config_df)`
目的： 把各分隊原始配置表展開成「每一台車一列」的初始狀態。

輸入： `fleet_config_df`
至少會有三欄：
- `station_name`
- `vehicle_count`
- `als_count`

要做的事：
- 依每個分隊的 `vehicle_count` 展開成多台車
- 前 `als_count` 台設成 ALS，其餘設成 BLS
- 每台車要有唯一 `ambulance_id`
- 初始都設成 `Available`
- 初始 `available_at` 可以設成很早的時間，例如 `1900-01-01`

建議輸出欄位：
- `ambulance_id`
- `station_name`
- `current_station`
- `home_station`
- `is_als`
- `status`
- `available_at`



2. `release_finished_units(fleet_state, current_time)`
目的： 在每一筆新案件進來時，先把已經完成任務的車改回可用。

輸入：
- `fleet_state`
- `current_time`

要做的事：
- 找出 `status == "On duty"` 且 `available_at <= current_time` 的車
- 把這些車改成 `Available`
- `current_station` 先直接設回 `home_station`
  - 這代表第一版先假設車結束任務後已回原分隊

輸出：
- 更新後的 `fleet_state`



3. `select_nearest_available(case_row, fleet_state, travel_lookup)`
目的： 依「最近且有空的車」規則，幫這筆案件選車。

輸入：
- `case_row`：主程式傳進來的一筆案件
  - 你至少會用到：
    - `accepted_dt`
    - `grid_id`
- `fleet_state`
- `travel_lookup`：字典，key = `(station_name, grid_id)`，value = `travel_minutes`

要做的事：
1. 先找 `Available` 的車
2. 對每台可用車，用 `(current_station, grid_id)` 去 `travel_lookup` 查去程時間
3. 若查不到 travel time，就跳過那台
4. 從可派車中選 `travel_minutes` 最短者
5. 若完全沒有可派車，就回傳 `None`

先做第一版就好：
- 不用處理雙軌
- 不用處理 ALS 限制
- 不用處理「等最早可用的車」
- 第一版只要做到：有空車就派最近的，沒空車就 `None`

若成功派車，回傳 dict：
- `ambulance_id`
- `station_name`
- `dispatch_dt_sim` = `accepted_dt`
- `arrival_dt_sim` = `accepted_dt + travel_minutes`
- `travel_min_sim`
- `wait_min_sim` = 0
- `response_min_sim` = `travel_min_sim`



4. `mark_unit_busy(fleet_state, ambulance_id, mission_end_time)`
目的：把剛被派出去的那台車改成忙碌。

輸入：
- `fleet_state`
- `ambulance_id`
- `mission_end_time`

要做的事：
- 找到對應 `ambulance_id`
- 把 `status` 改成 `On duty`
- 把 `available_at` 改成 `mission_end_time`

輸出：
- 更新後的 `fleet_state`



現在（可能）先不用做的事：
- 雙軌派遣
- 嚴重案件判斷
- ALS / BLS 限制派遣
- 等最早可用的車
- 醫院選擇
- 網格到醫院、醫院回分隊的移動時間
- 分梯次模擬
也許先把「單一分隊救護 + 最近可用車」跑通試試看



對接時要確認的事
1. `fleet_state` 到底是要用 DataFrame 還是 list of dict
2. 四個函數的輸入輸出格式要一致
3. `select_nearest_available()` 回傳的 key 名稱要跟主程式完全一致
4. `station_name` 命名要統一，避免 `陽明` / `陽明山` 這種對不起來
