# -*- coding: utf-8 -*-
"""
diagnose_speed_anomalies.py
──────────────────────────
分析 tdx_section_live_raw.jsonl 裡 TravelSpeed 接近 0 或異常的資料，
判斷這些是「真實壅塞（合理）」還是「感測器沒讀到值、系統補的無效佔位資料」。

判斷邏輯：
    - TravelSpeed = 0 且 TravelTime = 0 → 高度可疑是「無資料佔位」
      （真的塞到完全不動，通過這段路的時間不可能是 0 秒，除非根本沒車通過或沒量到）
    - TravelSpeed = 0 但 TravelTime > 0 → 較可能是真實嚴重壅塞（合理）
    - 同時比對這些異常值是集中在深夜（較可能是無資料）還是尖峰時段（較可能是真實塞車）

這是為了追查 WMAPE 偏高、以及訓練時一直出現的
「RuntimeWarning: divide by zero encountered in divide」警告，
找出根本原因並決定資料清理策略。

使用方式：
    python diagnose_speed_anomalies.py
"""

import os
import json
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
RAW_JSONL = os.path.join(PARENT_DIR, "fetch_tdx_section_live", "tdx_section_live_raw.jsonl")


def main():
    if not os.path.isfile(RAW_JSONL):
        print(f"❌ 找不到檔案：{RAW_JSONL}")
        return

    print(f"讀取：{RAW_JSONL}")

    section_ids = []
    speeds = []
    travel_times = []
    hours = []
    has_vd_list = []
    has_etag_list = []

    line_count = 0
    with open(RAW_JSONL, encoding="utf-8") as f:
        for line in f:
            line_count += 1
            line = line.strip().lstrip("\ufeff")
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue

            speed = r.get("TravelSpeed")
            if speed is None:
                continue

            section_ids.append(r.get("SectionID"))
            speeds.append(speed)
            travel_times.append(r.get("TravelTime"))

            t = r.get("DataCollectTime", "")
            hour = int(t[11:13]) if len(t) >= 13 else -1
            hours.append(hour)

            ds = r.get("DataSources") or {}
            has_vd_list.append(ds.get("HasVD", 0))
            has_etag_list.append(ds.get("HasETAG", 0))

            if line_count % 2_000_000 == 0:
                print(f"  → 已讀取 {line_count:,} 行...")

    print(f"讀取完成，共 {line_count:,} 行")

    df = pd.DataFrame({
        "SectionID": pd.Categorical(section_ids),
        "TravelSpeed": pd.array(speeds, dtype="float32"),
        "TravelTime": pd.array(travel_times, dtype="float32"),
        "Hour": pd.array(hours, dtype="int16"),
        "HasVD": pd.array(has_vd_list, dtype="int8"),
        "HasETAG": pd.array(has_etag_list, dtype="int8"),
    })
    del section_ids, speeds, travel_times, hours, has_vd_list, has_etag_list
    print(f"總筆數：{len(df):,}\n")

    # ── 1. 整體車速分布 ──────────────────────────────
    print(f"{'='*60}")
    print("TravelSpeed 整體分布")
    print(f"{'='*60}")
    print(df["TravelSpeed"].describe().to_string())
    print()
    zero_speed = (df["TravelSpeed"] == 0).sum()
    low_speed = (df["TravelSpeed"] < 5).sum() - zero_speed
    print(f"車速剛好 = 0：{zero_speed:,} 筆（{zero_speed/len(df):.2%}）")
    print(f"車速 0 < speed < 5：{low_speed:,} 筆（{low_speed/len(df):.2%}）")

    # ── 2. 車速=0 時，TravelTime 是不是也剛好是 0（無資料佔位的強烈訊號） ──────────────────────────────
    print(f"\n{'='*60}")
    print("車速 = 0 的資料裡，TravelTime 的分布狀況")
    print(f"{'='*60}")
    zero_speed_df = df[df["TravelSpeed"] == 0]
    zero_speed_zero_time = (zero_speed_df["TravelTime"] == 0).sum()
    zero_speed_nonzero_time = (zero_speed_df["TravelTime"] > 0).sum()
    print(f"車速=0 且 TravelTime=0（高度可疑是「無資料佔位」）：{zero_speed_zero_time:,} 筆"
          f"（占所有車速=0 資料的 {zero_speed_zero_time/max(len(zero_speed_df),1):.1%}）")
    print(f"車速=0 但 TravelTime>0（較可能是真實嚴重壅塞）：{zero_speed_nonzero_time:,} 筆"
          f"（占所有車速=0 資料的 {zero_speed_nonzero_time/max(len(zero_speed_df),1):.1%}）")

    # ── 3. 車速=0 的資料來源分布（是不是特定感測器來源比較容易出現0值） ──────────────────────────────
    print(f"\n{'='*60}")
    print("車速 = 0 的資料，其 DataSources 分布")
    print(f"{'='*60}")
    print(f"其中有用到 VD 的比例：{zero_speed_df['HasVD'].mean():.1%}")
    print(f"其中有用到 ETag 的比例：{zero_speed_df['HasETAG'].mean():.1%}")
    print(f"（對照組）全體資料中有用到 VD 的比例：{df['HasVD'].mean():.1%}")
    print(f"（對照組）全體資料中有用到 ETag 的比例：{df['HasETAG'].mean():.1%}")

    # ── 4. 車速=0 是否集中在深夜（無資料嫌疑）還是尖峰時段（真實壅塞嫌疑） ──────────────────────────────
    print(f"\n{'='*60}")
    print("車速 = 0 的資料，依「一天中的小時」分布")
    print(f"{'='*60}")
    hour_dist = zero_speed_df["Hour"].value_counts().sort_index()
    print(hour_dist.to_string())

    # ── 5. 哪些路段的車速=0 特別多 ──────────────────────────────
    print(f"\n{'='*60}")
    print("車速 = 0 筆數最多的前 15 個路段")
    print(f"{'='*60}")
    section_zero_counts = zero_speed_df["SectionID"].value_counts().head(15)
    print(section_zero_counts.to_string())

    # ── 6. 如果排除「車速=0 且 TravelTime=0」這種高度可疑的無效資料，剩下多少 ──────────────────────────────
    print(f"\n{'='*60}")
    print("清理建議")
    print(f"{'='*60}")
    suspicious_mask = (df["TravelSpeed"] == 0) & (df["TravelTime"] == 0)
    print(f"符合「車速=0 且 TravelTime=0」（建議視為無效值排除）的筆數：{suspicious_mask.sum():,}"
          f"（占全體資料 {suspicious_mask.sum()/len(df):.3%}）")
    print(f"排除這些之後，剩餘筆數：{(~suspicious_mask).sum():,}")


if __name__ == "__main__":
    main()
