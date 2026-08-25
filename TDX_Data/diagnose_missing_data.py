# -*- coding: utf-8 -*-
"""
diagnose_missing_data.py
──────────────────────────
分析 tdx_section_live_raw.jsonl 的缺值分布，找出：
    1. 哪些路段（SectionID）缺值最嚴重
    2. 缺值是不是集中在特定時段（例如深夜、特定幾天）
    3. 如果依「缺值比例」篩選掉最糟的路段，資料品質會如何改善

這是為了追查 build_taichung_stgcn_dataset.py 執行時看到的
「降採樣後缺值比例：52.8%」這個警訊，找出根本原因。

使用方式：
    python diagnose_missing_data.py
"""

import os
import json
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
RAW_JSONL = os.path.join(PARENT_DIR, "fetch_tdx_section_live", "tdx_section_live_raw.jsonl")

RESAMPLE_INTERVAL = "5min"
MISSING_THRESHOLD_CANDIDATES = [0.1, 0.2, 0.3, 0.4, 0.5]  # 測試幾種篩選門檻，看篩掉後剩多少路段


def main():
    if not os.path.isfile(RAW_JSONL):
        print(f"❌ 找不到檔案：{RAW_JSONL}")
        return

    print(f"讀取：{RAW_JSONL}")

    # 🌟 只擷取需要的 3 個欄位，不把整包 JSON 物件（含用不到的欄位）存進記憶體。
    # 資料量大時（例如半年份、將近 3000 萬筆），這樣可以把記憶體用量降到原本的 1/5~1/8，
    # 避免電腦記憶體不足當機。
    section_ids_list = []
    times_list = []
    speeds_list = []
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
            section_ids_list.append(r.get("SectionID"))
            times_list.append(r.get("DataCollectTime"))
            speeds_list.append(r.get("TravelSpeed"))

            if line_count % 2_000_000 == 0:
                print(f"  → 已讀取 {line_count:,} 行...")

    print(f"讀取完成，共 {line_count:,} 行，正在建立資料表（這一步也可能要花一點時間）...")

    # 用有效率的資料型態建立 DataFrame：
    # SectionID 用 category（重複值極多，category 比一般字串省記憶體很多）
    # TravelSpeed 用 float32（不需要 float64 的精度，省一半記憶體）
    df = pd.DataFrame({
        "SectionID": pd.Categorical(section_ids_list),
        "DataCollectTime": pd.to_datetime(times_list),
        "TravelSpeed": pd.array(speeds_list, dtype="float32"),
    })
    del section_ids_list, times_list, speeds_list  # 用完就釋放，進一步節省記憶體
    print(f"總筆數：{len(df):,}")

    pivot = df.pivot_table(index="DataCollectTime", columns="SectionID", values="TravelSpeed", aggfunc="mean")
    pivot = pivot.resample(RESAMPLE_INTERVAL).mean()

    print(f"\n完整時間範圍：{pivot.index.min()} ~ {pivot.index.max()}")
    print(f"共 {len(pivot)} 個時間點 × {pivot.shape[1]} 個路段")

    # ── 1. 每個路段的缺值比例 ──────────────────────────────
    missing_per_section = pivot.isna().mean().sort_values(ascending=False)
    print(f"\n{'='*60}")
    print("每個路段的缺值比例（由高到低，只列前 15 個最糟的）")
    print(f"{'='*60}")
    print(missing_per_section.head(15).to_string())

    print(f"\n整體缺值比例：{pivot.isna().values.mean():.1%}")
    print(f"缺值比例的分布：")
    print(f"  中位數：{missing_per_section.median():.1%}")
    print(f"  最好的路段：{missing_per_section.min():.1%}")
    print(f"  最差的路段：{missing_per_section.max():.1%}")

    # ── 2. 缺值是否集中在特定時段 ──────────────────────────────
    print(f"\n{'='*60}")
    print("缺值比例依「一天中的小時」分布（看是否集中在深夜等特定時段）")
    print(f"{'='*60}")
    missing_by_hour = pivot.isna().groupby(pivot.index.hour).mean().mean(axis=1)
    print(missing_by_hour.to_string())

    # ── 3. 缺值是否集中在特定日期 ──────────────────────────────
    print(f"\n{'='*60}")
    print("缺值比例依「日期」分布（看是否某幾天整個沒資料）")
    print(f"{'='*60}")
    missing_by_date = pivot.isna().groupby(pivot.index.date).mean().mean(axis=1)
    print(missing_by_date.to_string())

    # ── 4. 測試不同篩選門檻，篩掉最糟的路段後會剩多少 ──────────────────────────────
    print(f"\n{'='*60}")
    print("如果篩掉「缺值比例超過門檻」的路段，會剩下幾個路段：")
    print(f"{'='*60}")
    for threshold in MISSING_THRESHOLD_CANDIDATES:
        kept = (missing_per_section <= threshold).sum()
        print(f"  門檻 {threshold:.0%}：保留 {kept} / {len(missing_per_section)} 個路段")


if __name__ == "__main__":
    main()