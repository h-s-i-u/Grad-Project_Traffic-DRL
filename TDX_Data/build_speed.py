# -*- coding: utf-8 -*-
"""
build_speed.py  ← 取代原本 build_taichung_stgcn_dataset.py 的「時序」部分
──────────────────────────────────────────────────────────────
只處理跟時間序列有關的事：讀原始車速、過濾無效值、降採樣、補值、
剔除缺值過高的路段，最後把 build_network.py 算好的距離矩陣切成對應的鄰接矩陣。

與原版的三個差異：
  1. 不再需要 graph_nodes/edges（OSM 路網）——距離已由 build_network.py 算好
  2. 多輸出一份 **缺值遮罩 taichung_mask.npy**：實測降採樣後仍有約 30% 是補值，
     若把補值一起算進 MAE，成績會虛低且無法與 METR-LA（7.13% 缺失）並列比較。
     評估時只在 mask=True 的位置計分才有意義。
  3. 剔除缺值比例過高的路段（預設 >50%）——那些路段的「預測」幾乎都在預測補值

輸入：
    TDX_Data/tdx_section_live_raw.jsonl   （檔案：路段+時間+車速，約 11.9 GB，串流讀取）
    Map/taichung_dist.npy                  （build_network.py：路段間沿道路距離）
    Map/taichung_section_index_full.csv    （build_network.py：路段順序）
    Map/network_meta.json                  （build_network.py：σ、κ）

輸出（皆存到 Map/）：
    taichung_vel.csv              檔案E：(時間 × 路段) 車速矩陣，格式對齊 METR-LA
    taichung_mask.npy             新增：同形狀的布林矩陣，True = 真實觀測、False = 補值
    taichung_timestamps.csv       檔案H：每一列對應的實際時間
    taichung_adj.npy              檔案F：最終鄰接矩陣（已對齊篩選後的路段集合）
    taichung_section_index.csv    檔案G：最終欄位對照表
    speed_meta.json               缺值統計、被剔除的路段清單

使用方式：
    python build_speed.py
    python build_speed.py --max-missing 0.5    # 缺值超過 50% 的路段剔除
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
MAP_DIR = os.path.join(ROOT_DIR, "Map")

RESAMPLE_INTERVAL = "5min"   # 對齊 METR-LA / STGCN 的 5 分鐘慣例
MAX_MISSING_RATIO = 0.50     # 缺值比例超過這個值的路段直接剔除
MAX_SPEED_KMH = 120.0        # 超過此速度視為感測器異常（市區道路不可能），轉為缺值


def read_raw_speeds(path, valid_ids):
    """串流讀取 JSONL，只取需要的 4 個欄位。

    原始檔約 11.9 GB／2,800 萬行，整包 JSON 物件（含 DataSources 巢狀欄位）
    全載進記憶體會爆掉，所以逐行解析後只留下用得到的欄位。
    多讀 TravelTime 是為了判斷 TravelSpeed=0 究竟是真實壅塞還是無效佔位值。
    """
    sids, times, speeds, ttimes = [], [], [], []
    n_line = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            n_line += 1
            line = line.strip().lstrip("﻿")
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("SectionID") in valid_ids:
                sids.append(r["SectionID"])
                times.append(r.get("DataCollectTime"))
                speeds.append(r.get("TravelSpeed"))
                ttimes.append(r.get("TravelTime"))
            if n_line % 2_000_000 == 0:
                print(f"      → 已讀 {n_line:,} 行，保留 {len(sids):,} 筆...")

    df = pd.DataFrame({
        "SectionID": sids,
        "DataCollectTime": times,
        "TravelSpeed": pd.array(speeds, dtype="float32"),
        "TravelTime": pd.array(ttimes, dtype="float32"),
    })
    del sids, times, speeds, ttimes
    df["DataCollectTime"] = pd.to_datetime(df["DataCollectTime"])
    return df, n_line


def gaussian_adjacency(dist, sigma, kappa):
    """與 build_network.py 相同的核，但套用該檔算好的 σ 與 κ，確保兩邊一致"""
    safe = np.where(np.isfinite(dist), dist, np.inf)
    with np.errstate(over="ignore", invalid="ignore"):
        W = np.exp(-(safe ** 2) / (sigma ** 2))
    W[~np.isfinite(dist)] = 0.0
    if kappa is not None:
        W[safe > kappa] = 0.0
    np.fill_diagonal(W, 1.0)
    return W


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-missing", type=float, default=MAX_MISSING_RATIO,
                    help=f"缺值比例超過此值的路段剔除（預設 {MAX_MISSING_RATIO}）")
    ap.add_argument("--max-speed", type=float, default=MAX_SPEED_KMH,
                    help=f"超過此速度視為感測器異常並轉為缺值（km/h，預設 {MAX_SPEED_KMH:.0f}）")
    ap.add_argument("--resample", default=RESAMPLE_INTERVAL,
                    help=f"降採樣間隔（預設 {RESAMPLE_INTERVAL}）")
    ap.add_argument("--raw", default=os.path.join(SCRIPT_DIR, "tdx_section_live_raw.jsonl"),
                    help="fetch_tdx_section_live.py 的輸出路徑")
    args = ap.parse_args()

    dist_npy = os.path.join(MAP_DIR, "taichung_dist.npy")
    idx_csv = os.path.join(MAP_DIR, "taichung_section_index_full.csv")
    meta_json = os.path.join(MAP_DIR, "network_meta.json")
    for p, name in [(args.raw, "TDX 原始車速"), (dist_npy, "路段距離矩陣"),
                    (idx_csv, "路段清單"), (meta_json, "路網參數")]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"找不到{name}：{p}\n（距離／清單／參數請先執行 build_network.py）")

    print("[1/6] 讀取 build_network.py 的產物...")
    dist_full = np.load(dist_npy)
    idx_full = pd.read_csv(idx_csv, encoding="utf-8-sig")
    with open(meta_json, encoding="utf-8") as f:
        net_meta = json.load(f)
    order_full = list(idx_full["SectionID"])
    print(f"      → {len(order_full)} 個有效路段；σ={net_meta['sigma_m']:.1f}m、κ={net_meta['kappa_m']:.1f}m")

    print(f"[2/6] 串流讀取原始車速：{args.raw}")
    df, n_line = read_raw_speeds(args.raw, set(order_full))
    print(f"      → 共 {n_line:,} 行，保留 {len(df):,} 筆")
    if df.empty:
        raise ValueError("沒有讀到任何有效路段的資料，請確認 SectionID 是否對得起來。")

    # ETag 沒有偵測到有效車輛配對時，會同時把 TravelSpeed 與 TravelTime 填 0。
    # 這不是「時速 0」的真實壅塞，而是無效佔位值；照字面餵給模型會嚴重誤導訓練。
    invalid = (df["TravelSpeed"] == 0) & (df["TravelTime"] == 0)
    print(f"      → 無效佔位值（速度與時間同時為 0）：{int(invalid.sum()):,} 筆"
          f"（{invalid.mean():.1%}），轉為缺值")
    df.loc[invalid, "TravelSpeed"] = np.nan

    # 市區道路不可能出現的高速（實測 max 192 km/h）同樣是感測器異常。
    # 這裡轉成缺值而不是 clip：clip 會留下一個「捏造的觀測」並被 mask 標成 True，
    # 等於拿假資料去評分；轉缺值則會被補值機制處理、且 mask 正確標成 False。
    too_fast = df["TravelSpeed"] > args.max_speed
    print(f"      → 超過 {args.max_speed:.0f} km/h 的異常值：{int(too_fast.sum()):,} 筆"
          f"（{too_fast.mean():.2%}），轉為缺值")
    df.loc[too_fast, "TravelSpeed"] = np.nan

    print("[3/6] 轉成寬表並降採樣...")
    pivot = df.pivot_table(index="DataCollectTime", columns="SectionID",
                           values="TravelSpeed", aggfunc="mean")
    pivot = pivot.reindex(columns=order_full)          # 欄位順序對齊距離矩陣
    pivot = pivot.resample(args.resample).mean()
    print(f"      → {pivot.shape[0]} 個時間點 × {pivot.shape[1]} 個路段")

    # ★ 遮罩必須在補值「之前」取，這是評估能否誠實計分的關鍵
    mask_full = pivot.notna().to_numpy()
    miss_by_section = 1.0 - mask_full.mean(axis=0)
    overall_missing = 1.0 - mask_full.mean()
    print(f"      → 降採樣後整體缺值比例：{overall_missing:.1%}")

    print(f"[4/6] 剔除缺值 > {args.max_missing:.0%} 的路段...")
    keep = miss_by_section <= args.max_missing
    dropped = [{"SectionID": order_full[i], "missing_ratio": round(float(miss_by_section[i]), 4)}
               for i in np.where(~keep)[0]]
    print(f"      → 保留 {int(keep.sum())} / {len(order_full)} 個路段"
          f"（剔除 {len(dropped)} 個）")
    if keep.sum() < 2:
        raise ValueError("篩選後剩下的路段太少，請放寬 --max-missing。")

    order = [s for s, k in zip(order_full, keep) if k]
    pivot = pivot.loc[:, order]
    mask = mask_full[:, keep]

    print("[5/6] 補值（僅影響 vel.csv，mask 仍記錄原始觀測狀態）...")
    # 三層補值：前向 → 後向 → 該路段全域平均（處理整段皆缺的極端情況）
    pivot = pivot.ffill().bfill()
    pivot = pivot.fillna(pivot.mean(axis=0))
    print(f"      → 補值後缺值：{int(pivot.isna().sum().sum())} 格"
          f"（其中 {1 - mask.mean():.1%} 的格子是補出來的，評估時應以 mask 排除）")

    print("[6/6] 產生鄰接矩陣並輸出...")
    dist = dist_full[np.ix_(keep, keep)]
    W = gaussian_adjacency(dist, net_meta["sigma_m"], net_meta["kappa_m"])
    print(f"      → 鄰接密度 {float((W > 0).mean()):.1%}（目標 {net_meta['target_density']:.1%}）")

    p_vel = os.path.join(MAP_DIR, "taichung_vel.csv")
    pivot.to_csv(p_vel, index=False, encoding="utf-8-sig")
    print(f"✅ {p_vel}  {pivot.shape}")

    p_mask = os.path.join(MAP_DIR, "taichung_mask.npy")
    np.save(p_mask, mask)
    print(f"✅ {p_mask}  {mask.shape}（True = 真實觀測）")

    p_ts = os.path.join(MAP_DIR, "taichung_timestamps.csv")
    pd.Series(pivot.index, name="DataCollectTime").to_csv(
        p_ts, index=False, encoding="utf-8-sig")
    print(f"✅ {p_ts}")

    p_adj = os.path.join(MAP_DIR, "taichung_adj.npy")
    np.save(p_adj, W)
    print(f"✅ {p_adj}  {W.shape}")

    p_idx = os.path.join(MAP_DIR, "taichung_section_index.csv")
    pd.DataFrame({"matrix_index": range(len(order)), "SectionID": order}).merge(
        idx_full[["SectionID", "RoadName", "SectionName"]], on="SectionID", how="left"
    ).to_csv(p_idx, index=False, encoding="utf-8-sig")
    print(f"✅ {p_idx}")

    p_meta = os.path.join(MAP_DIR, "speed_meta.json")
    with open(p_meta, "w", encoding="utf-8") as f:
        json.dump({
            "n_timesteps": int(pivot.shape[0]), "n_sections": int(pivot.shape[1]),
            "resample": args.resample,
            "invalid_placeholder_ratio": float(invalid.mean()),
            "over_speed_ratio": float(too_fast.mean()),
            "max_speed_kmh": float(args.max_speed),
            "missing_after_resample": float(overall_missing),
            "imputed_ratio_final": float(1 - mask.mean()),
            "max_missing_threshold": float(args.max_missing),
            "dropped_sections": dropped,
            "time_start": str(pivot.index[0]), "time_end": str(pivot.index[-1]),
            "timezone_note": "DataCollectTime 為 UTC，台灣時間需 +8 小時",
        }, f, ensure_ascii=False, indent=2)
    print(f"✅ {p_meta}")

    print(f"\n📐 最終資料集：{pivot.shape[0]} 個時間點 × {pivot.shape[1]} 個路段")
    print(f"📐 補值比例 {1 - mask.mean():.1%} —— 評估 MAE/RMSE 時務必用 mask 排除，"
          f"否則成績虛低且無法與 METR-LA 比較")
    print(f"\n下一步：python convert_to_stgcn_dataset.py")


if __name__ == "__main__":
    main()
