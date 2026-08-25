# -*- coding: utf-8 -*-
"""
build_taichung_stgcn_dataset.py
──────────────────────────
把前面幾支程式的產出組合起來，做出跟 hazdzz/STGCN 的
data/metr-la/vel.csv、adj.npz 完全對應格式的台中版本。

輸入：
    1. tdx_section_live_raw.jsonl      （fetch_tdx_section_live.py 的輸出：路段+時間+車速）
    2. section_to_edge_mapping.csv     （map_section_to_network.py 的輸出：篩選出有效路段）
    3. graph_nodes/edges_taichung.csv  （完整路網，用來算路段間的真實道路距離）

輸出（存到 Capture_Road_Node 資料夾）：
    1. taichung_vel.csv     → (時間, 路段) 的車速矩陣，格式對應 STGCN 的 vel.csv
    2. taichung_adj.npy     → (路段, 路段) 的鄰接矩陣，格式對應 STGCN 的 adj.npz
    3. taichung_section_index.csv → 矩陣欄位對應的 SectionID 對照表
    4. taichung_timestamps.csv → vel.csv 每一列對應的實際時間

使用方式：
    python build_taichung_stgcn_dataset.py
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

RESAMPLE_INTERVAL = "5min"   # 把原始 1 分鐘資料降採樣成 5 分鐘一筆，對齊 METR-LA / STGCN 常見設定
RELEVANT_THRESHOLD_M = 300


def build_distance_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
    node_ids = nodes_df['node_id'].tolist()
    n = len(node_ids)
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    valid_mask = edges_df['from_node'].isin(id_to_idx) & edges_df['to_node'].isin(id_to_idx)
    edges_df = edges_df[valid_mask].copy()
    edges_df = edges_df.groupby(['from_node', 'to_node'], as_index=False)['length_m'].min()

    row = edges_df['from_node'].map(id_to_idx).values
    col = edges_df['to_node'].map(id_to_idx).values
    data = edges_df['length_m'].astype(float).values

    dist_graph = csr_matrix((data, (row, col)), shape=(n, n))
    return dist_graph, id_to_idx


def build_gaussian_adjacency(dist_matrix: np.ndarray, sigma: float = None):
    finite_mask = np.isfinite(dist_matrix) & (~np.eye(dist_matrix.shape[0], dtype=bool))
    finite_dists = dist_matrix[finite_mask]
    if sigma is None:
        sigma = float(finite_dists.std()) if len(finite_dists) > 0 else 1.0
    safe_dist = np.where(np.isfinite(dist_matrix), dist_matrix, np.inf)
    with np.errstate(over='ignore'):
        W = np.exp(-(safe_dist ** 2) / (sigma ** 2))
    W[~np.isfinite(dist_matrix)] = 0.0
    np.fill_diagonal(W, 1.0)
    return W, sigma


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

    DATA_DIR = os.path.join(PARENT_DIR, "Capture_Road_Node")
    NODES_CSV = os.path.join(DATA_DIR, "graph_nodes_taichung.csv")
    EDGES_CSV = os.path.join(DATA_DIR, "graph_edges_taichung.csv")
    MAPPING_CSV = os.path.join(DATA_DIR, "section_to_edge_mapping.csv")
    RAW_JSONL = os.path.join(PARENT_DIR, "fetch_tdx_section_live", "tdx_section_live_raw.jsonl")

    for path, name in [(NODES_CSV, "路網節點"), (EDGES_CSV, "路網路段"),
                        (MAPPING_CSV, "Section 對應表"), (RAW_JSONL, "TDX 路段車速原始資料")]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"找不到{name}檔案：{path}\n請確認前面幾支程式都已經跑過，或修改路徑設定。")

    # ── 1. 篩出有效路段 ──────────────────────────────
    print("[1/6] 讀取 Section 對應表，篩選有效路段...")
    mapping_df = pd.read_csv(MAPPING_CSV, encoding='utf-8-sig')
    relevant = mapping_df[mapping_df['distance_to_edge_m'] <= RELEVANT_THRESHOLD_M].copy()
    print(f"      → 有效路段數：{len(relevant)} / {len(mapping_df)}")
    if len(relevant) == 0:
        raise ValueError("沒有任何路段通過距離門檻，請確認 map_section_to_network.py 的結果。")

    valid_section_ids = set(relevant['SectionID'])

    # ── 2. 讀取原始車速時序資料，只保留有效路段 ──────────────────────────────
    print(f"[2/6] 讀取原始車速資料：{RAW_JSONL}")
    # 🌟 只擷取需要的 4 個欄位（SectionID、DataCollectTime、TravelSpeed、TravelTime），
    # 不把整包 JSON 物件（含 AuthorityCode、DataSources 巢狀物件等用不到的欄位）存進記憶體。
    # 資料量大時（例如半年份、將近 3000 萬筆），這樣可以把記憶體用量降到原本的 1/5~1/8，
    # 避免電腦記憶體不足當機。
    # 多讀 TravelTime 是為了判斷「TravelSpeed=0」是真實壅塞還是無效佔位值（見下方說明）。
    section_ids_list = []
    times_list = []
    speeds_list = []
    travel_times_list = []
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
            sid = r.get("SectionID")
            if sid in valid_section_ids:
                section_ids_list.append(sid)
                times_list.append(r.get("DataCollectTime"))
                speeds_list.append(r.get("TravelSpeed"))
                travel_times_list.append(r.get("TravelTime"))

            if line_count % 2_000_000 == 0:
                print(f"      → 已讀取 {line_count:,} 行，篩選後保留 {len(section_ids_list):,} 筆...")

    print(f"      → 篩選後筆數：{len(section_ids_list):,}")

    df = pd.DataFrame({
        "SectionID": section_ids_list,
        "DataCollectTime": times_list,
        "TravelSpeed": pd.array(speeds_list, dtype="float32"),
        "TravelTime": pd.array(travel_times_list, dtype="float32"),
    })
    # 用完就釋放這幾個中繼列表，進一步節省記憶體
    del section_ids_list, times_list, speeds_list, travel_times_list
    df["DataCollectTime"] = pd.to_datetime(df["DataCollectTime"])

    # 🌟 關鍵資料清理：TravelSpeed=0 且 TravelTime=0，代表這個路段這個時間點
    # ETag 系統沒有偵測到有效的車輛配對紀錄（沒算出東西），是無效佔位值，
    # 不是真實車速。診斷結果顯示，這種無效值占了全體資料高達 68%，
    # 如果照字面當成「時速 0」餵給模型，會嚴重誤導訓練，也是 WMAPE 一直
    # 偏高、一直出現 divide-by-zero 警告的主因。這裡把這種資料轉成缺值（NaN），
    # 讓後面的補值機制（前後補值）用鄰近真實有效的車速去填補，而不是當成真實的 0。
    invalid_mask = (df["TravelSpeed"] == 0) & (df["TravelTime"] == 0)
    invalid_count = invalid_mask.sum()
    print(f"      → 偵測到 {invalid_count:,} 筆無效佔位值（TravelSpeed=0 且 TravelTime=0），"
          f"占篩選後資料的 {invalid_count/len(df):.1%}，已轉成缺值處理")
    df.loc[invalid_mask, "TravelSpeed"] = np.nan

    # ── 3. 轉成 (時間 × 路段) 矩陣 ──────────────────────────────
    print("[3/6] 轉成寬表格式（時間 × 路段）...")
    # 同一路段同一時間如果有重複資料，取平均（NaN 會被 pandas 自動忽略，不會污染平均值）
    pivot = df.pivot_table(index="DataCollectTime", columns="SectionID", values="TravelSpeed", aggfunc="mean")

    # ── 4. 降採樣成固定時間間隔，並補值 ──────────────────────────────
    print(f"[4/6] 降採樣成每 {RESAMPLE_INTERVAL} 一筆，並補值...")
    pivot = pivot.resample(RESAMPLE_INTERVAL).mean()
    missing_ratio = pivot.isna().mean().mean()
    print(f"      → 降採樣後缺值比例：{missing_ratio:.1%}")
    # 缺值先用前一筆補（常見時序缺值處理），頭尾仍缺的用該路段平均車速補
    pivot = pivot.ffill().bfill()
    pivot = pivot.fillna(pivot.mean(axis=0))  # 如果整欄都是空的，退而求其次用全域平均

    section_order = list(pivot.columns)
    print(f"      → 最終矩陣大小：{pivot.shape[0]} 個時間點 × {pivot.shape[1]} 個路段")

    # ── 5. 算路段之間的真實道路距離，建立鄰接矩陣 ──────────────────────────────
    print("[5/6] 讀取完整路網，計算路段間的道路距離...")
    nodes_df = pd.read_csv(NODES_CSV)
    edges_df = pd.read_csv(EDGES_CSV)
    dist_graph, id_to_idx = build_distance_graph(nodes_df, edges_df)

    rep_nodes = []
    for sid in section_order:
        row = relevant[relevant['SectionID'] == sid].iloc[0]
        rep_nodes.append(id_to_idx[row['matched_from_node']])

    dist_from_sources = dijkstra(dist_graph, directed=True, indices=rep_nodes)
    dist_matrix = dist_from_sources[:, rep_nodes]
    dist_matrix = np.minimum(dist_matrix, dist_matrix.T)

    W, sigma_used = build_gaussian_adjacency(dist_matrix)
    print(f"      → 使用的 σ：{sigma_used:.1f} 公尺")

    # ── 6. 輸出 ──────────────────────────────
    print("[6/6] 輸出檔案...")
    vel_csv_path = os.path.join(DATA_DIR, "taichung_vel.csv")
    adj_npy_path = os.path.join(DATA_DIR, "taichung_adj.npy")
    index_csv_path = os.path.join(DATA_DIR, "taichung_section_index.csv")

    pivot.to_csv(vel_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 已輸出 {vel_csv_path}")

    # vel.csv 本身不含時間欄位（格式對齊 METR-LA），時間戳記另外存一份，
    # 方便之後想知道矩陣第 N 列對應到實際哪個時間點時查閱
    timestamps_path = os.path.join(DATA_DIR, "taichung_timestamps.csv")
    pd.Series(pivot.index, name="DataCollectTime").to_csv(timestamps_path, index=False, encoding='utf-8-sig')
    print(f"✅ 已輸出 {timestamps_path}（vel.csv 每一列對應的實際時間，供查閱用）")

    np.save(adj_npy_path, W)
    print(f"✅ 已輸出 {adj_npy_path}")

    index_df = pd.DataFrame({"matrix_index": range(len(section_order)), "SectionID": section_order})
    index_df = index_df.merge(relevant[['SectionID', 'RoadName', 'SectionName']], on='SectionID', how='left')
    index_df.to_csv(index_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 已輸出 {index_csv_path}")

    print(f"\n📐 最終資料集大小：{pivot.shape[0]} 個時間點 × {pivot.shape[1]} 個節點")
    print(f"📐 鄰接矩陣大小：{W.shape}")


if __name__ == "__main__":
    main()