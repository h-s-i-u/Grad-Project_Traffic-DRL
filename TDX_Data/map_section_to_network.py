# -*- coding: utf-8 -*-
"""
map_section_to_network.py
──────────────────────────
把 TDX「發布路段（Section）」對應（map matching）到你的 OSM 路網上最近的 edge。
邏輯跟 map_matching.py（VD 版）完全一樣，只是資料源換成 Section。

使用方式：
    python map_section_to_network.py

    資料夾結構假設：

        K/
        ├── map_section_to_network/
        │   └── map_section_to_network.py   ← 這支程式
        ├── Capture_Road_Node/
        │   ├── graph_nodes_taichung.csv
        │   └── graph_edges_taichung.csv
        └── fetch_tdx_section_metadata/
            └── tdx_section_metadata.csv     ← fetch_tdx_section_metadata.py 的輸出

    如果你的資料夾結構不同，改最下面 main() 裡的路徑就好。
"""

import os
import math
import numpy as np
import pandas as pd


def load_road_network(nodes_path: str, edges_path: str):
    print(f"[1/5] 讀取路網節點：{nodes_path}")
    nodes_df = pd.read_csv(nodes_path)
    print(f"[1/5] 讀取路網路段：{edges_path}")
    edges_df = pd.read_csv(edges_path)
    print(f"      → 節點數：{len(nodes_df)}，路段數：{len(edges_df)}")
    return nodes_df, edges_df


def latlon_to_local_meters(lat, lon, ref_lat):
    R = 6371000.0
    x = np.radians(lon) * R * math.cos(math.radians(ref_lat))
    y = np.radians(lat) * R
    return x, y


def point_to_segment_distance(px, py, ax, ay, bx, by):
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab_len_sq = abx**2 + aby**2
    ab_len_sq = np.where(ab_len_sq == 0, 1e-9, ab_len_sq)
    t = (apx * abx + apy * aby) / ab_len_sq
    t = np.clip(t, 0.0, 1.0)
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    dist = np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
    return dist


def map_match(section_df: pd.DataFrame, nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
    print("[2/5] 建立路網 edge 的座標...")

    ref_lat = nodes_df['latitude'].mean()
    node_x, node_y = latlon_to_local_meters(nodes_df['latitude'].values, nodes_df['longitude'].values, ref_lat)
    id_to_xy = dict(zip(nodes_df['node_id'], zip(node_x, node_y)))

    valid_mask = edges_df['from_node'].isin(id_to_xy) & edges_df['to_node'].isin(id_to_xy)
    dropped = int((~valid_mask).sum())
    if dropped > 0:
        print(f"      ⚠️ 有 {dropped} 條 edge 的端點不在節點清單中，已略過")
    edges_df = edges_df[valid_mask].reset_index(drop=True)

    ax = np.array([id_to_xy[n][0] for n in edges_df['from_node']])
    ay = np.array([id_to_xy[n][1] for n in edges_df['from_node']])
    bx = np.array([id_to_xy[n][0] for n in edges_df['to_node']])
    by = np.array([id_to_xy[n][1] for n in edges_df['to_node']])

    print(f"[3/5] 對 {len(section_df)} 個路段逐一比對最近的 edge（共比對 {len(edges_df)} 條 edge）...")

    sec_x, sec_y = latlon_to_local_meters(section_df['CenterLat'].values, section_df['CenterLon'].values, ref_lat)

    results = []
    for i in range(len(section_df)):
        dists = point_to_segment_distance(sec_x[i], sec_y[i], ax, ay, bx, by)
        best_idx = np.argmin(dists)
        best_dist = dists[best_idx]
        row = section_df.iloc[i]
        results.append({
            'SectionID': row['SectionID'],
            'RoadName': row.get('RoadName', ''),
            'SectionName': row.get('SectionName', ''),
            'section_lat': row['CenterLat'],
            'section_lon': row['CenterLon'],
            'matched_from_node': edges_df.iloc[best_idx]['from_node'],
            'matched_to_node': edges_df.iloc[best_idx]['to_node'],
            'distance_to_edge_m': round(float(best_dist), 1),
        })

    return pd.DataFrame(results)


def main():
    RELEVANT_THRESHOLD_M = 300

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

    DATA_DIR = os.path.join(PARENT_DIR, "Capture_Road_Node")
    NODES_CSV = os.path.join(DATA_DIR, "graph_nodes_taichung.csv")
    EDGES_CSV = os.path.join(DATA_DIR, "graph_edges_taichung.csv")

    SECTION_META_CSV = os.path.join(PARENT_DIR, "fetch_tdx_section_metadata", "tdx_section_metadata.csv")
    OUTPUT_CSV = os.path.join(DATA_DIR, "section_to_edge_mapping.csv")

    for path, name in [(NODES_CSV, "路網節點"), (EDGES_CSV, "路網路段"), (SECTION_META_CSV, "Section 座標資料")]:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"找不到{name}檔案：{path}\n"
                f"請先確認 fetch_tdx_section_metadata.py 已經跑過，或修改 main() 裡的路徑設定。"
            )

    nodes_df, edges_df = load_road_network(NODES_CSV, EDGES_CSV)

    print(f"[1/5] 讀取 Section 座標資料：{SECTION_META_CSV}")
    section_df = pd.read_csv(SECTION_META_CSV, encoding='utf-8-sig')
    section_df = section_df.dropna(subset=['CenterLat', 'CenterLon'])
    print(f"      → 路段數：{len(section_df)}")

    result_df = map_match(section_df, nodes_df, edges_df)

    print("[4/5] 輸出結果...")
    result_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"✅ 已輸出 {OUTPUT_CSV}")

    relevant = result_df[result_df['distance_to_edge_m'] <= RELEVANT_THRESHOLD_M]
    print(f"\n[5/5] 📊 統計：")
    print(f"   路段總數：{len(result_df)}")
    print(f"   距離路網 {RELEVANT_THRESHOLD_M} 公尺以內（有效路段）：{len(relevant)} 個")
    print(f"   距離路網 {RELEVANT_THRESHOLD_M} 公尺以上（可能不相關）：{len(result_df) - len(relevant)} 個")


if __name__ == "__main__":
    main()
