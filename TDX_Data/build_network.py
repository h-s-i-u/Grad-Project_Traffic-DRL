# -*- coding: utf-8 -*-
"""
build_network.py  ← 取代原本的 map_section_to_network.py（並接手鄰接矩陣的計算）
──────────────────────────────────────────────────────────────
把「所有跟地理／拓樸有關」的計算集中在這一支，產生後續只需要一次的靜態路網產物。

為什麼要這樣拆：
    原本鄰接矩陣是在 build_taichung_stgcn_dataset.py 裡算的，所以每次想微調
    σ 或距離門檻，都得重讀 11.9 GB 的 tdx_section_live_raw.jsonl。
    拆開後：
        本程式（靜態幾何）→ 只讀 CSV，幾秒鐘就能重跑
        build_speed.py（動態時序）→ 只需要本程式的產物 + raw.jsonl

輸入（皆可用 --nodes-csv / --edges-csv / --section-meta 覆寫）：
    Map/simplified_nodes_taichung.csv （檔案A：OSM 路口，預設為簡化後的路網）
    Map/simplified_edges_taichung.csv （檔案B：OSM 路段，一列一條「有向」邊）
    TDX_Data/tdx_section_metadata.csv （檔案C：TDX 發布路段座標）

    ⚠ 預設已改為 build_simplified_network.py 的產物（3,579 節點、已依 oneway 展開）。
      舊的 Map/graph_*_taichung.csv 仍可用 --nodes-csv/--edges-csv 指定以重現先前結果，
      但那份檔案不含單行道資訊。

輸出（皆存到 Map/）：
    section_to_edge_mapping.csv   檔案D：TDX↔OSM 一對一對應（沿用原格式，額外多幾欄）
    section_to_edges.csv          新增：一對「多」對應——一個 TDX 路段實際橫跨的所有 OSM edge
    taichung_dist.npy             新增：路段兩兩之間的「沿道路最短距離」矩陣（公尺）
    taichung_section_index_full.csv 通過距離門檻的路段清單（尚未做缺值篩選）
    network_meta.json             σ、距離門檻 κ、覆蓋率等，供 build_speed.py 沿用與報告引用

使用方式：
    python build_network.py
    python build_network.py --target-density 0.373   # 對齊 METR-LA 的鄰接密度
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
MAP_DIR = os.path.join(ROOT_DIR, "Map")

# TDX 路段中心點離 OSM 路網超過這個距離，視為不在框選範圍內，剔除
RELEVANT_THRESHOLD_M = 300
# 目標鄰接密度：METR-LA 的 adj_mx 實測非零比例為 37.3%，對齊它可讓兩個資料集的
# graph convolution 在「鄰居數量」這件事上條件相當，避免台中版因為全連通而稀釋空間結構
TARGET_DENSITY = 0.373
# 一對多比對時，若最短路徑長度超過「起訖點直線距離 × 這個倍數」，視為繞路太離譜而放棄
PATH_DETOUR_LIMIT = 3.0


# ─────────────────────────────────────────────────────────────
# 座標與幾何
# ─────────────────────────────────────────────────────────────
def latlon_to_local_meters(lat, lon, ref_lat):
    """經緯度轉成以公尺為單位的區域平面座標（範圍只有幾公里，這個近似誤差可忽略）"""
    R = 6371000.0
    x = np.radians(lon) * R * math.cos(math.radians(ref_lat))
    y = np.radians(lat) * R
    return x, y


def point_to_segment_distance(px, py, ax, ay, bx, by):
    """點到線段（不是無限延伸的直線）的最短距離，向量化計算"""
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab_len_sq = np.where((abx**2 + aby**2) == 0, 1e-9, abx**2 + aby**2)
    t = np.clip((apx * abx + apy * aby) / ab_len_sq, 0.0, 1.0)
    return np.sqrt((px - (ax + t * abx))**2 + (py - (ay + t * aby))**2)


# ─────────────────────────────────────────────────────────────
# 路網結構
# ─────────────────────────────────────────────────────────────
def build_distance_graph(nodes_df, edges_df):
    """把 OSM 路網組成 scipy 稀疏圖（權重＝路段長度，公尺），供 dijkstra 使用"""
    node_ids = nodes_df["node_id"].tolist()
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    ok = edges_df["from_node"].isin(id_to_idx) & edges_df["to_node"].isin(id_to_idx)
    e = edges_df[ok].copy()
    # 同一組起訖點若有多條平行邊，取最短的那條
    e = e.groupby(["from_node", "to_node"], as_index=False)["length_m"].min()

    graph = csr_matrix(
        (e["length_m"].astype(float).values,
         (e["from_node"].map(id_to_idx).values, e["to_node"].map(id_to_idx).values)),
        shape=(len(node_ids), len(node_ids)),
    )
    return graph, id_to_idx, node_ids


def match_sections_1to1(section_df, nodes_df, edges_df):
    """檔案D：每個 TDX 路段用「中心點」找最近的一條 OSM edge（沿用原本的邏輯）"""
    ref_lat = nodes_df["latitude"].mean()
    nx_, ny_ = latlon_to_local_meters(nodes_df["latitude"].values,
                                      nodes_df["longitude"].values, ref_lat)
    id_to_xy = dict(zip(nodes_df["node_id"], zip(nx_, ny_)))

    ok = edges_df["from_node"].isin(id_to_xy) & edges_df["to_node"].isin(id_to_xy)
    e = edges_df[ok].reset_index(drop=True)
    ax = np.array([id_to_xy[n][0] for n in e["from_node"]])
    ay = np.array([id_to_xy[n][1] for n in e["from_node"]])
    bx = np.array([id_to_xy[n][0] for n in e["to_node"]])
    by = np.array([id_to_xy[n][1] for n in e["to_node"]])

    sx, sy = latlon_to_local_meters(section_df["CenterLat"].values,
                                    section_df["CenterLon"].values, ref_lat)

    rows = []
    for i in range(len(section_df)):
        d = point_to_segment_distance(sx[i], sy[i], ax, ay, bx, by)
        j = int(np.argmin(d))
        r = section_df.iloc[i]
        rows.append({
            "SectionID": r["SectionID"],
            "RoadName": r.get("RoadName", ""),
            "SectionName": r.get("SectionName", ""),
            "section_lat": r["CenterLat"],
            "section_lon": r["CenterLon"],
            "matched_from_node": e.iloc[j]["from_node"],
            "matched_to_node": e.iloc[j]["to_node"],
            "distance_to_edge_m": round(float(d[j]), 1),
        })
    return pd.DataFrame(rows)


def nearest_nodes(lats, lons, nodes_df):
    """把一批座標各自貼到最近的 OSM 路口，回傳 (node_id 陣列, 距離公尺陣列)"""
    ref_lat = nodes_df["latitude"].mean()
    nx_, ny_ = latlon_to_local_meters(nodes_df["latitude"].values,
                                      nodes_df["longitude"].values, ref_lat)
    px, py = latlon_to_local_meters(np.asarray(lats, dtype=float),
                                    np.asarray(lons, dtype=float), ref_lat)
    ids = nodes_df["node_id"].values

    out_id, out_d = [], []
    for i in range(len(px)):
        d = np.sqrt((nx_ - px[i])**2 + (ny_ - py[i])**2)
        j = int(np.argmin(d))
        out_id.append(ids[j])
        out_d.append(float(d[j]))
    return np.array(out_id), np.array(out_d)


def match_sections_1tomany(section_df, nodes_df, graph, id_to_idx, node_ids, mapping_1to1):
    """一對多：一個 TDX 路段是「一整段路」（常橫跨數個路口），實際對應多條 OSM edge。

    做法：把路段的起點與終點各自貼到最近的路口，再在路網上求兩點間的最短路徑，
    路徑經過的每一條 edge 就是這個路段涵蓋的範圍。

    這對決策層（軌道 B）很重要——耦合時要知道預測車速該套用到「哪些邊」。
    只用一對一的話，212 個路段只能覆蓋 212 條邊；一對多可以覆蓋整段主幹道。
    """
    start_id, start_d = nearest_nodes(section_df["StartLat"], section_df["StartLon"], nodes_df)
    end_id, end_d = nearest_nodes(section_df["EndLat"], section_df["EndLon"], nodes_df)

    # 一次算完所有起點的最短路徑樹，比逐條呼叫 dijkstra 快很多
    src_idx = sorted({id_to_idx[n] for n in start_id if n in id_to_idx})
    src_pos = {v: i for i, v in enumerate(src_idx)}
    _, preds = dijkstra(graph, directed=True, indices=src_idx, return_predecessors=True)

    fallback = dict(zip(mapping_1to1["SectionID"],
                        zip(mapping_1to1["matched_from_node"], mapping_1to1["matched_to_node"])))

    rows, n_path, n_fallback = [], 0, 0
    for i, sid in enumerate(section_df["SectionID"].values):
        s, t = start_id[i], end_id[i]
        edges_seq = None

        if s in id_to_idx and t in id_to_idx and s != t:
            si, ti = id_to_idx[s], id_to_idx[t]
            # 從終點沿 predecessor 往回走，還原整條路徑
            path, cur, guard = [], ti, 0
            while cur != si and cur >= 0 and guard < 10000:
                path.append(cur)
                cur = preds[src_pos[si], cur]
                guard += 1
            if cur == si:
                path.append(si)
                path.reverse()
                # 檢查有沒有繞路繞得太離譜（貼點失敗時常見）
                straight = math.hypot(*(np.array(latlon_to_local_meters(
                    section_df.iloc[i]["EndLat"], section_df.iloc[i]["EndLon"],
                    nodes_df["latitude"].mean())) - np.array(latlon_to_local_meters(
                    section_df.iloc[i]["StartLat"], section_df.iloc[i]["StartLon"],
                    nodes_df["latitude"].mean()))))
                road_len = sum(graph[path[k], path[k + 1]] for k in range(len(path) - 1))
                if straight < 1.0 or road_len <= PATH_DETOUR_LIMIT * straight:
                    edges_seq = [(node_ids[path[k]], node_ids[path[k + 1]])
                                 for k in range(len(path) - 1)]

        if edges_seq:
            n_path += 1
        else:                                   # 找不到合理路徑 → 退回一對一的那條邊
            n_fallback += 1
            edges_seq = [fallback[sid]] if sid in fallback else []

        for seq, (f, t_) in enumerate(edges_seq):
            rows.append({"SectionID": sid, "seq": seq, "from_node": f, "to_node": t_})

    print(f"      → 一對多比對：{n_path} 個路段成功還原路徑，{n_fallback} 個退回單一邊")
    return pd.DataFrame(rows), start_d, end_d


# ─────────────────────────────────────────────────────────────
# 鄰接矩陣
# ─────────────────────────────────────────────────────────────
def choose_kappa(dist, target_density):
    """挑一個距離門檻 κ，使「距離 ≤ κ」的節點對比例剛好等於 target_density。

    計畫書 §4.2 的公式本來就寫 `Wij = exp(-dij²/σ²) if dij ≤ κ`，但原本的實作
    沒有套用門檻，導致 212×212 全部非零（100% 稠密）。全連通會讓 graph convolution
    在所有節點上平均，稀釋掉空間結構。
    """
    n = dist.shape[0]
    off = dist[~np.eye(n, dtype=bool)]
    k = int(np.clip(target_density * off.size, 1, off.size - 1))
    return float(np.partition(off, k)[k])


def gaussian_adjacency(dist, sigma=None, kappa=None):
    """Wij = exp(-dij²/σ²)（dij ≤ κ 才保留），對角線固定為 1"""
    n = dist.shape[0]
    off_finite = dist[(~np.eye(n, dtype=bool)) & np.isfinite(dist)]
    if sigma is None:
        sigma = float(off_finite.std()) if off_finite.size else 1.0

    safe = np.where(np.isfinite(dist), dist, np.inf)
    with np.errstate(over="ignore", invalid="ignore"):
        W = np.exp(-(safe ** 2) / (sigma ** 2))
    W[~np.isfinite(dist)] = 0.0
    if kappa is not None:
        W[safe > kappa] = 0.0
    np.fill_diagonal(W, 1.0)
    return W, sigma


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-density", type=float, default=TARGET_DENSITY,
                    help=f"鄰接矩陣的目標非零比例（預設 {TARGET_DENSITY}，對齊 METR-LA 的 37.3%%）")
    ap.add_argument("--relevant-threshold-m", type=float, default=RELEVANT_THRESHOLD_M,
                    help=f"TDX 路段離路網多遠就剔除（公尺，預設 {RELEVANT_THRESHOLD_M}）")
    ap.add_argument("--sigma", type=float, default=None,
                    help="高斯核 σ（公尺）。不指定則自動取可達距離的標準差")
    # Input paths are flags, not constants: the road network now comes from
    # build_simplified_network.py (3,579 nodes, one row per DIRECTED edge, one-ways
    # honoured) rather than the raw export. build_distance_graph already builds a
    # directed csr_matrix, so a directed edge list is what it wants -- no extra
    # handling is needed here. Point these at Map/graph_*_taichung.csv to reproduce
    # the earlier runs.
    ap.add_argument("--nodes-csv",
                    default=os.path.join(MAP_DIR, "simplified_nodes_taichung.csv"),
                    help="OSM node CSV (default: the simplified network)")
    ap.add_argument("--edges-csv",
                    default=os.path.join(MAP_DIR, "simplified_edges_taichung.csv"),
                    help="OSM edge CSV, one row per directed edge "
                         "(default: the simplified network)")
    ap.add_argument("--section-meta",
                    default=os.path.join(SCRIPT_DIR, "tdx_section_metadata.csv"),
                    help="TDX section coordinates")
    args = ap.parse_args()

    nodes_csv, edges_csv, meta_csv = args.nodes_csv, args.edges_csv, args.section_meta
    for p, name in [(nodes_csv, "OSM 路口"), (edges_csv, "OSM 路段"), (meta_csv, "TDX 路段座標")]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"找不到{name}：{p}")

    print("[1/6] 讀取輸入檔...")
    # 明確印出實際讀到的檔案：預設已改為簡化後的路網，
    # 若沿用舊檔會得到完全不同的 mapping 與距離矩陣，不該靠猜的
    print(f"      nodes : {nodes_csv}")
    print(f"      edges : {edges_csv}")
    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)
    section_df = pd.read_csv(meta_csv, encoding="utf-8-sig").dropna(
        subset=["CenterLat", "CenterLon", "StartLat", "StartLon", "EndLat", "EndLon"])
    print(f"      → OSM {len(nodes_df)} 節點 / {len(edges_df)} 邊；TDX {len(section_df)} 個路段")

    print("[2/6] 一對一比對（TDX 中心點 → 最近的 OSM edge）...")
    m1 = match_sections_1to1(section_df, nodes_df, edges_df)
    valid = m1[m1["distance_to_edge_m"] <= args.relevant_threshold_m].reset_index(drop=True)
    print(f"      → {len(valid)} / {len(m1)} 個路段在 {args.relevant_threshold_m:.0f} 公尺內（有效）")
    if valid.empty:
        raise ValueError("沒有任何路段通過距離門檻，請檢查 OSM 框選範圍是否涵蓋 TDX 路段。")

    print("[3/6] 建立路網距離圖...")
    graph, id_to_idx, node_ids = build_distance_graph(nodes_df, edges_df)

    print("[4/6] 一對多比對（起訖點貼路口 → 求最短路徑 → 展開成 edge 序列）...")
    sec_valid = section_df[section_df["SectionID"].isin(set(valid["SectionID"]))].reset_index(drop=True)
    m_many, start_d, end_d = match_sections_1tomany(
        sec_valid, nodes_df, graph, id_to_idx, node_ids, valid)

    print("[5/6] 計算路段兩兩之間的沿道路最短距離（dijkstra on 完整路網）...")
    section_order = list(valid["SectionID"])
    rep = [id_to_idx[valid.loc[valid.SectionID == s, "matched_from_node"].iloc[0]]
           for s in section_order]
    d_all = dijkstra(graph, directed=True, indices=rep)
    dist = d_all[:, rep]
    dist = np.minimum(dist, dist.T)          # 對稱化：取兩個方向較短者
    np.fill_diagonal(dist, 0.0)

    kappa = choose_kappa(dist, args.target_density)
    W, sigma = gaussian_adjacency(dist, sigma=args.sigma, kappa=kappa)
    density = float((W > 0).mean())
    reach = float(np.isfinite(dist).mean())
    print(f"      → σ = {sigma:.1f} 公尺；距離門檻 κ = {kappa:.1f} 公尺")
    print(f"      → 鄰接密度 {density:.1%}（目標 {args.target_density:.1%}）；可達節點對 {reach:.1%}")

    print("[6/6] 輸出...")
    os.makedirs(MAP_DIR, exist_ok=True)

    # 檔案D：沿用原欄位，額外補上「這個路段涵蓋幾條邊」與起訖點貼點誤差，方便檢查品質
    n_edges = m_many.groupby("SectionID").size()
    valid_out = valid.copy()
    valid_out["n_edges"] = valid_out["SectionID"].map(n_edges).fillna(0).astype(int)
    valid_out["start_snap_m"] = np.round(start_d, 1)
    valid_out["end_snap_m"] = np.round(end_d, 1)
    p_map = os.path.join(MAP_DIR, "section_to_edge_mapping.csv")
    # 未通過門檻的路段也一併留著（欄位對齊），保持與原本 244 筆的輸出一致
    m1_out = m1.merge(valid_out[["SectionID", "n_edges", "start_snap_m", "end_snap_m"]],
                      on="SectionID", how="left")
    m1_out.to_csv(p_map, index=False, encoding="utf-8-sig")
    print(f"✅ {p_map}（{len(m1_out)} 筆，其中 {len(valid)} 筆有效）")

    p_many = os.path.join(MAP_DIR, "section_to_edges.csv")
    m_many.to_csv(p_many, index=False, encoding="utf-8-sig")
    covered = m_many.groupby(["from_node", "to_node"]).ngroups
    print(f"✅ {p_many}（{len(m_many)} 筆對應，涵蓋 {covered} 條不重複的 OSM edge）")

    p_dist = os.path.join(MAP_DIR, "taichung_dist.npy")
    np.save(p_dist, dist)
    print(f"✅ {p_dist}  {dist.shape}（公尺，沿道路最短距離）")

    p_idx = os.path.join(MAP_DIR, "taichung_section_index_full.csv")
    pd.DataFrame({"matrix_index": range(len(section_order)),
                  "SectionID": section_order}).merge(
        valid[["SectionID", "RoadName", "SectionName"]], on="SectionID", how="left"
    ).to_csv(p_idx, index=False, encoding="utf-8-sig")
    print(f"✅ {p_idx}（{len(section_order)} 個路段，尚未做缺值篩選）")

    p_meta = os.path.join(MAP_DIR, "network_meta.json")
    with open(p_meta, "w", encoding="utf-8") as f:
        json.dump({
            "osm_nodes": int(len(nodes_df)), "osm_edges": int(len(edges_df)),
            "tdx_sections_total": int(len(m1)),
            "tdx_sections_valid": int(len(valid)),
            "relevant_threshold_m": float(args.relevant_threshold_m),
            "sigma_m": sigma, "kappa_m": kappa,
            "target_density": float(args.target_density),
            "achieved_density": density,
            "reachable_pair_ratio": reach,
            "edges_covered_by_sections": int(covered),
            "edge_side_coverage": float(covered / len(edges_df)),
            "section_order": section_order,
        }, f, ensure_ascii=False, indent=2)
    print(f"✅ {p_meta}")

    print(f"\n📐 路段側覆蓋率：{len(valid)}/{len(m1)} = {len(valid)/len(m1):.1%}"
          f"（有多少 TDX 路段能用）")
    print(f"📐 邊側覆蓋率：{covered}/{len(edges_df)} = {covered/len(edges_df):.1%}"
          f"（路網有多少邊拿得到預測；裁切路網後會大幅上升）")
    print("\n下一步：python build_speed.py")


if __name__ == "__main__":
    main()
