# -*- coding: utf-8 -*-
"""
build_adjacency_matrix.py
──────────────────────────
把 Capture_Road_Node.py 產出的 graph_nodes_taichung.csv / graph_edges_taichung.csv
轉換成 STGCN 可以直接使用的「鄰接矩陣」。

對應計畫書 4.2 節公式：
    Wij = exp( -dij² / σ² )   若 dij ≤ κ（門檻值，可選）
其中 dij 是「路網最短距離」（沿著實際道路走的距離，不是直線距離），
用 Floyd 演算法（節點數較多時自動改用等效但更快的 Dijkstra）在路網圖上計算。

輸出三個檔案：
    1. adjacency_taichung.npy       → (N, N) 的 numpy 矩陣，直接餵給 STGCN 訓練程式
    2. adjacency_taichung.csv       → 同樣的矩陣，但存成 CSV，方便你打開檢查
    3. adjacency_node_index.csv     → node_id 對應矩陣中第幾個 index 的對照表
                                       （訓練時一定要用這個對照表，才知道矩陣第 i 列/欄
                                        對應到原本地圖上的哪個路口）

使用方式：
    python build_adjacency_matrix.py

    這支程式會自動去讀「自己所在資料夾的上一層 → Capture_Road_Node 資料夾」
    裡的 graph_nodes_taichung.csv / graph_edges_taichung.csv，
    輸出的三個檔案也會存到同一個 Capture_Road_Node 資料夾裡。
    不管你從哪裡執行這支程式（例如用 VS Code 執行、或雙擊執行），路徑都不會跑掉。

    如果你的資料夾結構不同，改最下面 main() 裡的 DATA_DIR 就好。
"""

import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def load_graph_csvs(nodes_path: str, edges_path: str):
    """讀取 nodes / edges CSV，並回傳基本統計資訊"""
    print(f"[1/5] 讀取節點資料：{nodes_path}")
    nodes_df = pd.read_csv(nodes_path)
    print(f"[1/5] 讀取路段資料：{edges_path}")
    edges_df = pd.read_csv(edges_path)

    print(f"      → 節點數：{len(nodes_df)}")
    print(f"      → 路段數（原始，含重複邊）：{len(edges_df)}")
    return nodes_df, edges_df


def build_distance_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
    """
    把 edges_df 轉成稀疏矩陣（graph 的鄰接表示），邊權重是 length_m。
    這一步只是把「路段清單」轉成程式可以拿去算最短路徑的資料結構，
    還不是最終的 Wij 鄰接矩陣。
    """
    print("[2/5] 建立節點 ID ↔ 矩陣 index 對照表...")
    node_ids = nodes_df['node_id'].tolist()
    n = len(node_ids)
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # 過濾掉端點不在節點清單裡的邊（正常情況不會發生，這裡是保險機制）
    valid_mask = edges_df['from_node'].isin(id_to_idx) & edges_df['to_node'].isin(id_to_idx)
    dropped = int((~valid_mask).sum())
    if dropped > 0:
        print(f"      ⚠️ 有 {dropped} 條邊的端點不在節點清單中，已自動略過")
    edges_df = edges_df[valid_mask].copy()

    # 同一對 (from_node, to_node) 若有多筆資料（multigraph 的情況，
    # 例如同一路口間有兩條並行道路），只保留「距離最短」的那一條，
    # 因為我們要算的是「這兩個路口之間最快能怎麼走」。
    print("[3/5] 合併重複路段（同節點對取最短距離）...")
    before = len(edges_df)
    edges_df = (
        edges_df.groupby(['from_node', 'to_node'], as_index=False)['length_m']
        .min()
    )
    after = len(edges_df)
    if before != after:
        print(f"      → 合併後路段數：{after}（減少了 {before - after} 條重複邊）")

    row = edges_df['from_node'].map(id_to_idx).values
    col = edges_df['to_node'].map(id_to_idx).values
    data = edges_df['length_m'].astype(float).values

    # OSM 路網是有向的（單行道會只有一個方向的邊），
    # 用 csr_matrix 存成有向圖，之後算最短路徑時會照方向走。
    dist_graph = csr_matrix((data, (row, col)), shape=(n, n))

    return dist_graph, node_ids, id_to_idx


def compute_shortest_path_matrix(dist_graph: csr_matrix):
    """
    計算所有節點兩兩之間的「路網最短距離」。
    計畫書指定用 Floyd 演算法，但 Floyd 的計算量是 O(n^3)，
    節點數一多（例如超過 800 個）就會非常慢。
    所以這裡自動判斷：節點數不多就照計畫書用 Floyd-Warshall，
    節點數太多則改用數學上完全等價、但速度快很多的 Dijkstra-based 方法
    （scipy 內建，逐一從每個節點出發算最短路徑，結果跟 Floyd 一模一樣）。
    """
    n = dist_graph.shape[0]
    print(f"[4/5] 計算最短路徑矩陣（節點數 = {n}）...")

    if n <= 800:
        method = 'FW'  # Floyd-Warshall，符合計畫書描述
        print("      → 節點數不多，採用 Floyd-Warshall 演算法")
    else:
        method = 'D'   # Dijkstra，等價結果、速度快很多
        print("      → 節點數較多，Floyd-Warshall 會太慢，"
              "改用等效的 Dijkstra-based 全點對最短路徑演算法")

    # OSM 路網有單行道，這裡先照「有向圖」計算，
    # 走不到的節點對距離會是 inf（例如單行道逆向、或分屬不相連的路網區塊）
    dist_matrix = shortest_path(dist_graph, method=method, directed=True)

    # 因為 Wij 這種「空間相依權重」通常視為對稱關係（A、B 兩點的關聯程度
    # 不該因為方向不同而不同），這裡取兩個方向中「較短」的距離代表兩點的
    # 路網距離。如果你想改成嚴格依照單行道方向、不對稱，把這行拿掉即可。
    dist_matrix = np.minimum(dist_matrix, dist_matrix.T)

    n_unreachable = np.isinf(dist_matrix).sum()
    total_pairs = n * n
    print(f"      → 無法互通的節點對：{n_unreachable} / {total_pairs} "
          f"（{n_unreachable / total_pairs:.1%}，代表路網中有分離的區塊或單行道死巷）")

    return dist_matrix


def build_gaussian_adjacency(dist_matrix: np.ndarray, sigma: float = None, threshold_m: float = None):
    """
    套用 Gaussian kernel：Wij = exp(-dij² / σ²)，若 dij > threshold_m 則設為 0。

    sigma:        高斯核的寬度，數字越小，鄰接矩陣越「集中在鄰近節點」；
                   預設 None 時，自動用「所有可達節點對距離」的標準差。
    threshold_m:  距離門檻（單位：公尺）。超過這個距離的兩點視為不相關，
                   直接設為 0（讓矩陣稀疏一點，也更符合物理直覺——
                   離很遠的兩個路口，交通上通常沒有直接關聯）。
                   預設 None 時不做門檻篩選，只靠 Gaussian kernel 自然衰減。
    """
    print("[5/5] 套用 Gaussian kernel 計算 Wij...")

    finite_mask = np.isfinite(dist_matrix) & (~np.eye(dist_matrix.shape[0], dtype=bool))
    finite_dists = dist_matrix[finite_mask]

    if sigma is None:
        sigma = float(finite_dists.std())
        print(f"      → 自動計算 σ（可達節點對距離的標準差）= {sigma:.1f} 公尺")
    else:
        print(f"      → 使用者指定 σ = {sigma:.1f} 公尺")

    # 先把 inf 換成一個很大的數字，避免 exp 計算時出現 nan
    safe_dist = np.where(np.isfinite(dist_matrix), dist_matrix, np.inf)

    with np.errstate(over='ignore'):
        W = np.exp(-(safe_dist ** 2) / (sigma ** 2))

    # 走不到的節點對，關聯度強制設為 0（exp(-inf) 理論上也會趨近 0，這裡明確寫死避免浮點誤差）
    W[~np.isfinite(dist_matrix)] = 0.0

    # 自己對自己的關聯設為 1（節點對自身完全相關，這是 GCN 常見慣例，
    # 也讓後續計算 Laplacian 時每個節點至少跟自己有連結）
    np.fill_diagonal(W, 1.0)

    if threshold_m is not None:
        print(f"      → 套用距離門檻：超過 {threshold_m:.0f} 公尺的節點對關聯度設為 0")
        W[dist_matrix > threshold_m] = 0.0

    sparsity = (W == 0).sum() / W.size
    print(f"      → 完成。矩陣稀疏度（0 的比例）：{sparsity:.1%}")

    return W, sigma


def main():
    # ── 你可以在這裡調整參數 ──────────────────────────────
    # 這支程式檔案自己所在的資料夾
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # 往上一層，再進入 Capture_Road_Node 資料夾 → 這裡才是 CSV 實際存放的地方
    # 例如：.../K/build_adjacency_matrix.py 這支程式在跑，
    #      會去讀 .../K/Capture_Road_Node/graph_nodes_taichung.csv
    DATA_DIR = os.path.join(SCRIPT_DIR, os.pardir, "Capture_Road_Node")
    DATA_DIR = os.path.abspath(DATA_DIR)

    NODES_CSV = os.path.join(DATA_DIR, "graph_nodes_taichung.csv")
    EDGES_CSV = os.path.join(DATA_DIR, "graph_edges_taichung.csv")

    SIGMA = None          # None = 自動計算；也可以自己填一個數字（單位：公尺），例如 500
    THRESHOLD_M = None    # None = 不篩選；也可以填距離門檻（單位：公尺），例如 2000
    # ──────────────────────────────────────────────────

    print(f"📂 讀取資料夾：{DATA_DIR}")
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"找不到資料夾：{DATA_DIR}\n"
            f"請確認這支程式的位置，跟 Capture_Road_Node 資料夾是不是同一層（互為兄弟資料夾）。"
        )

    nodes_df, edges_df = load_graph_csvs(NODES_CSV, EDGES_CSV)
    dist_graph, node_ids, id_to_idx = build_distance_graph(nodes_df, edges_df)
    dist_matrix = compute_shortest_path_matrix(dist_graph)
    W, sigma_used = build_gaussian_adjacency(dist_matrix, sigma=SIGMA, threshold_m=THRESHOLD_M)

    # ── 輸出三個檔案（存到跟 CSV 同一個 Capture_Road_Node 資料夾裡）──
    npy_path = os.path.join(DATA_DIR, "adjacency_taichung.npy")
    csv_path = os.path.join(DATA_DIR, "adjacency_taichung.csv")
    index_path = os.path.join(DATA_DIR, "adjacency_node_index.csv")

    np.save(npy_path, W)
    print(f"\n✅ 已輸出 {npy_path}（訓練程式直接讀這個）")

    W_df = pd.DataFrame(W, index=node_ids, columns=node_ids)
    W_df.to_csv(csv_path)
    print(f"✅ 已輸出 {csv_path}（可以打開檢查數值）")

    index_df = pd.DataFrame({
        "matrix_index": range(len(node_ids)),
        "node_id": node_ids,
    })
    index_df.to_csv(index_path, index=False)
    print(f"✅ 已輸出 {index_path}（node_id 對照矩陣 index 用）")

    print(f"\n📐 最終矩陣大小：{W.shape[0]} × {W.shape[1]}")
    print(f"📐 使用的 σ：{sigma_used:.1f} 公尺")


if __name__ == "__main__":
    main()