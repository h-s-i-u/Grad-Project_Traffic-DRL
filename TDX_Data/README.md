# TDX_Data — 台中路況資料處理 pipeline

把 TDX 的路段車速資料與 OSM 路網結合，產生 STGCN／STGAT 可直接訓練的資料集。

---

## 快速開始

```bash
cd TDX_Data

# 0) 金鑰：複製 .env.example → .env，填入 TDX 的 Client Id / Secret
cp .env.example .env

# 1) 抓資料（metadata 幾秒；live 會跑很久，約 11.9 GB）
python fetch_tdx_section_metadata.py
python fetch_tdx_section_live.py

# 2) 建路網（靜態幾何，只讀 CSV，數秒可重跑）
python build_network.py

# 3) 建車速矩陣（動態時序，讀 11.9 GB，較久）
python build_speed.py

# 4) 轉成兩個模型各自的格式（兩者資料佈局完全不同）
python convert_to_stgcn_dataset.py     # → STGCN/data/taichung/{vel.csv, adj.npz, mask.npy}
python convert_to_stgat_dataset.py     # → STGAT/data/taichung/{train,val,test}.npz + adj_mx_dijsk.pkl
```

⚠️ **`convert_to_stgcn_dataset.py` 之後必須手動改 `STGCN/script/dataloader.py` 的
`n_vertex`（目前為 175）**，否則訓練會 shape mismatch。STGAT 不需要改碼——
`--num_of_vertices` 是 CLI 參數。

---

## Pipeline 結構

**設計原則：靜態幾何與動態時序分開**。原本兩者混在同一支程式，每次微調 σ／κ 都要
重讀 11.9 GB 的原始檔；拆開後調參數只要重跑 `build_network.py`（數秒）。

```
Map/graph_nodes_taichung.csv ─┐        （組員 A 以 Capture_Road_Node.py 產生）
Map/graph_edges_taichung.csv ─┤
                              │
fetch_tdx_section_metadata.py ┴─► build_network.py ──► section_to_edge_mapping.csv
    └► tdx_section_metadata.csv         【靜態幾何】    section_to_edges.csv  ★
                                                        taichung_dist.npy
                                                        taichung_section_index_full.csv
                                                        network_meta.json
                                                              │
fetch_tdx_section_live.py ────────────────────────────────────┤
    └► tdx_section_live_raw.jsonl ─► build_speed.py ◄─────────┘
              (11.9 GB)              【動態時序】
                                          │
                                          ├► taichung_vel.csv
                                          ├► taichung_mask.npy  ★
                                          ├► taichung_adj.npy
                                          ├► taichung_section_index.csv
                                          ├► taichung_timestamps.csv
                                          └► speed_meta.json
                                                    │
                            ┌───────────────────────┴───────────────────────┐
                convert_to_stgcn_dataset.py                    convert_to_stgat_dataset.py
                            │                                               │
                  STGCN/data/taichung/                          STGAT/data/taichung/
                {vel.csv, adj.npz, mask.npy}              {train,val,test}.npz（含 y_mask）
                                                             adj_mx_dijsk.pkl
```

★ = 相對原版新增的產物

---

## 各程式說明

| 程式 | 輸入 | 輸出 | 備註 |
|---|---|---|---|
| `fetch_tdx_section_metadata.py` | TDX API | `tdx_section_metadata.csv`（244 路段座標） | 金鑰讀自 `.env` |
| `fetch_tdx_section_live.py` | TDX API | `tdx_section_live_raw.jsonl`（約 11.9 GB） | 自動切 7 天批次、批次間等待避免超過呼叫頻率 |
| **`build_network.py`** | 路網 CSV + metadata | mapping ×2、距離矩陣、`network_meta.json` | 只讀 CSV，**數秒可重跑** |
| **`build_speed.py`** | `raw.jsonl` + 上一步產物 | 車速矩陣、**mask**、鄰接矩陣、索引 | **不需要** OSM 路網檔 |
| `convert_to_stgcn_dataset.py` | `build_speed.py` 產物 | `STGCN/data/taichung/` | `.npy` → scipy sparse `.npz` |
| **`convert_to_stgat_dataset.py`** | 同上 | `STGAT/data/taichung/` | 切成 DCRNN 視窗格式，額外加 **time-of-day** 特徵 |

#### 兩個轉檔程式的差異

| | STGCN | STGAT |
|---|---|---|
| 資料佈局 | `vel.csv`（時間 × 路段矩陣） | `{train,val,test}.npz`（已切好視窗） |
| 特徵數 | **1**（速度） | **2**（速度 + time-of-day） |
| 輸出時程 | **單步** → 15/30/60 分鐘各需訓練一個模型 | **12 步一次輸出** → 只需訓練一個 |
| 節點數設定 | 需手動改 `dataloader.py` 的 `n_vertex` | `--num_of_vertices` CLI 參數 |
| 切分 | `main.py` 內寫死 70/15/15 | 同樣用 70/15/15（刻意對齊，兩者 MAE 才可並列比較） |

> **time-of-day 的時區**：原始時間戳是 UTC，轉檔時已 **+8 轉成台灣時間**，讓日週期對齊
> 實際尖峰時段。STGCN 的 `vel.csv` 沒有這個特徵——這是兩份開源實作的既有差異，
> 解讀兩者 MAE 差距時需納入考量。

### 輔助工具（非主線）

| 程式 | 用途 |
|---|---|
| `diagnose_missing_data.py` | 缺值分布診斷（曾用來抓出 `$top` 設太小導致資料被截斷的 bug） |
| `diagnose_speed_anomalies.py` | 車速異常診斷（就是這支發現 68% 無效佔位值） |
| `build_adjacency_matrix.py` | 完整 7,489 節點路網的鄰接矩陣工具，給決策／SUMO 組員自行調參用 |

---

## 關鍵參數

```bash
# 鄰接矩陣密度（對齊 METR-LA 的 37.3%，避免全連通稀釋空間結構）
python build_network.py --target-density 0.373

# TDX 路段離路網多遠就剔除
python build_network.py --relevant-threshold-m 300

# 缺值超過此比例的路段剔除
python build_speed.py --max-missing 0.5

# 超過此速度視為感測器異常（轉為缺值，不是 clip）
python build_speed.py --max-speed 120
```

---

## ⚠️ 三個一定要知道的資料特性

### 1. 原始資料有 67% 是無效佔位值

`TravelSpeed=0` 且 `TravelTime=0` 表示 ETag 沒偵測到有效車輛配對，**不是「時速 0」的真實壅塞**。
照字面餵給模型會嚴重誤導訓練。`build_speed.py` 會自動轉成缺值。

```
原始 TravelSpeed
  │ 67.2% 無效佔位值            → 轉 NaN
  │ 0.21% 超過 120 km/h（感測器異常，實測 max 192）→ 轉 NaN
  ▼ 5 分鐘降採樣取平均（窗口內若仍有真實值即可救回）
  │ 仍有 30.5% 缺值
  ▼ 剔除缺值 >50% 的路段（212 → 175）
  │ 補值率降到 23.7%
  ▼ 三層補值：ffill → bfill → 該路段全域平均
taichung_vel.csv（表面上 0% 缺失）
```

### 2. 🔴 評估必須用 mask，否則成績虛低

最終仍有約 **23.7% 的儲存格是補值**（測試集為 25.5%；METR-LA 只有 7.13% 缺失，是其 3 倍）。
mask（`True` = 真實觀測）在**補值之前**取得，兩個模型都已備妥現成工具：

```bash
cd STGCN && python evaluate_masked.py --dataset taichung        # 單步，各時程分開跑
cd STGAT && python evaluate_masked_taichung.py                  # 一次輸出 12 個 horizon
```

兩者都會印出**遮罩前後並排 + persistence 基準 + PASS/FAIL 判定**。

> **兩套遮罩機制不可混用**：METR-LA 的缺失以 `0` 表示且保留原值，靠 loss 的
> `null_val=0.0` 遮罩；台中資料已被填補、**沒有 0 可遮罩**，必須用外部 mask。
> 若在台中資料上沿用 `null_val=0.0`，遮罩會完全失效（資料裡沒有 0）——
> 這正是 STGAT 當初 MAE 卡在 8.5 的那個 bug 的翻版。

**實測：遮罩的影響會隨 horizon 反轉**（STGAT，3 epochs）：

| horizon | 未遮罩 | 遮罩後 | 差異 |
|---|---|---|---|
| 1（5 min） | 2.7773 | 3.2948 | **+18.6%** |
| 6（30 min） | 3.7692 | 3.7226 | −1.2% |
| 12（60 min） | 4.1152 | 3.8638 | −6.1% |
| **平均** | 3.6988 | 3.7051 | **+0.2%** |

短時程的 ffill 值幾乎等於答案（`t+1` 的補值就是 `t` 的值），所以被灌水最嚴重；
時程拉長後補值反而變成雜訊。**兩者在平均值上互相抵銷（+0.2%）——只看平均會完全錯過這件事**，
必須逐 horizon 檢視。

### 2b. ⚠️ MAPE 的 float32 陷阱

`torch.Tensor()` 會把 float64 降成 float32，使真實的 `0 km/h` 在 z-score 逆轉換後變成
**約 1.27e-6** 而非精確 0。用 `y != 0` 過濾會完全失效，單一格就能讓 MAPE 爆成 **6356%**。
兩支評估工具都改用**物理門檻 1.0 km/h**，並同時輸出不受零值影響的 WMAPE 互相佐證。

### 3. 節點數是 175，不是 212

剔除高缺值路段後只剩 175 個。**必須同步修改**：

```python
# STGCN/script/dataloader.py
elif dataset_name == 'taichung':
    n_vertex = 175        # ← 隨 build_speed.py 的實際輸出調整
```

否則訓練會 shape mismatch。`convert_to_stgcn_dataset.py` 會偵測並提醒。

---

## 其他注意事項

| 項目 | 說明 |
|---|---|
| **時區** | `DataCollectTime` 為 **UTC**，台灣時間需 **+8 小時**。做尖峰時段分析前務必轉換 |
| **速度單位** | **km/h**（METR-LA 是 mph，不可混用） |
| **一對多對應** | 一個 TDX 路段橫跨數個路口，`section_to_edges.csv` 記錄它涵蓋的所有 OSM edge（平均 3.1 條）。決策層耦合時用這份，覆蓋率比一對一高 2.9 倍 |
| **金鑰** | 放在 `.env`（已被 `.gitignore` 忽略）。⚠️ 舊金鑰曾以明文進過 git 歷史，建議至 TDX 後台重新產生 |
| **`.jsonl` 不入版控** | 11.9 GB，需自行執行 `fetch_tdx_section_live.py` 產生 |

---

## 實測數據（2026-08-06）

| 項目 | 數值 |
|---|---|
| 原始資料 | 28,023,445 行 / 176 天 / 11.9 GB |
| TDX 路段 | 244 個 → **212** 通過 300 m 門檻 → **175** 通過缺值門檻 |
| 時間範圍 | 2026-01-27 ~ 07-21，5 分鐘間隔，**50,283 步**（> METR-LA 的 34,272） |
| 無效佔位值 | 67.2% |
| 降採樣後缺值 | 30.5% → 剔除後**補值率 23.7%** |
| 鄰接矩陣 | 175×175，σ=2,038 m、κ=3,025 m、密度 **40.0%** |
| 邊側覆蓋 | 524 條 OSM edge（完整路網 20,390 條的 2.6%；裁切後會大幅上升） |
| STGAT 資料集 | 50,260 樣本 × 12→12 步 × 175 路段 × 2 特徵（train/val/test = 35,182 / 7,539 / 7,539） |

### Smoke test 結果（3 epochs，遮罩後）

| 模型 | 15 min MAE | 30 min | 60 min | vs persistence |
|---|---|---|---|---|
| **STGAT** | **3.6135** | 3.7226 | 3.8638 | **−23.7%**（平均） |
| STGCN | 3.9187 | — | — | −10.7% |
| persistence | 4.3863 | 4.7836 | 5.2545 | — |

兩者都通過「勝過 persistence」與（STGAT）「MAE 隨 horizon 遞增」的判定。

> ⚠️ STGAT 目前領先 STGCN 約 7.8%，但**兩者輸入特徵不對等**（STGAT 多了 time-of-day）。
> 這也意味著 `integration/config.py` 的集成權重 `W_STGCN=0.7 / W_STGAT=0.3`
> ——當初因 STGAT 未收斂而設——**在台中資料上可能是反的**，待完整訓練後重新檢討。

---

相關文件：`paper_work/實驗設計.md`（實驗方法與評估協定）、`paper_work/原始資料欄位對應表.pdf`（各欄位定義）
