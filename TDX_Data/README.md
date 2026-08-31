# TDX_Data — 台中路況資料處理 pipeline

把 TDX 的路段車速資料與 OSM 路網結合，產生 STGCN／STGAT 可直接訓練的資料集，
以及決策層用的路由場域。

---

## 快速開始

```bash
cd TDX_Data

# 0) 金鑰：複製 .env.example → .env，填入 TDX 的 Client Id / Secret
cp .env.example .env

# 1) 抓資料（metadata 幾秒；live 會跑很久，約 11.9 GB）
python fetch_tdx_section_metadata.py
python fetch_tdx_section_live.py

# 2) 簡化路網（39,920 → 9,904 節點，自帶旅行時間不變性證明）
python build_simplified_network.py

# 3) 建幾何（靜態，只讀 CSV，數秒可重跑）
python build_network.py

# 4) 建車速矩陣（動態時序，讀 11.9 GB，較久）
python build_speed.py

# 5) 轉成兩個模型各自的格式（兩者資料佈局完全不同）
python convert_to_stgcn_dataset.py     # → STGCN/data/taichung/{vel.csv, adj.npz, mask.npy}
python convert_to_stgat_dataset.py     # → STGAT/data/taichung/{train,val,test}.npz + adj_mx_dijsk.pkl

# 6) 決策層的路由場域（9,904 → 840 節點）
python build_arena.py
```

⚠️ **步驟 5 之後必須手動確認 `STGCN/script/dataloader.py` 的 `n_vertex`（目前為 202）**，
否則訓練會 shape mismatch。STGAT 不需要改碼——`--num_of_vertices` 是 CLI 參數。

---

## Pipeline 結構

**兩個設計原則**：

1. **靜態幾何與動態時序分開**。原本兩者混在同一支程式，每次微調 σ／κ 都要重讀 11.9 GB
   的原始檔；拆開後調參數只要重跑 `build_network.py`（數秒）。
2. **預測與決策用的路網不是同一張**。預測的圖建在 **TDX 路段**上，地圖只決定
   「哪些路段通過 map-matching」——**地圖越大對預測越有利**；決策則需要一個小而集中的
   場域，否則 RL 的 credit assignment 會被路徑長度拖垮。詳見下方「三層路網」。

```
Map/Map_fined/*.csv ────► build_simplified_network.py ──► simplified_{nodes,edges}_taichung.csv
（組員以 Capture_Road_Node.py 產生）        【路網簡化】      simplified_meta.json
  39,920 節點 / 43,711 段                                          │
                                                                   │
fetch_tdx_section_metadata.py ─────────► build_network.py ◄────────┘
    └► tdx_section_metadata.csv            【靜態幾何】
                                                │
                     ┌──────────────────────────┼───────────────────────┐
                     ▼                          ▼                       ▼
        section_to_edge_mapping.csv    taichung_dist.npy        network_meta.json
        section_to_edges.csv  ★        （244×244 沿路網距離）    （σ／κ／覆蓋率）
                                                │
fetch_tdx_section_live.py ──────────────────────┤
    └► tdx_section_live_raw.jsonl ─► build_speed.py ◄─┘
              (11.9 GB)              【動態時序】
                                          │
                                          ├► taichung_vel.csv      （50,283 × 202）
                                          ├► taichung_mask.npy  ★  （True = 真實觀測）
                                          ├► taichung_adj.npy
                                          ├► taichung_section_index.csv
                                          ├► taichung_timestamps.csv
                                          └► speed_meta.json
                                                    │
                            ┌───────────────────────┼───────────────────────┐
                convert_to_stgcn_dataset.py   convert_to_stgat_dataset.py   │
                            │                       │                       │
                  STGCN/data/taichung/     STGAT/data/taichung/     build_arena.py ★
                {vel.csv, adj.npz, mask}  {train,val,test}.npz     【決策層場域】
                                          adj_mx_dijsk.pkl                  │
                                                                  arena_{nodes,edges}_taichung.csv
                                                                  arena_meta.json
```

★ = 相對原版新增的產物

### 三層路網，各司其職

| 層 | 規模 | 用途 |
|---|---:|---|
| ① 完整圖 `Map_fined` | 39,920 節點 / 43,711 段（1,748.8 km） | 原始交付、Demo 繪圖 |
| ② 簡化圖 | **9,904 / 27,022 有向邊** | 軌道 A 的 map-matching 與路網距離 |
| ③ 實驗場域 arena | **840 / 1,690** | 軌道 B 的訓練與評估 |

**節點 id 全程沿用原始 OSM id**，所以場域上算出的路徑可直接畫在完整地圖上。

🔴 **地圖的框選範圍會永久決定 `n_vertex`**：地圖小 → 通過 map-matching 的路段少 →
**模型的節點數在訓練當下就鎖死**，之後換多大的地圖都補不回來。
（歷史：舊地圖東界短 3.1 km 且缺北屯區，244 個路段只有 212 個匹配得上；
`Map_fined` v3 擴大框選後 **244/244 全數匹配**，存活路段由 175 升到 202。）

---

## 各程式說明

| 程式 | 輸入 | 輸出 | 備註 |
|---|---|---|---|
| `fetch_tdx_section_metadata.py` | TDX API | `tdx_section_metadata.csv`（244 路段座標） | 金鑰讀自 `.env` |
| `fetch_tdx_section_live.py` | TDX API | `tdx_section_live_raw.jsonl`（約 11.9 GB） | 自動切 7 天批次、批次間等待避免超過呼叫頻率 |
| **`build_simplified_network.py`** | `Map_fined` CSV | 簡化後的有向邊 CSV | 39,920 → 9,904 節點；**內建旅行時間不變性證明** |
| **`build_network.py`** | 簡化圖 + metadata | mapping ×2、距離矩陣、`network_meta.json` | 只讀 CSV，**數秒可重跑** |
| **`build_speed.py`** | `raw.jsonl` + 上一步產物 | 車速矩陣、**mask**、鄰接矩陣、索引 | **不需要** OSM 路網檔 |
| `convert_to_stgcn_dataset.py` | `build_speed.py` 產物 | `STGCN/data/taichung/` | `.npy` → scipy sparse `.npz` |
| **`convert_to_stgat_dataset.py`** | 同上 | `STGAT/data/taichung/` | 切成 DCRNN 視窗格式，額外加 **time-of-day** 特徵 |
| **`build_arena.py`** | 簡化圖 + mapping | `arena_*_taichung.csv`、`arena_meta.json` | 決策層場域；**取子圖後會再簡化一次** |

### `build_simplified_network.py` 的三個設計要點

`Map_fined` 每 ~44 m 一個節點，其中 **73.2% 是「一進一出」的中途節點**——
逐節點決策的 policy 在那裡沒有選擇可做，卻要付出一次決策、一次編碼、一次 PPO transition。

1. **先依 `oneway` 展開再判斷中途節點**——節點算不算「中途」取決於方向。
   盲目雙向化會憑空生出約三分之一的逆向邊，讓分流看起來比實際容易。
2. **速限用「長度加權調和平均」** `S = Σℓᵢ / Σ(ℓᵢ/vᵢ)`——**唯一能讓 `t0` 完全不變的算法**。
   算術平均或取最小會讓全網每條邊的 `t0` 悄悄偏移。
3. **輸出一列一條有向邊**——`taichung_loader.py` 不讀 `oneway` 欄，
   沿用「一列一路段」會讓它把每條路都當單行道。

**內建正確性證明**：比對原圖與簡化圖之間 300 組 OD 的自由流最短路徑時間，
實測**最大相對誤差 1.51e-15**（機器精度）、0 組不可達；超過 `1e-6` 直接 `exit 1`。
另設「不可達比例 > 2% 直接失敗」的連通性硬檢查。

> 🔴 **為何需要第二道檢查**：第一版的孤兒環偵測把**被正確合併掉的鏈中間節點**誤判成孤兒
> 又加回去，等於撤銷簡化（300 組 OD 有 248 組不可達），**卻仍印出 `PASS`**。
> **檢查只跑在沒壞的部分上，會給出假的通過訊號。**

### `build_arena.py` 的兩條保護規則

取子圖之後**必須再簡化一次**——切邊會把原本的路口變回中途節點（實測 47.4%），
導致 **54.5% 的決策點只有一個候選**（動作被迫，policy 梯度恆為零）。

但**兩類東西不能合併**，共同原因是**外部有程式用 id 直接定址它們**：

| 保護對象 | 為什麼 |
|---|---|
| 含**種子邊**的鏈 | `section_to_edges.csv` 用 `(u, v)` 掛預測，合併會**靜默切斷 TDX 訊號** |
| **Demo 端點**（東海大學、台中車站） | Demo 用 node id 指定起訖 |

> ⚠️ 第二條是漏掉後才補的——首次執行時**台中車站剛好是個中途節點，被合併掉了**。

### 兩個轉檔程式的差異

| | STGCN | STGAT |
|---|---|---|
| 資料佈局 | `vel.csv`（時間 × 路段矩陣） | `{train,val,test}.npz`（已切好視窗） |
| 特徵數 | **1**（速度） | **2**（速度 + time-of-day） |
| 輸出時程 | **單步** → 15/30/60 分鐘各需訓練一個模型 | **12 步一次輸出** → 只需訓練一個 |
| 節點數設定 | 需手動改 `dataloader.py` 的 `n_vertex` | `--num_of_vertices` CLI 參數 |
| 切分 | `main.py` 內寫死 70/15/15 | 同樣用 70/15/15（刻意對齊，兩者 MAE 才可並列比較） |

> **time-of-day 的時區**：原始時間戳是 UTC，轉檔時已 **+8 轉成台灣時間**，讓日週期對齊
> 實際尖峰時段。
>
> 🔴 **STGCN 的 `vel.csv` 沒有這個特徵，而那不只是「輸入不對等」**：計畫書指派 STGCN Path
> 萃取「規則性（尖峰時段、星期週期）」，而週期性正是 time-of-day 定義的東西——
> 實測 STGCN 在最 routine 的分桶與平日尖峰**輸給一個連模型都不是的歷史平均**。
> `fusion/` 的雙路模型已補上這個通道（`--stgcn-tod`）。詳見 `實驗記錄` §13.20 ③。

### 輔助工具（非主線）

| 程式 | 用途 |
|---|---|
| `diagnose_missing_data.py` | 缺值分布診斷（曾用來抓出 `$top` 設太小導致資料被截斷的 bug） |
| `diagnose_speed_anomalies.py` | 車速異常診斷（就是這支發現 68% 無效佔位值） |
| `build_adjacency_matrix.py` | 鄰接矩陣工具，給決策／SUMO 組員自行調參用 |
| `map_section_to_network.py`、`build_taichung_stgcn_dataset.py` | **已被 `build_network.py` / `build_speed.py` 取代**，保留供對照 |

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

# 場域：主幹道種子的車道數門檻；--no-resimplify 可關掉取子圖後的再簡化
python build_arena.py --min-lanes 3
```

---

## ⚠️ 三個一定要知道的資料特性

### 1. 原始資料有 68% 是無效佔位值

`TravelSpeed=0` 且 `TravelTime=0` 表示 ETag 沒偵測到有效車輛配對，**不是「時速 0」的真實壅塞**。
照字面餵給模型會嚴重誤導訓練。`build_speed.py` 會自動轉成缺值。

```
原始 TravelSpeed（28,023,445 行）
  │ 67.94% 無效佔位值                                    → 轉 NaN
  │  0.06% 超過 120 km/h（感測器異常，實測 max 192）      → 轉 NaN
  ▼ 5 分鐘降採樣取平均（窗口內若仍有真實值即可救回）
  │ 仍有 31.2% 缺值
  ▼ 剔除缺值 >50% 的路段（244 → 202，剔除 42 個）
  │ 補值率降到 24.6%
  ▼ 三層補值：ffill → bfill → 該路段全域平均
taichung_vel.csv（50,283 × 202，表面上 0% 缺失）
```

> **異常值為何轉缺值而非 clip**：clip 會留下一個「捏造的觀測」並被 mask 標成 `True`，
> 等於拿假資料去評分；轉缺值則交由補值機制處理，且 mask 正確標成 `False`。

### 2. 🔴 評估必須用 mask，否則成績虛低

最終仍有約 **24.6% 的儲存格是補值**（METR-LA 只有 7.13% 缺失，是其 3.5 倍）。
mask（`True` = 真實觀測）在**補值之前**取得，兩個模型都已備妥現成工具：

```bash
cd STGCN && python evaluate_masked.py --dataset taichung --n-pred 3   # 單步，各時程分開跑
cd STGAT && python evaluate_masked_taichung.py                        # 一次輸出 12 個 horizon
```

兩者都會印出**遮罩前後並排 + persistence 基準 + PASS/FAIL 判定**。

> **兩套遮罩機制不可混用**：METR-LA 的缺失以 `0` 表示且保留原值，靠 loss 的
> `null_val=0.0` 遮罩；台中資料已被填補、**沒有 0 可遮罩**，必須用外部 mask。
> 若在台中資料上沿用 `null_val=0.0`，遮罩會完全失效（資料裡沒有 0）——
> 這正是 STGAT 當初 MAE 卡在 8.5 的那個 bug 的翻版。

**實測：遮罩的影響會隨 horizon 反轉**（202 節點版正式訓練）：

| horizon | STGCN 遮罩差 | STGAT 遮罩差 |
|---|---:|---:|
| 15 min | +0.7% | **+9.3%** |
| 30 min | −5.4% | −0.9% |
| 60 min | −9.7% | −7.6% |

短時程的 ffill 值幾乎等於答案（`t+1` 的補值就是 `t` 的值），所以被灌水最嚴重；
時程拉長後補值反而變成雜訊。**兩端會在 12 步平均上互相抵銷（STGAT 僅 +0.5%）——
只看平均會完全錯過這件事**，必須逐 horizon 檢視。**兩個模型 × 兩份資料集皆同一型態。**

#### 2b. ⚠️ MAPE 的 float32 陷阱

`torch.Tensor()` 會把 float64 降成 float32，使真實的 `0 km/h` 在 z-score 逆轉換後變成
**約 1.27e-6** 而非精確 0。用 `y != 0` 過濾會完全失效，單一格就能讓 MAPE 爆成 **6356%**。
兩支評估工具都改用**物理門檻 1.0 km/h**，並同時輸出不受零值影響的 WMAPE 互相佐證。

### 3. 節點數是 202，不是 244

剔除 42 個高缺值路段後只剩 202 個。**必須同步修改**：

```python
# STGCN/script/dataloader.py
elif dataset_name == 'taichung':
    n_vertex = 202        # ← 隨 build_speed.py 的實際輸出調整
```

否則訓練會 shape mismatch。`convert_to_stgcn_dataset.py` 會偵測並提醒。
**每次重跑資料鏈都要複查這個值**——它寫死在程式裡，而鄰接矩陣從磁碟載入。

---

## 其他注意事項

| 項目 | 說明 |
|---|---|
| **時區** | `DataCollectTime` 為 **UTC**，台灣時間需 **+8 小時**。做尖峰／星期分析前務必轉換 |
| **速度單位** | **km/h**（METR-LA 是 mph，不可混用） |
| **一對多對應** | 一個 TDX 路段橫跨數個路口，`section_to_edges.csv` 記錄它涵蓋的所有 OSM edge。244 個路段展開後共 **715 條唯一邊**，其中屬於 202 個存活路段的有 **601 條** |
| **金鑰** | 放在 `.env`（已被 `.gitignore` 忽略）。⚠️ 舊金鑰曾以明文進過 git 歷史，**建議至 TDX 後台重新產生** |
| **`.jsonl` 不入版控** | 11.9 GB，需自行執行 `fetch_tdx_section_live.py` 產生 |
| **Windows 的 int32 陷阱** | `astype(int)` 在 Windows 是 int32，而 **84.7% 的 OSM id 超過 2³¹−1**。一律用 `to_numpy(dtype="int64")` |

---

## 實測數據（2026-08-13 資料鏈重建後）

| 項目 | 數值 |
|---|---|
| 原始資料 | 28,023,445 行 / 176 天 / 11.9 GB |
| 時間範圍 | 2026-01-27 ~ 07-21，5 分鐘間隔，**50,283 步**（> METR-LA 的 34,272） |
| TDX 路段 | 244 個 → **244 通過 300 m 門檻（100%）** → **202 通過缺值門檻** |
| 路網簡化 | 39,920 / 43,711 → **9,904 節點 / 27,022 有向邊**（合併 27,058 個中途節點） |
| 無效佔位值 | 67.94% |
| 降採樣後缺值 | 31.2% → 剔除 42 個路段後**補值率 24.6%** |
| 鄰接矩陣 | 202×202，σ=**2,283.4 m**、κ=**3,341.8 m**、密度 **37.56%**（目標 37.3%） |
| 種子邊 | **715** 條（244 路段）／ **601** 條（202 存活路段） |
| 邊側覆蓋（完整圖） | 601 / 27,022 = **2.2%** |
| **邊側覆蓋（場域）** | **416 / 1,690 = 24.6%（按邊）**、**14.1%（按長度）** |
| 場域 | **840 節點 / 1,690 有向邊**，平均 28.7 跳，141 / 202 個路段有代表 |
| STGAT 資料集 | 12→12 步 × 202 路段 × 2 特徵（train/val/test = **35,182 / 7,539 / 7,539**） |

### 正式訓練結果（202 節點版，masked，僅計真實觀測 73.8%）

| 模型 | 15 min | 30 min | 60 min |
|---|---:|---:|---:|
| **Fusion**（雙路閘控，一個模型三時程） | **3.3786** | **3.4799** | **3.5579** |
| STGAT | 3.3802 | 3.5127 | 3.6276 |
| STGCN | 3.5560 | 3.7535 | 3.9549 |
| HA（歷史平均） | 4.0486 | 4.0489 | 4.0496 |
| persistence | 4.2872 | 4.6744 | 5.1281 |

🔴 **報告寫法：相對 persistence 與相對 HA 兩組都要列。** 相對 persistence 的領先隨時程
**擴大**（−21% → −29%），相對 HA 卻**縮小**（−16.5% → −10.4%）——因為 HA 不隨時程退化。
只引用前者會把「persistence 變爛」誤記成「模型變好」。

> **集成權重**：`W_STGCN/W_STGAT = 0.2/0.8`，val 上挑選、**test 上確認**
> （test 最佳為 0.15，兩者差 0.07%，未過擬合）。但固定權重集成只勝過單獨 STGAT 0.29%，
> 兩模型誤差相關 0.908——這正是改做 Gated Fusion 的實證動機。

---

## 🔴 一個必須知道的下游事實：**預測對路由沒有可測量的影響**

決策層量到的結論，會影響這份 pipeline 的優先順序：

```
場域按邊數的覆蓋率 24.6%，但 按長度 只有 14.1%
而 Dijkstra 比的是 時間 = 長度 / 速度 —— 長度才是有效權重
```

無預測的邊全部乘上同一個 fallback 常數，**均勻縮放不改變路徑排序**；有預測的邊只佔
路徑成本的 14.1%，擾動約 2%，小於競爭路線之間的典型差距。
**實測 86% 的路徑用 `t0` 與用 `tpred` 完全相同**，且四個獨立佐證都指向同一結論
（詳見 `實驗記錄` §13.13、§13.24）。

**這不是 pipeline 的 bug，是資料可得性的限制**——TDX 的 202 個路段對應約 600 條邊，
而路網要強連通且提供替代道路至少需要 1,690 條。**縮小場域已實測無效**
（節點縮到 58%、按邊覆蓋率翻倍，結果零變化），因為按長度的覆蓋率不會跟著動。

> **對本 pipeline 的意涵**：再花力氣提高「按邊數」的覆蓋率沒有幫助。
> 有幫助的是**更多 TDX 路段**（更多真實觀測的道路長度），那取決於資料源而非處理程式。

---

相關文件：`paper_work/實驗設計.md`（實驗方法與評估協定）、
`paper_work/實驗記錄_DRL決策模組.md`（完整歷程與所有實測數字）、
`paper_work/原始資料欄位對應表.pdf`（各欄位定義）
