# 開發記錄 — 互動 Demo（`demo/`）

> **這份只記 demo 這個應用程式。** 量到的、關於 agent 或模型本身的東西不在這裡：
> - ⑦ 對起始負載敏感、beam-8 從空網路的 served 平手 → `實驗記錄_DRL決策模組.md` **§13.28**
> - `capacity_scale` 隱含 154 秒觀測窗、ρ 是流量比 → `實驗設計.md` **§4.4**
>
> **相關**：`demo/README.md`（英文，程式介面與端點契約）。
> SUMO 那一側的交接文件（`交接_SUMO_20260831.md`）未隨版本庫發布，本文件已把用得到的論證直接寫入
>
> 【必讀】**這個頁面上的任何數字都不是結果。** 報告的正式數字一律由 `run_compare.py` 產生。
>
> **標記慣例**：`【必讀】` = 會出錯但不報錯的地方，報告中以**紅色**螢光筆標示；`【但書】` = 結論成立但有條件，**橘色**；`★`／`★★`／`★★★` = 重要性

---

## 1. 為什麼先做一個不需要 SUMO 的 demo

計畫書 §4.5 的視覺化層是「SUMO + Web dashboard」，而 SUMO 那半**組內沒有人跑過**——
`setRoute`、`getLastStepVehicleNumber`、gui 邊著色全部來自文件而非實測。
若把整個 demo 押在它上面，M4 的時程風險就綁在一個沒人驗證過的相依上。

而**最有說服力的畫面是「兩張並排的負載熱圖在同一個封路事件下分岔」，
而那完全不需要模擬器**——`metrics.edge_loads()` 早就產出「每條邊多少台車」，節點又都帶
lat/lon。所以 `demo/` 做成獨立可運作，SUMO 之後只是把「彩色的邊」升級成「看得到車在跑」。

**代價**：`FakeBackend` 裡的車輛推進是自己寫的，不是微觀模擬——沒有跟車、路口、號誌。
所以頁面上必須明寫「這不是結果」，README 和頁面底部都有。

---

## 2. 架構：一個介面，兩個實作

```
index.html  ──HTTP/JSON──►  app.py (FastAPI)  ──►  Backend（shared.py）
                                                    ├── FakeBackend   app.py，無模擬器
                                                    └── SumoBackend   controller.py，兩個 SUMO 實例
                                                          │
                                              shared.Funnel / plan_all / common_subset
                                                          │
                                                    reroute_service.Router
                                                          │
                                                    policies.py（未修改）
```

`Backend` 只有五個方法：`geometry` / `roads` / `state` / `close_road` / `reset`，定義在
`shared.py`。`--backend sumo` 會 `from controller import SumoBackend`。
**需求產生、逐策略批次指派、共同子集三件事也在 `shared.py`**——兩個 backend 呼叫同一個
函式，所以「SUMO 那格」和「無模擬器那格」拿到的是同一批 800 趟、同一個共同子集（§9）。

四個端點都是純 JSON over HTTP，所以換 React 前端不用動後端（`demo/README.md` 有對照表）。
顯示字串刻意放在 `index.html` 的 `LABEL` 表而不是 API 回傳裡。

---

## 3. 三個「不報錯、但讓對照組失效」的錯

這三個都是我犯的，而且**畫面上看起來一切正常**。記在這裡是因為它們是同一類：
**demo 的正確性不在於它會不會動，而在於兩格是不是同一個實驗。**

### 3.1 兩個世界都在跑羊群路線

`_spawn_initial` 與 respawn 都寫死 `pol.policy_prediction_greedy`。初始車隊跑完之後，
**右邊那格 100% 是 ④ 的路線**，⑦ 只在按下封路那一瞬間碰過當時在途的車。

截圖上 **已達 78,185 台、模擬時間 43,258 s**——初始車隊早在幾萬秒前就到站了。
於是 ⑦ 的 worst-ρ（1.200）比 ④（1.131）還差，那只是兩個羊群世界因歷史不同而產生的漂移。

**根因不是「呼叫錯函式」。** 就算改成呼叫 DRL 也一樣：eq.4 是靠**車輛間的耦合**運作的
（第 k 台車繞開前 k−1 台填滿的路），零星補車每次只有幾台，每台讀到的 `rho` 幾乎都是 0，
agent 等於在無資訊下決策。

**修法：改成回合制。** 一回合 = 800 台車，每個世界用自己的策略做一次**完整批次指派**，
負載在指派過程中自己疊出來——就是 `run_compare` 的算法。跑到 `--refresh`（預設 0.85）
抵達後兩個世界同步換下一回合。

### 3.2 兩格的車隊大小不一樣

改成回合制之後 ⑦ 確實變好了（worst-ρ 0.737 vs 1.200），**但那個數字有一半是假的**：

```
④  可指派 768/800
⑦  可指派 641/800     ← 少 127 台，被我直接從車隊裡丟掉
```

⑦ 在封路情境下 greedy 會走死路（§13.23 量過 143/766），而我把排不出路線的車直接跳過。
**車少 → 負載少 → ρ 低。** 右邊的 0.737 有一部分只是因為車比較少。

**修法：共同子集。** 只有**每個策略都排得出路線**的行程才進車隊，兩邊拿到完全相同的一批車。
**這正是報告的做法**——§13.23 的比較就是在「766 趟的共同子集」上做的。
被排除的趟數顯示在頂端。

⑦ 排不出的那些**仍然顯示出來**（`此策略可指派` 欄），因為那是它的弱點，
不該變成「車比較少所以 ρ 比較低」的優勢。

### 3.3 指派與改道共用同一個欄位

面板出現 `798/436`——分子來自**回合開始的指派**（該策略排出 798 趟，分母應為 800），
分母和秒數卻被**封路改道**的統計蓋掉（當下 436 台在途）。兩個毫不相干的事件。

**修法**：`plan_stats` 與 `last_reroute` 分開存、分成兩欄顯示，新回合開始時後者歸零。

---

## 4. 路型還原：改了三次才對

`demo/build_geometry.py`。arena 的一條邊是**合併鏈**，CSV 只留下兩個端點，
所以直接畫會是直線。全部 1,690 條的「真實路長 / 直線弦長」中位數是 1.000，
但差的那幾條剛好最長最顯眼——**十甲東路 4,279.5 m 的路畫成 3,391 m 的一條直線橫過市區**。

| 次序 | 做法 | 結果 | 為什麼錯 |
|---|---|---|---|
| 1 | 兩端點之間找**最短路** | 72.2%，**失敗的幾乎全是 >1 km 的邊** | 被合併的鏈**不是**最短路；長邊必然存在更短的替代路線 |
| 2 | 加上「中途節點必須不在子圖節點集」的約束 | **和第 1 次完全相同的統計數字** | 約束本身正確，但圖是斷的 |
| 3 | 修 `oneway` 展開 | **99.8%**，residual 中位數 0.0000% | 正確 |
| 3b | 對剩下 3 條改用「找長度對的」枚舉 | 99.9% | 子圖裁掉的邊在完整圖裡還在，最小化會挑到捷徑 |
| 3c | `--loose`：最後 2 條（同一條路兩向）用盡力而為的形狀 | 100%，並列出長度比 | |

**第 2 次的失敗訊息已經在告訴我答案了**：`no chain in Map_fined` 是「一條路都走不到」，
不是「找到但選錯」。那是**圖結構壞掉**的訊號，我當時應該直接去讀
`build_simplified_network.py`，而不是先改搜尋策略——**在錯的圖上做正確的搜尋，
結果一模一樣。**

### 根因是一個約定差異

```
Map_fined      一列一「路段」，方向在 oneway 欄（0% 的 (u,v) 有反向；29,906/43,711 是 no）
simplified_*   合併的輸出，已經一列一「方向」（84.3%）
arena_*        同上（52.5%，其餘是真的單行）
```

我用同一種方式讀三個檔案，於是 fine 圖裡**每條雙向路都只剩單向**，鏈一碰到反向路段就斷。
**這正好解釋為什麼失敗率隨長度上升**——跳數越多，撞到反向段的機率越高。

> **這和 `.meta.json` sidecar、`null_val=0.0`、`togo_refresh` 是同一類坑**：
> 兩邊都是合理的設計，混用時不會報錯，只會安靜地讓一半的東西失效。

### 為什麼 `arena_geometry.json` 要進版控

`build_geometry.py` 讀 `Map/simplified_*` 與 `Map/Map_fined/`，**兩者都不在 repo 裡**
（只有 arena 的兩份 CSV 是 tracked）。所以 clone 下來的人**跑不了這支腳本**。
169 KB、確定性輸出，在 `.gitignore` 的 `*.json` 後面加了一條例外。

---

## 5. 其他決定

| | |
|---|---|
| **底圖換掉** | CARTO 的 `basemaps.cartocdn.com` 現在要 API key，不給就在每張圖磚上蓋滿 `API KEY REQUIRED`。改用 OSM 官方圖磚（免金鑰），CSS filter 轉深色——filter 只作用在圖磚影像，不影響上層 canvas 的路網 |
| **Attribution** | 我一開始把 Leaflet 的 attribution control 關掉了。OSM（ODbL）要求標示來源，改成頁面底部一行常駐 credit（兩張地圖不必在兩個角落印同樣的字兩次） |
| **封路配色** | 洋紅虛線。負載色階是藍→綠→琥珀→橘→紅，**紅色已經是「最壅塞」**，而封路是另一種事實。且封閉邊**在負載掉到 0 之後仍保持該樣式**，否則剛關掉的路會褪成和「沒人走的路」一樣的灰 |
| **`GET /state` 不等鎖** | `close_road()` 會握著鎖整個重算過程（⑦ 約 6 秒）。原本 `/state` 會等鎖，**訪客一按下封路畫面就凍結 6 秒**——正好是它該顯示「重新規劃中」的時候 |
| **Canvas 而非 SVG** | 每秒重上色 1,690 條 polyline × 2 格，SVG renderer 撐不住。且只重繪飽和度真的變了的邊 |
| **`--vehicles` 預設 800** | `capacity_scale = 0.0429` 是對 800 車校準的。300 車時沒有一條邊跨過 `RHO_THRESHOLD = 0.85`，eq.4 的飽和項恆為 0，**兩格會長得一樣** |
| **`--episodes N`** | 跑完 N 回合停在最後一幀。攤位要 0（一直跑），但要把 served% 拿去和報告對讀就得用 1 |
| **離線攤位** | 頁面從 CDN 載 Leaflet，沒網路整頁死掉。`app.py` 啟動時檢查 `vendor/leaflet.js` 並印出 vendoring 指令。圖磚掉了不要緊——底圖空白但所有道路照常畫，polyline 來自我們自己的圖 |

---

## 6. 與報告數字的關係

【但書】**頁面上的數字不會等於報告裡的數字**，方向一致而已。三個原因：

1. **需求的亂數不同**：`make_demand` 用 numpy 的 RNG，`FakeBackend._episode` 用
   `random.Random(f"{seed}-episode-{n}")`。分布相同（S2 漏斗、800 台、4 個 hub），抽到的車不同。
2. **封路時機不同**：報告的 S3 用 `at` 把封閉錯開在派遣序列中；demo 的按鈕是 `at=0.0`，
   **下一次指派時整條路已經是關的**，等於 §13.17 那張表的「closed from the start = upper bound」。
3. **行程長度不同**：車輛放在路徑中途，剩餘行程約為完整長度的一半。

**能對讀的是比值，不是絕對值。** 實測一次：⑦ 的 worst-ρ 相對 ④ 是 **−23.9%**
（0.600 vs 0.788），報告的 S3 是 −27.5%——方向與量級一致。

---

## 7. 現況與待做

| 項目 | 狀態 |
|---|---|
| `demo/app.py`、`index.html`、`build_geometry.py`、`arena_geometry.json` | 完成 |
| `demo/shared.py` | 完成（09-04）：兩個 backend 共用的契約與需求產生器（§9） |
| `demo/controller.py`（`SumoBackend`） | **已寫（09-04），mock 自測通過；【必讀】尚未在真的 SUMO 上跑過**——這台機器沒裝。§9 列出實機自測要確認的四個假設 |
| React 版 | 待做。API 已框架無關，換前端只動 `index.html` 加 `app.py` 兩行 |
| 攤位前要做 | vendoring Leaflet；用 `--episodes 1` 對一次數字；決定 `--speed` 與 `--refresh`；情境圖層（實驗記錄 §18.5 Q，未定案） |

---

## 8. SUMO 的接點（09-02 補；此前只存在於對話中）

`reroute_service.py` 從頭到尾不 import SUMO，接縫只有 `demo/app.py` 的 `Backend` 五個方法。
`geometry()` / `roads()` 是一行轉發給 `Router`，**真正要寫的只有 `state()` / `close_road()` /
`reset()`**。

### 每個模擬步（背景執行緒）

```python
traci.simulationStep()
on_edge = {v: traci.vehicle.getRoadID(v) for v in traci.vehicle.getIDList()}
lw.observe(traci.simulation.getTime(), on_edge)      # lw = LoadWindow()
```

### `state()`（1 Hz 輪詢）

```python
st = router.network_state(lw.counts())   # worst_rho / gini_load / frac_saturated / edges
```

【必讀】**ATT 不從這裡拿。** `network_state` 沒有它，因為離線的 ATT 是 BPR 模型的推估；
live 的真值只有 SUMO 量得到（`tripinfo`，或 arrival − depart）。

### `close_road(road)`

```python
edges = router.road_edges(road)                    # "臺灣大道" -> 83 個 edge_id
closed |= set(edges)                               # 只記在 router 這邊；不要 setDisallowed（§9.5）

active = [(v, traci.vehicle.getRoadID(v), dest_osmid[v],
           traci.vehicle.getRoute(v)[traci.vehicle.getRouteIndex(v):])
          for v in traci.vehicle.getIDList()]
routes = router.reroute(active, closed=all_closed, policy=key)
for veh, route in routes.items():
    traci.vehicle.setRoute(veh, route)             # route[0] 必定是該車當前邊
```

**`routes` 裡沒有的車不要動**——「不在裡面」是「保持原路線」，不是「無路可走」。
診斷在 `router.last_stats`（`no_path` / `dead_end` / `max_hops` / `unaffected`）。

### 【必讀】兩個 pane 是兩個獨立的 SUMO 實例

一個模擬跑不出兩種策略——④ 和 ⑦ 產生的車流狀態不同。TraCI 支援多連線：

```python
traci.start(["sumo", "-c", "s2_4_herding.sumocfg"], label="herding")
traci.start(["sumo", "-c", "s2_7_drl.sumocfg"],     label="drl")
traci.switch("herding"); traci.simulationStep()
traci.switch("drl");     traci.simulationStep()
```

每個 world 各自一個 `LoadWindow` 與一組 `dest_osmid`，對應 `FakeBackend.worlds`。

> **退路**：只讓 ⑦ 跑 SUMO、④ 那格續用 `FakeBackend` 的 World。
> 但**那兩格就不再是同一種模擬**，畫面上必須標明。

### 每回合的需求產生器：已補（09-04）

`demo/` 是回合制的（見 §3.1），但產生每回合需求的 `FakeBackend._episode()` 原本是私有方法，
`SumoBackend` 拿不到。09-02 列的兩條路是「提升成 `Router.demand()`」或「重播同一份 `.rou.xml`」；
實際走的是第三條：**搬到 `demo/shared.py` 的 `Funnel`**，不動 `integration/`。
理由：回合是 demo 的概念（`run_compare` 有自己的 `make_demand`），放進 `Router` 等於讓
路由服務知道「攤位在跑第幾回合」；放在 demo 層兩個 backend 一樣共用，而 `integration/`
的檔案一行都不用改。

### 建網（一次）

```bash
cd integration
python export_sumo.py --drl checkpoints/taichung/drl_fusion_togo25.pt                       --policies 4,7 --vehicles 800 --window 600
cd sumo && sh build_net.sh          # 一行 netconvert -> taichung.net.xml
```

【必讀】**不要從 OSM 重抽路網。** edge id 是 `<from_osmid>_<to_osmid>`，`.net.xml` 與路線
由同一支程式產生所以恆等；重抽會讓 netconvert 自己生一套 id，那張對照表就是舊交接文件
掛了兩個月沒解決的問題。

---

## 9. `controller.py`：SUMO 即時版（09-04 寫成，實機待驗）

09-04 定案 demo 走**即時版**（兩格的車真的在 SUMO 裡跑），不是預錄短片。§8 的接縫照著寫成
`SumoBackend`；這裡記的是實作時做的決定、mock 自測看到什麼、以及**還沒被真的 SUMO 驗過的
四個假設**。

### 9.1 結構

| 層 | 內容 |
|---|---|
| `shared.py` | `Backend`、`PANES`、`Funnel`（每回合需求，字串種子 → 跨機器同一批）、`plan_all`（逐策略批次指派）、`common_subset`。`FakeBackend` 改為呼叫這些，diff 是刪 30 行 |
| `controller._Sim` | **所有 TraCI 呼叫都在這一個類別裡**：start / step / add / where / set_route / remove / close_edges。上面的邏輯不碰 traci |
| `controller._MockSim` | 同五個呼叫、沒有 SUMO：車每 10 步前進一條邊。`--selftest --mock` 跑的是它 |
| `controller.SumoWorld` | 一格：`LoadWindow`、存活集合、本回合集合、目的地表、封路與改道 |
| `controller.SumoBackend` | 兩個 `_Sim`（label = pane key）、一條執行緒交替 step、回合制與 `FakeBackend` 相同 |

### 9.2 三個刻意與 `FakeBackend` 不同的地方

| | `FakeBackend` | `SumoBackend` | 為什麼 |
|---|---|---|---|
| **車怎麼進場** | 直接放在路徑中途（同一個起始比例） | SUMO 放不了半路的車：**路徑從同一個比例處截斷，車從截斷點出發**。800 台在 t=0 一起插入會在第一條邊排隊，面板多一欄 `pending` | 兩格截在同一處，比較仍成立；排隊是 SUMO 的真實行為 |
| **無路可走的車** | 走到封閉邊時停下計 `stranded` | **封路當下就移除並計 `stranded`**（剩餘路徑碰到封閉邊、且 router 沒給新路線的那些） | 同一批車、只是提早計。不移除的話 SUMO 會在 `--time-to-teleport` 後把它**瞬移**過封閉路段——既不誠實也看不見。已設 `-1` 關掉瞬移 |
| **上一回合的殘車** | 換回合時整個車隊被換掉（15% 在途車消失） | **繼續開**，只是不再屬於本回合：算進 `driving`，不算進新回合的 `arrived`／`fleet` | 車在畫面上憑空消失比帳目複雜更糟 |

另外兩個只有這邊才有的欄位：`pending`（上表）與 `rejected`（SUMO 拒收的路線，只在
netconvert 掉了某條邊時發生，**應為 0**；自測有檢查，因為一格拒收、另一格接受會靜靜地讓兩格
不再是同一個實驗）。

### 9.3 TraCI 的成本

每步要一份 `{車: 所在邊}` 快照餵 `LoadWindow`。逐車 `getRoadID()` 是 800 次來回；改用
**訂閱**：`getDepartedIDList()` → 對新進場的車 `subscribe(VAR_ROAD_ID)` →
`getAllSubscriptionResults()` 一次拿全部。**每步三次呼叫，與車數無關**——這是 5 倍速跑得動
和跑不動的差別。改道時才逐車 `getRoute()`／`getRouteIndex()`，一次封路一次。

### 9.4 mock 自測看到的

`python controller.py --selftest --mock --vehicles 200 --speed 100 --drive 60`（herding vs oracle，
沒有 checkpoint 時自動改用這對）：

```
fleets {'herding': 199, 'oracle': 199}, dropped 1          ← 共同子集，兩格相同
herding  ep 1 t 150 driving 111 pending 0 arrived 88 rejected 0
oracle   ep 1 t 150 driving 101 pending 0 arrived 98 rejected 0
closing 臺灣大道
herding  active 111 routed 103 applied 103 setRoute-failed 0 stranded 2 no_path 2
oracle   active 101 routed  93 applied  93 setRoute-failed 0 stranded 4 no_path 4
reset → episode 1, closed = []
PASS: 0 problem(s)
```

第一版跑出 `driving 2 / pending 199 / arrived 0`——不是 bug，是 mock 的車一步一條邊、
一回合不到一秒就跑完，自測讀到的是剛派出的下一回合。但它暴露了真問題：**換回合時上一回合
的殘車抵達會算進新回合的 `arrived`**。修法是多一個 `current` 集合（§9.2 第三列）。
順手把 mock 改成每邊 10 步，讓回合長得足以在中途封路。

### 9.5 第一次實機（09-04）：四個假設驗掉三個，第四個換成另一個問題

`python controller.py --selftest --drl …`，真的 `sumo`、headless、734 台／格：

| 假設 | 結果 |
|---|---|
| 匯出的 1,690 條邊 id 全部活過 netconvert | **成立**：`rejected 0` |
| departed → subscribe → `getAllSubscriptionResults()` 每步拿得到全部在途車 | **成立**：driving 485 + arrived 249 = fleet 734，一台不差 |
| `setRoute()` 對在途車可用 | **成立**：herding 425 台重排，397 台接受 |
| `setRoute()` 對還沒進場的車可用 | **沒測到**：封路時 `pending 0`，沒有這種車 |

失敗的 28（herding）／41（drl）台，原因全是同一句：

```
No connection between edge '8349349076_13021803239' and edge '13021803239_5520441521'
```

**第一次的診斷是錯的，而且錯到動了 `integration/`。** 我看到「這兩條邊在我們的圖上相連、
SUMO 卻說沒 connection」，就認定是 netconvert 用幾何角度猜轉向時把合併鏈的直線弦當成迴轉、
沒建連線，於是在 `export_sumo.py` 加了逐車道明列的 `taichung.con.xml`（3,655 對邊）。
第二次實機（09-05）：`grep` 證實那對連線**確實在 `.net.xml` 裡**，SUMO **照樣拒絕同一批路線**。
連線從來不是問題。`export_sumo.py` 已 `git checkout` 還原。

正確的診斷來自兩個唯讀探針：

| 探針 | 結果 |
|---|---|
| 被拒的 23 對邊，對照 `edges_by_road(g, "臺灣大道")` | **「to」那條邊 23／23 全是封閉邊**，「from」9／23 是 |
| 離線呼叫 `Router.reroute()`（128 台車，起點在走廊上或走廊口，三種策略） | **回傳的路線沒有任何一條含封閉邊**（0／0／0） |

所以 SUMO 檢查的**不是我們送進去的路線**。`replaceRouteEdges` 會把車**已經開過的路段**接在
新路線前面（`getRoute()` 一直回傳含歷史的完整路線，就是這個緣故），然後 `hasValidRoute`
**從歷史的第一條邊開始驗**——而我們在改道前對封閉邊做了 `setDisallowed(["passenger"])`。
任何一台這回合曾經開過臺灣大道的車，不管現在人在哪，新路線都會因為歷史裡有一條「不准
passenger 走」的邊被整條打回票。20／32 台就是有那段歷史的車。

**修法（09-05）：封路不寫進 SUMO 的 permission。** 封閉只存在於 router（遮蔽邊）與 §9.2
的擱淺規則。訪客看到的一樣：沒有新車開進去、路上的車開離。sumo-gui 不會把路畫成封閉，
網頁會。

【必讀】**不能用的修法：讓 SUMO 接手改道。** 路線被拒時交給 SUMO 自己的 Dijkstra，等於右邊
那格有一部分路線悄悄變成 SUMO 的決策——跟 `實驗記錄` 記過的「U-turn fallback 用 ④ 的權重」
是同一類錯：對比被稀釋，畫面上看不出來。

**教訓**：「SUMO 說 A 和 B 沒 connection」和「我們的路線裡有 A→B」是兩件事，第一次我沒去
確認第二件就去改建網的程式。兩個探針各三十秒，該在提議改 `integration/` 之前跑，不是之後。

### 9.6 同一次跑出來的另外兩件

**「Vehicle 'e0_v16' is not known」22 筆／19 筆，正好等於 `stranded`。** 第二次實機加了
`gone`／`remove-failed` 計數後兩者都是 0——所以不是 SUMO 丟了車，是**我們自己 `remove()`
掉的車，訂閱還留著**，SUMO 下一步評估訂閱時對不存在的車各報一次錯（回應的是訂閱結果，
命令代碼看起來像 GET）。`remove()` 前先 `unsubscribe()`。

**一台車「Vehicle is on junction-internal edge leading elsewhere」。** 車在路口內部、已經
選定另一個出口，SUMO 不接受從別的出口走的新路線。這種車記為 `deferred`，`step()` 看到它回到
一般路段（road id 不以 `:` 開頭）就從它實際到的那條邊重新問一次 router。不處理的話它會照舊路線
開向封閉路段。

**第三次實機看的行**：`setRoute-failed 0`（`deferred` 可以不為 0，但事後 `still to retry 0`）、
`gone 0`、`remove-failed 0`、不再有 `not known`。
