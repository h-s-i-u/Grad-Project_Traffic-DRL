# Third-party code and data

This repository is a university coursework project that vendors **two** upstream
implementations (`STGAT/`, `STGCN/`) and redistributes small derived extracts of public
datasets. Licensing is therefore **not uniform across directories**, and this file
records what applies where.

> Written in good faith by the authors after reading each upstream's terms. It is not
> legal advice. If you are an upstream author or data provider and something here is
> wrong, please open an issue and we will correct it or remove the material.

---

## Code

| Path | Origin | Terms |
|---|---|---|
| `integration/` · `fusion/` · `TDX_Data/` · `CWA/` · `Map/Capture_Road_Node.py` · `paper_work/*.md` · `README.md` | **this project** | **MIT** — see [`LICENSE`](LICENSE) |
| `STGCN/` | [hazdzz/STGCN](https://github.com/hazdzz/STGCN) | **LGPL-2.1** (`STGCN/LICENSE`). Our modifications to files inside this directory are covered by LGPL-2.1, not by our MIT licence |
| *(not vendored)* | [Lei-Kun/DRL-…-routing-problems](https://github.com/Lei-Kun/DRL-and-graph-neural-network-for-routing-problems) | **MIT**. The Residual E-GAT was **re-implemented** in `integration/policies.py` from the paper; no upstream code is redistributed here |
| `STGAT/` | [xyk0058/STGAT](https://github.com/xyk0058/STGAT) | ⚠️ **no licence file upstream** — see below |
| `STGAT/*_taichung.py`, `STGAT/transfer_taichung.py`, `STGCN/evaluate_masked.py`, `STGCN/run_infer_taichung.py` | **this project** | **MIT** (written by us; they sit in those directories only because they import the upstream packages) |

### ⚠️ On `STGAT/`

The upstream repository ships **no licence file**, which under default copyright means
all rights are reserved by its authors. It is included here so that this coursework is
reproducible end to end, with attribution, and with no claim of ownership. If you intend
to reuse the STGAT code for anything beyond reading this project, obtain it from the
upstream repository and clarify terms with its authors rather than relying on this copy.

### Which files we changed inside the upstream directories

Documented in `paper_work/實驗記錄_DRL決策模組.md` §17.2. In summary:

- `STGAT/model/stgat.py` — registered the attention heads as `nn.ModuleList`
- `STGAT/train.py` — removed two leftover debug `break` statements; added `--seed` and `--init-from`
- `STGCN/script/dataloader.py` — added a `taichung` branch

---

## Data

| Data | Source | Terms | What that means here |
|---|---|---|---|
| `Map/*.csv`, `Map/*.json`, `Map/*.npy` (road network, adjacency, arena) | **OpenStreetMap** via OSMnx | **ODbL 1.0** | These are a *derived database*. ODbL requires **attribution** and that any redistributed derived database stay under **ODbL**. Attribution: **© OpenStreetMap contributors** |
| `TDX_Data/` outputs, `Map/taichung_vel.csv`, `integration/taichung_pred_edges.csv` (section speeds) | **TDX — 交通部運輸資料流通服務平臺** (Ministry of Transportation and Communications, Taiwan) | per TDX's service terms; the platform's open datasets are generally released under **政府資料開放授權條款** (Open Government Data License, Taiwan), which permits redistribution and derivatives **with attribution** | Attribution: **資料來源：交通部運輸資料流通服務平臺（TDX）**. Raw downloads are **not** redistributed here — only derived, aggregated speed matrices |
| `integration/data/adj_mx_dijsk.pkl`, `STGAT/data/METR-LA/*`, `STGCN/data/metr-la/*` | **METR-LA**, from Li et al. (DCRNN) | released by its authors for research use | Redistributed only in the small derived form the pipeline needs; the full dataset is available from the DCRNN authors |
| `CWA/` rainfall | **中央氣象署 (Central Weather Administration)** Open Data | 政府資料開放授權條款 | Evaluated and then **deliberately excluded** from the system (README, "Climate features"); only the analysis scripts are published |

### Attribution, in the form the licences ask for

> Road network © **OpenStreetMap** contributors, available under the
> **Open Database Licence (ODbL) 1.0**.
> Traffic speed data 資料來源：**交通部運輸資料流通服務平臺（TDX）**.
> Rainfall data 資料來源：**中央氣象署**.

### Not redistributed

The raw TDX download (~11.9 GB), the model checkpoints, and the papers referenced in the
README are not in this repository. `TDX_Data/README.md` documents how to rebuild the data
chain from the source APIs; you will need your own TDX credentials in `TDX_Data/.env`.
