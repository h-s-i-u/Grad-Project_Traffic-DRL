# -*- coding: utf-8 -*-
"""
convert_to_stgcn_dataset.py
──────────────────────────
把 build_speed.py 的產物轉成 hazdzz/STGCN 直接可讀的資料夾格式：

    STGCN/data/taichung/
        ├── vel.csv     （直接複製，檔名對齊 METR-LA）
        ├── adj.npz     （.npy 密集矩陣 → scipy sparse 壓縮格式）
        ├── mask.npy    （缺值遮罩，供「只在真實觀測上計分」的評估使用）
        └── section_index.csv / timestamps.csv（對照用，方便解讀模型輸出）

為什麼 adj 要轉檔：STGCN 內部是用 scipy.sparse.load_npz() 讀鄰接矩陣，
跟 numpy 的 np.load() 是不同格式，單純改副檔名不會生效。

為什麼要帶上 mask：台中資料在降採樣後仍有約 30% 是補值（METR-LA 只有 7.13% 缺失）。
若把補值一起算進 MAE，成績會虛低、也無法與 METR-LA 或論文並列比較。

使用方式：
    python convert_to_stgcn_dataset.py
    python convert_to_stgcn_dataset.py --stgcn-dir /path/to/STGCN
"""

import argparse
import os
import shutil

import numpy as np
import scipy.sparse as sp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
MAP_DIR = os.path.join(ROOT_DIR, "Map")

DATASET_NAME = "taichung"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stgcn-dir", default=os.path.join(ROOT_DIR, "STGCN"),
                    help="STGCN 專案路徑（預設為本 repo 內的 STGCN/）")
    ap.add_argument("--source-dir", default=MAP_DIR,
                    help="build_speed.py 的輸出資料夾（預設 Map/）")
    args = ap.parse_args()

    src_vel = os.path.join(args.source_dir, "taichung_vel.csv")
    src_adj = os.path.join(args.source_dir, "taichung_adj.npy")
    for p, name in [(src_vel, "taichung_vel.csv"), (src_adj, "taichung_adj.npy")]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"找不到 {name}：{p}\n請先執行 build_speed.py。")

    target_dir = os.path.join(args.stgcn_dir, "data", DATASET_NAME)
    os.makedirs(target_dir, exist_ok=True)
    print(f"目標資料夾：{target_dir}")

    # 1. vel.csv：STGCN 的 dataloader 是寫死這個檔名去讀的
    dst_vel = os.path.join(target_dir, "vel.csv")
    shutil.copy(src_vel, dst_vel)
    print(f"✅ vel.csv → {dst_vel}")

    # 2. adj：numpy 密集 .npy → scipy sparse .npz
    print("正在轉換鄰接矩陣格式（.npy → scipy sparse .npz）...")
    adj_dense = np.load(src_adj)
    dst_adj = os.path.join(target_dir, "adj.npz")
    sp.save_npz(dst_adj, sp.csr_matrix(adj_dense))
    if np.allclose(sp.load_npz(dst_adj).toarray(), adj_dense):
        print(f"✅ adj.npz → {dst_adj}（已驗證與原始 .npy 數值一致）")
    else:
        print("⚠️ 轉換後內容與原始檔案有差異，請檢查")

    # 3. 一併帶過去的輔助檔（有就複製，沒有不影響訓練）
    for name in ("taichung_mask.npy", "taichung_section_index.csv", "taichung_timestamps.csv"):
        src = os.path.join(args.source_dir, name)
        if os.path.isfile(src):
            dst = os.path.join(target_dir, name.replace("taichung_", ""))
            shutil.copy(src, dst)
            print(f"✅ {os.path.basename(dst)} → {dst}")
        elif name == "taichung_mask.npy":
            print("⚠️ 找不到 taichung_mask.npy —— 將無法做「只在真實觀測上計分」的評估")

    n_vertex = adj_dense.shape[0]
    print(f"\n📐 鄰接矩陣：{adj_dense.shape}，非零比例 {float((adj_dense > 0).mean()):.1%}")
    print("\n完成！接下來：")
    print(f"   cd {args.stgcn_dir}")
    print(f"   python main.py --dataset {DATASET_NAME} --epochs 3     # 先小 epoch 驗證流程")
    if n_vertex != 212:
        print(f"\n⚠️ 節點數為 {n_vertex}（不是 212）——因為 build_speed.py 剔除了缺值過高的路段。")
        print(f"   請把 STGCN/script/dataloader.py 裡 taichung 的 n_vertex 改成 {n_vertex}，"
              f"否則會 shape mismatch。")


if __name__ == "__main__":
    main()
