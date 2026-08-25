import os
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from tkintermapview import TkinterMapView
import osmnx as ox
import pandas as pd

# 🌟 1. 修復工作目錄路徑的問題 (正確使用 __file__)
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
os.chdir(current_dir)

# 用來記錄點擊的座標組
click_coords = []

# 🌟 進度顯示相關的全域狀態
is_processing = False      # 目前是否正在背景執行處理
process_start_time = None  # 本次處理開始的時間戳記
is_awaiting_download = False  # 是否正卡在「等待 OSM 伺服器回應」這個不確定時長的階段

# osmnx 2.0 之後，graph_from_bbox 的 bbox 參數格式改為 (west, south, east, north)，
# 舊版 (1.x) 則是 (north, south, east, west)。這裡自動判斷版本以避免抓錯範圍。
_OSMNX_MAJOR_VERSION = int(ox.__version__.split('.')[0])

def get_graph_from_bbox(north, south, east, west, network_type='drive'):
    """依照目前安裝的 osmnx 版本，用正確的 bbox 順序呼叫 graph_from_bbox"""
    if _OSMNX_MAJOR_VERSION >= 2:
        # osmnx >= 2.0: bbox = (west, south, east, north)
        return ox.graph_from_bbox(bbox=(west, south, east, north), network_type=network_type)
    else:
        # osmnx < 2.0（已棄用寫法，但仍相容）
        return ox.graph_from_bbox(north=north, south=south, east=east, west=west, network_type=network_type)

def clean_lanes(val):
    """獨立的車道數清洗功能，處理 OSM 的髒資料"""
    # 🌟 修正：list 必須先被攔截處理，因為 pd.isna() 對 list 會回傳一個
    # 布林「陣列」而不是單一 True/False，導致後面的 if 判斷式報錯：
    # "The truth value of an array with more than one element is ambiguous."
    if isinstance(val, list):
        nums = [int(x) for x in val if str(x).isdigit()]
        return max(nums) if nums else 1
    if pd.isna(val):
        return 1
    if isinstance(val, str) and ',' in val:
        nums = [int(x) for x in val.split(',') if x.isdigit()]
        return max(nums) if nums else 1
    try:
        return int(float(val))
    except:
        return 1

def calculate_hcm_capacity(row):
    """根據 HCM 理論精準估算路段容量 (capacity)"""
    N = row['lanes'] # N: 車道數
    
    C0 = 1900.0      # C0: 理想飽和流率 (市區幹道)
    fw = 0.95        # fw: 車道寬度與淨空因子 
    f_HV = 0.90      # f_HV: 重車因子 (台灣大道公車客運多)
    fp = 0.93        # fp: 行人與機車干擾因子 (商圈、車站干擾大)
    fc = 0.90        # fc: 地區特性因子 (路邊臨停、號誌密集)
    
    # 公式: C = C0 * N * fw * f_HV * fp * fc
    capacity = C0 * N * fw * f_HV * fp * fc
    return round(capacity)

def set_progress_ui(percent, text, color="orange"):
    """實際更新進度條與狀態文字（只能在主執行緒呼叫）"""
    # 已經進入有明確百分比的階段，改成「確定進度」模式
    if progress_bar['mode'] != 'determinate':
        progress_bar.stop()
        progress_bar.config(mode='determinate')
    progress_bar['value'] = percent
    percent_label.config(text=f"{percent}%")
    status_label.config(text=text, fg=color)


def update_progress(percent, text, color="orange"):
    """給背景執行緒呼叫的安全版本，透過 root.after 把更新丟回主執行緒"""
    root.after(0, lambda: set_progress_ui(percent, text, color))


def tick_elapsed():
    """每秒執行一次：更新已耗時秒數，並在下載階段等太久時給出提示，
    這樣使用者可以確認程式仍在運作、沒有卡死。"""
    global is_processing, is_awaiting_download

    if not is_processing:
        return  # 已結束，停止計時迴圈

    elapsed = int(time.time() - process_start_time)
    elapsed_label.config(text=f"⏱ 已執行 {elapsed} 秒")

    # 只有在「等待 OSM 伺服器回應」這個無法預估時長的階段，才需要用時間長短來提醒使用者
    if is_awaiting_download:
        if elapsed >= 90:
            status_label.config(
                text=f"⏳ 仍在等待 OSM 伺服器回應（已 {elapsed} 秒）...\n"
                     f"若持續過久，可能是範圍過大或網路不穩，可考慮取消後縮小範圍重試",
                fg="#CC6600"
            )
        elif elapsed >= 30:
            status_label.config(
                text=f"⏳ 正在下載並處理路網資料，已等待 {elapsed} 秒，請稍候...",
                fg="orange"
            )

    root.after(1000, tick_elapsed)


def process_osm_data(north, south, east, west):
    """核心處理邏輯：接收點選的經緯度並處理路網資料"""
    global is_awaiting_download
    try:
        print(f"\n🚀 開始處理範圍：北:{north}, 南:{south}, 東:{east}, 西:{west}")
        print("正在透過 OSMnx 獲取並建立路網圖，這可能需要幾十秒，請稍候...")

        # 下載階段耗時不定（要看範圍大小、OSM 伺服器狀況），無法給出準確百分比，
        # 所以先切到「跑動進度條」模式，讓使用者知道程式還活著、不是卡死
        is_awaiting_download = True
        root.after(0, lambda: (progress_bar.config(mode='indeterminate'), progress_bar.start(12)))
        update_progress(0, "⏳ 正在連線 OSM 伺服器下載路網資料，請稍候...")

        # 透過 OSMnx 自動獲取範圍內的「可行駛道路 (drive)」
        graph = get_graph_from_bbox(north, south, east, west, network_type='drive')

        # 下載完成，之後每一步都是本地運算，可以給出明確百分比了
        is_awaiting_download = False
        update_progress(35, "✅ 路網下載完成，正在轉換資料格式...")

        nodes, edges = ox.graph_to_gdfs(graph)

        # 重新命名欄位以對齊 STGCN 輸入需求
        edges = edges.reset_index()
        edges = edges.rename(columns={
            'u': 'from_node',
            'v': 'to_node',
            'length': 'length_m'
        })

        update_progress(50, "🔧 正在清洗車道數（lanes）資料...")
        # 處理車道數 (lanes)
        if 'lanes' in edges.columns:
            edges['lanes'] = edges['lanes'].apply(clean_lanes)
        else:
            edges['lanes'] = 1  

        update_progress(60, "🔧 正在清洗速限（maxspeed）資料...")
        # 處理自由車速 (free_flow_speed_kmh)
        def clean_maxspeed(val):
            try:
                if isinstance(val, list):
                    return float(val[0]) # 如果有多個速限，取第一個
                return float(val)
            except:
                return 50.0  # 台中市區預設速限 50 km/h

        if 'maxspeed' in edges.columns:
            edges['free_flow_speed_kmh'] = edges['maxspeed'].apply(clean_maxspeed)
        else:
            edges['free_flow_speed_kmh'] = 50.0

        update_progress(70, "📊 正在依 HCM 理論計算路段容量（capacity）...")
        # 透過 HCM 理論函數估算路段容量 (capacity)
        edges['capacity'] = edges.apply(calculate_hcm_capacity, axis=1)

        # 萃取並輸出最終您定義的欄位
        final_edges = edges[['from_node', 'to_node', 'length_m', 'free_flow_speed_kmh', 'lanes', 'capacity']]
        
        update_progress(85, "🗺️ 正在整理節點（nodes）資料...")
        # 🌟 處理節點資料 (nodes)
        nodes = nodes.reset_index()
        # 保留節點 ID、經度 (x)、緯度 (y)
        final_nodes = nodes[['osmid', 'y', 'x']].copy()
        final_nodes = final_nodes.rename(columns={
            'osmid': 'node_id',
            'y': 'latitude',
            'x': 'longitude'
        })

        update_progress(95, "💾 正在匯出 CSV 檔案...")
        # 輸出邊和節點資料
        edges_filename = "graph_edges_taichung.csv"
        nodes_filename = "graph_nodes_taichung.csv"
        final_edges.to_csv(edges_filename, index=False)
        final_nodes.to_csv(nodes_filename, index=False)
        
        # 取得完整路徑
        edges_fullpath = os.path.abspath(edges_filename)
        nodes_fullpath = os.path.abspath(nodes_filename)
        
        success_msg = f"✅ 成功匯出靜態地圖資料：\n\n📄 路段資料：{edges_filename}\n   ({len(final_edges)} 條路段)\n\n📄 節點資料：{nodes_filename}\n   ({len(final_nodes)} 個節點)\n\n💾 儲存位置：\n{os.path.dirname(edges_fullpath)}"
        print(success_msg)
        print(f"\n完整路徑：\n  {edges_fullpath}\n  {nodes_fullpath}")

        update_progress(100, "✅ 匯出完成！", color="green")

        # 這個函式是在背景執行緒中執行的，Tkinter 元件只能在主執行緒操作，
        # 所以用 root.after(0, ...) 把 UI 更新丟回主執行緒
        root.after(0, lambda: on_extraction_success(success_msg))

    except Exception as e:
        error_msg = f"處理時發生錯誤: {e}"
        print(error_msg)
        root.after(0, lambda: on_extraction_error(error_msg))


def on_extraction_success(success_msg):
    """在主執行緒中執行：處理成功後的 UI 更新"""
    global is_processing, is_awaiting_download
    is_processing = False
    is_awaiting_download = False
    status_label.config(text="✅ 匯出完成，可重新選取其他範圍", fg="green")
    btn_run.config(state=tk.NORMAL)
    btn_reset.config(state=tk.NORMAL)
    messagebox.showinfo("處理成功", success_msg)


def on_extraction_error(error_msg):
    """在主執行緒中執行：處理失敗後的 UI 更新"""
    global is_processing, is_awaiting_download
    is_processing = False
    is_awaiting_download = False
    progress_bar.stop()
    progress_bar.config(mode='determinate')
    progress_bar['value'] = 0
    percent_label.config(text="0%")
    status_label.config(text="❌ 發生錯誤，請重新選取範圍後再試一次", fg="red")
    btn_run.config(state=tk.NORMAL)
    btn_reset.config(state=tk.NORMAL)
    messagebox.showerror("錯誤", error_msg)

def reset_selection():
    """重新選取功能：清除所有已選取的座標和標記"""
    global click_coords
    click_coords = []
    
    # 清除地圖上所有標記和路徑
    map_widget.delete_all_marker()
    map_widget.delete_all_path()
    
    # 重置 UI 狀態
    status_label.config(text="請點擊地圖【左上角】設定起點", fg="blue")
    coords_label.config(text="")
    btn_run.config(state=tk.DISABLED)
    btn_reset.config(state=tk.DISABLED)

    # 重置進度條與計時顯示
    progress_bar.stop()
    progress_bar.config(mode='determinate')
    progress_bar['value'] = 0
    percent_label.config(text="0%")
    elapsed_label.config(text="")
    
    print("🔄 已清除選取範圍，請重新選擇")

def run_extraction():
    """按鈕觸發的執行函式，將座標轉交給核心邏輯（在背景執行緒執行，避免凍結 UI）"""
    global is_processing, process_start_time
    if len(click_coords) == 2:
        lat1, lon1 = click_coords[0]
        lat2, lon2 = click_coords[1]
        
        north = max(lat1, lat2)
        south = min(lat1, lat2)
        east = max(lon1, lon2)
        west = min(lon1, lon2)
        
        # 下載/處理期間先鎖住按鈕，避免使用者重複點擊觸發多個下載
        btn_run.config(state=tk.DISABLED)
        btn_reset.config(state=tk.DISABLED)
        status_label.config(text="⏳ 正在下載並處理路網資料，請稍候...", fg="orange")

        # 🌟 初始化進度顯示狀態，並啟動每秒更新一次的計時迴圈
        is_processing = True
        process_start_time = time.time()
        progress_bar['value'] = 0
        percent_label.config(text="0%")
        elapsed_label.config(text="⏱ 已執行 0 秒")
        tick_elapsed()
        
        thread = threading.Thread(
            target=process_osm_data,
            args=(north, south, east, west),
            daemon=True
        )
        thread.start()

def add_marker_event(coords):
    """地圖滑鼠點擊事件：紀錄座標並畫出紅色選取框"""
    global click_coords
    lat, lon = coords
    
    if len(click_coords) < 2:
        click_coords.append((lat, lon))
        map_widget.set_marker(lat, lon, text=f"點 {len(click_coords)}")
        
        if len(click_coords) == 1:
            status_label.config(text="請點擊地圖【右下角】設定終點", fg="blue")
            # 顯示第一個點的經緯度
            coords_label.config(text=f"📍 起點座標：緯度 {lat:.5f}　經度 {lon:.5f}")
            
        elif len(click_coords) == 2:
            status_label.config(text="✔ 已框選範圍！", fg="green")
            btn_run.config(state=tk.NORMAL) # 啟用執行按鈕
            btn_reset.config(state=tk.NORMAL) # 啟用重置按鈕
            
            # ✨ 新功能：畫出紅色的範圍框
            lat1, lon1 = click_coords[0]
            lat2, lon2 = click_coords[1]
            north = max(lat1, lat2)
            south = min(lat1, lat2)
            east = max(lon1, lon2)
            west = min(lon1, lon2)

            # 顯示框選範圍的北/南/東/西四個邊界經緯度
            coords_label.config(
                text=f"北:{north:.5f}　南:{south:.5f}　東:{east:.5f}　西:{west:.5f}"
            )
            
            # 依序定義矩形的四個頂點 (左上 -> 右上 -> 右下 -> 左下 -> 左上) 來畫出封閉框線
            box_path = [
                (north, west),
                (north, east),
                (south, east),
                (south, west),
                (north, west)
            ]
            
            # 在地圖上繪製紅色的框線 (寬度 3)
            map_widget.set_path(box_path, color="red", width=3)

# --- 建立 Tkinter UI 介面 ---
root = tk.Tk()
root.title("OSM 節點框選工具 (DRL 交通專用)")
root.geometry("900x700")

status_label = tk.Label(root, text="請點擊地圖【左上角】設定起點", font=("Arial", 12, "bold"), fg="blue", height=2)
status_label.pack(side=tk.TOP, fill=tk.X)

# 🌟 座標顯示標籤：顯示目前框選範圍的北/南/東/西經緯度
coords_label = tk.Label(root, text="", font=("Consolas", 10), fg="#333333")
coords_label.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))

# 🌟 進度顯示區塊：進度條 + 百分比 + 已耗時秒數，讓使用者確認執行進度、以及有沒有卡住
progress_frame = tk.Frame(root)
progress_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 5))

progress_bar = ttk.Progressbar(progress_frame, mode='determinate', maximum=100, value=0)
progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

percent_label = tk.Label(progress_frame, text="0%", font=("Arial", 10, "bold"), width=5, anchor="w")
percent_label.pack(side=tk.LEFT)

elapsed_label = tk.Label(progress_frame, text="", font=("Arial", 10), fg="#666666", width=16, anchor="e")
elapsed_label.pack(side=tk.LEFT)

# 建立按鈕容器框架
button_frame = tk.Frame(root)
button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

# 重新選取按鈕
btn_reset = tk.Button(button_frame, text="🔄 重新選取", font=("Arial", 11, "bold"), bg="#FF9800", fg="white", state=tk.DISABLED, height=2, command=reset_selection)
btn_reset.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

# 執行按鈕
btn_run = tk.Button(button_frame, text="🚀 開始下載範圍圖資並導出 CSV", font=("Arial", 11, "bold"), bg="#4CAF50", fg="white", state=tk.DISABLED, height=2, command=run_extraction)
btn_run.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

map_widget = TkinterMapView(root, width=900, height=600, corner_radius=0)
map_widget.pack(fill=tk.BOTH, expand=True)

# 預設中心為台中東海大學附近
map_widget.set_position(24.179, 120.590) 
map_widget.set_zoom(15)
map_widget.add_left_click_map_command(add_marker_event)

root.mainloop()