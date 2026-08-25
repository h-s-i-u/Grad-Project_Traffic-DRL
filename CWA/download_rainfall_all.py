# -*- coding: utf-8 -*-
"""
CODiS 氣候觀測資料查詢服務 - 五個測站降雨量(及其他氣象要素)自動下載腳本
資料來源：https://codis.cwa.gov.tw/StationData

這支是「一次抓五個測站」的整合版，不用每個資料夾各跑一次。執行這支
腳本就會依序抓：北區(臺中站)、南屯、大肚、西屯、龍井，並且自動把
CSV 存進各自對應的資料夾（資料夾不存在會自動建立），資料夾名稱固定
放在腳本同一層目錄下。

各測站對應資訊（已實際測試過可以正常抓到資料）：
- 北區  -> 中央氣象署「臺中」有人測站，站碼 467490，type=cwb
          （臺中市氣象站清單中沒有站名直接叫「北區」的測站；查詢站址
          發現「臺中」站地址為「北區精武路295號」，位於臺中市北區，
          資料從日治時期就有，經與你確認後採用此站）
- 南屯區 -> 南屯自動氣象站，站碼 C0F9U0，type=auto_C0
- 大肚  -> 大肚農業站，站碼 C2F000，type=agr
          （原本的大肚自動氣象站 C0F000 已於 2023-06-29 撤站，
          現在的 C2F000 農業站是 2024-01-08 才啟用，涵蓋
          2026 年這段期間，所以改用這一個）
- 西屯區 -> 西屯自動氣象站，站碼 C0F9T0，type=auto_C0
- 龍井  -> 龍井自動氣象站，站碼 C0F9R0，type=auto_C0

如果之後要再加測站，直接在下面 STATIONS 這個列表多加一筆即可，
格式跟其他幾筆一樣。

需要先安裝 requests 套件：pip install requests

已知問題：SSLCertVerificationError / Missing Subject Key Identifier
------------------------------------------------------------------
如果執行時看到類似
    SSLError(SSLCertVerificationError(..., 'Missing Subject Key Identifier'))
這是台灣政府憑證管理中心(GRCA)簽發的憑證鏈缺少 Subject Key Identifier
這個欄位，新版 OpenSSL（3.2 以後，Python 3.12/3.13 在 Windows 上常會
帶到這個版本）驗證變嚴格後就會擋下來，很多 .gov.tw 網站都有這個通病，
不是你電腦或程式有問題。瀏覽器不會擋是因為瀏覽器的憑證檢查邏輯比較
寬鬆。這支腳本已經把這個網域的 SSL 驗證關掉來繞過這個已知問題（資料
是公開的氣象觀測資料，非敏感個資，關閉驗證風險很低）。
"""

import csv
import os
import time
import warnings
from datetime import datetime, timedelta

import requests
import urllib3

# 繞過 GRCA (台灣政府憑證) 憑證鏈缺少 Subject Key Identifier 造成的
# SSLCertVerificationError，詳見上方檔頭說明。
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# ------------------------- 設定區 -------------------------
START_DATE = datetime(2026, 1, 28)
END_DATE = datetime(2026, 7, 27)

# 腳本所在目錄，所有輸出資料夾都建在這裡面
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STATIONS = [
    {
        "name": "北區(臺中站)",
        "file_prefix": "北區",
        "stn_id": "467490",
        "stn_type": "cwb",
        "folder": "北區2026.01.28 to 2026.07.27降雨資料",
    },
    {
        "name": "南屯",
        "file_prefix": "南屯",
        "stn_id": "C0F9U0",
        "stn_type": "auto_C0",
        "folder": "南屯區2026.01.28 to 2026.07.27降雨資料",
    },
    {
        "name": "大肚",
        "file_prefix": "大肚",
        "stn_id": "C2F000",
        "stn_type": "agr",
        "folder": "大肚2026.01.28 to 2026.07.27降雨資料",
    },
    {
        "name": "西屯",
        "file_prefix": "西屯",
        "stn_id": "C0F9T0",
        "stn_type": "auto_C0",
        "folder": "西屯區2026.01.28 to 2026.07.27降雨資料",
    },
    {
        "name": "龍井",
        "file_prefix": "龍井",
        "stn_id": "C0F9R0",
        "stn_type": "auto_C0",
        "folder": "龍井2026.01.28 to 2026.07.27降雨資料",
    },
]

API_URL = "https://codis.cwa.gov.tw/api/station?"
REQUEST_DELAY_SEC = 0.4   # 每次呼叫間隔，避免對伺服器造成太大負擔
MAX_RETRIES = 3
# -----------------------------------------------------------

HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://codis.cwa.gov.tw/StationData",
    "Origin": "https://codis.cwa.gov.tw",
}


def daterange(start_date, end_date):
    days = (end_date - start_date).days
    for i in range(days + 1):
        yield start_date + timedelta(days=i)


def fetch_one_day(session, stn_id, stn_type, day):
    """呼叫 API 抓單一天的逐時資料，回傳該天 dts (list of hourly dict)。"""
    date_str = day.strftime("%Y-%m-%d")
    payload = {
        "date": f"{date_str}T00:00:00+08:00",
        "type": "report_date",
        "stn_ID": stn_id,
        "stn_type": stn_type,
        "more": "",
        "start": f"{date_str}T00:00:00",
        "end": f"{date_str}T23:59:59",
        "item": "",
    }

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.post(
                API_URL, data=payload, headers=HEADERS, timeout=20, verify=False
            )
            resp.raise_for_status()
            j = resp.json()
            if j.get("code") != 200:
                raise RuntimeError(f"API 回傳錯誤: {j}")
            data = j.get("data") or []
            if not data:
                return []
            return data[0].get("dts", [])
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"    ⚠️  {date_str} 第 {attempt} 次嘗試失敗: {e}")
            time.sleep(1.5 * attempt)
    print(f"    ❌ {date_str} 抓取失敗，跳過（錯誤: {last_err}）")
    return None


def g(d, *keys):
    """安全地取巢狀 dict 的值，取不到回傳 None。"""
    for k in keys:
        if d is None:
            return None
        d = d.get(k)
    return d


def process_station(session, station):
    name = station["name"]
    stn_id = station["stn_id"]
    stn_type = station["stn_type"]
    out_dir = os.path.join(BASE_DIR, station["folder"])
    os.makedirs(out_dir, exist_ok=True)

    hourly_csv = os.path.join(
        out_dir, f"{station['file_prefix']}_降雨量_逐時資料_20260128_20260727.csv"
    )
    daily_csv = os.path.join(
        out_dir, f"{station['file_prefix']}_每日降雨量總計_20260128_20260727.csv"
    )

    print(f"\n===== 開始處理【{name}】(站碼 {stn_id}) =====")

    rows = []
    daily_totals = []
    failed_days = []

    for day in daterange(START_DATE, END_DATE):
        date_str = day.strftime("%Y-%m-%d")
        print(f"  正在處理 {date_str} ...", end=" ")

        dts = fetch_one_day(session, stn_id, stn_type, day)

        if dts is None:
            failed_days.append(date_str)
            time.sleep(REQUEST_DELAY_SEC)
            continue

        if not dts:
            print("（當天無資料）")
            time.sleep(REQUEST_DELAY_SEC)
            continue

        day_precp_total = 0.0
        day_has_data = False

        for hour_data in dts:
            data_time = hour_data.get("DataTime", "")
            hour_str = data_time[11:13] if len(data_time) >= 13 else ""

            raw_precp = g(hour_data, "Precipitation", "Accumulation")
            # CWA 資料裡，累積雨量若小於 0（例如 -999.x）代表缺測/估計值
            # 的特殊註記碼，不是真的負雨量，網頁上會顯示成 "&" 這種符號
            # 而不是數字。這裡比照網頁的做法，這類值當成缺測處理（CSV
            # 留空、且不計入當天總雨量）。
            if isinstance(raw_precp, (int, float)) and raw_precp >= 0:
                precp = raw_precp
            else:
                precp = ""

            rows.append({
                "Date": date_str,
                "ObsTime": hour_str,
                "StnPres": g(hour_data, "StationPressure", "Instantaneous"),
                "Temperature": g(hour_data, "AirTemperature", "Instantaneous"),
                "RH": g(hour_data, "RelativeHumidity", "Instantaneous"),
                "WS": g(hour_data, "WindSpeed", "Mean"),
                "WD": g(hour_data, "WindDirection", "Mean"),
                "WSGust": g(hour_data, "PeakGust", "Maximum"),
                "WDGust": g(hour_data, "PeakGust", "Direction"),
                "Precp": precp,
            })

            if isinstance(precp, (int, float)):
                day_precp_total += precp
                day_has_data = True

        daily_totals.append({
            "Date": date_str,
            "Precp_Total_mm": round(day_precp_total, 1) if day_has_data else "",
        })

        print(f"✅ 完成（{len(dts)} 筆）")
        time.sleep(REQUEST_DELAY_SEC)

    if rows:
        with open(hourly_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "Date", "ObsTime", "StnPres", "Temperature", "RH",
                "WS", "WD", "WSGust", "WDGust", "Precp",
            ])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  📄 逐時資料已存成: {hourly_csv}（共 {len(rows)} 筆）")

    if daily_totals:
        with open(daily_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["Date", "Precp_Total_mm"])
            writer.writeheader()
            writer.writerows(daily_totals)
        print(f"  📄 每日降雨量總計已存成: {daily_csv}")

    if failed_days:
        print(f"  ⚠️  以下 {len(failed_days)} 天抓取失敗，可重新執行腳本"
              f"（已抓到的資料不受影響）：{', '.join(failed_days)}")


def main():
    print(f"🌐 準備抓取 {len(STATIONS)} 個測站 "
          f"{START_DATE:%Y-%m-%d} ~ {END_DATE:%Y-%m-%d} 的逐時資料")

    session = requests.Session()

    for station in STATIONS:
        process_station(session, station)

    print("\n🎉 全部測站處理完畢！")


if __name__ == "__main__":
    main()
