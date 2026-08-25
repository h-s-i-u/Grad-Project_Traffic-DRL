# -*- coding: utf-8 -*-
"""
fetch_tdx_section_metadata.py
──────────────────────────
抓取台中「發布路段（Section）」的座標資料。

跟 tdx_section_live_raw.jsonl 不同：那份是「路段 + 時間 + 車速」的動態資料，
這支程式抓的是「路段本身在哪裡（起訖點經緯度）」的靜態資料，
兩者要用 SectionID 對起來，才能知道每一筆車速資料實際對應到地圖上的哪裡。

使用方式：
    python fetch_tdx_section_metadata.py
"""

import os
import json
import requests
import urllib3
import pandas as pd

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_dotenv(path: str) -> None:
    """讀取 .env（一行一組 KEY=VALUE）並寫進環境變數。

    自己實作而不用 python-dotenv，是為了不多裝套件；已存在的環境變數優先，
    所以在 CI／伺服器上可以直接用系統環境變數覆蓋，不必改檔案。
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


# 金鑰改由 .env 讀取，不寫死在程式碼裡（.env 已被 .gitignore 忽略，不會進版控）
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))
CLIENT_ID = os.environ.get("TDX_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("TDX_CLIENT_SECRET", "")

TOKEN_URL = "https://tdx.transportdata.tw/auth/realms/TDXConnect/protocol/openid-connect/token"
CITY = "Taichung"
DATES = "2026-07-20~2026-07-26"  # 跟 fetch_tdx_section_live.py 用同一個區間即可，只是要抓到路段清單

OUTPUT_CSV = os.path.join(SCRIPT_DIR, "tdx_section_metadata.csv")


def get_access_token() -> str:
    payload = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }
    response = requests.post(TOKEN_URL, data=payload, timeout=30, verify=False)
    response.raise_for_status()
    return response.json()["access_token"]


def main():
    if not CLIENT_ID or not CLIENT_SECRET:
        print("❌ 找不到 TDX 金鑰。請在 TDX_Data/.env 內設定 "
              "TDX_CLIENT_ID 與 TDX_CLIENT_SECRET（格式可參考 .env.example）")
        return

    print("正在取得 access token...")
    token = get_access_token()
    print("✅ 取得成功")

    url = f"https://tdx.transportdata.tw/api/historical/v2/Historical/Road/Traffic/Section/City/{CITY}"
    params = {"Dates": DATES, "$top": 100000, "$format": "JSONL"}
    headers = {
        "authorization": f"Bearer {token}",
        "Accept-Encoding": "identity",
        "Accept": "text/ndjson",
    }

    print("正在下載路段座標資料...")
    response = requests.get(url, params=params, headers=headers, timeout=60, verify=False)
    response.raise_for_status()

    rows = []
    seen_ids = set()
    for line in response.text.strip().split("\n"):
        line = line.strip().lstrip("\ufeff")
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue

        sid = r.get("SectionID")
        if sid is None or sid in seen_ids:
            continue  # 同一個路段在不同天可能重複出現，只留第一筆座標資料就夠了
        seen_ids.add(sid)

        start = r.get("SectionStart", {}) or {}
        end = r.get("SectionEnd", {}) or {}
        rows.append({
            "SectionID": sid,
            "RoadName": r.get("RoadName", ""),
            "SectionName": r.get("SectionName", ""),
            "StartLat": start.get("PositionLat"),
            "StartLon": start.get("PositionLon"),
            "EndLat": end.get("PositionLat"),
            "EndLon": end.get("PositionLon"),
        })

    df = pd.DataFrame(rows)
    # 中心點座標，之後 map matching 用得到
    df["CenterLat"] = (df["StartLat"] + df["EndLat"]) / 2
    df["CenterLon"] = (df["StartLon"] + df["EndLon"]) / 2

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ 已輸出 {OUTPUT_CSV}")
    print(f"共 {len(df)} 個不重複路段")


if __name__ == "__main__":
    main()
