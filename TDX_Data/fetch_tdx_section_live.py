# -*- coding: utf-8 -*-
"""
fetch_tdx_section_live.py
──────────────────────────
下載 TDX「發布路段即時路況歷史資料」（Road/Traffic/Live/City/{City}）。

TDX 這個 API 一次查詢最多只能涵蓋 7 天，所以這支程式會：
    1. 把你指定的整段日期範圍（例如一個月），自動切成一塊一塊「不超過 7 天」的區間
    2. 依序抓取每一塊，中間會依你的會員等級的呼叫頻率限制自動等待，避免被擋
    3. 把所有區塊的資料合併、去除重複，輸出成同一份資料集

為了避免資料量太大時把電腦記憶體塞爆（一週的台中資料大約 200MB+），
這支程式採用「邊抓邊寫檔案」的方式，不會把所有資料一次載進記憶體。

使用方式：
    python fetch_tdx_section_live.py
"""

import os
import json
import time
import requests
import urllib3
from datetime import date, timedelta

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

# ── 你可以在這裡調整想抓的整體日期範圍 ──────────────────────────────
# 注意：TDX 歷史資料只提供到「昨天」為止（不含當天），日期範圍不要包含今天
START_DATE = date(2026, 1, 28)   # 開始日期（往回抓約 6 個月）
END_DATE = date(2026, 7, 27)     # 結束日期（含這一天，設為「昨天」）
# ──────────────────────────────────────────────────────────

MAX_CHUNK_DAYS = 7          # $top 已經調到 1000 萬（見下方 fetch_chunk），
                             # 244 個路段 × 7 天 × 每天約 1440 分鐘 ≈ 246 萬筆，遠低於這個上限，
                             # 不會再被截斷，改回 7 天一批可以減少批次數、加快整體抓取速度
SECONDS_BETWEEN_REQUESTS = 15  # 每次呼叫 API 之間等待的秒數，避免超過「基礎會員 5次/分鐘」的限制

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "tdx_section_live_raw.jsonl")


def get_access_token() -> str:
    payload = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }
    response = requests.post(TOKEN_URL, data=payload, timeout=30, verify=False)
    response.raise_for_status()
    return response.json()["access_token"]


def split_into_chunks(start: date, end: date, max_days: int):
    """把 [start, end] 這段日期範圍，切成一塊一塊最多 max_days 天的區間"""
    chunks = []
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=max_days - 1), end)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks


def fetch_chunk(token: str, start: date, end: date) -> str:
    """抓一個區間的資料，回傳原始文字內容（一行一筆 JSON）"""
    dates_param = f"{start.isoformat()}~{end.isoformat()}" if start != end else start.isoformat()

    url = f"https://tdx.transportdata.tw/api/historical/v2/Historical/Road/Traffic/Live/City/{CITY}"
    params = {"Dates": dates_param, "$top": 10000000, "$format": "JSONL"}
    headers = {
        "authorization": f"Bearer {token}",
        "Accept-Encoding": "identity",
        "Accept": "text/ndjson",
    }

    print(f"  正在下載 {dates_param} ...")
    response = requests.get(url, params=params, headers=headers, timeout=300, verify=False)
    response.raise_for_status()
    line_count = response.text.count("\n") + 1
    print(f"  → 下載完成，{len(response.content):,} 位元組，約 {line_count:,} 行"
          f"（如果這個數字剛好卡在整數關卡，例如很接近某個整數百萬，要懷疑是不是又被截斷了）")
    return response.text


def main():
    if CLIENT_ID.startswith("請填入"):
        print("❌ 請先把 CLIENT_ID / CLIENT_SECRET 填好再執行")
        return

    if START_DATE > END_DATE:
        print("❌ START_DATE 不能晚於 END_DATE，請檢查設定")
        return

    chunks = split_into_chunks(START_DATE, END_DATE, MAX_CHUNK_DAYS)
    print(f"日期範圍 {START_DATE} ~ {END_DATE}，共切成 {len(chunks)} 個批次（每批最多 {MAX_CHUNK_DAYS} 天）：")
    for i, (s, e) in enumerate(chunks, 1):
        print(f"  批次 {i}：{s} ~ {e}")

    print("\n正在取得 access token...")
    token = get_access_token()
    print("✅ 取得成功\n")

    # 用來去重：同一筆 (SectionID, DataCollectTime) 可能因為批次區間邊界重疊而重複出現，
    # 只在記憶體裡存這個「鍵」的集合（不存整筆資料），控制記憶體用量
    seen_keys = set()
    total_written = 0
    total_skipped_duplicate = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
            print(f"[批次 {i}/{len(chunks)}]")
            text = fetch_chunk(token, chunk_start, chunk_end)

            chunk_written = 0
            for line in text.strip().split("\n"):
                line = line.strip().lstrip("\ufeff")
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue

                sid = r.get("SectionID")
                t = r.get("DataCollectTime")
                key = (sid, t)
                if key in seen_keys:
                    total_skipped_duplicate += 1
                    continue
                seen_keys.add(key)

                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                chunk_written += 1

            total_written += chunk_written
            print(f"  → 這批新增 {chunk_written:,} 筆（累積目前共 {total_written:,} 筆）")

            # 批次之間等待，避免超過帳號的呼叫頻率限制
            if i < len(chunks):
                print(f"  等待 {SECONDS_BETWEEN_REQUESTS} 秒，避免超過呼叫頻率限制...\n")
                time.sleep(SECONDS_BETWEEN_REQUESTS)

    print(f"\n{'='*60}")
    print(f"✅ 全部完成，已輸出 {OUTPUT_PATH}")
    print(f"   總筆數：{total_written:,}")
    print(f"   批次邊界重疊、已自動跳過的重複筆數：{total_skipped_duplicate:,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()