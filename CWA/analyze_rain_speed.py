# -*- coding: utf-8 -*-
"""
analyze_rain_speed.py
─────────────────────
Does rainfall measurably slow the Taichung road network down?

Answer this BEFORE wiring rain into the prediction models. Adding a feature channel
means retraining STGCN and STGAT at three horizons each, re-searching the ensemble
weight, and re-running the whole inference chain. Rain covers only ~6% of the hours in
this period, so it is entirely possible the answer is "no measurable effect" -- and
that is a perfectly reportable finding for 計劃書 §4.2, not a failure.

Design (three confounders, three controls):

  1. Time of day dominates speed variation -- rush hour swamps any weather effect.
     -> Every observation is compared against a baseline computed from DRY hours in
        the SAME (section, hour-of-day, weekday/weekend) cell. The reported number is
        a relative deviation from that baseline, never a raw speed difference.

  2. Sections differ enormously in their typical speed.
     -> The section is part of the stratum, so a wet-hour observation is only ever
        compared against the same road's own dry-hour behaviour.

  3. All 175 sections share one weather reading each hour, so cell-level counts
     wildly overstate the sample size.
     -> Deviations are averaged to ONE number per timestep first; the error bars come
        from the spread across timesteps, not across cells.

The headline test is dose-response, not a rain/no-rain split: a binary difference can
come from almost anything, but speed falling monotonically as rain intensifies is hard
to explain any other way.

Time alignment (easy to get wrong, and it fails silently):
    TDX  DataCollectTime is UTC.
    CWA  ObsTime 1..24 is LOCAL (UTC+8) and labels the hour ENDING at that time --
         ObsTime 1 covers 00:01-01:00, ObsTime 24 covers 23:01-24:00.
    So a UTC timestamp maps to local = utc + 8h, then (local - 1 second) gives the
    date and hour whose bucket it belongs to. An 8-hour error here would line rush
    hour up with the middle of the night and nothing would raise an error.

Missing data:
    - Speed: only cells with taichung_mask.npy == True are used (23.7% of the matrix
      is ffill-imputed and would mostly reproduce the previous step).
    - Rain: CODiS leaves missing/estimated hours BLANK on purpose so they are not read
      as 0 mm. Those hours are dropped, never zero-filled.

Usage:
    cd CWA
    python analyze_rain_speed.py
    python analyze_rain_speed.py --rain-agg max     # "it rained somewhere in the city"
    python analyze_rain_speed.py --min-dry 30
"""

import argparse
import glob
import os
import re
from datetime import timedelta, timezone

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP_DIR = os.path.join(ROOT_DIR, "Map")
CWA_DIR = os.path.dirname(os.path.abspath(__file__))

TZ_OFFSET_HOURS = 8          # TDX is UTC; CODiS ObsTime is local
# Rain intensity buckets, mm per hour. The open-ended top bucket is where an effect
# should show up first if there is one at all.
BUCKETS = [(0.0, 0.0, "0（無雨）"), (0.0, 1.0, "0–1"), (1.0, 5.0, "1–5"),
           (5.0, 10.0, "5–10"), (10.0, np.inf, "> 10")]


def load_stations(cwa_dir):
    """{station name: DataFrame[Date, ObsTime, Precp, ...]}, with ObsTime repaired.

    The ObsTime column in these files is WRONG for the last hour of every day.
    download_rainfall_all.py:208 takes the label by slicing the hour field out of the
    API's DataTime (`data_time[11:13]`), and hour 24 -- the 23:01-24:00 window, which
    the query bounds as `...T23:59:59` -- yields "23". So every day carries two rows
    labelled 23 and none labelled 24, in all five stations, 181 days each.

    The rows themselves are correct and in chronological order; only the label is
    wrong (2026-01-28 has 23:00 at 15.0 degrees and 24:00 at 14.8). So the hour is
    rebuilt from each row's position within its day, and the number of rows that
    disagree with the file is reported -- it should equal the number of days, and
    anything larger means this assumption does not hold.

    Worth fixing at source too, otherwise every consumer of this data has to know.
    """
    out, fixes = {}, {}
    for path in sorted(glob.glob(os.path.join(cwa_dir, "*", "*逐時*.csv"))):
        folder = os.path.basename(os.path.dirname(path))
        name = re.split(r"\d{4}", folder)[0] or folder
        df = pd.read_csv(path)
        missing = {"Date", "ObsTime", "Precp"} - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}")
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

        per_day = df.groupby("Date", sort=False).size()
        odd = per_day[per_day != 24]
        if len(odd):
            raise ValueError(
                f"{name}: {len(odd)} day(s) do not have exactly 24 rows "
                f"(e.g. {odd.index[0]} has {odd.iloc[0]}). The hour cannot be rebuilt "
                f"from row position -- inspect the file before going further.")
        from_file = df["ObsTime"].astype("int64")
        # int64 on purpose: the lookup key comes from a pandas .dt.hour, which is
        # int32 on Windows, and a MultiIndex level dtype mismatch makes reindex()
        # match NOTHING and return an all-NaN column -- no error, just no analysis.
        df["ObsTime"] = (df.groupby("Date", sort=False).cumcount() + 1).astype("int64")
        n_fixed = int((df["ObsTime"] != from_file).sum())
        if n_fixed > len(per_day):
            raise ValueError(
                f"{name}: {n_fixed} rows disagree with the file's ObsTime but there "
                f"are only {len(per_day)} days. Expected exactly one bad label per "
                f"day (the 24th hour); the file is wrong in some other way too.")
        fixes[name] = n_fixed
        out[name] = df
    if not out:
        raise FileNotFoundError(f"no '*逐時*.csv' under {cwa_dir}/*/")
    return out, fixes


def station_matrix(stations):
    """(date, ObsTime) x station wide table of Precp, for agreement + aggregation."""
    wide = None
    for name, df in stations.items():
        s = df.set_index(["Date", "ObsTime"])["Precp"].rename(name)
        if s.index.has_duplicates:
            raise ValueError(f"station {name} has duplicate (Date, ObsTime) rows; "
                             f"reindex would raise later")
        wide = s.to_frame() if wide is None else wide.join(s, how="outer")
    return wide.sort_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map-dir", default=MAP_DIR)
    ap.add_argument("--cwa-dir", default=CWA_DIR)
    ap.add_argument("--rain-agg", default="mean", choices=["mean", "max"],
                    help="how to combine the 5 stations into one city-wide number")
    ap.add_argument("--min-dry", type=int, default=20,
                    help="minimum dry observations before a stratum gets a baseline")
    ap.add_argument("--min-cells", type=int, default=30,
                    help="minimum sections observed before a timestep is usable")
    ap.add_argument("--lags", default="0,1,2",
                    help="hours of lag to test (speed now vs rain N hours ago)")
    cli = ap.parse_args()

    # ---------- speed ----------
    vel_p = os.path.join(cli.map_dir, "taichung_vel.csv")
    ts_p = os.path.join(cli.map_dir, "taichung_timestamps.csv")
    mask_p = os.path.join(cli.map_dir, "taichung_mask.npy")
    for p in (vel_p, ts_p, mask_p):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{p} not found -- run TDX_Data/build_speed.py first")

    vel = pd.read_csv(vel_p, encoding="utf-8-sig").to_numpy(np.float32)
    mask = np.load(mask_p)
    ts = pd.read_csv(ts_p, encoding="utf-8-sig")
    utc = pd.to_datetime(ts.iloc[:, 0], utc=True)
    T, N = vel.shape
    if mask.shape != vel.shape or len(utc) != T:
        raise ValueError(f"shape mismatch: vel {vel.shape}, mask {mask.shape}, "
                         f"timestamps {len(utc)}")

    # ---------- rain ----------
    stations, fixes = load_stations(cli.cwa_dir)
    wide = station_matrix(stations)

    print("=== 輸入 ===")
    print(f"  車速  {T:,} 步 × {N} 路段（UTC）"
          f"  {utc.min():%Y-%m-%d} ~ {utc.max():%Y-%m-%d}")
    print(f"  真實觀測 {int(mask.sum()):,} / {mask.size:,}（{mask.mean():.1%}），"
          f"其餘為補值，全部排除")
    print(f"  降雨  {len(stations)} 站 × {len(wide):,} 小時（本地時 UTC+8）："
          f"{', '.join(stations)}")
    if any(fixes.values()):
        n_days = len(set(next(iter(stations.values()))["Date"]))
        print(f"  ⚠ 已修正 ObsTime 標籤：每站 {max(fixes.values())} 列"
              f"（= {n_days} 天，每天最後一小時）")
        print(f"    原始檔把第 24 小時（23:01–24:00）標成 23，與真正的第 23 小時撞號；")
        print(f"    資料值正確，僅標籤錯誤，已依當天列序重建。"
              f"建議也在 download_rainfall_all.py 修掉。")

    print(f"\n=== 測站一致性 ===")
    corr = wide.corr(min_periods=100)
    iu = np.triu_indices(len(corr), k=1)
    pair = corr.to_numpy()[iu]
    print(f"  站間逐時降雨相關係數：中位數 {np.nanmedian(pair):.3f}，"
          f"範圍 {np.nanmin(pair):.3f}–{np.nanmax(pair):.3f}")
    if np.nanmedian(pair) >= 0.5:
        print(f"  → 五站高度一致，以全市{cli.rain_agg}代表區域降雨是合理的"
              f"（測站經緯度未知，無法做逐路段指派）")
    else:
        print(f"  ⚠ 站間一致性偏低，降雨在空間上分佈不均；全市聚合會低估局部強度，"
              f"結論宜保守")

    city = (wide.mean(axis=1, skipna=True) if cli.rain_agg == "mean"
            else wide.max(axis=1, skipna=True))
    city = city.where(wide.notna().any(axis=1))     # all-station-missing stays NaN

    # ---------- align ----------
    # local = utc + 8h; the bucket is the hour ENDING at ObsTime, so step back one
    # second before reading off the date and hour.
    # stdlib timezone, not pd.FixedOffset: the latter is not public pandas API and is
    # absent in some versions (AttributeError under the WSL environment).
    local = utc.dt.tz_convert(timezone(timedelta(hours=TZ_OFFSET_HOURS)))
    shifted = local - pd.Timedelta(seconds=1)
    key = pd.MultiIndex.from_arrays([shifted.dt.strftime("%Y-%m-%d"),
                                     (shifted.dt.hour + 1).astype("int64")])
    rain_all = city.reindex(key).to_numpy(dtype=float)          # (T,) NaN where absent
    # A dtype or timezone slip would leave every lookup unmatched. Catch it loudly
    # here rather than reporting "no rain in the entire period" later.
    if not np.isfinite(rain_all).any():
        raise SystemExit(
            "ERROR  no TDX timestep matched a CWA hour. Check the index dtypes and "
            f"that the periods overlap:\n"
            f"  TDX local {local.min():%Y-%m-%d %H:%M} ~ {local.max():%Y-%m-%d %H:%M}\n"
            f"  CWA       {wide.index.get_level_values(0).min()} ~ "
            f"{wide.index.get_level_values(0).max()}")
    hour = local.dt.hour.to_numpy()                              # local hour-of-day
    weekend = (local.dt.dayofweek >= 5).to_numpy().astype(np.int8)

    covered = ~np.isnan(rain_all)
    print(f"\n=== 時間對齊 ===")
    print(f"  {covered.sum():,} / {T:,} 個時間步有降雨資料（{covered.mean():.1%}）")
    print(f"  未涵蓋 {int((~covered).sum()):,} 步：TDX 起點早於 CWA，或該小時為缺測")
    ex = np.flatnonzero(covered)[:1]
    if ex.size:
        i = int(ex[0])
        print(f"  對齊示例：UTC {utc.iloc[i]:%Y-%m-%d %H:%M} → 本地 "
              f"{local.iloc[i]:%Y-%m-%d %H:%M} → CWA {key[i][0]} ObsTime {key[i][1]}")

    lags = [int(x) for x in cli.lags.split(",")]
    steps_per_hour = 12          # TDX is sampled every 5 minutes

    # ---------- per-lag analysis ----------
    for lag in lags:
        rain = (rain_all if lag == 0 else
                np.concatenate([np.full(lag * steps_per_hour, np.nan),
                                rain_all[:-lag * steps_per_hour]]))
        usable = ~np.isnan(rain)
        cell = mask & usable[:, None]
        ti, si = np.nonzero(cell)
        if ti.size == 0:
            print(f"\n=== lag {lag}h：無可用資料 ===")
            continue
        speed = vel[ti, si]

        # Baseline: same section, same hour-of-day, same weekday/weekend, DRY only.
        n_strat = N * 48
        strat = si.astype(np.int64) * 48 + hour[ti] * 2 + weekend[ti]
        dry = rain[ti] == 0.0
        cnt = np.bincount(strat[dry], minlength=n_strat)
        tot = np.bincount(strat[dry], weights=speed[dry].astype(np.float64),
                          minlength=n_strat)
        base = np.where(cnt >= cli.min_dry, tot / np.maximum(cnt, 1), np.nan)

        b = base[strat]
        ok = np.isfinite(b) & (b > 0)
        dev = speed[ok] / b[ok] - 1.0                    # relative deviation

        # Collapse to one number per timestep: all sections share the same weather, so
        # treating cells as independent would inflate the sample size ~100x.
        dsum = np.bincount(ti[ok], weights=dev, minlength=T)
        dcnt = np.bincount(ti[ok], minlength=T)
        good = dcnt >= cli.min_cells
        dev_t = np.where(good, dsum / np.maximum(dcnt, 1), np.nan)

        print(f"\n=== 降雨對車速的影響（lag {lag}h，控制路段 × 小時 × 平日假日）===")
        print(f"  {'時雨量 (mm)':<12}{'時間步':>8}{'小時':>7}{'觀測數':>11}"
              f"{'平均車速':>10}{'相對基準':>12}{'標準誤':>9}")
        print("  " + "-" * 69)
        rows = []
        for lo, hi, label in BUCKETS:
            sel = (rain == lo) if lo == hi else ((rain > lo) & (rain <= hi))
            sel &= good
            n_t = int(sel.sum())
            if n_t == 0:
                print(f"  {label:<12}{0:>8}{'—':>7}{'—':>11}{'—':>10}{'—':>12}{'—':>9}")
                rows.append((label, 0, np.nan, np.nan))
                continue
            d = dev_t[sel]
            m = float(np.nanmean(d))
            se = float(np.nanstd(d, ddof=1) / np.sqrt(n_t)) if n_t > 1 else np.nan
            n_obs = int(dcnt[sel].sum())
            mspeed = float(np.average(vel[sel][mask[sel]].mean()) if mask[sel].any()
                           else np.nan)
            print(f"  {label:<12}{n_t:>8,}{n_t / steps_per_hour:>7.0f}{n_obs:>11,}"
                  f"{mspeed:>10.2f}{m * 100:>11.2f}%{se * 100:>8.2f}%")
            rows.append((label, n_t, m, se))

        # --- verdict for this lag ---
        wet = [r for r in rows if r[0] != "0（無雨）" and r[1] > 0 and np.isfinite(r[2])]
        if len(wet) < 2:
            print("  → 有雨樣本太少，無法判定")
            continue
        means = [r[2] for r in wet]
        monotone = all(means[i] >= means[i + 1] for i in range(len(means) - 1))
        strongest = wet[-1]
        sig = (abs(strongest[2]) > 2 * strongest[3]) if np.isfinite(strongest[3]) else False
        print(f"  劑量反應（雨越大越慢）：{'✅ 單調成立' if monotone else '❌ 不單調'}")
        print(f"  最強效應：{strongest[0]} mm/h → {strongest[2] * 100:+.2f}% "
              f"（{'超過' if sig else '未超過'} 2 倍標準誤）")

    print(f"\n=== 怎麼判讀 ===")
    print("  值得投入（重訓兩個模型 × 三時程）的條件是三項同時成立：")
    print("    ① 劑量反應單調  ② 最強效應超過 2 倍標準誤  ③ 效應幅度有實務意義（≳3%）")
    print("  若不成立，這本身就是可寫進報告的發現——「本期間降雨僅佔約 6% 的時數，")
    print("  且未觀察到顯著車速影響」，計劃書 §4.2 的氣候特徵即可據此說明取捨。")
    print("\n  注意：全市聚合會低估局部強度（測站經緯度未知，無法逐路段指派），")
    print("  因此這個檢定偏保守——量到效應可信，量不到則不能完全排除。")


if __name__ == "__main__":
    main()
