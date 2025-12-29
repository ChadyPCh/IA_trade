"""
src/data/validate_dataset.py

Valida integridad de datos RAW y PROCESSED:
- Rango, orden, duplicados
- Frecuencia 5m y huecos
- Reglas OHLCV
- Outliers de retornos
- Consistencia RAW vs PROCESSED

Uso:
python -m src.data.validate_dataset --symbol ETHUSDT --interval 5 --start 2024-01-01 --end 2026-01-01
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def raw_path(symbol: str, interval: str) -> Path:
    return project_root() / "data" / "raw" / f"{symbol}_kline_{interval}m.csv"


def processed_path(symbol: str, interval: str) -> Path:
    return project_root() / "data" / "processed" / f"{symbol}_features_{interval}m.csv"


def parse_ymd(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def summarize_gaps(times: pd.Series, freq_min: int) -> dict:
    dt = times.diff()
    expected = pd.Timedelta(minutes=freq_min)

    gaps = dt[dt > expected]
    gap_count = int(gaps.shape[0])

    if gap_count == 0:
        return {"gap_count": 0, "max_gap": None, "top_gaps": []}

    top = gaps.sort_values(ascending=False).head(10)
    top_list = [(str(times.loc[idx - 1]), str(times.loc[idx]), str(top.loc[idx])) for idx in top.index if idx in times.index]

    return {
        "gap_count": gap_count,
        "max_gap": str(gaps.max()),
        "top_gaps": top_list,
    }


def validate_ohlc(df: pd.DataFrame) -> dict:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    bad_high = (h < np.maximum(o, c)).sum()
    bad_low = (l > np.minimum(o, c)).sum()
    neg_price = ((o < 0) | (h < 0) | (l < 0) | (c < 0)).sum()
    return {
        "bad_high_count": int(bad_high),
        "bad_low_count": int(bad_low),
        "negative_price_count": int(neg_price),
    }


def validate_basic(df: pd.DataFrame, freq_min: int, start: pd.Timestamp | None, end: pd.Timestamp | None) -> dict:
    out = {}

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    out["time_parse_na"] = int(df["time"].isna().sum())

    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    out["rows"] = int(len(df))
    out["min_time"] = str(df["time"].min()) if len(df) else None
    out["max_time"] = str(df["time"].max()) if len(df) else None

    out["is_sorted"] = bool(df["time"].is_monotonic_increasing)
    out["duplicates"] = int(df["time"].duplicated().sum())

    # rango esperado (si se provee)
    if start is not None and len(df):
        out["start_ok"] = bool(df["time"].min() <= start + pd.Timedelta(minutes=freq_min))
    if end is not None and len(df):
        out["end_ok"] = bool(df["time"].max() >= end - pd.Timedelta(minutes=freq_min))

    out["gaps"] = summarize_gaps(df["time"], freq_min=freq_min)

    # valores negativos en vol/turnover
    if "volume" in df.columns:
        out["negative_volume"] = int((df["volume"] < 0).sum())
    if "turnover" in df.columns:
        out["negative_turnover"] = int((df["turnover"] < 0).sum())

    return out


def validate_returns(df: pd.DataFrame) -> dict:
    # Calcula retornos si no existen
    close = df["close"].astype(float)
    ret = close.pct_change()

    # Umbral “grosero” para 5m (ajustable)
    extreme = (ret.abs() > 0.30).sum()  # >30% en 5m suele ser sospechoso
    max_abs = float(ret.abs().max()) if ret.notna().any() else 0.0

    return {
        "extreme_return_count(|ret|>30%)": int(extreme),
        "max_abs_return": max_abs,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str, default="ETHUSDT")
    p.add_argument("--interval", type=str, default="5")
    p.add_argument("--start", type=str, default="")
    p.add_argument("--end", type=str, default="")
    args = p.parse_args()

    symbol = args.symbol.strip().upper()
    interval = args.interval.strip()
    freq_min = int(interval)

    start = parse_ymd(args.start) if args.start else None
    end = parse_ymd(args.end) if args.end else None

    rp = raw_path(symbol, interval)
    pp = processed_path(symbol, interval)

    if not rp.exists():
        raise FileNotFoundError(f"No existe RAW: {rp}")
    if not pp.exists():
        raise FileNotFoundError(f"No existe PROCESSED: {pp}")

    df_raw = pd.read_csv(rp)
    df_proc = pd.read_csv(pp)

    print("=== RAW CHECK ===")
    raw_basic = validate_basic(df_raw, freq_min=freq_min, start=start, end=end)
    print(raw_basic)

    print("\n=== RAW OHLC RULES ===")
    print(validate_ohlc(df_raw))

    print("\n=== RAW RETURNS CHECK ===")
    print(validate_returns(df_raw))

    print("\n=== PROCESSED CHECK ===")
    proc_basic = validate_basic(df_proc, freq_min=freq_min, start=start, end=end)
    print(proc_basic)

    print("\n=== RAW vs PROCESSED CONSISTENCY ===")
    # comparar tamaño y time
    df_raw_t = pd.to_datetime(df_raw["time"], utc=True, errors="coerce")
    df_proc_t = pd.to_datetime(df_proc["time"], utc=True, errors="coerce")

    same_len = len(df_raw_t) == len(df_proc_t)
    same_time = False
    if same_len:
        same_time = bool((df_raw_t.fillna(pd.Timestamp(0, tz="UTC")).values == df_proc_t.fillna(pd.Timestamp(0, tz="UTC")).values).all())

    print({
        "raw_rows": len(df_raw),
        "processed_rows": len(df_proc),
        "same_len": same_len,
        "same_time_series": same_time,
        "processed_nan_count": int(df_proc.isna().sum().sum()),
        "processed_inf_count": int(np.isinf(df_proc.select_dtypes(include=[np.number])).sum().sum()),
    })


if __name__ == "__main__":
    main()
