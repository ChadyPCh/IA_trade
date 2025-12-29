"""
src/data/download_klines.py

Downloader de klines Bybit V5.
- Guarda CSV en data/raw/
- Reanuda si existe (backfill + dedupe)
- Usa paginación HACIA ATRÁS (end -> start) para estabilidad
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from src.broker.bybit_client import BybitClient


def parse_ymd(date_str: str) -> datetime:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.replace(tzinfo=timezone.utc)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def interval_minutes_to_timedelta(interval_str: str) -> timedelta:
    return timedelta(minutes=int(interval_str))


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_output_path(symbol: str, interval: str) -> Path:
    return project_root() / "data" / "raw" / f"{symbol}_kline_{interval}m.csv"


@dataclass
class ExistingDataInfo:
    exists: bool
    last_time: Optional[pd.Timestamp] = None


def inspect_existing_csv(path: Path) -> ExistingDataInfo:
    if not path.exists():
        return ExistingDataInfo(exists=False, last_time=None)

    try:
        df = pd.read_csv(path, usecols=["time"])
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time")
        if df.empty:
            return ExistingDataInfo(exists=True, last_time=None)
        return ExistingDataInfo(exists=True, last_time=df["time"].iloc[-1])
    except Exception:
        return ExistingDataInfo(exists=True, last_time=None)


def save_merged_csv(path: Path, df_new: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        df_old = pd.read_csv(path)
        df_old["time"] = pd.to_datetime(df_old["time"], utc=True, errors="coerce")
        df_new["time"] = pd.to_datetime(df_new["time"], utc=True, errors="coerce")
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new.copy()
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

    df = df.dropna(subset=["time"])
    df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)

    # Reordenar columnas (contrato)
    cols = ["time", "open", "high", "low", "close", "volume", "turnover"]
    df = df[cols]

    df.to_csv(path, index=False)


def rows_to_df(rows) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "turnover"])

    df = pd.DataFrame(
        {
            "time_ms": [r.start_time_ms for r in rows],
            "open": [r.open for r in rows],
            "high": [r.high for r in rows],
            "low": [r.low for r in rows],
            "close": [r.close for r in rows],
            "volume": [r.volume for r in rows],
            "turnover": [r.turnover for r in rows],
        }
    )
    df["time"] = pd.to_datetime(df["time_ms"], unit="ms", utc=True)
    df = df.drop(columns=["time_ms"]).sort_values("time").reset_index(drop=True)
    df = df[["time", "open", "high", "low", "close", "volume", "turnover"]]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Descargar velas (klines) de Bybit y guardarlas localmente.")
    parser.add_argument("--symbol", type=str, default="ETHUSDT")
    parser.add_argument("--interval", type=str, default="5")
    parser.add_argument("--start", type=str, default="2024-01-01")

    # ✅ IMPORTANTE: por el tema de tu reloj, aquí te recomiendo usar --end siempre
    parser.add_argument("--end", type=str, default="", help="YYYY-MM-DD UTC. Recomendado especificarlo.")

    parser.add_argument("--category", type=str, default="linear")
    parser.add_argument("--base-url", type=str, default="https://api.bybit.com")
    parser.add_argument("--output", type=str, default="")

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--backfill", type=int, default=200)

    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--sleep", type=float, default=0.35)
    parser.add_argument("--progress-every", type=int, default=10)

    args = parser.parse_args()

    symbol = args.symbol.strip().upper()
    interval = args.interval.strip()
    start_dt = parse_ymd(args.start.strip())

    # Si no se pasa end, usa ahora (pero tu PC parece adelantada)
    end_dt = parse_ymd(args.end.strip()) if args.end.strip() else utc_now()

    output_path = Path(args.output).resolve() if args.output.strip() else default_output_path(symbol, interval)

    if args.overwrite and output_path.exists():
        output_path.unlink()

    info = inspect_existing_csv(output_path)
    do_resume = (not args.overwrite) or args.resume

    if do_resume and info.exists and info.last_time is not None:
        # reanudamos desde la última vela guardada (con backfill)
        interval_td = interval_minutes_to_timedelta(interval)
        backfill_td = interval_td * args.backfill
        start_dt_resume = (info.last_time.to_pydatetime() - backfill_td).astimezone(timezone.utc)
        if start_dt_resume < start_dt:
            start_dt_resume = start_dt

        print(f"[RESUME] Archivo existe: {output_path}")
        print(f"[RESUME] Última vela guardada: {info.last_time}")
        print(f"[RESUME] Descargando desde: {start_dt_resume.isoformat()} hasta: {end_dt.isoformat()}")
        start_dt_effective = start_dt_resume
    else:
        print(f"[FULL] Descargando desde: {start_dt.isoformat()} hasta: {end_dt.isoformat()}")
        start_dt_effective = start_dt

    client = BybitClient(base_url=args.base_url.strip())

    rows = client.download_klines_backward(
        symbol=symbol,
        interval=interval,
        start_dt=start_dt_effective,
        end_dt=end_dt,
        category=args.category.strip(),
        limit=args.limit,
        sleep_s=args.sleep,
        progress_every=args.progress_every,
    )

    if not rows:
        print("[WARN] No se descargaron velas (rows vacío).")
        return

    df_new = rows_to_df(rows)
    save_merged_csv(output_path, df_new)

    df_final = pd.read_csv(output_path)
    df_final["time"] = pd.to_datetime(df_final["time"], utc=True, errors="coerce")
    df_final = df_final.dropna(subset=["time"]).sort_values("time")

    print(f"[OK] Guardado: {output_path}")
    print(f"[OK] Filas totales: {len(df_final)}")
    if not df_final.empty:
        print(f"[OK] Rango final: {df_final['time'].iloc[0]} -> {df_final['time'].iloc[-1]}")


if __name__ == "__main__":
    main()
