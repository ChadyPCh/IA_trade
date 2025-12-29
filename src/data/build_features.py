"""
src/data/build_features.py

Script para construir features a partir del CSV RAW descargado de Bybit.

Flujo:
1) Lee:  data/raw/ETHUSDT_kline_5m.csv
2) Calcula features (RSI, MACD, retornos, etc.) con src/features/feature_engineering.py
3) Guarda: data/processed/ETHUSDT_features_5m.csv

Este script existe para que train_offline.py NO tenga que conectarse a Bybit
y trabaje únicamente con datos locales ya procesados.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.features.feature_engineering import build_features


def project_root() -> Path:
    # .../IA_trade/src/data/build_features.py -> parents[2] = IA_trade
    return Path(__file__).resolve().parents[2]


def default_raw_path(symbol: str, interval: str) -> Path:
    return project_root() / "data" / "raw" / f"{symbol}_kline_{interval}m.csv"


def default_processed_path(symbol: str, interval: str) -> Path:
    return project_root() / "data" / "processed" / f"{symbol}_features_{interval}m.csv"


def validate_raw_df(df: pd.DataFrame) -> None:
    required_cols = ["time", "open", "high", "low", "close", "volume", "turnover"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"RAW CSV inválido. Faltan columnas: {missing}")

    # Validar types básicos
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    if df["time"].isna().any():
        raise ValueError("RAW CSV inválido: columna 'time' tiene valores no parseables a datetime UTC.")

    # Asegurar orden
    if not df["time"].is_monotonic_increasing:
        # no es error fatal, pero lo ordenamos antes de features
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Construir features desde CSV RAW.")
    parser.add_argument("--symbol", type=str, default="ETHUSDT", help="Símbolo, ej: ETHUSDT")
    parser.add_argument("--interval", type=str, default="5", help="Intervalo en minutos, ej: 5")
    parser.add_argument(
        "--raw",
        type=str,
        default="",
        help="Ruta al CSV RAW. Vacío = data/raw/<symbol>_kline_<interval>m.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Ruta al CSV PROCESSED. Vacío = data/processed/<symbol>_features_<interval>m.csv",
    )

    args = parser.parse_args()

    symbol = args.symbol.strip().upper()
    interval = args.interval.strip()

    raw_path = Path(args.raw).resolve() if args.raw.strip() else default_raw_path(symbol, interval)
    out_path = Path(args.out).resolve() if args.out.strip() else default_processed_path(symbol, interval)

    if not raw_path.exists():
        raise FileNotFoundError(f"No existe RAW CSV: {raw_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[RAW] Leyendo: {raw_path}")
    df_raw = pd.read_csv(raw_path)

    validate_raw_df(df_raw)

    # Asegurar formato y orden
    df_raw["time"] = pd.to_datetime(df_raw["time"], utc=True, errors="coerce")
    df_raw = df_raw.sort_values("time").reset_index(drop=True)

    print(f"[FEAT] Construyendo features...")
    df_feat = build_features(df_raw)

    # Garantizar que 'time' siga siendo datetime serializable y que esté primera
    df_feat["time"] = pd.to_datetime(df_feat["time"], utc=True, errors="coerce")
    # Reordenar: time primero
    cols = list(df_feat.columns)
    if "time" in cols:
        cols = ["time"] + [c for c in cols if c != "time"]
        df_feat = df_feat[cols]

    df_feat.to_csv(out_path, index=False)

    print(f"[OK] Guardado PROCESSED: {out_path}")
    print(f"[OK] Filas: {len(df_feat)} | Columnas: {len(df_feat.columns)}")
    print(f"[OK] Rango: {df_feat['time'].iloc[0]} -> {df_feat['time'].iloc[-1]}")


if __name__ == "__main__":
    main()
