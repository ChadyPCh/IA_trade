"""
src/features/feature_engineering.py

Feature engineering para datos OHLCV.

Contrato de entrada (df):
- Columnas mínimas requeridas:
  time, open, high, low, close, volume, turnover

Salida:
- DF original + columnas de features (RSI, MACD, EMAs, retornos, rangos, vol_norm, etc.)
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye features a partir de OHLCV.

    df debe tener: time, open, high, low, close, volume, turnover
    """
    required = ["time", "open", "high", "low", "close", "volume", "turnover"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en RAW para features: {missing}")

    out = df.copy()

    # Asegurar time datetime UTC (no rompe si ya lo es)
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    out = out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    # Retornos
    out["return_1"] = out["close"].pct_change().fillna(0.0)
    out["log_return_1"] = np.log(out["close"].replace(0, np.nan)).diff().replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # EMAs + MACD
    out["ema_12"] = _ema(out["close"], 12)
    out["ema_26"] = _ema(out["close"], 26)
    out["macd"] = (out["ema_12"] - out["ema_26"]).fillna(0.0)
    out["macd_signal"] = _ema(out["macd"], 9).fillna(0.0)
    out["macd_hist"] = (out["macd"] - out["macd_signal"]).fillna(0.0)

    # RSI
    out["rsi_14"] = _rsi(out["close"], 14)

    # Rangos
    out["hl_range"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    out["oc_range"] = (out["close"] - out["open"]) / out["open"].replace(0, np.nan)
    out["hl_range"] = out["hl_range"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out["oc_range"] = out["oc_range"].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # Volumen normalizado
    vol_roll = out["volume"].rolling(50).mean()
    out["vol_norm"] = (out["volume"] / vol_roll.replace(0, np.nan)).replace([np.inf, -np.inf], 0.0).fillna(1.0)

    # Limpieza final
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return out
