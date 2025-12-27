"""
config/settings.py

Configuración central del proyecto IA_trade.
Aquí definimos parámetros generales para entrenamiento, live trading y logging.
"""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    # Trading
    SYMBOL: str = "ETHUSDT"
    TIMEFRAME_TRAIN: str = "5"     # velas (por ejemplo 5 minutos si la fuente lo define así)
    TIMEFRAME_LIVE: str = "1"      # live suele ser 1m
    MAX_OPEN_POSITIONS: int = 1    # solo ETH, normalmente 1

    # Risk
    RISK_PER_TRADE: float = 0.005  # 0.5%
    CAPITAL_FRACTION_PER_TRADE: float = 0.50  # 50% del capital total por operación

    # Paths
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    ETH_MODELS_DIR: Path = MODELS_DIR / "eth"
    CHECKPOINTS_DIR: Path = MODELS_DIR / "checkpoints"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"

    # Files
    TRADES_CSV: Path = LOGS_DIR / "trades.csv"
    DECISIONS_LOG: Path = LOGS_DIR / "decisions.log"
    ERRORS_LOG: Path = LOGS_DIR / "errors.log"


SETTINGS = Settings()
