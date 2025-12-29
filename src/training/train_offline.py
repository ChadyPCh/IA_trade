"""
src/training/train_offline.py

Entrenamiento OFFLINE (RL) con Stable-Baselines3 PPO usando data/processed.
- NO llama a Bybit.
- Usa episodios aleatorios para "muchas variantes".
- Valida con los últimos meses.

Uso recomendado:
python -m src.training.train_offline --symbol ETHUSDT --interval 5 --val-months 3 --timesteps 2000000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from src.env.trading_environment import TradingEnvironment


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def processed_path(symbol: str, interval: str) -> Path:
    return project_root() / "data" / "processed" / f"{symbol}_features_{interval}m.csv"


def models_dir(symbol: str) -> Path:
    return project_root() / "models" / symbol.lower()


def split_train_val_by_months(df: pd.DataFrame, val_months: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    max_t = df["time"].max()
    cutoff = max_t - pd.DateOffset(months=val_months)

    train_df = df[df["time"] < cutoff].reset_index(drop=True)
    val_df = df[df["time"] >= cutoff].reset_index(drop=True)

    if len(train_df) < 10000 or len(val_df) < 1000:
        raise ValueError(f"Split muy pequeño. train={len(train_df)} val={len(val_df)} cutoff={cutoff}")

    return train_df, val_df


def pick_feature_cols(df: pd.DataFrame) -> List[str]:
    # Solo features (sin OHLCV crudo) para observación base estable
    # Si luego quieres incluir OHLC, se agrega aquí sin cambiar el entorno.
    preferred = [
        "return_1", "log_return_1",
        "ema_12", "ema_26",
        "macd", "macd_signal", "macd_hist",
        "rsi_14",
        "hl_range", "oc_range",
        "vol_norm",
    ]
    missing = [c for c in preferred if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan features esperadas en processed: {missing}")
    return preferred


def make_env(df: pd.DataFrame, feature_cols: List[str], lookback: int, episode_len: int, max_trade_steps: int, seed: int):
    def _init():
        env = TradingEnvironment(
            df=df,
            feature_cols=feature_cols,
            price_col="close",
            lookback=lookback,
            episode_len=episode_len,
            max_trade_steps=max_trade_steps,  # 36 = 3h en 5m
            target_profit=0.01,
            stop_loss=0.01,
            reward_scale=1.0,
            step_pnl_weight=0.05,
            time_penalty=0.001,
            seed=seed,
        )
        return env
    return _init


def main() -> None:
    p = argparse.ArgumentParser(description="Entrenamiento offline RL (PPO) usando data/processed.")
    p.add_argument("--symbol", type=str, default="ETHUSDT")
    p.add_argument("--interval", type=str, default="5")
    p.add_argument("--val-months", type=int, default=3)

    p.add_argument("--lookback", type=int, default=60)         # 5h de contexto
    p.add_argument("--episode-len", type=int, default=2000)     # variedad (muchas ventanas aleatorias)
    p.add_argument("--max-trade-steps", type=int, default=36)   # 3h en 5m

    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--vecnorm", action="store_true", help="Usar VecNormalize (recomendado).")
    p.add_argument("--no-vecnorm", action="store_true", help="No usar VecNormalize.")

    args = p.parse_args()

    symbol = args.symbol.strip().upper()
    interval = args.interval.strip()

    path = processed_path(symbol, interval)
    if not path.exists():
        raise FileNotFoundError(f"No existe processed CSV: {path}")

    print(f"[LOAD] {path}")
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    train_df, val_df = split_train_val_by_months(df, val_months=args.val_months)
    feature_cols = pick_feature_cols(df)

    print(f"[SPLIT] train={len(train_df)} | val={len(val_df)} | val_months={args.val_months}")
    print(f"[FEAT] cols={len(feature_cols)} -> {feature_cols}")

    # Env train (episodios aleatorios)
    env_train = DummyVecEnv([make_env(train_df, feature_cols, args.lookback, args.episode_len, args.max_trade_steps, args.seed)])

    use_vecnorm = args.vecnorm or (not args.no_vecnorm)
    if use_vecnorm:
        env_train = VecNormalize(env_train, norm_obs=True, norm_reward=True, clip_obs=10.0)

    out_dir = models_dir(symbol)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=100_000,
        save_path=str(out_dir / "checkpoints"),
        name_prefix="ppo",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env_train,
        verbose=1,
        seed=args.seed,
        n_steps=2048,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        ent_coef=0.0,
        clip_range=0.2,
        tensorboard_log=str(out_dir / "tb"),
    )

    print(f"[TRAIN] timesteps={args.timesteps} lookback={args.lookback} episode_len={args.episode_len} max_trade_steps={args.max_trade_steps}")
    model.learn(total_timesteps=args.timesteps, callback=checkpoint_cb)

    # Guardar modelo final + VecNormalize
    model_path = out_dir / "ppo_offline.zip"
    model.save(str(model_path))
    print(f"[SAVE] model -> {model_path}")

    if use_vecnorm and isinstance(env_train, VecNormalize):
        vn_path = out_dir / "vecnormalize.pkl"
        env_train.save(str(vn_path))
        print(f"[SAVE] vecnormalize -> {vn_path}")

    print("[DONE] Entrenamiento offline terminado.")


if __name__ == "__main__":
    main()
