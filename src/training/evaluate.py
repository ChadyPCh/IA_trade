"""
src/training/evaluate.py

Evaluación simple del modelo PPO sobre el tramo de validación (últimos meses).
- No llama a Bybit.
- Reporta: reward promedio, trades, winrate y pnl promedio por episodio.

Uso:
python -m src.training.evaluate --symbol ETHUSDT --interval 5 --val-months 3 --episodes 30
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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
    return train_df, val_df


def pick_feature_cols(df: pd.DataFrame) -> List[str]:
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
            max_trade_steps=max_trade_steps,
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
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str, default="ETHUSDT")
    p.add_argument("--interval", type=str, default="5")
    p.add_argument("--val-months", type=int, default=3)

    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--episode-len", type=int, default=2000)
    p.add_argument("--max-trade-steps", type=int, default=36)

    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=123)

    args = p.parse_args()

    symbol = args.symbol.strip().upper()
    interval = args.interval.strip()

    path = processed_path(symbol, interval)
    if not path.exists():
        raise FileNotFoundError(f"No existe processed CSV: {path}")

    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    _, val_df = split_train_val_by_months(df, val_months=args.val_months)
    feature_cols = pick_feature_cols(df)

    mdir = models_dir(symbol)
    model_path = mdir / "ppo_offline.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"No existe modelo: {model_path}")

    # Env validación
    env = DummyVecEnv([make_env(val_df, feature_cols, args.lookback, args.episode_len, args.max_trade_steps, args.seed)])

    # Cargar vecnormalize si existe
    vn_path = mdir / "vecnormalize.pkl"
    if vn_path.exists():
        env = VecNormalize.load(str(vn_path), env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(str(model_path))

    rewards = []
    trades = []
    winrates = []
    pnls = []

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, dones, infos = env.step(action)
            ep_reward += float(r[0])
            done = bool(dones[0])

            # al final del episodio, el entorno devuelve métricas
            if done:
                info = infos[0]
                trades.append(int(info.get("episode_trades", 0)))
                winrates.append(float(info.get("episode_winrate", 0.0)))
                pnls.append(float(info.get("episode_total_pnl", 0.0)))

        rewards.append(ep_reward)

    print("=== EVAL SUMMARY ===")
    print(f"episodes: {args.episodes}")
    print(f"avg_reward: {np.mean(rewards):.4f} | std: {np.std(rewards):.4f}")
    print(f"avg_trades/ep: {np.mean(trades):.2f}")
    print(f"avg_winrate: {np.mean(winrates):.3f}")
    print(f"avg_total_pnl/ep: {np.mean(pnls):.4f}")


if __name__ == "__main__":
    main()
