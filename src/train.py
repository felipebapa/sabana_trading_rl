"""Entrenamiento del agente DQN (sabana_trading_rl).

Uso:
    python -m src.train --config configs/reward_v1.yaml                # semillas del config
    python -m src.train --config configs/reward_v2.yaml --seeds 0 1 2
    python -m src.train --config configs/reward_v1.yaml --total-steps 5000 --seeds 0   # prueba rápida

Protocolo:
  * El agente aprende SOLO con el período `train` (episodios de ~1 año con inicio aleatorio).
  * Cada `eval_every` pasos se evalúa (política greedy) en `val`. El mejor checkpoint
    según `selection_metric` se guarda como model.pt; hay early stopping por paciencia.
  * El período `test` (held-out) NUNCA se toca aquí.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.agent import DQNAgent
from src.data import build_splits, load_prices
from src.environment import make_env
from src.evaluate import evaluate_policy
from src.utils import load_config, set_seed


def train(cfg: dict, seed: int, run_dir: Path, verbose: bool = True) -> dict:
    set_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    prices = load_prices(cfg["data"]["csv"])
    splits = build_splits(prices, cfg["data"])
    train_env = make_env(cfg, splits["train"], training=True)
    val_env = make_env(cfg, splits["val"], training=False)
    test_env = make_env(cfg, splits["test"], training=False)

    obs_dim = train_env.observation_space.shape[0]
    n_actions = train_env.action_space.n
    tc, ac = cfg["train"], cfg["agent"]
    total = int(tc["total_steps"])
    agent = DQNAgent(obs_dim, n_actions, ac, total_steps=total, seed=seed)

    metric = tc["selection_metric"]
    best_score, best_step, since_best = -np.inf, 0, 0
    recent_ep_rewards: deque[float] = deque(maxlen=10)
    losses: list[float] = []
    log_rows: list[dict] = []

    obs, info = train_env.reset(seed=seed)
    ep_reward = 0.0
    stopped_early = False

    for step in range(1, total + 1):
        eps = agent.epsilon(step)
        action = agent.act(obs, info, epsilon=eps)
        next_obs, reward, terminated, truncated, info = train_env.step(action)
        # Solo `terminated` corta el bootstrap; un truncamiento (fin de ventana) no.
        agent.buffer.add(obs, action, reward, next_obs, terminated)
        obs = next_obs
        ep_reward += reward

        if step % ac["train_freq"] == 0:
            loss = agent.update()
            if loss is not None:
                losses.append(loss)

        if terminated or truncated:
            recent_ep_rewards.append(ep_reward)
            ep_reward = 0.0
            obs, info = train_env.reset()

        if step % tc["eval_every"] == 0 or step == total:
            _, _, vm = evaluate_policy(val_env, agent)
            score = vm[metric]

            # --- solo diagnóstico de sobreajuste (NO se usa para decidir) ---
            _, _, tm = evaluate_policy(test_env, agent)

            row = {
                "step": step,
                "epsilon": round(eps, 4),
                "loss": float(np.mean(losses[-1000:])) if losses else np.nan,
                "train_ep_reward": float(np.mean(recent_ep_rewards)) if recent_ep_rewards else np.nan,
                **{f"val_{k}": vm[k] for k in ["cum_return", "sharpe", "max_drawdown", "n_trades", "avg_position"]},
                **{f"test_{k}": tm[k] for k in ["cum_return", "sharpe", "max_drawdown", "n_trades", "avg_position"]},
            }
            log_rows.append(row)

            # guardar checkpoint intermedio (opcional pero útil para la curva)
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            agent.save(ckpt_dir / f"step_{step:07d}.pt", extra={"step": step, "val_" + metric: score})

            if score > best_score:
                best_score, best_step, since_best = score, step, 0
                agent.save(run_dir / "model.pt", extra={"step": step, "val_" + metric: score, "seed": seed})
            else:
                since_best += 1

            if verbose:
                print(
                    f"[{cfg['name']} seed={seed}] step {step:>7}/{total} eps={eps:.2f} "
                    f"val_{metric}={score:+.3f} test_{metric}={tm[metric]:+.3f} "
                    f"val_ret={vm['cum_return']:+.1%} test_ret={tm['cum_return']:+.1%} "
                    f"{'*' if since_best == 0 else ''}"
                )
            if step >= tc["min_steps"] and since_best >= tc["patience"]:
                stopped_early = True
                if verbose:
                    print(f"  early stopping en step {step} (mejor: step {best_step})")
                break

    pd.DataFrame(log_rows).to_csv(run_dir / "train_log.csv", index=False)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    summary = {"seed": seed, "best_step": best_step, f"best_val_{metric}": best_score,
               "steps_run": step, "stopped_early": stopped_early}
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Entrena el agente DQN")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--csv", default=None, help="sobrescribe data.csv del config")
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.csv:
        cfg["data"]["csv"] = a.csv
    if a.total_steps:
        cfg["train"]["total_steps"] = a.total_steps
    if a.eval_every:
        cfg["train"]["eval_every"] = a.eval_every
    seeds = a.seeds if a.seeds is not None else cfg["train"]["seeds"]

    for s in seeds:
        train(cfg, s, Path(a.out) / cfg["name"] / f"seed_{s}")


if __name__ == "__main__":
    main()
