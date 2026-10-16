"""Entrenamiento DQN sin consultar el período held-out (test).

Uso:
    python -m src.train --config configs/reward_v4.yaml
    python -m src.train --config configs/reward_v4.yaml --seeds 0 1 2
    python -m src.train --config configs/reward_v4.yaml --total-steps 5000 --seeds 0

Durante el entrenamiento:
- Se aprende exclusivamente con train.
- Se seleccionan checkpoints y se aplica early stopping usando exclusivamente val.
- test NO se carga ni se evalúa en este módulo.
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

    # Es intencional: train.py solo construye train y val.
    train_env = make_env(cfg, splits["train"], training=True)
    val_env = make_env(cfg, splits["val"], training=False)

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
    step = 0

    for step in range(1, total + 1):
        eps = agent.epsilon(step)
        action = agent.act(obs, info, epsilon=eps)
        next_obs, reward, terminated, truncated, info = train_env.step(action)

        # En truncamiento externo se conserva el bootstrap; solo terminated corta el bootstrap.
        agent.buffer.add(obs, action, reward, next_obs, terminated)
        obs = next_obs
        ep_reward += reward

        if step % ac["train_freq"] == 0:
            loss = agent.update()
            if loss is not None:
                losses.append(float(loss))

        if terminated or truncated:
            recent_ep_rewards.append(ep_reward)
            ep_reward = 0.0
            obs, info = train_env.reset()

        if step % int(tc["eval_every"]) == 0 or step == total:
            _, _, vm = evaluate_policy(val_env, agent)
            score = float(vm[metric])

            row = {
                "step": step,
                "epsilon": float(eps),
                "loss_mean_last_1000": float(np.mean(losses[-1000:])) if losses else np.nan,
                "train_ep_reward_mean_last_10": (
                    float(np.mean(recent_ep_rewards)) if recent_ep_rewards else np.nan
                ),
                **{f"val_{k}": vm[k] for k in [
                    "cum_return", "sharpe", "max_drawdown", "n_trades", "avg_position"
                ]},
            }
            log_rows.append(row)

            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            agent.save(
                ckpt_dir / f"step_{step:07d}.pt",
                extra={"step": step, "val_" + metric: score, "seed": seed},
            )

            improved = score > best_score
            if improved:
                best_score, best_step, since_best = score, step, 0
                agent.save(
                    run_dir / "model.pt",
                    extra={"step": step, "val_" + metric: score, "seed": seed},
                )
            else:
                since_best += 1

            if verbose:
                print(
                    f"[{cfg['name']} seed={seed}] step {step:>7}/{total} "
                    f"eps={eps:.3f} val_{metric}={score:+.3f} "
                    f"val_ret={vm['cum_return']:+.1%}"
                    f"{' *' if improved else ''}"
                )

            if step >= int(tc["min_steps"]) and since_best >= int(tc["patience"]):
                stopped_early = True
                if verbose:
                    print(f"  early stopping en step {step}; mejor checkpoint: {best_step}")
                break

    pd.DataFrame(log_rows).to_csv(run_dir / "train_log.csv", index=False)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    summary = {
        "seed": seed,
        "best_step": best_step,
        f"best_val_{metric}": float(best_score),
        "steps_run": step,
        "stopped_early": stopped_early,
        "test_evaluated_during_training": False,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Entrena DQN usando train y selección con val")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--csv", default=None, help="sobrescribe el CSV definido en el config")
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
    for seed in seeds:
        train(cfg, seed, Path(a.out) / cfg["name"] / f"seed_{seed}")


if __name__ == "__main__":
    main()
