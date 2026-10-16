"""Evaluación de agentes entrenados y baselines bajo las MISMAS condiciones
(mismo período, mismos datos, mismos costos de transacción).

Uso:
    python -m src.evaluate --runs-dir runs --split val
    python -m src.evaluate --runs-dir runs --split test --final     # held-out

El período `test` (held-out) está protegido: exige `--final` para evitar usarlo por accidente
al elegir hiperparámetros o versiones de la recompensa.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.agent import DQNAgent
from src.baselines import make_baselines
from src.data import build_splits, load_prices
from src.environment import make_env
from src.metrics import compute_metrics, drawdown_series
from src.utils import load_config


# --------------------------------------------------------------------------- #
# Núcleo: correr un episodio completo con una política
# --------------------------------------------------------------------------- #
def run_episode(env, policy, seed: int | None = None) -> pd.DataFrame:
    """Corre un episodio y devuelve un DataFrame (una fila por paso, indexado por la
    fecha en la que se valora el portafolio, es decir t+1)."""
    obs, info = env.reset(seed=seed)
    start_date, init_value = info["date"], info["value"]
    rows = []
    done = False
    while not done:
        action = policy.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append(
            dict(
                date=info["date"],
                decision_date=info["decision_date"],
                action=info["action"],
                position=info["position"],
                turnover=info["turnover"],
                cost=info["cost"],
                asset_return=info["asset_return"],
                reward=reward,
                value=info["value"],
                drawdown=info["drawdown"],
            )
        )
        done = terminated or truncated
    log = pd.DataFrame(rows).set_index("date")
    log.attrs.update(start_date=start_date, initial_value=init_value)
    return log


def equity_curve(log: pd.DataFrame) -> pd.Series:
    first = pd.Series([log.attrs["initial_value"]], index=[log.attrs["start_date"]])
    eq = pd.concat([first, log["value"]])
    eq.name = "equity"
    return eq


def evaluate_policy(env, policy, seed: int | None = None):
    log = run_episode(env, policy, seed)
    eq = equity_curve(log)
    return log, eq, compute_metrics(eq, log, env.initial_capital, env.periods_per_year)


# --------------------------------------------------------------------------- #
# Reportes
# --------------------------------------------------------------------------- #
KEY_COLS = ["cum_return", "ann_return", "ann_vol", "sharpe", "max_drawdown",
            "total_cost", "n_trades", "avg_position", "pct_days_cash"]


def _style(label: str, i: int):
    fixed = {
        "buy_and_hold": dict(color="black", lw=2.0),
        "random": dict(color="gray", lw=1.2, ls=":"),
        "constante_50": dict(color="tab:olive", lw=1.2, ls="--"),
        "sma50_trend": dict(color="tab:brown", lw=1.2, ls="--"),
    }
    return fixed.get(label, dict(color=f"C{i}", lw=1.8))


def _mean_band(curves: list[pd.Series]) -> pd.DataFrame:
    df = pd.concat(curves, axis=1)
    return pd.DataFrame({"mean": df.mean(axis=1), "lo": df.min(axis=1), "hi": df.max(axis=1)})


def _plot_curves(results: dict, kind: str, path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (label, runs) in enumerate(results.items()):
        curves = [r[1] if kind == "equity" else drawdown_series(r[1]) for r in runs]
        band = _mean_band(curves)
        st = _style(label, i)
        suffix = f" (media de {len(runs)})" if len(runs) > 1 else ""
        ax.plot(band.index, band["mean"], label=label + suffix, **st)
        if len(runs) > 1:
            ax.fill_between(band.index, band["lo"], band["hi"], color=st["color"], alpha=0.12)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_positions(results: dict, close: pd.Series, path: Path, symbol: str = "precio") -> None:
    agents = [k for k in results if k.startswith("dqn_")]
    if not agents:
        return
    fig, axes = plt.subplots(len(agents) + 1, 1, figsize=(11, 2.2 * (len(agents) + 1) + 1), sharex=True)
    axes[0].plot(close.index, close.values, color="black", lw=1)
    axes[0].set_ylabel(symbol)
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3)
    for ax, label in zip(axes[1:], agents):
        log = results[label][0][0]  # primera semilla
        ax.fill_between(log.index, log["position"], step="post", alpha=0.6, color="tab:blue")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("posición")
        ax.set_title(f"{label} (semilla 0)", fontsize=9, loc="left")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _diagnostics(summary: pd.DataFrame) -> list[str]:
    msgs = []
    for label, row in summary.iterrows():
        if not label.startswith("dqn_"):
            continue
        if row["pct_days_cash"] > 0.9:
            msgs.append(f"[{label}] pasa >90 % del tiempo fuera del mercado: posible política 'no hacer nada'.")
        if row["trades_per_year"] > 50:
            msgs.append(f"[{label}] opera {row['trades_per_year']:.0f} veces/año: posible sobre-operación (costos).")
        if row["avg_position"] > 0.95:
            msgs.append(f"[{label}] casi siempre 100 % invertido: se comporta como buy & hold.")
    return msgs


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def evaluate_cli(
    config: str,
    runs_dir: str | None,
    split: str,
    final: bool,
    out: str,
    n_random: int = 20,
    cost_bps: float | None = None,
) -> None:
    if split == "test" and not final:
        raise SystemExit(
            "El split 'test' es el held-out: úsalo UNA vez con el agente ya definido.\n"
            "Si es tu intención, agrega --final."
        )
    cfg = load_config(config)
        # Sobrescribe el costo para el análisis de sensibilidad.
    if cost_bps is not None:
        if cost_bps < 0:
            raise ValueError("--cost-bps debe ser >= 0")
        cfg["env"]["cost_bps"] = float(cost_bps)

    prices = load_prices(cfg["data"]["csv"])
    splits = build_splits(prices, cfg["data"])
    env = make_env(cfg, splits[split], training=False)
    obs_dim = env.observation_space.shape[0]

    results: dict[str, list] = {}
    for name, policies in make_baselines(env, n_random).items():
        results[name] = [evaluate_policy(env, p) for p in policies]

    if runs_dir:
        for model in sorted(Path(runs_dir).glob("*/seed_*/model.pt")):
            agent = DQNAgent.load(model)
            if agent.obs_dim != obs_dim:
                raise ValueError(f"{model}: obs_dim {agent.obs_dim} != {obs_dim} (¿features distintas?)")
            results.setdefault(f"dqn_{model.parts[-3]}", []).append(evaluate_policy(env, agent))

    out_dir = Path(out) / split
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    rows = []
    for label, runs in results.items():
        for k, (log, eq, m) in enumerate(runs):
            rows.append({"method": label, "run": k, **m})
            if label.startswith("dqn_") or k == 0:
                log.to_csv(out_dir / "logs" / f"{label}_{k}.csv")
    per_run = pd.DataFrame(rows)
    per_run.to_csv(out_dir / "metrics_per_run.csv", index=False)

    num = per_run.drop(columns=["run"]).groupby("method", sort=False)
    summary = num.mean(numeric_only=True)
    summary_std = num.std(numeric_only=True).add_suffix("_std")
    pd.concat([summary, summary_std], axis=1).to_csv(out_dir / "metrics_summary.csv")

    _plot_curves(results, "equity", out_dir / "equity_curves.png",
                 f"Curva de capital — {cfg['data'].get('symbol', '')} {split} (capital inicial {env.initial_capital:,.0f})", "Valor del portafolio")
    _plot_curves(results, "drawdown", out_dir / "drawdowns.png", f"Drawdown — {split}", "Drawdown")
    _plot_positions(results, splits[split].close, out_dir / "positions.png", cfg["data"].get("symbol", "precio"))

    show = summary[KEY_COLS].copy()
    for c in ["cum_return", "ann_return", "ann_vol", "max_drawdown", "avg_position", "pct_days_cash"]:
        show[c] = (show[c] * 100).round(1)
    show["total_cost"] = show["total_cost"].round(0)
    show["sharpe"] = show["sharpe"].round(2)
    show["n_trades"] = show["n_trades"].round(0)
    show.columns = ["ret_acum%", "ret_anual%", "vol%", "sharpe", "max_dd%", "costos$", "n_ops", "pos_media%", "dias_cash%"]
    print(f"\n=== Resultados en '{split}' ({splits[split].close.index[0].date()} → {splits[split].close.index[-1].date()}) ===")
    print(show.to_string())
    for m in _diagnostics(summary):
        print("  ⚠", m)
    print(f"\nArchivos guardados en {out_dir}/")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evalúa agentes DQN y baselines")
    ap.add_argument("--config", default="configs/base.yaml", help="define datos, features y costos de la evaluación")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true", help="necesario para evaluar sobre el held-out (test)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument(
    "--cost-bps",
    type=float,
    default=None,
    help="Sobrescribe el costo de transacción para todos los métodos",
)
    a = ap.parse_args()
    evaluate_cli(
    a.config, a.runs_dir, a.split, a.final, a.out,
    a.n_random, a.cost_bps
)


if __name__ == "__main__":
    main()
