"""Curva de sobreajuste: val vs test de varias versiones.

Uso:
    python -m scripts.plot_overfitting --runs-dir runs --configs reward_v1 reward_v2 reward_v3 reward_v4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_logs(runs_dir: Path, config_name: str) -> list[pd.DataFrame]:
    logs = []
    for p in sorted((runs_dir / config_name).glob("seed_*/train_log.csv")):
        df = pd.read_csv(p)
        if "test_sharpe" not in df.columns:
            print(f"  ⚠ {p} no tiene columnas test_*. Reentrena con el train.py modificado.")
            continue
        df["seed"] = int(p.parent.name.split("_")[1])
        logs.append(df)
    return logs


def mean_curve(logs: list[pd.DataFrame], col: str):
    """Devuelve (steps, mean, lo, hi) de una columna."""
    all_steps = sorted(set().union(*[set(df["step"]) for df in logs]))
    out = []
    for step in all_steps:
        vals = [df.loc[df["step"] == step, col].values[0]
                for df in logs if step in df["step"].values]
        if vals:
            out.append((step, sum(vals)/len(vals), min(vals), max(vals)))
    if not out:
        return None
    return zip(*out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--configs", nargs="+", required=True,
                    help="ej: reward_v1 reward_v2 reward_v3 reward_v4")
    ap.add_argument("--metric", default="sharpe", choices=["sharpe", "cum_return", "max_drawdown"])
    ap.add_argument("--out", default="results/overfitting_comparison.png")
    a = ap.parse_args()

    runs_dir = Path(a.runs_dir)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    colors = plt.cm.tab10.colors

    for i, name in enumerate(a.configs):
        logs = load_logs(runs_dir, name)
        if not logs:
            continue

        val = mean_curve(logs, f"val_{a.metric}")
        test = mean_curve(logs, f"test_{a.metric}")
        if val is None or test is None:
            continue

        steps_v, v_mean, v_lo, v_hi = val
        steps_t, t_mean, t_lo, t_hi = test
        c = colors[i % len(colors)]

        # Panel izquierdo: validación
        axes[0].plot(steps_v, v_mean, label=name, color=c, lw=2)
        axes[0].fill_between(steps_v, v_lo, v_hi, color=c, alpha=0.12)

        # Panel derecho: test
        axes[1].plot(steps_t, t_mean, label=name, color=c, lw=2)
        axes[1].fill_between(steps_t, t_lo, t_hi, color=c, alpha=0.12)

    axes[0].set_title(f"Validación — {a.metric}")
    axes[1].set_title(f"Test (held-out) — {a.metric}")
    for ax in axes:
        ax.set_xlabel("Pasos de entrenamiento")
        ax.set_ylabel(a.metric)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Curva de sobreajuste: Val vs Test", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Guardado: {out}")


if __name__ == "__main__":
    main()