"""Curva de sobreajuste: val vs test a lo largo del entrenamiento.

Uso:
    python -m scripts.plot_overfitting --runs-dir runs --config-name reward_v4
    python -m scripts.plot_overfitting --runs-dir runs --config-name reward_v4 --metric sharpe
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


def plot_curves(logs: list[pd.DataFrame], metric: str, out: Path, title: str):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    val_col = f"val_{metric}"
    test_col = f"test_{metric}"

    # media ± rango entre semillas
    all_steps = sorted(set().union(*[set(df["step"]) for df in logs]))
    val_mat, test_mat = [], []
    for step in all_steps:
        vals = [df.loc[df["step"] == step, val_col].values[0]
                for df in logs if step in df["step"].values]
        tests = [df.loc[df["step"] == step, test_col].values[0]
                 for df in logs if step in df["step"].values]
        if vals:
            val_mat.append((step, sum(vals)/len(vals), min(vals), max(vals)))
            test_mat.append((step, sum(tests)/len(tests), min(tests), max(tests)))

    if not val_mat:
        print("No hay datos para graficar.")
        return

    steps, v_mean, v_lo, v_hi = zip(*val_mat)
    _, t_mean, t_lo, t_hi = zip(*test_mat)

    ax.plot(steps, v_mean, label=f"Val {metric}", color="C0", lw=2)
    ax.fill_between(steps, v_lo, v_hi, color="C0", alpha=0.15)
    ax.plot(steps, t_mean, label=f"Test {metric}", color="C1", lw=2)
    ax.fill_between(steps, t_lo, t_hi, color="C1", alpha=0.15)

    # marca el mejor step según validación (media)
    best_idx = max(range(len(v_mean)), key=lambda i: v_mean[i])
    ax.axvline(steps[best_idx], color="gray", ls="--", lw=1, label=f"Best val step ({steps[best_idx]})")

    ax.set_xlabel("Pasos de entrenamiento")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Guardado: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--config-name", required=True, help="ej: reward_v4")
    ap.add_argument("--metric", default="sharpe", choices=["sharpe", "cum_return", "max_drawdown"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    runs_dir = Path(a.runs_dir)
    logs = load_logs(runs_dir, a.config_name)
    if not logs:
        return

    out = Path(a.out) if a.out else runs_dir / a.config_name / f"overfitting_{a.metric}.png"
    plot_curves(
        logs,
        a.metric,
        out,
        title=f"Curva de sobreajuste — {a.config_name} ({a.metric})",
    )


if __name__ == "__main__":
    main()