"""Grafica métricas de validación registradas durante el entrenamiento.

Ejemplo:
    python scripts/plot_training.py runs/reward_v4/seed_0/train_log.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Ruta a train_log.csv")
    parser.add_argument("--out", default=None, help="PNG de salida (por defecto junto al CSV)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    required = {"step", "val_sharpe", "val_cum_return", "val_max_drawdown"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Faltan columnas en {csv_path}: {sorted(missing)}")

    out = Path(args.out) if args.out else csv_path.with_name("training_curves.png")
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(df["step"], df["val_sharpe"], marker="o")
    axes[0].set_ylabel("Sharpe (val)")
    axes[1].plot(df["step"], 100 * df["val_cum_return"], marker="o")
    axes[1].set_ylabel("Retorno acumulado val (%)")
    axes[2].plot(df["step"], 100 * df["val_max_drawdown"], marker="o")
    axes[2].set_ylabel("Máx. drawdown val (%)")
    axes[2].set_xlabel("Pasos de entrenamiento")

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Diagnóstico de entrenamiento (solo validación)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Gráfico guardado en: {out}")


if __name__ == "__main__":
    main()
