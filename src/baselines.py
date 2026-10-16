"""Estrategias de referencia (sabana_trading_rl). Todas exponen `act(obs, info=None) -> int`
y se evalúan en el mismo entorno (mismos datos y mismos costos) que el agente."""
from __future__ import annotations

import argparse

import numpy as np


class RandomAgent:
    """Elige una acción uniformemente al azar en cada paso."""

    def __init__(self, n_actions: int, seed: int = 0):
        self.n, self.rng = n_actions, np.random.default_rng(seed)

    def act(self, obs, info=None) -> int:
        return int(self.rng.integers(self.n))


class BuyAndHold:
    """Compra el 100 % el primer día y mantiene (paga el costo de entrada una vez)."""

    def __init__(self, full_action: int):
        self.a = full_action

    def act(self, obs, info=None) -> int:
        return self.a


class ConstantExposure:
    """Mantiene siempre la misma exposición (p. ej. 50 %)."""

    def __init__(self, action: int):
        self.a = action

    def act(self, obs, info=None) -> int:
        return self.a


class SMATrend:
    """Seguimiento de tendencia: 100 % invertido si close > SMA50, si no 0 %.
    Referencia de una regla simple con feature técnica (usa solo datos hasta t)."""

    def __init__(self, feature_names: list[str], full_action: int):
        if "dist_sma50" not in feature_names:
            raise ValueError("SMATrend requiere la feature 'dist_sma50'")
        self.i, self.full = feature_names.index("dist_sma50"), full_action

    def act(self, obs, info=None) -> int:
        return self.full if info["features_raw"][self.i] > 0 else 0


def make_baselines(env, n_random: int = 20) -> dict[str, list]:
    """Devuelve {nombre: [políticas]}; el azar usa varias semillas."""
    full = len(env.actions) - 1
    half = int(np.argmin(np.abs(env.actions - 0.5)))
    out = {
        "buy_and_hold": [BuyAndHold(full)],
        "constante_50": [ConstantExposure(half)],
        "random": [RandomAgent(len(env.actions), seed=s) for s in range(n_random)],
    }
    if "dist_sma50" in env.feature_names:
        out["sma50_trend"] = [SMATrend(env.feature_names, full)]
    return out


def main() -> None:
    """Evalúa solo los baselines: python -m src.baselines --split val"""
    from src.evaluate import evaluate_cli

    ap = argparse.ArgumentParser(description="Evalúa los baselines")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true", help="necesario para usar el held-out (test)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    evaluate_cli(config=a.config, runs_dir=None, split=a.split, final=a.final, out=a.out)


if __name__ == "__main__":
    main()
