"""Métricas de desempeño de una curva de capital."""
from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Drawdown en cada instante (valor <= 0): equity / máximo previo - 1."""
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown_series(equity).min())


def compute_metrics(equity: pd.Series, log: pd.DataFrame, initial_capital: float,
                    periods_per_year: float = 365.0) -> dict:
    """`equity` incluye el valor inicial; `log` es el registro por paso de run_episode."""
    rets = equity.pct_change().dropna()
    n = len(rets)
    cum = float(equity.iloc[-1] / initial_capital - 1.0)
    years = max(n / periods_per_year, 1e-9)
    ann_ret = float((equity.iloc[-1] / initial_capital) ** (1 / years) - 1.0)
    vol = float(rets.std(ddof=1) * np.sqrt(periods_per_year)) if n > 1 else 0.0
    sharpe = float(rets.mean() / rets.std(ddof=1) * np.sqrt(periods_per_year)) if rets.std(ddof=1) > 0 else 0.0
    total_cost = float(log["cost"].sum())
    return {
        "final_value": float(equity.iloc[-1]),
        "cum_return": cum,
        "ann_return": ann_ret,
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(equity),
        "total_cost": total_cost,
        "cost_pct_capital": total_cost / initial_capital,
        "n_trades": int((log["turnover"] > 1e-12).sum()),
        "trades_per_year": float((log["turnover"] > 1e-12).sum() / years),
        "avg_position": float(log["position"].mean()),
        "pct_days_cash": float((log["position"] == 0).mean()),
        "pct_days_full": float((log["position"] >= 0.999).mean()),
    }
