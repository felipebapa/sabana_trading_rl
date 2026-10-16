"""Auditoría anti-lookahead: ninguna feature en t puede depender de datos posteriores a t,
y la normalización debe ajustarse solo con el período de entrenamiento."""
import numpy as np
import pandas as pd

from src.data import build_splits, compute_features, synthetic_prices
from src.utils import load_config

CFG = load_config("configs/base.yaml")["data"]


def test_features_are_causal():
    """Calcular con la serie truncada en k da lo mismo que con la serie completa."""
    prices = synthetic_prices(n=1200, seed=1)
    full = compute_features(prices)
    k = 800
    part = compute_features(prices.iloc[:k])
    pd.testing.assert_frame_equal(part, full.loc[part.index], check_exact=False, rtol=1e-10, atol=1e-12)


def test_changing_the_future_does_not_change_the_past():
    prices = synthetic_prices(n=1200, seed=2)
    k = 700
    altered = prices.copy()
    altered.iloc[k:, :4] *= 3.0          # precios futuros totalmente distintos
    altered.iloc[k:, 4] *= 10.0          # y volumen también
    a, b = compute_features(prices), compute_features(altered)
    past = a.index[a.index < prices.index[k]]
    pd.testing.assert_frame_equal(a.loc[past], b.loc[past])


def test_normalization_uses_train_only():
    """Si alteramos val/test, las features de train NO cambian (media/std solo de train)."""
    prices = synthetic_prices(n=3300, seed=3)
    base = build_splits(prices, CFG)
    altered = prices.copy()
    altered.loc["2022-01-01":, ["open", "high", "low", "close"]] *= 5.0
    alt = build_splits(altered, CFG)
    pd.testing.assert_frame_equal(base["train"].features, alt["train"].features)


def test_splits_are_disjoint_and_ordered():
    s = build_splits(synthetic_prices(n=3300, seed=4), CFG)
    assert s["train"].close.index.max() < s["val"].close.index.min()
    assert s["val"].close.index.max() < s["test"].close.index.min()
