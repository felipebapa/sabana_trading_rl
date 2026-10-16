import numpy as np
import pandas as pd
import pytest

from src.environment import InvestmentEnv


def make(closes, **kw):
    idx = pd.bdate_range("2020-01-01", periods=len(closes))
    close = pd.Series(closes, index=idx, dtype=float)
    feats = pd.DataFrame({"f": 0.1 * np.arange(len(closes), dtype=float)}, index=idx)
    return InvestmentEnv(close, feats, **kw)


def test_timing_and_value_update():
    env = make([100, 110, 121, 133.1], cost_bps=10, initial_capital=100_000)
    obs, info = env.reset()
    assert info["value"] == 100_000 and obs[-1] == 0.0
    # decisión en t=0: 100 % invertido; resultado usa close[1]/close[0]
    obs, r, term, trunc, info = env.step(2)
    expected = 100_000 * (1 - 0.001) * 1.10
    assert info["value"] == pytest.approx(expected)
    assert info["cost"] == pytest.approx(100.0)
    assert obs[0] == pytest.approx(0.1)  # observación en t=1 -> features[1]
    assert obs[-1] == 1.0  # posición actual


def test_cost_only_when_position_changes():
    env = make([100, 100, 100, 100, 100], cost_bps=10)
    env.reset()
    _, _, _, _, i1 = env.step(2)   # 0 -> 1 : cuesta
    _, _, _, _, i2 = env.step(2)   # 1 -> 1 : gratis
    _, _, _, _, i3 = env.step(0)   # 1 -> 0 : cuesta
    assert i1["cost"] > 0 and i2["cost"] == 0 and i3["cost"] > 0
    assert i3["cum_cost"] == pytest.approx(i1["cost"] + i3["cost"])


def test_half_position_return():
    env = make([100, 110, 110], cost_bps=0)
    env.reset()
    _, _, _, _, info = env.step(1)  # 50 %
    assert info["value"] == pytest.approx(100_000 * (1 + 0.5 * 0.10))


def test_reward_is_net_log_return_scaled():
    env = make([100, 110, 121], cost_bps=10, reward_cfg={"scale": 100.0})
    env.reset()
    _, r, *_ = env.step(2)
    assert r == pytest.approx(100.0 * np.log(0.999 * 1.10))


def test_drawdown_and_turnover_penalties():
    env = make([100, 90, 90], cost_bps=0, reward_cfg={"dd_coef": 2.0, "turnover_penalty": 0.01})
    env.reset()
    _, r, *_ = env.step(2)  # cae 10 % -> dd = 0.1
    assert r == pytest.approx(np.log(0.9) - 2.0 * 0.10 - 0.01 * 1.0)


def test_episode_ends_with_truncation_at_end_of_data():
    env = make([100, 101, 102, 103, 104])
    env.reset()
    steps, done = 0, False
    while not done:
        _, _, term, trunc, _ = env.step(0)
        done = term or trunc
        steps += 1
    assert steps == 4 and trunc and not term


def test_random_start_respects_episode_length():
    env = make(list(range(100, 200)), episode_length=20, random_start=True)
    for seed in range(10):
        env.reset(seed=seed)
        steps, done = 0, False
        while not done:
            _, _, term, trunc, _ = env.step(1)
            done = term or trunc
            steps += 1
        assert steps == 20


def test_gymnasium_api_check():
    from gymnasium.utils.env_checker import check_env

    check_env(make(list(np.linspace(100, 150, 50))), skip_render_check=True)


def test_cash_earns_risk_free_per_bar():
    env = make([100, 100, 100], cost_bps=0, rf_annual=0.365, periods_per_year=365)
    env.reset()
    _, _, _, _, info = env.step(0)  # se queda en efectivo
    assert info["value"] == pytest.approx(100_000 * (1.365 ** (1 / 365)))
