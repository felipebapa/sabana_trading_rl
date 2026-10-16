import numpy as np
import pytest

from src.agent import DQNAgent
from src.utils import load_config

AC = load_config("configs/base.yaml")["agent"]


def test_agent_update_and_act(tmp_path):
    agent = DQNAgent(obs_dim=5, n_actions=3, cfg={**AC, "learning_starts": 10}, total_steps=100, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        agent.buffer.add(rng.normal(size=5), int(rng.integers(3)), float(rng.normal()), rng.normal(size=5), False)
    losses = [agent.update() for _ in range(20)]
    assert all(l is not None and np.isfinite(l) for l in losses)
    assert agent.act(rng.normal(size=5)) in (0, 1, 2)
    # guardar y cargar da la misma política
    agent.save(tmp_path / "m.pt")
    loaded = DQNAgent.load(tmp_path / "m.pt")
    x = rng.normal(size=5)
    assert agent.act(x) == loaded.act(x)


def test_epsilon_schedule():
    agent = DQNAgent(5, 3, AC, total_steps=1000, seed=0)
    assert agent.epsilon(0) == pytest.approx(AC["eps_start"])
    assert agent.epsilon(10_000) == pytest.approx(AC["eps_end"])
