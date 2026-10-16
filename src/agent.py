"""Agente DQN implementado desde cero en PyTorch.

Componentes: red Q (MLP), replay buffer, red objetivo (actualización dura),
política epsilon-greedy con decaimiento lineal y pérdida Huber.
Opcional: Double DQN (`agent.double_dqn: true`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Buffer circular con arreglos de numpy."""

    def __init__(self, capacity: int, obs_dim: int, seed: int = 0):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, done) -> None:
        i = self.idx
        self.obs[i], self.act[i], self.rew[i] = obs, action, reward
        self.next_obs[i], self.done[i] = next_obs, float(done)
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        ids = self.rng.integers(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.obs[ids]),
            torch.from_numpy(self.act[ids]).unsqueeze(1),
            torch.from_numpy(self.rew[ids]).unsqueeze(1),
            torch.from_numpy(self.next_obs[ids]),
            torch.from_numpy(self.done[ids]).unsqueeze(1),
        )

    def __len__(self) -> int:
        return self.size


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: dict, total_steps: int, seed: int = 0):
        self.obs_dim, self.n_actions, self.cfg = obs_dim, n_actions, cfg
        self.device = torch.device("cpu")  # redes pequeñas: CPU es suficiente y más reproducible
        self.q = QNetwork(obs_dim, n_actions, cfg["hidden"]).to(self.device)
        self.target = QNetwork(obs_dim, n_actions, cfg["hidden"]).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg["lr"])
        self.buffer = ReplayBuffer(cfg["buffer_size"], obs_dim, seed)
        self.rng = np.random.default_rng(seed + 12345)
        self.eps_decay_steps = max(1, int(cfg["eps_decay_frac"] * total_steps))
        self.updates = 0

    # --- política ------------------------------------------------------- #
    def epsilon(self, step: int) -> float:
        c = self.cfg
        frac = min(1.0, step / self.eps_decay_steps)
        return c["eps_start"] + frac * (c["eps_end"] - c["eps_start"])

    @torch.no_grad()
    def act(self, obs: np.ndarray, info: dict | None = None, epsilon: float = 0.0) -> int:
        """Epsilon-greedy. Con epsilon=0 es la política greedy (evaluación)."""
        if epsilon > 0.0 and self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.q(x).argmax(dim=1).item())

    # --- aprendizaje ---------------------------------------------------- #
    def update(self) -> float | None:
        c = self.cfg
        if len(self.buffer) < max(c["batch_size"], c["learning_starts"]):
            return None
        s, a, r, s2, d = self.buffer.sample(c["batch_size"])
        q_sa = self.q(s).gather(1, a)
        with torch.no_grad():
            if c["double_dqn"]:
                a2 = self.q(s2).argmax(dim=1, keepdim=True)
                q_next = self.target(s2).gather(1, a2)
            else:
                q_next = self.target(s2).max(dim=1, keepdim=True)[0]
            y = r + c["gamma"] * (1.0 - d) * q_next
        loss = F.smooth_l1_loss(q_sa, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), c["grad_clip"])
        self.opt.step()
        self.updates += 1
        if self.updates % c["target_update_every"] == 0:
            self.target.load_state_dict(self.q.state_dict())
        return float(loss.item())

    # --- persistencia --------------------------------------------------- #
    def save(self, path: str | Path, extra: dict | None = None) -> None:
        torch.save(
            {
                "state_dict": self.q.state_dict(),
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
                "hidden": self.cfg["hidden"],
                "extra": extra or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "DQNAgent":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = {
            "hidden": ckpt["hidden"], "lr": 1e-3, "buffer_size": 1, "eps_decay_frac": 1.0,
            "eps_start": 0.0, "eps_end": 0.0,
        }
        agent = cls(ckpt["obs_dim"], ckpt["n_actions"], cfg, total_steps=1)
        agent.q.load_state_dict(ckpt["state_dict"])
        agent.q.eval()
        return agent
