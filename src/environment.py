"""Entorno de simulación de inversión (Gymnasium).

CONVENCIÓN TEMPORAL (lookahead-safe)
------------------------------------
Sea t el índice de una vela (la fila t es la vela que abrió en dates[t] y cierra una vela después).
La decisión se toma en el CIERRE de la vela t, cuando su `close` ya es conocido.

    1. El agente OBSERVA  obs_t = [features_t, posición_actual]
       (features_t usa solo datos hasta el cierre de t).
    2. El agente DECIDE   a_t -> posición objetivo w_t ∈ actions.
       La operación se ejecuta al precio de cierre de t (supuesto simplificador).
       Costo:  cost_t = V_t * |w_t - w_{t-1}| * cost_bps/1e4
    3. El precio CAMBIA   close_t -> close_{t+1}:
       V_{t+1} = (V_t - cost_t) * (1 + w_t * r_{t+1} + (1 - w_t) * rf_por_vela)
       con r_{t+1} = close_{t+1}/close_t - 1.
    4. RECOMPENSA         R_{t+1} = scale * [ log(V_{t+1}/V_t)            (neto de costos)
                                              - risk_aversion * log_ret^2
                                              - dd_coef * max(0, dd_{t+1} - dd_t)
                                              - turnover_penalty * |w_t - w_{t-1}| ]
    5. Nueva observación obs_{t+1}.

close_{t+1} SOLO se usa para calcular el resultado de la acción (paso 3),
nunca forma parte de la observación con la que se decidió.

El fin de los datos / del episodio se señala con `truncated` (no es un estado
terminal del problema). `terminated` solo ocurre si el capital llega a 0.
"""
from __future__ import annotations

from typing import Sequence

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class InvestmentEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        close: pd.Series,
        features: pd.DataFrame,
        raw_features: pd.DataFrame | None = None,
        actions: Sequence[float] = (0.0, 0.5, 1.0),
        initial_capital: float = 100_000.0,
        cost_bps: float = 10.0,
        rf_annual: float = 0.0,
        periods_per_year: float = 365.0,
        reward_cfg: dict | None = None,
        episode_length: int | None = None,
        random_start: bool = False,
    ):
        super().__init__()
        if len(close) != len(features):
            raise ValueError("close y features deben tener la misma longitud")
        self.dates = pd.DatetimeIndex(close.index)
        self.close = close.to_numpy(dtype=np.float64)
        self.features = features.to_numpy(dtype=np.float32)
        self.raw_features = (
            raw_features.to_numpy(dtype=np.float32) if raw_features is not None else self.features
        )
        self.feature_names = list(features.columns)

        self.actions = np.asarray(actions, dtype=np.float64)
        self.initial_capital = float(initial_capital)
        self.cost_rate = float(cost_bps) / 1e4
        self.periods_per_year = float(periods_per_year)
        self.rf_step = (1.0 + float(rf_annual)) ** (1.0 / self.periods_per_year) - 1.0

        rc = reward_cfg or {}
        self.r_scale = float(rc.get("scale", 1.0))
        self.r_risk = float(rc.get("risk_aversion", 0.0))
        self.r_dd = float(rc.get("dd_coef", 0.0))
        self.r_turn = float(rc.get("turnover_penalty", 0.0))

        n = len(self.close)
        max_len = n - 1  # cada paso necesita close[t+1]
        self.episode_length = max_len if not episode_length else min(int(episode_length), max_len)
        self.random_start = bool(random_start)

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.features.shape[1] + 1,), dtype=np.float32
        )

    # ------------------------------------------------------------------ #
    def _obs(self) -> np.ndarray:
        return np.concatenate([self.features[self.t], [self.position]]).astype(np.float32)

    def _info(self, **extra) -> dict:
        info = dict(
            date=self.dates[self.t],
            value=self.value,
            position=self.position,
            drawdown=1.0 - self.value / self.peak,
            cum_cost=self.cum_cost,
            features_raw=self.raw_features[self.t].copy(),  # solo datos hasta t
        )
        info.update(extra)
        return info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        n = len(self.close)
        if self.random_start:
            self.t = int(self.np_random.integers(0, n - self.episode_length))
        else:
            self.t = 0
        self.t_end = self.t + self.episode_length
        self.steps = 0
        self.value = self.initial_capital
        self.peak = self.value
        self.position = 0.0          # siempre se comienza en efectivo
        self.cum_cost = 0.0
        return self._obs(), self._info()

    def step(self, action: int):
        w = float(self.actions[int(action)])
        prev_w = self.position
        turnover = abs(w - prev_w)

        # (2) costo de transacción sobre el capital negociado, ejecutado en close_t
        cost = self.value * turnover * self.cost_rate
        v_after_cost = self.value - cost

        # (3) el precio se mueve de t a t+1
        asset_ret = self.close[self.t + 1] / self.close[self.t] - 1.0
        port_ret = w * asset_ret + (1.0 - w) * self.rf_step
        new_value = v_after_cost * (1.0 + port_ret)

        # (4) recompensa
        log_ret = float(np.log(max(new_value, 1e-12) / self.value))
        old_dd = 1.0 - self.value / self.peak
        new_peak = max(self.peak, new_value)
        new_dd = 1.0 - new_value / new_peak
        reward = (
            log_ret
            - self.r_risk * log_ret**2
            - self.r_dd * max(0.0, new_dd - old_dd)
            - self.r_turn * turnover
        ) * self.r_scale

        # avanzar
        prev_value = self.value
        self.value, self.peak = new_value, new_peak
        self.position = w
        self.cum_cost += cost
        self.t += 1
        self.steps += 1

        terminated = bool(self.value <= 0.0)
        truncated = bool(self.steps >= self.episode_length or self.t >= len(self.close) - 1)
        info = self._info(
            decision_date=self.dates[self.t - 1],
            action=int(action),
            turnover=turnover,
            cost=cost,
            asset_return=asset_ret,
            portfolio_return=new_value / prev_value - 1.0,
            log_return=log_ret,
        )
        return self._obs(), float(reward), terminated, truncated, info


def make_env(cfg: dict, split, training: bool = False) -> InvestmentEnv:
    """Construye el entorno de un Split a partir de la configuración."""
    e = cfg["env"]
    return InvestmentEnv(
        close=split.close,
        features=split.features,
        raw_features=split.raw_features,
        actions=e["actions"],
        initial_capital=e["initial_capital"],
        cost_bps=e["cost_bps"],
        rf_annual=e.get("rf_annual", 0.0),
        periods_per_year=e.get("periods_per_year", 365.0),
        reward_cfg=cfg["reward"],
        episode_length=e.get("train_episode_length") if training else None,
        random_start=training,
    )
