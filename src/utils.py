"""Utilidades: semillas y carga de configuración con herencia (`extends`)."""
from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import yaml


def set_seed(seed: int) -> None:
    """Fija todas las fuentes de aleatoriedad relevantes."""
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _deep_update(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    """Carga un YAML. Si tiene `extends: otro.yaml`, hereda y sobrescribe campos."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    parent = cfg.pop("extends", None)
    if parent:
        base = load_config(path.parent / parent)
        cfg = _deep_update(base, cfg)
    return cfg
