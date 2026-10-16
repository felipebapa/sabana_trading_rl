"""Pruebas de la capa de datos: formato del downloader, velas incompletas, huecos y descarga simulada."""
import pickle
from datetime import timezone

import numpy as np
import pandas as pd
import pytest

from src import data as D


def fake_downloader_frame(n=10, start="2024-01-01", timeframe="1d"):
    """Imita el DataFrame que guarda gym_trading_env.downloader (índice date_open + date_close)."""
    idx = pd.date_range(start, periods=n, freq=D.bar_delta(timeframe), name="date_open")
    close = 100 + np.arange(n, dtype=float)
    df = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 10.0}, index=idx
    )
    df["date_close"] = idx + D.bar_delta(timeframe)
    return df


def test_pickle_to_prices_format():
    p = D.pickle_to_prices(fake_downloader_frame(), now=pd.Timestamp("2030-01-01"))
    assert list(p.columns) == D.REQUIRED_COLS
    assert p.index.name == "timestamp" and len(p) == 10


def test_incomplete_last_candle_is_dropped():
    df = fake_downloader_frame(10, "2024-01-01")  # la última vela abre 2024-01-10 y cierra 2024-01-11
    p = D.pickle_to_prices(df, now=pd.Timestamp("2024-01-10 12:00"))
    assert p.index.max() == pd.Timestamp("2024-01-09")   # la del día 10 aún no cerró


def test_count_missing_bars():
    p = D.pickle_to_prices(fake_downloader_frame(10), now=pd.Timestamp("2030-01-01"))
    assert D.count_missing_bars(p, "1d") == 0
    assert D.count_missing_bars(p.drop(p.index[4]), "1d") == 1


def test_download_uses_utc_and_keeps_first_candle(tmp_path, monkeypatch):
    """Simula el downloader: comprueba since/until en UTC (independientes de la zona horaria)
    y que `since` queda justo antes de la primera vela pedida."""
    import sys
    import types

    seen = {}

    def fake_download(exchange_names, symbols, timeframe, dir, since, until):
        seen.update(since=since, until=until, exchange=exchange_names[0])
        fake_downloader_frame(5).to_pickle(f"{dir}/{exchange_names[0]}-{symbols[0].replace('/', '')}-{timeframe}.pkl")

    mod = types.ModuleType("gym_trading_env.downloader")
    mod.download = fake_download
    monkeypatch.setitem(sys.modules, "gym_trading_env", types.ModuleType("gym_trading_env"))
    monkeypatch.setitem(sys.modules, "gym_trading_env.downloader", mod)

    path = D.download_crypto("binance", "BTC/USDT", "1d", "2018-01-01", "2026-01-01", tmp_path)
    assert path.name == "binance-BTCUSDT-1d.pkl"
    assert seen["since"].tzinfo == timezone.utc and seen["until"].tzinfo == timezone.utc
    first_candle_ms = pd.Timestamp("2018-01-01", tz="UTC").timestamp() * 1e3
    assert seen["since"].timestamp() * 1e3 < first_candle_ms          # estrictamente antes
    assert first_candle_ms - seen["since"].timestamp() * 1e3 < 5     # pero solo por ~1 ms


def test_prepare_dataset_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "download_crypto", lambda *a, **k: _write_pkl(tmp_path))
    out = tmp_path / "x.csv"
    D.prepare_dataset("binance", "BTC/USDT", "1d", "2024-01-01", "2024-02-01", out, raw_dir=tmp_path)
    loaded = D.load_prices(out)
    assert list(loaded.columns) == D.REQUIRED_COLS and len(loaded) == 10


def _write_pkl(tmp_path):
    p = tmp_path / "b.pkl"
    fake_downloader_frame(10).to_pickle(p)
    return p
