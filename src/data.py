"""Datos de cripto: descarga (downloader de gym-trading-env), carga, features causales
y partición temporal.

lookahead: cada feature en la fila t se calcula solo con datos hasta t (ventanas rolling/ewm hacia atrás). La normalización (z-score) usa únicamente media y desviación del período de ENTRENAMIENTO.

CONVENCIÓN DE TIEMPO: el índice de cada fila es la hora de apertura de la vela (UTC), igual que `date_open` en el downloader. La vela cierra una vela después, y el agente decide al cierre de esa vela (con `close` ya conocido).

CSV esperado:  timestamp, open, high, low, close, volume
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]
WARMUP = 50          # filas descartadas al inicio: la ventana más larga es SMA50
CLIP = 5.0           # los z-scores se recortan a [-5, 5]


# --------------------------------------------------------------------------- #
# Descarga con el downloader de gym-trading-env (ccxt)
# --------------------------------------------------------------------------- #
def _utc(date_str: str) -> datetime:
    """Fecha 'YYYY-MM-DD' como datetime en UTC.

    El downloader llama a `.timestamp()` sobre el datetime recibido: con un datetime
    "ingenuo" lo interpreta en la zona horaria de TU computador y el resultado cambiaría
    de una máquina a otra. Con tzinfo=UTC es determinista.
    """
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def download_crypto(exchange: str, symbol: str, timeframe: str, start: str, end: str,
                    raw_dir: str | Path = "data/raw") -> Path:
    """Descarga velas OHLCV y devuelve la ruta del .pickle crudo del downloader."""
    try:
        from gym_trading_env.downloader import download
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Falta gym-trading-env para descargar datos. Instalar con:  pip install -e \".[data]\""
        ) from e

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    # El downloader descarta la vela cuyo timestamp es igual a `since`
    # (usa `since < timestamp_open`). Restamos 1 ms para conservar la vela de `start`.
    since = _utc(start) - timedelta(milliseconds=1)
    until = _utc(end)  # exclusivo
    download(
        exchange_names=[exchange],
        symbols=[symbol],
        timeframe=timeframe,
        dir=str(raw_dir),
        since=since,
        until=until,
    )
    path = raw_dir / f"{exchange}-{symbol.replace('/', '')}-{timeframe}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"El downloader no generó {path}")
    return path


def pickle_to_prices(df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Convierte el DataFrame del downloader (índice `date_open`, columna `date_close`)
    al formato del proyecto, descartando la última vela si aún no ha cerrado."""
    df = df.copy()
    if "date_open" in df.columns:
        df = df.set_index("date_open")
    df.index = pd.to_datetime(df.index)
    df.index.name = "timestamp"
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en los datos descargados: {sorted(missing)}")

    if "date_close" in df.columns:
        now = pd.Timestamp.now(tz="UTC").tz_localize(None) if now is None else now
        df = df[pd.to_datetime(df["date_close"]) <= now]  # vela incompleta => su close no es definitivo

    df = df[REQUIRED_COLS].astype(float)
    df = df[~df.index.duplicated(keep="first")].sort_index().dropna()
    return df[df["close"] > 0]


def bar_delta(timeframe: str) -> pd.Timedelta:
    """'1d' -> Timedelta de 1 día (formato de ccxt: m, h, d, w)."""
    units = {"m": "min", "h": "h", "d": "D", "w": "W"}
    return pd.Timedelta(int(timeframe[:-1]), units[timeframe[-1]])


def count_missing_bars(prices: pd.DataFrame, timeframe: str) -> int:
    """Velas que faltan entre la primera y la última (cripto opera 24/7: debería ser ~0)."""
    expected = pd.date_range(prices.index.min(), prices.index.max(), freq=bar_delta(timeframe))
    return int(len(expected.difference(prices.index)))


def prepare_dataset(exchange: str, symbol: str, timeframe: str, start: str, end: str,
                    out: str | Path, raw_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Descarga -> convierte -> guarda el CSV del proyecto (y muestra su hash SHA-256)."""
    pkl = download_crypto(exchange, symbol, timeframe, start, end, raw_dir)
    # Solo se leen pickles generados por nosotros (pickle no es seguro con archivos ajenos).
    prices = pickle_to_prices(pd.read_pickle(pkl))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    gaps = count_missing_bars(prices, timeframe)
    print(f"{symbol} ({exchange}, {timeframe}): {len(prices)} velas, "
          f"{prices.index.min()} -> {prices.index.max()}  guardado en {out}")
    print(f"SHA-256 del CSV: {sha}")
    if gaps:
        print(f"AVISO: faltan {gaps} velas entre la primera y la última (huecos del exchange).")
    return prices


# --------------------------------------------------------------------------- #
# Carga
# --------------------------------------------------------------------------- #
def load_prices(path: str | Path) -> pd.DataFrame:
    """Lee un CSV `timestamp,open,high,low,close,volume` y lo valida."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = {"timestamp", *REQUIRED_COLS} - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {path}: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df = (
        df.sort_values("timestamp")
        .drop_duplicates("timestamp")
        .set_index("timestamp")[REQUIRED_COLS]
        .astype(float)
        .dropna()
    )
    if (df["close"] <= 0).any():
        raise ValueError("Hay precios de cierre <= 0")
    return df


def synthetic_prices(n: int = 3300, seed: int = 0, start: str = "2017-08-17") -> pd.DataFrame:
    """Serie sintética tipo cripto (velas diarias 24/7, volatilidad alta con regímenes).
    Solo para tests y pruebas de humo."""
    rng = np.random.default_rng(seed)
    vol = np.where((np.arange(n) // 200) % 3 == 2, 0.05, 0.03)
    rets = rng.normal(0.0008, vol)
    close = 100 * np.exp(np.cumsum(rets))
    idx = pd.date_range(start, periods=n, freq="D", name="timestamp")
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.002, n)),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(10_000, 60_000, n).astype(float),
        },
        index=idx,
    )


# --------------------------------------------------------------------------- #
# Features (todas causales; ventanas en velas)
# --------------------------------------------------------------------------- #
def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """RSI de Wilder (suavizado exponencial hacia atrás => causal)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def compute_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Calcula todas las features disponibles (sin normalizar).

    Con velas diarias, una ventana de N velas = N días. Como cripto opera 24/7,
    7 y 30 velas son una semana y un mes calendario.

    Justificación de cada variable:
      ret_1       retorno logarítmico de 1 vela: momentum / reversión de muy corto plazo.
      ret_7       retorno de 7 velas (~1 semana): momentum de corto plazo.
      ret_30      retorno de 30 velas (~1 mes): momentum de mediano plazo y fase del ciclo.
      dist_sma20  precio / SMA20 - 1: cuán extendido está el precio frente a su tendencia corta.
      dist_sma50  precio / SMA50 - 1: régimen alcista/bajista (referencia clásica de tendencia).
      vol_20      desviación estándar de retornos de 20 velas: régimen de riesgo; en cripto la
                  volatilidad se agrupa en rachas y las grandes caídas ocurren con volatilidad alta.
      rsi_14      RSI reescalado a [-1, 1]: sobrecompra / sobreventa.
      vol_ratio   log(volumen / media 20 velas del volumen): interés/estrés del mercado.
    """
    close = prices["close"]
    volume = prices["volume"].replace(0.0, np.nan)
    logret = np.log(close).diff()

    f = pd.DataFrame(index=prices.index)
    f["ret_1"] = logret
    f["ret_7"] = np.log(close / close.shift(7))
    f["ret_30"] = np.log(close / close.shift(30))
    f["dist_sma20"] = close / close.rolling(20).mean() - 1.0
    f["dist_sma50"] = close / close.rolling(50).mean() - 1.0
    f["vol_20"] = logret.rolling(20).std()
    f["rsi_14"] = (_rsi(close, 14) - 50.0) / 50.0
    f["vol_ratio"] = np.log(volume / volume.rolling(20).mean())
    f = f.replace([np.inf, -np.inf], np.nan)
    # Descarta el calentamiento inicial; huecos posteriores se rellenan solo hacia
    # adelante (valor pasado => causal).
    return f.iloc[WARMUP:].ffill().fillna(0.0)


# --------------------------------------------------------------------------- #
# Partición temporal + normalización solo con train
# --------------------------------------------------------------------------- #
@dataclass
class Split:
    name: str
    close: pd.Series            # precio de cierre alineado con las features
    features: pd.DataFrame      # features normalizadas (lo que ve el agente)
    raw_features: pd.DataFrame  # features sin normalizar (para baselines y análisis)

    @property
    def feature_names(self) -> list[str]:
        return list(self.features.columns)

    def __len__(self) -> int:
        return len(self.close)


def build_splits(prices: pd.DataFrame, data_cfg: dict) -> dict[str, Split]:
    """Devuelve {'train','val','test'} -> Split, con normalización ajustada solo en train."""
    names = list(data_cfg["features"])
    raw_all = compute_features(prices)[names]
    close_all = prices["close"].reindex(raw_all.index)

    bounds = {k: (pd.Timestamp(v[0]), pd.Timestamp(v[1]) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1))
              for k, v in data_cfg["splits"].items()}
    order = ["train", "val", "test"]
    for a, b in zip(order[:-1], order[1:]):
        if not bounds[a][1] < bounds[b][0]:
            raise ValueError(f"Los períodos '{a}' y '{b}' se solapan o no están en orden")

    tr0, tr1 = bounds["train"]
    raw_train = raw_all.loc[tr0:tr1]
    if raw_train.empty:
        raise ValueError("El período de entrenamiento no tiene datos")
    mean = raw_train.mean()
    std = raw_train.std().replace(0.0, 1.0)

    splits: dict[str, Split] = {}
    for name in order:
        s, e = bounds[name]
        raw = raw_all.loc[s:e]
        if len(raw) < 30:
            raise ValueError(f"El período '{name}' tiene muy pocos datos ({len(raw)} filas)")
        norm = ((raw - mean) / std).clip(-CLIP, CLIP)
        splits[name] = Split(name, close_all.loc[raw.index], norm, raw)
    return splits


# --------------------------------------------------------------------------- #
# CLI:  python -m src.data      (usa data.download de configs/base.yaml)
# --------------------------------------------------------------------------- #
def main() -> None:
    from src.utils import load_config

    ap = argparse.ArgumentParser(description="Descarga velas de cripto con el downloader de gym-trading-env")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--exchange")
    ap.add_argument("--symbol")
    ap.add_argument("--timeframe")
    ap.add_argument("--start")
    ap.add_argument("--end", help="exclusivo")
    ap.add_argument("--out")
    ap.add_argument("--raw-dir", default="data/raw")
    a = ap.parse_args()

    cfg = load_config(a.config)["data"]
    dl = cfg["download"]
    prepare_dataset(
        exchange=a.exchange or dl["exchange"],
        symbol=a.symbol or cfg["symbol"],
        timeframe=a.timeframe or dl["timeframe"],
        start=a.start or str(dl["start"]),
        end=a.end or str(dl["end"]),
        out=a.out or cfg["csv"],
        raw_dir=a.raw_dir,
    )


if __name__ == "__main__":
    main()
