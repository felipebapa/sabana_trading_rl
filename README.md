# sabana_trading_rl - Agente inversor con Reinforcement Learning (DQN) en cripto

Un agente DQN aprende cuánto capital exponer a **BTC/USDT** (0 %, 50 % o 100 %) a partir de velas históricas
diarias, y se compara contra estrategias simples (comprar y mantener, azar, exposición constante, tendencia SMA50)
en un período **held-out** que no vio durante el entrenamiento.

```
Observar el mercado → Tomar una acción → El precio cambia → Recompensa (neta de costos) → Volver a decidir
```

Los datos se descargan con el downloader de [Gym Trading Env](https://gym-trading-env.readthedocs.io/en/latest/download.html)
(ccxt, solo cripto). El **entorno de simulación, el agente DQN y la evaluación son propios** (`src/`).

## Estructura

```
.
├── configs/            base.yaml + reward_v1/v2/v3.yaml (las versiones de la recompensa)
├── data/               BTCUSDT_1d.csv (procesado, se versiona) y data/raw/ (pickle crudo, no se versiona)
├── notebooks/eda.ipynb análisis exploratorio
├── src/
│   ├── data.py         descarga, conversión, features causales, partición temporal, normalización solo con train
│   ├── environment.py  entorno Gymnasium lookahead-safe (InvestmentEnv)
│   ├── agent.py        DQN desde cero (red Q, replay buffer, red objetivo, epsilon-greedy)
│   ├── train.py        entrenamiento + validación + early stopping + checkpoints
│   ├── evaluate.py     evaluación agente vs baselines, métricas, gráficos
│   ├── baselines.py    azar, buy & hold, 50 % constante, tendencia SMA50
│   ├── metrics.py      retorno, drawdown, Sharpe, costos, etc.
│   └── utils.py        semillas y carga de configuración
├── tests/              anti-lookahead, entorno, datos (formato del downloader) y agente
├── pyproject.toml
└── README.md
```

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: source .venv/Scripts/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu    # opcional: torch liviano (CPU)
python -m pip install -e ".[dev]"
pytest -q
```

Python ≥ 3.10 (probado con 3.12). El código corre en CPU. En Windows, mantén el proyecto en una ruta corta
(p. ej. `C:\dev\sabana_trading_rl`) y fuera de OneDrive para evitar errores de rutas largas.

## Datos

Dataset: **BTC/USDT**, velas diarias (`1d`) de **Binance**, 2017-08-01 → 2025-12-31, descargadas con
`gym_trading_env.downloader`. La librería solo cubre cripto: los datos de acciones aparecen como "coming soon" en su documentación.

```bash
python -m pip install -e ".[data]"     # instala gym-trading-env (fija ccxt==3.0.59 y flask==2.2.5)
python -m src.data                     # lee data.download de configs/base.yaml
```

Esto genera `data/raw/binance-BTCUSDT-1d.pkl` (crudo) y `data/BTCUSDT_1d.csv` con el formato del proyecto:

```
timestamp,open,high,low,close,volume
```

`timestamp` es la hora de **apertura** de la vela en UTC. El comando imprime el **hash SHA-256 del CSV**, el número
de velas y si hay huecos. Versiona el CSV para que cualquiera reproduzca exactamente el mismo experimento (si ya lo
tienes, no necesitas instalar `.[data]`).

Detalles del downloader que el código ya maneja:

* **Zona horaria.** El downloader interpreta `since`/`until` en la zona horaria de tu computador; aquí se pasan en UTC explícito, así el resultado no depende de la máquina.
* **Primera vela.** Descarta la vela cuyo timestamp es igual a `since`; se compensa restando 1 ms.
* **Vela incompleta.** Si pides datos hasta hoy, la última vela puede no haber cerrado; se descarta si `date_close` es futuro.
* **Exchange bloqueado.** Si Binance no responde en tu país, cambia `data.download.exchange` a `bitfinex2` o `huobi` en `configs/base.yaml`.

Otro activo o timeframe: cambia `symbol`, `download.timeframe` y `env.periods_per_year` (1h → 8760; las ventanas de las features están en velas).

### Partición temporal (`configs/base.yaml`)

| Período | Fechas | Régimen | Uso |
|---|---|---|---|
| train | 2018 – 2021 | mercado bajista 2018, lateral 2019, caída y auge 2020-21 | el agente aprende |
| val | 2022 – 2023 | caída de 2022 y recuperación | hiperparámetros, versión de la recompensa, early stopping |
| **test (held-out)** | 2024 – 2025 | auge posterior a 2023 | evaluación final, **una sola vez** |

`evaluate --split test` exige `--final` para evitar usar el held-out por accidente.
Los datos de 2017 solo sirven de calentamiento para los indicadores.

## Ejecutar

```bash
# 1) Entrenar cada versión de la recompensa (5 semillas por defecto, ver train.seeds)
python -m src.train --config configs/reward_v1.yaml
python -m src.train --config configs/reward_v2.yaml
python -m src.train --config configs/reward_v3.yaml
python -m src.train --config configs/reward_v4.yaml
#    prueba rápida:  python -m src.train --config configs/reward_v1.yaml --seeds 0 --total-steps 5000

# 2) Evaluar en VALIDACIÓN (aquí se decide qué recompensa/hiperparámetros usar)
python -m src.evaluate --runs-dir runs --split val

# 3) Solo baselines
python -m src.baselines --split val

# 4) Evaluación final sobre el held-out (una vez, con el agente ya definido)
python -m src.evaluate --runs-dir runs --split test --final
```

Salidas en `runs/<config>/seed_<n>/` (`model.pt`, `train_log.csv`, `config.yaml`, `summary.json`) y en
`results/<split>/`: `metrics_summary.csv`, `metrics_per_run.csv`, `equity_curves.png`, `drawdowns.png`,
`positions.png` y `logs/*.csv` (posición/acción/costo por vela, útil para buscar comportamientos anómalos).

## Diseño

### Cronología de una decisión (lookahead-safe)

En el **cierre** de la vela *t* (su `close` ya es conocido):

1. El agente observa `[features_t, posición_actual]` (solo datos hasta *t*).
2. Elige la posición objetivo `w_t ∈ {0, 0.5, 1}`; se ejecuta al precio de cierre de *t* y paga `|w_t − w_{t−1}| · V_t · cost_bps/10⁴`.
3. El precio pasa de `close_t` a `close_{t+1}`: `V_{t+1} = (V_t − costo) · (1 + w_t · r_{t+1} + (1 − w_t) · rf)`.
4. Recompensa y nueva observación `obs_{t+1}`.

`close_{t+1}` solo se usa para calcular el resultado de la acción, nunca para decidirla. El episodio siempre
empieza en efectivo (0 %), así que buy & hold también paga el costo de entrada. El fin de los datos/ventana es
`truncated` (el DQN hace bootstrap); `terminated` solo ocurriría con capital 0.

Este timing coincide con el de `TradingEnv` de Gym Trading Env (la acción se ejecuta al precio de la posición actual y luego el índice avanza).
Se mantiene un entorno propio porque el enunciado pide construirlo y porque así los costos, la recompensa y las pruebas anti-lookahead quedan bajo control.

### Estado

`[ret_1, ret_7, ret_30, dist_sma20, dist_sma50, vol_20, rsi_14, vol_ratio, posición_actual]` (9 dimensiones).
Cada feature está definida y justificada en la docstring de `compute_features` (`src/data.py`). Como cripto opera 24/7, 7 y 30 velas son una semana y un mes calendario.

Control de lookahead:
* Todas las features son ventanas hacia atrás (rolling/ewm); las primeras 50 velas se descartan (calentamiento).
* Los indicadores se calculan sobre la serie completa, de modo que val/test usen el historial previo real (disponible en ese momento).
* La normalización (z-score, recorte a ±5) usa **solo** media y desviación del período train.
* `tests/test_lookahead.py` lo verifica: truncar la serie no cambia las features pasadas, alterar el futuro no cambia el pasado, y alterar val/test no cambia el train normalizado.

### Acciones

`{0 %, 50 %, 100 %}` → `Discrete(3)`. Solo largo, sin apalancamiento ni posición corta; el cash rinde `rf_annual` (0 por defecto).
Con tres niveles hay pocos parámetros que aprender y el costo por cambio es fácil de interpretar; no puede
vender en corto, apalancarse ni ajustar la posición con más finura.

### Costos

10 bps (0,10 %) sobre el capital negociado en cada cambio de posición, equivalente a la comisión spot típica de Binance.
No incluye deslizamiento (*slippage*) ni impacto de mercado.

### Recompensa (iteración)

```
R_{t+1} = scale · [ log(V_{t+1}/V_t) − risk_aversion · log_ret² − dd_coef · max(0, Δdrawdown) − turnover_penalty · |Δw| ]
```

| Versión | Config | Idea |
|---|---|---|
| V1 | `reward_v1.yaml` | solo retorno logarítmico neto de costos (ingenua) |
| V2 | `reward_v2.yaml` | V1 + penalización al aumento del drawdown (`dd_coef = 1.0`) |
| V3 | `reward_v3.yaml` | V2 + penalización por rotación (`turnover_penalty = 0.002`) |

Flujo: entrenar → mirar `positions.png` y el diagnóstico impreso por `evaluate` en **val** (fuera del mercado, sobre-operación, imita buy & hold) →
ajustar los coeficientes → reentrenar → comparar. Los coeficientes por defecto son un punto de partida, no un resultado: cámbialos solo con validación y documenta cada cambio.

### DQN y hiperparámetros (`configs/base.yaml`)

| Parámetro | Valor |
|---|---|
| Red | MLP 9 → 64 → 64 → 3, ReLU |
| Optimizador / pérdida | Adam, lr 5e-4 / Huber |
| γ | 0.99 |
| Batch / buffer | 64 / 50 000 |
| Pasos | 100 000 (episodios de 365 velas con inicio aleatorio) |
| Exploración | ε-greedy 1.0 → 0.05, lineal en el 50 % de los pasos |
| Red objetivo | copia dura cada 500 actualizaciones |
| Double DQN | desactivado (`agent.double_dqn`) |
| Selección de modelo | mejor `val_sharpe`, evaluado cada 5 000 pasos |
| Criterio de parada | paciencia de 8 evaluaciones sin mejora, no antes de 30 000 pasos |
| Semillas | 0–4 (`train.seeds`) |

## Reproducibilidad

* Semillas fijadas para `random`, `numpy` y `torch`; el entorno y el replay buffer usan la misma semilla; entrenamiento en CPU.
* `config.yaml` se guarda junto a cada modelo en `runs/`; el hash SHA-256 del CSV identifica los datos exactos.
* El agente aleatorio se promedia sobre 20 semillas (`--n-random`).
* Con las mismas versiones de librerías y el mismo CSV, una semilla da el mismo resultado; entre plataformas distintas puede haber pequeñas diferencias numéricas.

## Limitaciones conocidas

* Ejecución al precio de cierre de la misma vela, sin deslizamiento ni impacto de mercado.
* Un solo activo y un único par train/test: los resultados dependen del período. Con velas diarias, train tiene ~1 460 velas, pocas para una red neuronal: vigila el sobreajuste con la validación.
* BTC/USDT es muy volátil (drawdowns de buy & hold de −60 % o más), lo que cambia el balance riesgo/retorno frente a un índice bursátil: un agente que iguala el retorno de buy & hold con menor drawdown ya es un resultado válido.
* El downloader fija versiones antiguas de `ccxt` y `flask`. Instálalo en el mismo entorno solo si necesitas descargar datos.
