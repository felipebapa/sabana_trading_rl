# Sabana Trading RL

## Agente inversor DQN para BTC/USDT

Proyecto académico de Reinforcement Learning que entrena un agente DQN para seleccionar una exposición objetivo de **0 %, 50 % o 100 %** a BTC/USDT con datos diarios históricos. El proyecto incluye un entorno de simulación, cuatro variantes de recompensa, baselines, pruebas y reportes reproducibles.

> **Alcance:** esto es una simulación educativa, no una recomendación de inversión ni una estrategia validada para operar dinero real.

## Pregunta del proyecto

¿Puede un agente de RL aprender una política que generalice a datos posteriores no usados para entrenar, y que aporte valor frente a estrategias simples una vez considerados riesgo y costos?

La respuesta debe juzgarse comparando retorno, Sharpe, drawdown, costos y estabilidad entre semillas; no solo por el retorno acumulado.

## Estructura

```text
.
├── configs/             # configuración base y variantes reward_v1...reward_v4
├── data/
│   └── BTCUSDT_1d.csv   # dataset procesado versionado
├── notebooks/
│   └── eda.ipynb        # análisis exploratorio
├── results/             # resultados y gráficos guardados para consulta
├── src/
│   ├── data.py          # carga, features causales y partición temporal
│   ├── environment.py   # entorno de inversión compatible con Gymnasium
│   ├── agent.py         # DQN, replay buffer y red objetivo
│   ├── train.py         # entrenamiento: train + selección en val, nunca test
│   ├── evaluate.py      # evaluación de agentes y baselines
│   ├── baselines.py     # aleatorio, buy & hold, 50 % y SMA50
│   ├── metrics.py       # retorno, Sharpe, drawdown, costos
│   └── utils.py         # configuración y semillas
├── tests/
├── scripts/
│   └── plot_training.py # curvas de validación durante el entrenamiento
├── pyproject.toml
└── README.md
```

## Instalación

Requiere Python >= 3.10. En Windows se puede usar Git Bash.

```bash
python -m venv .venv
# Git Bash en Windows:
source .venv/Scripts/activate
# macOS/Linux:
# source .venv/bin/activate

python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
pytest -q
```

Si ya tienes PyTorch instalado y funcional, no necesitas reinstalarlo.

## Datos y partición temporal

El dataset del proyecto es BTC/USDT, velas diarias OHLCV de Binance, en `data/BTCUSDT_1d.csv`. El archivo procesado se incluye para que no sea obligatorio volver a descargarlo. El código de descarga está en `src/data.py` y el extra opcional es:

```bash
python -m pip install -e ".[data]"
python -m src.data
```

La descarga requiere conexión al exchange y puede variar con el tiempo; para reproducir resultados publicados, usa el CSV versionado y verifica su SHA-256.

La configuración actual define:

| Split | Período | Uso |
|---|---|---|
| `train` | 2018–2021 | Aprendizaje del agente |
| `val` | 2022–2023 | Selección de checkpoint, early stopping y diseño de recompensa |
| `test` | 2024–2025 | Evaluación final solo después de fijar el agente |

Las fechas se configuran en `configs/base.yaml`. Las particiones son cronológicas y no deben solaparse. Las features se calculan con ventanas retrospectivas y el escalador se ajusta solo con `train`.

### Advertencia sobre el test publicado anteriormente

Las versiones anteriores del código evaluaban el split `test` repetidamente durante el entrenamiento, y las métricas de ese período se consultaron al analizar y comparar las versiones del agente. Por tanto, **2024–2025 ya no debe describirse como un held-out completamente intacto**. La corrección de `src/train.py` impide nuevas consultas accidentales, pero no puede deshacer el acceso histórico.

Para una afirmación fuerte de generalización, define un período futuro que no se consulte durante el desarrollo o utiliza un protocolo walk-forward con ventanas temporales predefinidas. No selecciones hiperparámetros con los resultados del período final.

## Ejecutar los experimentos

### 1. Entrenar

El entrenamiento solo usa `train`. La selección del mejor checkpoint y el early stopping utilizan `val`. El archivo `src/train.py` no construye ni evalúa el entorno de test.

```bash
python -m src.train --config configs/reward_v1.yaml
python -m src.train --config configs/reward_v2.yaml
python -m src.train --config configs/reward_v3.yaml
python -m src.train --config configs/reward_v4.yaml
```

Prueba rápida:

```bash
python -m src.train --config configs/reward_v4.yaml --seeds 0 --total-steps 5000
```

Cada ejecución guarda `model.pt`, `train_log.csv`, `config.yaml`, `summary.json` y checkpoints intermedios bajo `runs/<config>/seed_<n>/`.

### 2. Graficar el proceso de entrenamiento

Esto muestra las métricas de validación durante los checkpoints. No grafica el test ni debe usarse para cambiar decisiones después de consultar el período final.

```bash
python scripts/plot_training.py runs/reward_v4/seed_0/train_log.csv
```

La pérdida de DQN es un diagnóstico del ajuste de la red, pero por sí sola no demuestra que la política sea rentable o generalice.

### 3. Comparar en validación

```bash
python -m src.evaluate --runs-dir runs --split val --out results
python -m src.baselines --split val
```

La validación se puede consultar durante el desarrollo, pero la selección repetida sobre ella puede producir sobreajuste a validación. Registra qué cambios se hicieron y por qué.

### 4. Sensibilidad a costos de transacción

Aplica el mismo costo a todos los agentes y baselines. Primero agrega el pequeño cambio descrito en `patches/evaluate_cost_sensitivity.patch` a `src/evaluate.py`.

```bash
python -m src.evaluate --runs-dir runs --split val --out results_cost10 --cost-bps 10
python -m src.evaluate --runs-dir runs --split val --out results_cost20 --cost-bps 20
python -m src.evaluate --runs-dir runs --split val --out results_cost30 --cost-bps 30
```

Usa estas pruebas para entender la sensibilidad, no para escoger una configuración a partir del test. Las salidas se separan por carpeta para no sobrescribir los resultados.

### 5. Evaluación final

Solo después de fijar versión de recompensa, hiperparámetros, semillas y modelo:

```bash
python -m src.evaluate --runs-dir runs --split test --final --out results_final
```

Ejecuta esto una vez para el análisis final acordado. La bandera `--final` es una protección contra errores, no una garantía matemática de que el período nunca se haya visto.

## Formulación del entorno

### Estado

El estado incluye indicadores causales calculados con datos disponibles hasta el momento de decisión:

- retornos logarítmicos de 1, 7 y 30 velas;
- distancia del precio a SMA20 y SMA50;
- volatilidad de 20 velas;
- RSI de 14 velas;
- ratio de volumen;
- posición actual.

La posición se incluye porque la acción económica depende de la exposición previa: cambiar de 0 % a 100 % cuesta más que mantener la posición.

### Acciones

`{0.0, 0.5, 1.0}` significa mantener 0 %, 50 % o 100 % del capital expuesto al activo. El agente no opera en corto, no usa apalancamiento y no puede seleccionar exposiciones intermedias.

### Cronología y limitación de ejecución

El entorno actual observa las features de la vela `t`, decide una posición objetivo y simula el retorno desde `close_t` hasta `close_{t+1}`. La operación se carga al `close_t`. El cierre futuro se usa para calcular el resultado de la acción, no para construir la observación.

Esto evita el uso directo del retorno futuro en la decisión, pero **ejecutar al cierre que acaba de utilizarse para decidir es un supuesto simplificador**. En mercados reales el precio de ejecución puede diferir. Antes de presentar resultados como simulación realista, una mejora es ejecutar en `open_{t+1}` y calcular el rendimiento desde ese precio, con pruebas que garanticen la alineación temporal.

### Costos y recompensa

El costo base es 10 bps (0,10 %) sobre el capital negociado al cambiar la exposición. No incluye slippage ni impacto de mercado. La recompensa combina retorno logarítmico neto de costos con penalizaciones configurables por riesgo, aumento de drawdown y rotación.

| Variante | Cambio principal |
|---|---|
| V1 | Retorno logarítmico neto de costos |
| V2 | Penalización por aumento de drawdown |
| V3 | Penalización pequeña por rotación |
| V4 | Penalización más fuerte por rotación |

Las penalizaciones son elecciones de diseño; una recompensa mayor no implica necesariamente mejor rendimiento financiero. Por eso se comparan las métricas financieras y las posiciones, no solo la recompensa acumulada.

## DQN y reproducibilidad

Los hiperparámetros están en `configs/base.yaml` y sus configuraciones derivadas. El agente usa una red MLP, replay buffer, red objetivo y exploración epsilon-greedy. Las semillas de Python/NumPy/PyTorch se fijan cuando es posible y la configuración se guarda junto al modelo.

Para reproducir:

1. Usa la misma versión del CSV procesado.
2. Instala las dependencias del proyecto.
3. Ejecuta las pruebas con `pytest -q`.
4. Entrena con la configuración y semillas documentadas.
5. Evalúa primero en validación.
6. Mantén el período final fuera de la selección.

## Métricas y limitaciones

Reporta al menos retorno acumulado, Sharpe, máximo drawdown, costos, número de operaciones, curva de capital y evolución de las posiciones. Presenta resultados por semilla además de media y desviación estándar.

Limitaciones conocidas:

- un solo activo y un histórico limitado;
- los regímenes del mercado pueden cambiar;
- las semillas pueden producir políticas muy distintas;
- la validación puede sobreajustarse por probar muchas variantes;
- la ejecución actual usa el cierre como supuesto simplificador;
- no se modelan slippage, impacto de mercado, impuestos, spreads variables ni restricciones de liquidez;
- un resultado histórico positivo no garantiza rentabilidad futura.

## Resultados

Los gráficos y CSV en `results/` son artefactos de análisis que se pueden inspeccionar sin volver a entrenar. Si cambias el código, las features, los costos o las fechas, vuelve a generar las métricas y marca claramente qué versión del experimento produjo cada archivo. No presentes métricas históricas generadas por el protocolo anterior como una evaluación independiente nueva.

## Licencia y uso

Proyecto académico y educativo. No constituye asesoría financiera. El trading de criptoactivos implica riesgo elevado de pérdida de capital.
