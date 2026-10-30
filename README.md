# Miniaplicación de apoyo en ciberseguridad con Claude

Miniaplicación en Python que consume la API de Claude (Sonnet 5.5) mediante `requests` para apoyar a equipos técnicos de pequeñas empresas en el entendimiento y remediación de vulnerabilidades. Incluye un banco de prompts de SOC y una prueba que mide qué tan consistentes son las respuestas del modelo.

La elección del modelo está justificada en [docs/decision_arquitectura.md](docs/decision_arquitectura.md) y el diagrama en [docs/diagrama_arquitectura.png](docs/diagrama_arquitectura.png).

## Requisitos

- Python 3.14 (versión con la que se desarrolló; las dependencias están fijadas para ella)
- Una API key de Anthropic ([console.anthropic.com](https://console.anthropic.com))
- Opcional: una API key de Gemini, solo para `src/api_gemini.py`

## Instalación

```bash
git clone <url-del-repositorio>
cd <carpeta-del-repositorio>

python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

## Configuración

Copia la plantilla .env.example y completa tus claves (`.env` está en `.gitignore`, nunca lo subas al repositorio):

```bash
cp .env.example .env             # Linux / macOS
# copy .env.example .env         # Windows
```

| Variable | Obligatoria | Uso |
|---|---|---|
| `CLAUDE_API_KEY` | sí | API key de Anthropic |
| `GEMINI_API_KEY` | no | Solo para `src/api_gemini.py` |

## Ejecución

Todos los comandos se ejecutan desde la raíz del proyecto.

**Prueba de conexión con Claude** — envía un prompt corto y muestra la respuesta, el costo de la operación y el costo acumulado de la sesión:

```bash
python src/main.py
```

**Prueba de latencia** — conexión más 10 peticiones seguidas, con la latencia promedio:

```bash
python src/api_claude_refactor.py
```

**Prueba de consistencia** — envía cada prompt de `JSON/promp_bank.JSON` 3 veces y valida las respuestas:

```bash
PYTHONPATH=src python test/test_consistencia.py          # Linux / macOS
# $env:PYTHONPATH="src"; python test/test_consistencia.py   # Windows (PowerShell)
```

`PYTHONPATH=src` es necesario porque la prueba importa las funciones de `src/`. Ten en cuenta que hace 9 llamadas a la API (3 prompts × 3 intentos), con costo.

### Gemini (solo comparativo)

El modelo elegido para la miniaplicación es **Claude Sonnet 5.5**. `src/api_gemini.py` (modelo `gemini-3.5-flash-lite`; `gemini-3.1-pro-preview`) se mantiene únicamente para comparar resultados, costo y latencia contra Gemini; no forma parte del flujo principal. Requiere `GEMINI_API_KEY` en `.env`.

```bash
python src/api_gemini.py                                          # conexión + latencia
PYTHONPATH=src python test/test_consistencia_gemini.py            # consistencia (Linux / macOS)
# $env:PYTHONPATH="src"; python test/test_consistencia_gemini.py  # Windows (PowerShell)
```

La prueba usa `JSON/promp_bank_gemini.JSON` (mismos prompts, con el schema en formato Gemini) y guarda el resultado en `JSON/registro_resultados_gemini.json`, para comparar directamente con el de Claude.

### Costos

El costo mostrado usa los precios por millón de tokens registrados en [docs/decision_arquitectura.md](docs/decision_arquitectura.md):

| Modelo | Entrada | Salida |
|---|---|---|
| Claude Sonnet 5.5 | USD 2 | USD 10 |
| Gemini 3.1 Pro (prompts ≤ 200k tokens) | USD 2 | USD 12 |

Si los precios cambian, actualízalos en `INPUT_PRICING` y `OUTPUT_PRICING` de `src/api_claude_refactor.py` (Claude) y en `get_gemini_response` (Gemini).

## Reintentos y costo de la sesión

**Reintento con backoff exponencial**: `with_retry` espera 2 s y duplica la espera en cada intento (2, 4, 8...), con un tope de 30 s y un jitter de hasta 10 %. Solo se reintenta `TimeoutError` (red, 429, 5xx); un 401/403 falla de inmediato. Al agotar los intentos se lanza `RuntimeError`.

**Costo acumulado**: `registrar_costo` suma cada operación exitosa a `costo_total_sesion` e imprime `Operación: N tokens, costo estimado: $X (acumulado: $Y)`; `src/main.py` imprime `Costo acumulado de la sesión` al terminar. Una operación fallida no suma costo, y el acumulado se reinicia en cada ejecución. `registrar_costo` es seguro en hilos: protege el acumulado con un lock (`_lock_costo`), lo que importa en el modo concurrente del ensayo.

**Ensayo simulado de rate limit** — 10 peticiones con un 429 simulado en la petición 6; NO usa la red ni la API real (`requests.post` y `time.sleep` están mockeados):

```bash
python scripts/ensayo_rate_limit_simulado.py
```

El registro está en [docs/registro_rate_limit.md](docs/registro_rate_limit.md). Ese ensayo es simulado: no es un 429 real del proveedor. El ensayo real está en [docs/registro_rate_limit_real.md](docs/registro_rate_limit_real.md) y no alcanzó ningún 429 en 30 peticiones secuenciales.

## Pipeline de texto

`src/pipeline_texto.py` normaliza el texto antes de compararlo con una respuesta de referencia:

| Función | Qué hace |
|---|---|
| `limpiar_texto` | Pasa a minúsculas, quita la puntuación y colapsa los espacios; lanza `TypeError` si no recibe un `str` |
| `tokenizar` | Separa el texto limpio en palabras |
| `procesar_texto` | `limpiar_texto` + `tokenizar` |
| `vectorizar` | Cuenta las repeticiones de cada token (`Counter`) |
| `coincidencia_por_palabras` | Proporción de palabras distintas de la referencia presentes en la respuesta (0.0 si la referencia está vacía) |
| `comparar_antes_despues` | Coincidencia con el texto crudo (`split`) y con el texto procesado |

**Límites de la métrica**: no distingue el orden de las palabras ("el perro persiguió al gato" y "el gato persiguió al perro" coinciden al 100 %) ni las repeticiones, porque `coincidencia_por_palabras` sigue comparando conjuntos de palabras distintas (las repeticiones sí se cuentan en [Métricas de texto](#métricas-de-texto-precisión-cobertura-y-f1)); los signos de apertura (`¿`, `¡`) se eliminan con el resto de la puntuación, y los decimales pierden el punto: `9.0` pasa a `90`.

**Pipeline sobre salidas del cliente** — envía las 3 consultas de `JSON/casos_pipeline.json` a Claude y compara cada respuesta con su referencia, antes y después de procesar el texto:

```bash
python scripts/ejecutar_pipeline.py --confirmar-api-real
```

Hace 3 llamadas **REALES** a la API de Claude, con un costo aproximado de USD 0.005. El tope de gasto por defecto es `--tope-usd 0.05`; sin `--confirmar-api-real` el script sale con código 2 sin tocar la red. Escribe el registro en `docs/registro_pipeline.md`.

**Resultados de la corrida real** (2026-10-28, 3 casos, costo total USD 0.00354; detalle en [docs/registro_pipeline.md](docs/registro_pipeline.md)):

| Caso | Coincidencia texto crudo | Coincidencia texto procesado |
|---|---|---|
| 1 | 77.3 % | 81.0 % |
| 2 | 45.0 % | 63.2 % |
| 3 | 50.0 % | 66.7 % |

El texto procesado mejora la coincidencia en los 3 casos. Ningún caso llega a 100 % porque las respuestas del modelo son más largas que la referencia y usan sinónimos.

## Métricas de texto (precisión, cobertura y F1)

`src/metricas_texto.py` compara los tokens de la respuesta con los de la referencia contando las repeticiones (intersección de `Counter`):

| Función | Qué hace |
|---|---|
| `contar_coincidencias` | Tokens compartidos, cada uno hasta el mínimo de veces que aparece en ambos lados; lanza `TypeError` si recibe algo que no sea una lista |
| `calcular_precision_cobertura_f1` | Precisión (coincidencias / tokens de la respuesta), cobertura (coincidencias / tokens de la referencia) y F1; 0.0 si alguna lista está vacía |
| `calcular_metricas` | `procesar_texto` a ambos textos + `calcular_precision_cobertura_f1` |
| `resumir_metricas` | Promedio simple de precisión, cobertura y F1 (0.0 con lista vacía) |
| `ordenar_peores_primero` | Lista nueva ordenada por F1 ascendente |
| `seleccionar_muestra_revision` | Los n peores y los n mejores por F1 (n = 5 por defecto), sin duplicados |

**Ejemplo**: referencia "el gato duerme en el sofa" (6 tokens) y respuesta "el gato duerme en el sillon todo el dia" (9 tokens) comparten 5 tokens ("el" cuenta 2 veces, no 3): precisión 5/9 = 0.56, cobertura 5/6 = 0.83 y F1 0.67.

**Límites de la métrica**:

- **Sinónimos**: "el auto es veloz" contra "el vehículo es rápido" da F1 0.5, aunque digan lo mismo.
- **Dato equivocado**: "python fue creado en 1991" contra "python fue creado en 1989" da F1 0.8, aunque el año esté mal.
- **Negación ausente**: "python no es un lenguaje compilado" contra "python es un lenguaje compilado" da F1 0.909, aunque la respuesta afirme lo contrario.

**Métricas sobre salidas del cliente** — envía las consultas de `JSON/casos_metricas.json` (20 casos de ciberseguridad) a Claude y calcula precisión, cobertura y F1 de cada respuesta:

```bash
python scripts/ejecutar_metricas.py --confirmar-api-real
```

- Hace llamadas **REALES** a la API de Claude, con costo.
- `--confirmar-api-real` es obligatorio: sin él el script sale con código 2 sin tocar la red.
- `--tope-usd` es US$ 0.05 por defecto y como máximo; un valor mayor o menor o igual a 0 sale con código 1.
- Acepta como máximo 25 casos; con más sale con código 1 antes de llamar a la API.
- Escribe los resultados en `docs/resultados_metricas.json` (`--salida` lo cambia) y reescribe el archivo después de cada caso, para no perder lo hecho si algo falla a mitad.
- Aún no hay resultados de una corrida real.

## Ensayo real de rate limit

Envía N peticiones seguidas a Claude pidiendo una respuesta de una sola palabra, para intentar provocar un 429 real del proveedor. Registra la petición del primer 429, las esperas del backoff, las llamadas HTTP y el costo:

```bash
python scripts/ensayo_rate_limit_real.py --confirmar-api-real
python scripts/ensayo_rate_limit_real.py --confirmar-api-real --concurrencia 10 --n 60
```

- `--confirmar-api-real` es obligatorio: sin él el script sale con código 2 sin tocar la red.
- `--n` fija el número de peticiones (por defecto 30; máximo 30 en modo secuencial y 100 en modo concurrente).
- `--concurrencia K` (por defecto 1 = secuencial; máximo 20) envía las peticiones en olas de K peticiones simultáneas; cada ola espera a que terminen todas antes de lanzar la siguiente. El tope de gasto se comprueba antes de cada ola, así que el sobregiro máximo posible es una ola completa.
- El modo concurrente (`--concurrencia` mayor que 1) escribe por defecto en `docs/registro_rate_limit_real_concurrente.md`, sin pisar el registro secuencial; `--salida` cambia el archivo.
- `--tope-usd 0.05` es el tope de gasto por defecto; al alcanzarlo el ensayo se corta.
- El costo esperado es muy bajo (respuestas de ~1 token).
- Puede NO aparecer ningún 429; en ese caso el registro lo dice así.

**Resultado del ensayo secuencial real** (2026-10-28): 30 peticiones en 33.6 s, 30 llamadas HTTP, 0 respuestas 429, 0 esperas y costo total USD 0.00318. No se alcanzó ningún 429 en 30 peticiones secuenciales (unas 54 peticiones por minuto); detalle en [docs/registro_rate_limit_real.md](docs/registro_rate_limit_real.md).

**Resultado del ensayo concurrente real** (2026-10-28): 100 peticiones en 5 olas de 20 simultáneas, 100 llamadas HTTP, 0 respuestas 429, 0 esperas, 0 fallos definitivos, duración 10.7 s (unas 560 peticiones por minuto) y costo total USD 0.01060. No se alcanzó ningún 429 en 100 peticiones; detalle en [docs/registro_rate_limit_real_concurrente.md](docs/registro_rate_limit_real_concurrente.md). Con la cuota actual de la cuenta no se pudo observar un 429 real, así que el comportamiento del backoff queda documentado por el ensayo simulado ([docs/registro_rate_limit.md](docs/registro_rate_limit.md)).

Genera `docs/registro_rate_limit_real.md` (secuencial) o `docs/registro_rate_limit_real_concurrente.md` (concurrente); no pisa [docs/registro_rate_limit.md](docs/registro_rate_limit.md), que es el ensayo simulado.

## Pruebas unitarias

Desde la raíz del proyecto:

```bash
.venv/bin/python -m pytest -v
# python -m pytest -v            # con el entorno virtual activo
```

No usan la API real ni la red (todo está mockeado) y corren en menos de 2 segundos. `pytest.ini` limita la suite a `test/unit`; la prueba de consistencia sigue siendo el script aparte con `PYTHONPATH=src` y con costo (9 llamadas).

## Prueba de consistencia

Cada intento se valida en este orden y se registra el primer fallo:

1. **JSON inválido**: la respuesta no se puede leer como JSON.
2. **Claves faltantes**: falta alguna de `claves_esperadas`.
3. **Valor incorrecto**: algún valor no coincide con `respuesta_esperada` (las listas se comparan sin importar el orden).

El resultado queda en `JSON/registro_resultados.json`, con éxitos, intentos y el motivo de cada fallo por prompt.

### Agregar un prompt al banco

Cada entrada de `JSON/promp_bank.JSON` tiene esta forma:

| Campo | Obligatorio | Descripción |
|---|---|---|
| `nombre` | sí | Identificador del prompt en el registro |
| `system` | sí | Instrucción de sistema enviada a Claude |
| `ejemplo_usuario` | sí | Mensaje del usuario |
| `claves_esperadas` | sí | Claves que debe traer el JSON de respuesta |
| `respuesta_esperada` | no | Valores exactos a comprobar (solo los objetivos) |
| `output_schema` | no | JSON Schema para salidas estructuradas; requiere `"additionalProperties": false` |

Usa datos de prueba: dominios `.test` e IPs de documentación (`192.0.2.x`, `198.51.100.x`).

## Estructura

```
pytest.ini                 limita pytest a test/unit
src/
  api_claude_refactor.py   cliente de Claude: arma la petición, la envía con reintento, calcula y acumula el costo
  api_gemini.py            cliente de Gemini, solo comparativo (alternativa descartada, ver docs)
  main.py                  prueba de conexión
  pipeline_texto.py        limpieza, tokenización y coincidencia por palabras
  metricas_texto.py        precisión, cobertura y F1 sobre tokens, contando repeticiones
scripts/
  ensayo_rate_limit_simulado.py  ensayo de rate limit simulado con mocks
  ejecutar_pipeline.py           pipeline de texto sobre respuestas reales de Claude (con costo)
  ejecutar_metricas.py           precisión, cobertura y F1 sobre respuestas reales de Claude (con costo)
  ensayo_rate_limit_real.py      ensayo de rate limit real contra Claude (con costo)
test/
  unit/test_api_claude_refactor.py     pruebas unitarias con mocks
  unit/test_api_gemini.py              pruebas minimas del manejo de errores del cliente Gemini, con mocks
  unit/test_pipeline_texto.py          pruebas del pipeline de texto
  unit/test_ejecutar_pipeline.py       pruebas del script del pipeline, con mocks
  unit/test_metricas_texto.py          pruebas de las métricas de texto
  unit/test_ejecutar_metricas.py       pruebas del script de métricas, con mocks
  unit/test_ensayo_rate_limit_real.py  pruebas del ensayo real, con mocks
  test_consistencia.py         prueba de consistencia con Claude
  test_consistencia_gemini.py  misma prueba con Gemini (comparativa)
JSON/
  promp_bank.JSON                  banco de prompts para Claude
  promp_bank_gemini.JSON           banco de prompts para Gemini
  registro_resultados.json         resultado de la última prueba con Claude
  registro_resultados_gemini.json  resultado de la última prueba con Gemini
  casos_pipeline.json              consultas y referencias del pipeline de texto
  casos_metricas.json              consultas y referencias de las métricas de texto
docs/                      decisión de arquitectura, diagrama y evidencias
  registro_rate_limit.md   registro del ensayo de rate limit (simulado)
  registro_pipeline.md     registro del pipeline de texto (se genera al correr el script)
  registro_rate_limit_real.md  registro del ensayo de rate limit real (se genera al correr el ensayo)
  registro_rate_limit_real_concurrente.md  registro del ensayo concurrente (se genera al correr el ensayo con --concurrencia mayor que 1)
```

## Problemas comunes

| Error | Causa probable |
|---|---|
| `Credencial inválida o sin permiso` | La API key (`CLAUDE_API_KEY` o `GEMINI_API_KEY`) falta o es inválida en `.env`, o el modelo no está disponible para tu clave (401/403) |
| `Ocurrio un error de conexión: 400` | Petición mal formada, revisa el `output_schema` del prompt |
| `ModuleNotFoundError: No module named 'api_claude_refactor'` | Falta `PYTHONPATH=src` al ejecutar la prueba de consistencia |
