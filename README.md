# Miniaplicación de apoyo en ciberseguridad con Claude

Miniaplicación en Python que consume la API de Claude (Sonnet 5.5) mediante `requests` para apoyar a equipos técnicos de pequeñas empresas en el entendimiento y remediación de vulnerabilidades. Incluye un banco de prompts de SOC y una prueba que mide qué tan consistentes son las respuestas del modelo.

La elección del modelo está justificada en [docs/decision_arquitectura.md](docs/decision_arquitectura.md) y el diagrama en [docs/diagrama_arquitectura.png](docs/diagrama_arquitectura.png).

## Requisitos

- Python 3.14 (versión con la que se desarrolló; las dependencias están fijadas para ella)
- Una API key de Anthropic ([console.anthropic.com](https://console.anthropic.com))
- Opcional: una API key de Gemini, solo para `src/api_gemini.py` y `scripts/ejecutar_cadenas.py --proveedor gemini`
- El agente usa `langchain-classic` (`AgentExecutor`), fijado en `requirements.txt`

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
| `GEMINI_API_KEY` | no | Solo para `src/api_gemini.py` y `scripts/ejecutar_cadenas.py --proveedor gemini` |

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
| Gemini 3.5 Flash-Lite (informados por el usuario, no verificados contra la documentación oficial ni registrados en ese documento) | USD 0.30 | USD 2.50 |

Si los precios cambian, actualízalos en `INPUT_PRICING` y `OUTPUT_PRICING` de `src/api_claude_refactor.py` (Claude), en `get_gemini_response` (Gemini) y en `PRECIOS_GEMINI` de `scripts/ejecutar_cadenas.py` (Gemini 3.5 Flash-Lite en las cadenas).

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

**Límites de la métrica**: no distingue el orden de las palabras ("el perro persiguió al gato" y "el gato persiguió al perro" coinciden al 100 %) ni las repeticiones, porque `coincidencia_por_palabras` sigue comparando conjuntos de palabras distintas (las repeticiones sí se cuentan en [Métricas de texto](#métricas-de-texto-precisión-cobertura-y-f1)); los signos de apertura (`¿`, `¡`) se eliminan con el resto de la puntuación, y los decimales pierden el punto: `9.0` pasa a `90`. Los guiones también se borran y unen lo que separaban: `CVE-2021-44228` pasa a `cve202144228` y `AES-256` a `aes256`.

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
- **Guiones**: la limpieza borra los guiones, así que un identificador como `CVE-2021-44228` queda como un solo token (`cve202144228`): con un dígito distinto cuenta entero como fallado. Además "AES 256" (con espacio) no coincide con "AES-256": "AES-256 usa una clave de 256 bits." contra "AES 256 usa clave de 256 bits" da F1 0.71 (5/7).

**Métricas sobre salidas del cliente** — envía las consultas de `JSON/casos_metricas.json` (20 casos de ciberseguridad) a Claude y calcula precisión, cobertura y F1 de cada respuesta:

```bash
python scripts/ejecutar_metricas.py --confirmar-api-real
```

- Hace llamadas **REALES** a la API de Claude, con costo.
- `--confirmar-api-real` es obligatorio: sin él el script sale con código 2 sin tocar la red.
- `--tope-usd` es US$ 0.05 por defecto y como máximo; un valor mayor o menor o igual a 0 sale con código 1.
- Acepta como máximo 25 casos; con más sale con código 1 antes de llamar a la API.
- Escribe los resultados en `docs/resultados_metricas.json` (`--salida` lo cambia) y reescribe el archivo después de cada caso, para no perder lo hecho si algo falla a mitad.

**Resultados de la corrida real** (2026-10-30, 20 casos, costo estimado USD 0.018 con una llamada adicional por una consulta que se enfocó; detalle en [docs/resultados_metricas.json](docs/resultados_metricas.json)): precisión promedio 0.40, cobertura promedio 0.80 y F1 promedio 0.52. Las 20 respuestas resultaron correctas y aun así 19 tienen precisión menor que cobertura: la métrica penaliza la longitud, no los errores. La revisión manual de 10 casos (5 peores y 5 mejores, con rúbrica de corrección factual, completitud y contradicción) está en [docs/revision_manual.json](docs/revision_manual.json) y lo que la métrica no puede decir, con ejemplos, en [docs/limites_metrica.md](docs/limites_metrica.md).

## LangChain: chain de tres pasos

`src/cadena_tres_pasos.py` encadena tres prompts con `langchain-anthropic`. Convierte un aviso de seguridad en una checklist de remediación para alguien sin conocimientos técnicos:

| Paso | Entrada | Responsabilidad |
|---|---|---|
| 1 | `{texto}` | Extraer los 3 puntos clave, uno por línea, sin desarrollarlos |
| 2 | `{puntos}` | Reescribirlos en lenguaje simple |
| 3 | `{texto_simple}` | Formatearlos como checklist en viñetas markdown, máximo 3 líneas |

`construir_cadena(modelo)` devuelve la chain completa sin ejecutarla y `construir_pasos(modelo)` las tres sub-chains por separado. Ningún módulo crea el modelo ni lee la clave al importarse: `crear_modelo()` (`src/modelo_langchain.py`) lo crea cuando se le llama y las pruebas inyectan un mock. Los reintentos son los de LangChain (`max_retries=2` del modelo), no `with_retry`. `claude-sonnet-5-5` no admite `temperature` distinta de la de por defecto (ni `top_k`), por eso `crear_modelo` no la configura; el control del determinismo de la clase no aplica a este modelo.

Se prueba con `test/unit/test_cadena_tres_pasos.py`, sin red. La corrida real está en [LangChain: corrida real con topes](#langchain-corrida-real-con-topes).

## LangChain: enrutamiento

`src/enrutamiento.py` clasifica los mensajes dirigidos a un equipo de seguridad y envía cada uno a la ruta con el prompt que le corresponde:

| Categoría | Cuándo | Qué pide la ruta |
|---|---|---|
| `incidente` | Algo sospechoso o roto ahora | Contener, priorizar y dar pasos inmediatos |
| `vulnerabilidad` | Un hallazgo o CVE a evaluar | Evaluar la severidad y la remediación |
| `consulta` | Pregunta general (**ruta por defecto**) | Responder con precisión |

**Límite de longitud del incidente**: el prompt de la ruta `incidente` pide un máximo de 10 pasos numerados y menos de 350 palabras, porque con `max_tokens` 2048 la respuesta sin ese límite se cortó (en la segunda corrida real la ruta llegó al tope de 2048 tokens de salida).

**Límite de longitud de la vulnerabilidad**: el prompt de la ruta `vulnerabilidad` pide un máximo de 8 pasos o secciones y menos de 400 palabras, porque la tercera corrida real, anterior a este límite, mostró su respuesta cortada por `max_tokens`. La cuarta corrida real, con este límite, salió completa.

**Riesgo de la ruta por defecto**: la clasificación tolera mayúsculas, espacios y puntuación en los extremos, pero no interpreta frases. Una categoría inesperada (por ejemplo, si el modelo responde con una frase en vez de una palabra) cae en `consulta` y solo deja un `logger.warning` (`Categoria inesperada`), así que un incidente mal clasificado se respondería como una pregunta general. Para depurar, mira qué devuelve la clasificación por sí sola:

```python
paso_clasificacion = construir_paso_clasificacion(modelo)
paso_clasificacion.invoke({"mensaje": "..."})
```

## LangChain: fallos controlados

`src/enrutamiento.py` tiene dos capas, y la primera actúa antes que la segunda:

| Capa | Cuándo usarla | Qué devuelve |
|---|---|---|
| `construir_ruta_segura` (`with_fallbacks`) | Falla un paso que tiene una alternativa razonable: una ruta | El primer respaldo que funcione (la misma plantilla con `modelo_secundario`, si se pasó) y, al final, `RESPUESTA_GENERICA` |
| `procesar_mensaje_seguro` (`try/except`) | Falla lo que no tiene respaldo, como la clasificación: sin categoría no hay ruta posible | Registra el error con `logger.exception` y devuelve un texto fijo |

`with_fallbacks` solo reacciona a una excepción: **no cubre un paso colgado salvo que el modelo tenga timeout**. `crear_modelo` ya lo fija (30 s por defecto). No hay un modelo secundario fijado: `modelo_secundario` es un parámetro opcional.

**Checklist de orquestación**:

| Revisado | Pieza | Prueba que la cubre |
|---|---|---|
| [x] | Clasificación | `test_clasificacion_normaliza_espacios_y_mayusculas` |
| [x] | Bifurcación | `test_categoria_inventada_cae_en_la_ruta_por_defecto` |
| [x] | Rutas | `test_ruta_elegida_recibe_el_mensaje_original_y_la_categoria` |
| [x] | Flujo completo | `test_flujo_completo_el_respaldo_de_la_ruta_actua_antes_del_try_except` |

## LangChain: corrida real con topes

Pasa el aviso y los 3 mensajes ficticios de `JSON/casos_cadenas.json` por la chain de tres pasos y el enrutamiento:

```bash
python scripts/ejecutar_cadenas.py --confirmar-api-real
```

- Hace 9 llamadas **REALES** a la API de Claude (3 de la chain y 2 por mensaje: clasificación y ruta), con costo. Con el máximo de 2048 tokens de salida, las 9 llamadas sumarían unos USD 0.18 teóricos sin contar la entrada, pero el tope se revisa entre casos y el caso más caro realista ronda USD 0.02-0.03.
- `--confirmar-api-real` es obligatorio: sin él el script sale con código 2 sin cargar LangChain ni tocar la red.
- `--tope-usd` es US$ 0.10 por defecto y como máximo; un valor mayor o menor o igual a 0 sale con código 1. Se comprueba entre casos, así que el caso en curso puede sobrepasarlo.
- `--max-tokens` es 2048 por defecto y como máximo; un valor mayor o menor o igual a 0 sale con código 1 antes de crear el modelo.
- Cada resultado del JSON trae el campo `truncado`: `true` si alguna llamada de ese caso terminó cortada por el límite de tokens. Para esos casos se imprime `AVISO: respuesta cortada por max_tokens.`, sin cambiar el código de salida.
- `MAX_LLAMADAS` (12) es un límite fijo en el código: si los casos requieren más llamadas, sale con código 1 antes de crear el modelo. Cuenta las llamadas del plan, no los reintentos internos del modelo (`max_retries=2`).
- El costo se calcula con `UsageMetadataCallbackHandler` y los precios de `src/api_claude_refactor.py`: es una **estimación**, no el cobro real.
- Escribe los resultados en `docs/resultados_cadenas.json` (`--salida` lo cambia) y reescribe el archivo después de cada caso, de forma atómica (escribe un `.tmp` y lo reemplaza, así un fallo al serializar no deja el JSON truncado). Informa si una categoría obtenida difiere de la esperada y si una respuesta es un texto de respaldo.

**Con Gemini** (`src/modelo_gemini.py`, modelo `gemini-3.5-flash-lite`):

```bash
python scripts/ejecutar_cadenas.py --confirmar-api-real --proveedor gemini
```

- Requiere `GEMINI_API_KEY` en `.env`. `--proveedor` acepta `claude` (por defecto) o `gemini`; las demás opciones y topes son los mismos, y `--max-tokens` es aquí `max_output_tokens`.
- No se configura `temperature` porque este modelo usa muestreo fijo.
- El costo se calcula con los precios que informó el usuario (entrada USD 0.30, salida USD 2.50 por millón de tokens) y se rotula como **estimación** en el campo `nota_costo` del JSON, junto con `proveedor`. La consola imprime la nota de costo del proveedor elegido.
- **Corrida real de Gemini** (2026-11-06, `gemini-3.5-flash-lite`, `--max-tokens 2048`, tope USD 0.05; registro en [docs/resultados_cadenas_gemini.json](docs/resultados_cadenas_gemini.json)): 9 llamadas, costo estimado USD 0.00504 (1151 tokens de entrada y 1876 de salida) con los precios que informó el usuario. Las 3 categorías coincidieron, ningún caso cayó en un texto de respaldo y ninguna respuesta salió cortada. Las respuestas son más breves que las de Claude (incidente 282 palabras, vulnerabilidad 294, consulta 258). Dos reservas: el costo no se contrastó con la consola de Google, y no se verificó si `candidatesTokenCount` incluye los tokens de razonamiento del modelo, que se cobran como salida; si no los incluye, el costo real sería mayor que el estimado. (En esta corrida la consola todavía imprimía la nota de costo de Claude; se corrigió después, y la del JSON siempre fue la correcta.)

**Límites conocidos** (documentados, no corregidos):

- **El tope se revisa entre casos**: el primer caso siempre corre y el caso en curso puede pasarse. Con tope 1e-9 el aviso completo corrió igual: 3 llamadas, USD 0.0012 estimados (dato de la revisión de QA contra un servidor falso local, no de la API real; no es una corrida real). Con los casos del repo y `max_tokens` 2048, las 9 llamadas al máximo serían unos USD 0.18 teóricos; es un cálculo, no una medición.
- **Timeout de 30 s con `max_tokens` 2048** (sin problemas en las corridas reales): `crear_modelo` mantiene `timeout=30`. Una respuesta larga podría agotarlo; en una ruta, el respaldo entregaría `RESPUESTA_GENERICA` y no quedaría marcada como `truncado`, sino con `hubo_fallo`. En las tres corridas reales no ocurrió.
- **`MAX_LLAMADAS` no cuenta peticiones HTTP**: cuenta las llamadas del plan (9). Con `max_retries=2`, un servidor que falla produce hasta 3 intentos por llamada, y el medidor de uso no ve las peticiones fallidas.
- **Clave de solo espacios**: `crear_modelo` solo rechaza la cadena vacía. Una clave de espacios o tabs pasa y el problema aparece como error de conexión en la primera llamada.
- **Medidor de uso contrastado con la consola**: el medidor se contrastó con la consola de Anthropic en la primera corrida real: la consola marcó 1.492 tokens de entrada y 3.795 de salida, igual a lo medido por el script. El costo sigue siendo una estimación con los precios del proyecto.

**Corridas reales con Claude** (9 llamadas cada una, Claude Sonnet 5.5; el costo es una estimación del medidor con los precios del proyecto):

| Corrida | `max_tokens` | Tope USD | Costo estimado | Registro |
|---|---|---|---|---|
| 1 (2026-11-04) | 1024 | 0.05 | 0.04093 | [docs/resultados_cadenas.json](docs/resultados_cadenas.json) |
| 2 (2026-11-06) | 2048 | 0.10 | 0.05561 | [docs/resultados_cadenas_2048.json](docs/resultados_cadenas_2048.json) |
| 3 (2026-11-06) | 2048 | 0.10 | 0.05489 | [docs/resultados_cadenas_v3.json](docs/resultados_cadenas_v3.json) |
| 4 (2026-11-06) | 2048 | 0.10 | 0.04557 | [docs/resultados_cadenas_v4.json](docs/resultados_cadenas_v4.json) |

- **Corrida 1**: las 3 respuestas de las rutas gastaron entre 1030 y 1123 tokens de salida y llegaron cortadas a mitad de frase. El caso de vulnerabilidad no se evaluó: el modelo respondió que `CVE-0000-00002` no es un identificador válido.
- **Corrida 2**: con 2048 tokens, el incidente siguió cortado (la ruta llegó al tope de 2048 y el caso sumó 2054 tokens de salida con la clasificación); vulnerabilidad y consulta salieron completas, pero la vulnerabilidad volvió a negarse a dar una severidad por el ID ficticio.
- **Corrida 3** (prompt de incidente con límite de longitud y caso de vulnerabilidad con ID de laboratorio, descripción y CVSS inventados): las 3 categorías coincidieron con las esperadas y ningún caso cayó en un texto de respaldo. El incidente salió completo (1090 tokens de salida, unas 369 palabras: el límite de 350 es una instrucción al modelo, no un tope exacto). El modelo evaluó el caso de vulnerabilidad (severidad crítica según el CVSS dado), pero esa respuesta llegó cortada por `max_tokens` (el caso sumó 2200 tokens de salida) y el script la marcó con `truncado` y el AVISO. Es el único caso cortado de la corrida.
- **Corrida 4** (con el límite de longitud también en la ruta de vulnerabilidad): ningún caso salió cortado (`truncado` falso en los cuatro), las 3 categorías coincidieron y ninguno cayó en un texto de respaldo. Incidente: 1030 tokens de salida (363 palabras); vulnerabilidad: 1267 (374 palabras, cerró con un plazo de 7 días y revisión semanal); consulta: 1138. El límite de 400 palabras es una instrucción al modelo y no se mide en el script.
- En ninguna corrida se agotó el `timeout` de 30 s (`hubo_fallo` fue falso en todos los casos).
- Los resultados dependen del modelo y varían entre corridas; son cuatro muestras, no una medición estadística.

## LangChain: agente con herramientas y memoria

`src/herramientas_agente.py` y `src/agente.py` arman un agente que decide cuándo usar una herramienta (Bloques 21 y 22). Usa `AgentExecutor` y `create_tool_calling_agent` de `langchain-classic`, como la clase, con Claude Sonnet 5.5. Ningún módulo crea el modelo ni lee la clave al importarse: el modelo siempre entra como parámetro (`crear_modelo()` de `src/modelo_langchain.py`).

| Herramienta | Qué hace |
|---|---|
| `buscar_en_documentacion` | Responde sobre cómo funciona la miniaplicación (reintentos, costo, topes) con textos fijos del proyecto |
| `calcular` | Aritmética con un evaluador propio sobre `ast` (sin `eval`): `+ - * / // %`, signo y paréntesis; sin `**`, nombres ni llamadas; máximo 100 caracteres. Rechaza lo no permitido con `Expresión no permitida.`, devuelve `No se puede dividir entre cero.` y `Resultado fuera de rango.` si el resultado no es finito. Los floats enteros mayores o iguales a 2^53 se muestran con `str(float)` (por ejemplo `1e+23`), no con su expansión binaria |
| `consultar_cve_de_laboratorio` | Consulta un CVE ficticio (`CVE-2099-10001` o `CVE-2099-10002`) con `args_schema` (`cve_id` e `incluir_mitigacion`) y nombre explícito. Normaliza el identificador y, si no existe, refleja a lo más 64 caracteres de él |
| `aislar_host_de_laboratorio` | **Riesgo alto, simulada** (Bloque 23): solo acepta hosts `*.ejemplo.test`, pide aprobación humana y registra cada decisión. No toca ninguna red, proceso ni archivo; la acción real no existe. Ver [Supervisión humana](#supervisión-humana-y-trazas-bloque-23) |

**Desvío de la clase**: la clase calcula con `eval()` filtrado por caracteres; aquí se reemplaza por el evaluador de `ast` por seguridad.

**Memoria**: `AlmacenHistoriales` guarda un historial por `session_id` y `construir_agente_con_memoria` usa `RunnableWithMessageHistory`. LangChain marca esa clase y `InMemoryChatMessageHistory` como **deprecadas** (esta última se eliminará en la versión 2.0.0) y recomienda la persistencia de LangGraph y `create_agent`; se usan igual porque la clase las usa y son lo más simple sobre `AgentExecutor`. La alternativa moderna queda documentada, no implementada. Cada invocación exige `config={"configurable": {"session_id": ...}}`; sin él lanza `ValueError`.

**Salida normalizada**: Claude a veces responde con una lista de bloques (por ejemplo un bloque `thinking` con firma opaca más el bloque `text`) en vez de texto. `normalizar_salida` deja solo el texto antes de la memoria; sin esa normalización el historial guardaba dicts sueltos y el segundo turno de la sesión fallaba con `ValueError`. Se encontró en la primera corrida real y se corrigió después (ver abajo).

Se prueba sin red con `test/unit/modelo_falso_herramientas.py`, un modelo falso con guion que sí implementa `bind_tools` (el `GenericFakeChatModel` no lo tiene).

### Corrida real del agente

```bash
python scripts/ejecutar_agente.py --confirmar-api-real
```

Procesa los 10 turnos de `JSON/casos_agente.json` (una consulta por herramienta, una que encadena dos, una general, una conversación de 2 turnos, una consulta de otra sesión para comprobar el aislamiento, un pedido de aislar un host de laboratorio y otro sobre un host fuera de laboratorio) y escribe `docs/resultados_agente.json` (`--salida` lo cambia). La primera corrida real tuvo solo los 8 primeros turnos.

- `--confirmar-api-real` es obligatorio: sin él sale con código 2 sin cargar LangChain ni leer la clave.
- `--tope-usd` es US$ 0.10 por defecto y como máximo; `--max-tokens` es 1024 por defecto y 2048 como máximo.
- **Los topes se revisan antes de cada llamada al modelo**, no solo entre turnos: un callback lanza `LimiteExcedido` antes de una llamada que superaría `MAX_LLAMADAS` (30) o el tope de gasto, así que el sobregiro máximo es el costo de una llamada. Además hay `MAX_ITERACIONES` (4 por turno), `MAX_TURNOS` (10) y un máximo de 2000 caracteres por mensaje.
- **Sin datos de uso no hay tope de gasto**: si un turno termina sin error pero el medidor no registró tokens, la corrida se detiene (`cortado_por_falta_de_uso`) en vez de seguir con costo 0.
- El JSON trae por turno la respuesta (siempre texto), las herramientas usadas y esperadas, la traza de pasos con su observación, llamadas, tokens, costo estimado, `truncado`, `hubo_fallo` y los mensajes de la sesión; y el total de llamadas y los cortes. Se guarda después de cada turno, de forma atómica. Un enlace duro de `--salida` al archivo de casos se detecta y se rechaza.
- El script desactiva el streaming del ejecutor (`stream_runnable=False`).
- `--aprobacion {denegar,interactiva}` (por defecto `denegar`), `--tiempo-aprobacion` (segundos, mayor que 0 y como máximo 300; 60 por defecto), `--trazas <ruta>` (por defecto `trazas/trazas_agente.jsonl`) y `--sin-trazas`. Detalle en la sección siguiente.

**Primera corrida real** (2026-11-11, `--max-tokens 1024`, tope USD 0.10; registro en [docs/resultados_agente.json](docs/resultados_agente.json)): 15 llamadas, 13570 tokens de entrada y 1388 de salida, costo estimado USD 0.04102. No hubo cortes ni fallos y ningún tope se activó. El costo se calcula con los precios del proyecto y es una estimación; no se contrastó con la consola de Anthropic.

| Turno | Llamadas | Tokens (entrada / salida) | Costo estimado (USD) |
|---|---|---|---|
| documentacion | 2 | 1778 / 281 | 0.00637 |
| calculo | 2 | 1659 / 65 | 0.00397 |
| cve | 2 | 1819 / 224 | 0.00588 |
| encadenada | 4 | 3926 / 281 | 0.01066 |
| general | 1 | 804 / 97 | 0.00258 |
| conversacion_1 | 2 | 1814 / 206 | 0.00569 |
| conversacion_2 | 1 | 954 / 108 | 0.00299 |
| aislamiento | 1 | 816 / 126 | 0.00289 |

- Las herramientas elegidas coincidieron con las esperadas en 7 de 8 turnos. En `encadenada` el modelo hizo una búsqueda extra: su primera consulta ("precio de salida por millón de tokens") no contenía la palabra "costo", que es la clave que busca la herramienta, y el script la marcó como `DIFERENCIA de herramientas`. La respuesta final fue correcta (USD 2,50 por 250000 tokens de salida).
- La consulta de otra sesión respondió que no tenía registro de ningún CVE previo: el aislamiento entre sesiones funcionó con el modelo real.
- **Hallazgo de la corrida**: en `conversacion_2` la respuesta llegó como lista de bloques con un bloque `thinking`, y la sesión quedó con 5 mensajes en vez de 4. En esa sesión no había un tercer turno, así que no falló, pero el segundo turno tras una respuesta así lo habría hecho. Se corrigió con `normalizar_salida`; el JSON versionado es el registro de la corrida **antes** de la corrección (su `respuesta` de ese turno sigue siendo una lista). La corrección se verificó con modelos falsos y un servidor falso local, no con una segunda corrida real.

**Límites conocidos del agente** (documentados, no corregidos; vienen de la revisión de QA contra servidores falsos, salvo que se indique otra cosa):

- **Reintentos del SDK**: `MAX_LLAMADAS` cuenta llamadas al modelo, no peticiones HTTP. Con un servidor que responde 500, 8 llamadas contadas produjeron 24 peticiones (3 por llamada, `max_retries=2`). Las peticiones fallidas no las ve el medidor y no hay tope de tiempo total. No se sabe si el proveedor cobra los intentos que expiran.
- **Un 401 no corta la corrida**: intenta todos los turnos (una petición sin costo por turno) y termina con código 1.
- **Argumentos de herramienta inválidos** (por ejemplo `expresion` numérica): el `ValidationError` mata ese turno; el historial queda intacto y el turno siguiente corre. Con el esquema de Claude es improbable.
- **Corte por `max_iterations`**: la respuesta `Agent stopped due to max iterations.` se guarda como respuesta del asistente y los turnos siguientes de esa sesión la ven.
- **`session_id`**: no se normalizan; `"a"` y `"a "` son sesiones distintas.
- **`--casos` inexistente o con JSON roto** termina con traza, sin mensaje `Error:` (patrón heredado de `ejecutar_cadenas.py`).
- **`normalizar_salida`** falla con `TypeError` si un bloque `text` trae `None`; Claude no lo hace. Ninguna prueba distingue incluir el texto de un bloque `thinking` no vacío (en la corrida real vino vacío).
- **Sin verificar con la API real**: el streaming (el script lo desactiva), la facturación de intentos fallidos y la corrección de los bloques `thinking`.

## Supervisión humana y trazas (Bloque 23)

El agente tiene una herramienta de riesgo alto, `aislar_host_de_laboratorio`, **simulada**: acepta solo hosts `*.ejemplo.test` y devuelve "aislado (simulado: no se modificó ninguna red)". Sirve para tener algo real que supervisar sin tocar nada. El documento completo es [docs/limites_autonomia_agente.md](docs/limites_autonomia_agente.md): herramientas por nivel de riesgo (`RIESGO`), qué requiere aprobación y por qué, revisión de trazas, qué hacer ante un comportamiento inesperado, controles automáticos y límites conocidos. Lo que sigue es el resumen.

**Aprobación humana** (`src/supervision.py`):

- `--aprobacion denegar` (por defecto): nadie aprueba, así que una corrida desatendida nunca ejecuta una acción de riesgo alto; el intento queda registrado como no aprobado. `--aprobacion interactiva` pregunta por consola con plazo (`--tiempo-aprobacion`).
- Solo la respuesta `s` aprueba (mayúsculas y espacios se ignoran). Plazo vencido, fin de entrada, línea parcial, línea de más de 1024 bytes, tiempo inválido, descriptor cerrado o cualquier otro error cuentan como **no aprobado**; solo `KeyboardInterrupt` se propaga. El plazo cubre toda la lectura (`os.read` con `select`) y en una terminal se descarta lo tecleado antes de preguntar. Un aprobador personalizado solo aprueba si devuelve exactamente `True`.
- El motivo que escribe el modelo se sanea (una línea, solo caracteres imprimibles, máximo 200) y se muestra entre comillas como "texto del modelo, no verificado". Se encontró que sin sanear podía borrar la línea del mensaje de aprobación con secuencias ESC y simular una aprobación previa.
- **Topes**: 3 aprobaciones solicitadas por turno y 10 por corrida; un host ya negado en el turno no se vuelve a preguntar. Peor caso desatendido con `--tiempo-aprobacion 300`: 10 x 300 s = 50 minutos. Un host fuera de laboratorio se rechaza sin pedir aprobación, pero el intento queda registrado (clave `rechazo`) porque es una señal de seguridad. Cada decisión queda en `aprobaciones` del JSON por turno y de la corrida.
- **Ctrl-C** guarda lo ya decidido (JSON y traza con `"interrumpido": true`) y el script sale con código 130.

**Trazas** (`src/trazas.py`, `scripts/revisar_trazas.py`): una línea JSONL por turno en `trazas/trazas_agente.jsonl` con la entrada completa, las acciones, la respuesta y las aprobaciones. Quedan fuera de git (`trazas/`, `*trazas*.jsonl`, `trazas*/`), se crean con permisos 0600/0700 y no tienen rotación. `--trazas` se rechaza si coincide con `--casos` o `--salida`, o si apunta a un archivo que no es JSONL.

```bash
python scripts/revisar_trazas.py                    # lee trazas/trazas_agente.jsonl
python scripts/revisar_trazas.py --trazas <ruta>
```

Marca candidatos a revisar, no concluye nada: la misma herramienta usada dos veces seguidas, una herramienta de riesgo alto sin ningún registro de aprobación y una herramienta ausente de `RIESGO`. La revisión es un proceso humano (una muestra semanal y todas las trazas con herramientas de riesgo alto); nada lo ejecuta solo.

**Cómo se verificó**: la lectura de la aprobación se probó con pipes reales y con una terminal simulada (`pty`), con un Ctrl-C real y un servidor falso local; la QA independiente (`mira`) encontró 10 hallazgos (1 medio, 9 bajos) y se corrigieron todos salvo los que quedaron como límites conocidos. **Sin verificar con la API real**: cómo se comporta Claude tras un "no" ni con varias llamadas de herramienta en una misma respuesta (los topes cubren ambos casos), y la aprobación con una persona en una sesión larga.

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
  modelo_langchain.py      crea el ChatAnthropic de Sonnet 5.5 a pedido
  modelo_gemini.py         crea el ChatGoogleGenerativeAI de Gemini 3.5 Flash-Lite a pedido
  cadena_tres_pasos.py     chain de tres pasos: aviso, puntos clave, lenguaje simple, checklist
  enrutamiento.py          clasifica mensajes, los envía a la ruta de su categoría y controla los fallos
  herramientas_agente.py   herramientas del agente: documentación, cálculo seguro con ast, CVE de laboratorio y aislamiento simulado con aprobación
  agente.py                agente con herramientas, traza de decisión, memoria por sesión y normalización de la salida
  supervision.py           aprobación humana con plazo, registro de decisiones y topes de aprobaciones por turno y por corrida
  trazas.py                trazas JSONL por turno y marcas para su revisión
scripts/
  ensayo_rate_limit_simulado.py  ensayo de rate limit simulado con mocks
  ejecutar_pipeline.py           pipeline de texto sobre respuestas reales de Claude (con costo)
  ejecutar_metricas.py           precisión, cobertura y F1 sobre respuestas reales de Claude (con costo)
  ensayo_rate_limit_real.py      ensayo de rate limit real contra Claude (con costo)
  ejecutar_cadenas.py            chain de tres pasos y enrutamiento sobre respuestas reales de Claude o Gemini (con costo)
  ejecutar_agente.py             agente con herramientas y memoria sobre respuestas reales de Claude, con topes de llamadas y de gasto, aprobación y trazas (con costo)
  revisar_trazas.py              revisa las trazas del agente y marca candidatos a revisar (sin costo)
test/
  unit/test_api_claude_refactor.py     pruebas unitarias con mocks
  unit/test_api_gemini.py              pruebas minimas del manejo de errores del cliente Gemini, con mocks
  unit/test_pipeline_texto.py          pruebas del pipeline de texto
  unit/test_ejecutar_pipeline.py       pruebas del script del pipeline, con mocks
  unit/test_metricas_texto.py          pruebas de las métricas de texto
  unit/test_ejecutar_metricas.py       pruebas del script de métricas, con mocks
  unit/test_ensayo_rate_limit_real.py  pruebas del ensayo real, con mocks
  unit/test_modelo_langchain.py        pruebas de crear_modelo y de su validación, sin red
  unit/test_modelo_gemini.py           pruebas de crear_modelo_gemini y del armado de su petición, sin red
  unit/test_cadena_tres_pasos.py       pruebas de la chain de tres pasos, con mocks
  unit/test_enrutamiento.py            pruebas del enrutamiento condicional, con mocks
  unit/test_orquestacion.py            pruebas de los fallos controlados y del flujo completo, con mocks
  unit/test_ejecutar_cadenas.py        pruebas del script de las cadenas, con un modelo falso
  unit/modelo_falso_herramientas.py    modelo falso con guion y bind_tools para probar el agente (sin pruebas propias)
  unit/test_herramientas_agente.py     pruebas de las herramientas, del evaluador seguro y de la herramienta de riesgo alto
  unit/test_agente.py                  pruebas del agente: elección de herramienta, traza, max_iterations
  unit/test_memoria_agente.py          pruebas de la memoria: historial, aislamiento de sesiones y normalización
  unit/test_ejecutar_agente.py         pruebas del script del agente, con un modelo falso
  unit/test_supervision.py             pruebas de la aprobación (pipes, pty), el registro y los topes
  unit/test_trazas.py                  pruebas de las trazas, su lectura validada y su revisión
  unit/test_limites_autonomia.py       pruebas que cruzan el documento de límites con el código
  unit/test_conftest_guardian.py       prueba del guardián que impide que la suite escriba en las trazas reales
  unit/conftest.py                     guardián automático: falla una prueba que escriba en trazas/trazas_agente.jsonl
  test_consistencia.py         prueba de consistencia con Claude
  test_consistencia_gemini.py  misma prueba con Gemini (comparativa)
JSON/
  promp_bank.JSON                  banco de prompts para Claude
  promp_bank_gemini.JSON           banco de prompts para Gemini
  registro_resultados.json         resultado de la última prueba con Claude
  registro_resultados_gemini.json  resultado de la última prueba con Gemini
  casos_pipeline.json              consultas y referencias del pipeline de texto
  casos_metricas.json              consultas y referencias de las métricas de texto
  casos_cadenas.json               aviso y mensajes ficticios de las cadenas de LangChain
  casos_agente.json                10 turnos de la corrida real del agente (8 de herramientas y memoria, 2 de aislamiento)
docs/                      decisión de arquitectura, diagrama y evidencias
  registro_rate_limit.md   registro del ensayo de rate limit (simulado)
  registro_pipeline.md     registro del pipeline de texto (se genera al correr el script)
  registro_rate_limit_real.md  registro del ensayo de rate limit real (se genera al correr el ensayo)
  registro_rate_limit_real_concurrente.md  registro del ensayo concurrente (se genera al correr el ensayo con --concurrencia mayor que 1)
  resultados_metricas.json  resultados de la corrida real de métricas (se genera al correr ejecutar_metricas.py)
  resultados_cadenas.json  resultados de la corrida real de las cadenas (se genera al correr ejecutar_cadenas.py)
  resultados_agente.json   resultados de la primera corrida real del agente, 8 turnos (se genera al correr ejecutar_agente.py)
  limites_autonomia_agente.md  límites de autonomía del agente: riesgo por herramienta, aprobación, trazas y límites conocidos
  revision_manual.json     revisión manual de 10 casos con la rúbrica de tres preguntas
  limites_metrica.md       límites conocidos de la métrica y recomendación de uso
```

## Problemas comunes

| Error | Causa probable |
|---|---|
| `Credencial inválida o sin permiso` | La API key (`CLAUDE_API_KEY` o `GEMINI_API_KEY`) falta o es inválida en `.env`, o el modelo no está disponible para tu clave (401/403) |
| `Ocurrio un error de conexión: 400` | Petición mal formada, revisa el `output_schema` del prompt |
| `ModuleNotFoundError: No module named 'api_claude_refactor'` | Falta `PYTHONPATH=src` al ejecutar la prueba de consistencia |
