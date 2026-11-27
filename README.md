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
| Gemini 3.5 Flash-Lite (página oficial de precios de Gemini, consultada el 2026-11-23; no registrados en ese documento) | USD 0.30 | USD 2.50 |

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
- El costo se calcula con los precios de la página oficial de precios de Gemini, consultada el 2026-11-23 (entrada USD 0.30, salida USD 2.50 por millón de tokens) y se rotula como **estimación** en el campo `nota_costo` del JSON, junto con `proveedor`. La consola imprime la nota de costo del proveedor elegido.
- **Corrida real de Gemini** (2026-11-06, `gemini-3.5-flash-lite`, `--max-tokens 2048`, tope USD 0.05; registro en [docs/resultados_cadenas_gemini.json](docs/resultados_cadenas_gemini.json)): 9 llamadas, costo estimado USD 0.00504 (1151 tokens de entrada y 1876 de salida) con esos precios (la `nota_costo` del JSON de esa corrida conserva el texto anterior, "informados por el usuario": los precios se verificaron después, el 2026-11-23). Las 3 categorías coincidieron, ningún caso cayó en un texto de respaldo y ninguna respuesta salió cortada. Las respuestas son más breves que las de Claude (incidente 282 palabras, vulnerabilidad 294, consulta 258). Dos reservas: el costo no se contrastó con la consola de Google, y no se verificó si `candidatesTokenCount` incluye los tokens de razonamiento del modelo, que se cobran como salida; si no los incluye, el costo real sería mayor que el estimado. (En esta corrida la consola todavía imprimía la nota de costo de Claude; se corrigió después, y la del JSON siempre fue la correcta.)

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

**Con Gemini** (`src/modelo_gemini.py`, modelo `gemini-3.5-flash-lite`):

```bash
python scripts/ejecutar_agente.py --confirmar-api-real --proveedor gemini
```

- Requiere `GEMINI_API_KEY` en `.env`. `--proveedor` acepta `claude` (por defecto) o `gemini`; sin `--confirmar-api-real` sale con código 2 sin importar `modelo_gemini` ni LangChain ni leer ninguna clave. Los topes y las demás opciones son los mismos, y `--max-tokens` es aquí `max_output_tokens`.
- La salida por defecto es `docs/resultados_agente_gemini.json`, para no pisar los resultados reales de Claude; un `--salida` explícito se respeta siempre. El JSON trae `proveedor` y la `nota_costo` de Gemini, y la consola imprime esa misma nota.
- El modelo se crea con nivel de razonamiento `medium` (`thinking_level="medium"`), porque la documentación de Gemini lo recomienda para agentes con herramientas; el nivel por defecto del modelo es `minimal`. El razonamiento se factura como salida (USD 2,50 por millón de tokens), así que `medium` cuesta más que `minimal`.
- El costo se estima con los precios de Gemini 3.5 Flash-Lite de la página oficial de precios de Gemini, consultada el 2026-11-23 (`PRECIOS_GEMINI` de `scripts/ejecutar_cadenas.py`; entrada USD 0.30 y salida USD 2.50 por millón de tokens), como en las cadenas.

**Corrida real con Gemini** (2026-11-23, `gemini-3.5-flash-lite`, `thinking_level` medium, `--max-tokens 1024`, tope USD 0.10, aprobación `denegar`; registro en [docs/resultados_agente_gemini.json](docs/resultados_agente_gemini.json)): 19 llamadas, 9866 tokens de entrada y 3479 de salida (la salida incluye el razonamiento), costo estimado USD 0.01166. Ningún tope ni corte se activó. Es UNA corrida por proveedor, así que no hay varianza medida. El costo es una estimación del script con los precios de la página oficial de Gemini consultada el 2026-11-23; no se contrastó con la consola de Google. La `nota_costo` de ese JSON conserva el texto anterior ("informó el usuario") porque la corrida se hizo antes de actualizar la nota. La segunda corrida de Claude tuvo 18 llamadas y USD 0.06608 (precios y conteo de tokens distintos); la diferencia de unas 5.7 veces es una estimación entre tablas de precios distintas, no un cobro comparado.

| Turno | Gemini: llamadas / tokens de entrada / tokens de salida / costo estimado (USD) | Claude (segunda corrida): lo mismo | Herramientas = esperadas (Gemini / Claude) |
|---|---|---|---|
| documentacion | 2 / 984 / 353 / 0.00118 | 2 / 2082 / 271 / 0.00687 | sí / sí |
| calculo | 2 / 874 / 109 / 0.00053 | 2 / 1963 / 66 / 0.00459 | sí / sí |
| cve | 2 / 1128 / 403 / 0.00135 | 2 / 2123 / 224 / 0.00649 | sí / sí |
| encadenada | 3 / 1664 / 362 / 0.00140 | 4 / 4534 / 322 / 0.01229 | sí / no |
| general | 1 / 393 / 274 / 0.00080 | 1 / 956 / 97 / 0.00288 | sí / sí |
| conversacion_1 | ERROR 503 / 2 / 401 / 169 / 0.00054 | 2 / 2118 / 217 / 0.00641 | error 503 / sí |
| conversacion_2 | 1 / 400 / 292 / 0.00085 | 1 / 1117 / 382 / 0.00605 | sí / sí |
| aislamiento | 1 / 398 / 164 / 0.00053 | 1 / 968 / 114 / 0.00308 | sí / sí |
| aislamiento_con_aprobacion | 2 / 1104 / 302 / 0.00109 | 2 / 2166 / 487 / 0.00920 | sí / sí |
| aislamiento_host_no_permitido | 3 / 2520 / 1051 / 0.00338 | 1 / 966 / 629 / 0.00822 | no / no |

- **Confirmado con la API real**: tool calling, firmas de pensamiento en las llamadas encadenadas (sin errores 4xx), `candidateCount` 1 aceptado (la guía de migración decía que no estaba soportado) y `thinking_level` medium aceptado. Que el nivel de razonamiento se aplique no es observable en la respuesta: está aceptado, no verificado como aplicado. Una aprobación de riesgo alto se registró y se denegó, y la herramienta no se ejecutó.
- **Fallo transitorio**: `conversacion_1` falló con 503 UNAVAILABLE de la API (el script siguió). El turno no entró al historial (`mensajes_en_sesion` 0), así que `conversacion_2` no probó la memoria en esa corrida. Se repitieron solo esos dos turnos (tope USD 0.01, 3 llamadas, USD 0.00159 estimado): `conversacion_2` respondió "Media (CVSS 5.3)" sin usar herramientas, es decir, recordó el CVE del turno anterior. La memoria entre turnos con Gemini queda comprobada en una corrida; el registro de la repetición está en [docs/resultados_agente_gemini_conversacion.json](docs/resultados_agente_gemini_conversacion.json).
- **Host fuera de laboratorio**: en `aislamiento_host_no_permitido` Gemini llamó dos veces a `buscar_en_documentacion` y rechazó con texto sin usar `aislar_host_de_laboratorio` (3 llamadas); Claude no usó ninguna herramienta. Ninguno de los dos ejercitó el rechazo automático de la herramienta, y no hubo ningún aislamiento: no es un fallo de seguridad, pero el criterio de herramientas esperadas de ese turno no se cumple con ningún proveedor.
- `encadenada`: Gemini usó las dos herramientas esperadas en 3 llamadas; Claude (segunda corrida) hizo una búsqueda extra (4 llamadas).
- **No verificado**: que `thinking_level` se aplique, el costo contra la consola de Google (el razonamiento se factura como salida y el script solo suma lo que la API informa) y la varianza entre corridas.

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

**Segunda corrida real, con aprobación interactiva** (2026-11-11, 10 turnos, `--aprobacion interactiva --tiempo-aprobacion 60 --max-tokens 1024`, tope USD 0.10; registro en [docs/resultados_agente_v2.json](docs/resultados_agente_v2.json)): 18 llamadas, 18993 tokens de entrada y 2809 de salida, costo estimado USD 0.06608 (se estimó USD 0.066 antes de correr). Ningún tope se activó, ningún turno falló ni quedó truncado, y las respuestas llegaron como texto (la sesión de dos turnos terminó con 4 mensajes).

- **Aprobación**: el turno `aislamiento_con_aprobacion` pidió una sola aprobación (la persona respondió `s`) y la herramienta devolvió "aislado (simulado)". La decisión quedó en `aprobaciones` del turno y de la corrida y en la traza de la sesión `s-soc`. El motivo lo redactó el modelo y quedó registrado ya saneado.
- **Host fuera de laboratorio**: en `aislamiento_host_no_permitido` el modelo **no llamó a la herramienta**: se negó por sí mismo (dijo que solo sirve para hosts `.ejemplo.test` y que la lentitud no justifica aislar un servidor). El script lo marcó como diferencia de herramientas (se esperaba `aislar_host_de_laboratorio`). Por eso el rechazo automático de un host ajeno **no se ejercitó con el modelo real**; solo se probó con modelos y servidores falsos.
- **Revisión de trazas**: `scripts/revisar_trazas.py` revisó 10 registros y marcó solo `s-enc` (la misma búsqueda dos veces seguidas, la diferencia de herramientas ya conocida). Ninguna marca de riesgo alto. El archivo de trazas quedó con permisos 0600 y su directorio con 0700.
- **No observado en esta corrida** (siguen sin verificar con la API real): que el modelo repita el pedido tras un "no" y que emita varias llamadas de herramienta en una respuesta. Las llamadas por turno coincidieron con las herramientas usadas (1 por herramienta, 3 en `encadenada` con 4 llamadas).
- El costo es una estimación con los precios del proyecto, no contrastada con la consola.

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

**Cómo se verificó**: la lectura de la aprobación se probó con pipes reales y con una terminal simulada (`pty`), con un Ctrl-C real y un servidor falso local; la QA independiente (`mira`) encontró 10 hallazgos (1 medio, 9 bajos) y se corrigieron todos salvo los que quedaron como límites conocidos. **Sin verificar con la API real**: cómo se comporta Claude tras un "no" ni con varias llamadas de herramienta en una misma respuesta (los topes cubren ambos casos), y la aprobación con una persona en una sesión larga (la segunda corrida real pidió una sola aprobación).

## RAG: indexación y recuperación con citas (Bloques 24 a 26)

Un corpus propio se indexa en un vector store persistente (Chroma) con embeddings de Gemini (`gemini-embedding-001`, vectores de 3072 dimensiones) y se consulta con recuperación semántica; Claude responde solo con el contexto recuperado y cita sus fuentes. Los embeddings requieren Gemini porque Anthropic no ofrece un modelo de embeddings; la respuesta la da Claude. Dependencias: `chromadb` 1.5.9 y `langchain-chroma` 1.1.0 (verificadas en Python 3.14 con `pip check` limpio).

**Corpus** (`JSON/corpus_miniapp.json`): 10 documentos de una empresa ficticia, Taller Aurora: política de parcheo, avisos de los dos CVE ficticios del agente, playbooks de tráfico saliente y de phishing, contraseñas y MFA, respaldos, clasificación de incidentes y dos documentos sobre el propio agente (aprobación humana, límites y trazas). Los valores son inventados (por ejemplo "4 días hábiles" para vulnerabilidades críticas) para que Claude no pueda responderlos de memoria: si una respuesta los cita con su fuente, la recuperación funcionó. No hay datos de ninguna organización real.

**Indexación** (`src/indexacion.py`, `src/embeddings_gemini.py`, `scripts/indexar_corpus.py`):

- Cada documento se parte con `RecursiveCharacterTextSplitter` (200 caracteres con solapamiento de 30 por defecto) y cada chunk lleva metadata `fuente`, `documento` y `fragmento`, con ids deterministas `<documento>_<n>`. Reindexar el mismo corpus no duplica; los chunks de un documento editado se reemplazan y los de un documento quitado se borran **después** de agregar, de modo que si la adición falla el índice anterior queda intacto.
- Colección explícita (`corpus_miniapp`) y ruta absoluta: Chroma crea sin avisar un índice vacío si la ruta relativa apunta a otro lugar, así que `cargar_vectorstore` comprueba que el directorio existe y que la colección no está vacía. El índice vive en `vectorstore/` (fuera de git) y se valida el nombre de la colección, los ids, el máximo de 50 documentos y de 200 chunks.
- **Bloqueo de indexación**: dos indexaciones simultáneas sobre el mismo directorio con corpus distintos dejaban el índice vacío (cada proceso borraba como huérfano lo que agregó el otro; reproducido por la QA). Ahora `indexar` toma un `flock` no bloqueante sobre un archivo `<directorio>.lock` hermano del índice (ignorado por git) y la segunda indexación falla de inmediato con "Ya hay otra indexación en curso". La consulta no toma el bloqueo. Solo Linux/macOS.
- El modo revisión y la indexación comparten las mismas validaciones (máximo de 200 chunks, nombre de colección, chunks vacíos, duplicados o sin fuente): lo que la revisión aprueba la indexación lo acepta. Si la revisión falla imprime el informe y después `Error:` con código 1. `~` en las rutas se expande y las abreviaturas de argparse (`--conf`) están desactivadas, para que nada active una llamada real por accidente.
- Un error 429 se reintenta con esperas de 5, 10 y 20 s; una clave rechazada o un 403 salen con un mensaje claro y un error desconocido se informa sin traceback.
- Sin `--confirmar-api-real` el script funciona en **modo revisión**: divide el corpus, verifica la integridad (sin vacíos, sin duplicados, sin chunks sin fuente) e imprime los chunks sin llamar a ninguna API. `--comparar-tamanos 120,200,300` compara tamaños de chunk (con solapamiento 30: 50, 29 y 20 chunks, con 5408, 4865 y 4628 caracteres). Con el flag indexa de verdad.

```bash
python scripts/indexar_corpus.py                           # modo revisión, sin API
python scripts/indexar_corpus.py --comparar-tamanos 120,200,300
python scripts/indexar_corpus.py --confirmar-api-real      # llama a la API de embeddings de Gemini
```

La primera indexación real creó 29 chunks (4865 caracteres) con una sola petición de embeddings. No se estima un costo en dólares porque el proyecto no tiene el precio de ese modelo.

**Recuperación y citas** (`src/recuperacion.py`, `scripts/consultar_corpus.py`, `JSON/casos_recuperacion.json`):

- `recuperar` busca los k fragmentos más parecidos con su **distancia** (menor es más parecido). `formatear_contexto` los numera `[1]..[k]` y guarda el mapa número -> fuente; `extraer_citas` separa las citas válidas, las inválidas (un `[99]` que el modelo inventa) y detecta la ausencia de citas. Reconoce `[1]`, listas (`[1,2]`, `[1, 2]`) y rangos ascendentes de hasta 10 (`[1-3]`, `[1–3]`); solo cuentan dígitos ASCII sin ceros a la izquierda (`[01]`, `[１]`, `[+1]`, `[1.0]` no son citas) y se ignoran `arr[2]` y lo que esté dentro de bloques de código.
- El prompt trata el contexto como **datos, no instrucciones**: va entre `<contexto>` y `</contexto>`, una regla lo declara material de consulta y las etiquetas que traiga un chunk se neutralizan (`<contexto>`, `</CONTEXTO>`, `< / contexto >`, `</contexto foo>`, una etiqueta sin cerrar y la de ancho completo, tras normalizar a NFKC) y se le quitan los saltos de línea, de modo que ninguna línea de un chunk pueda imitar `[2]`, `Pregunta:` o `Respuesta:`. Si el contexto no alcanza el modelo debe responder con una frase fija ("El contexto no alcanza para responder esa pregunta."), lo que permite comprobar por código si la admitió: la respuesta debe EMPEZAR con la frase (con o sin punto, comillas o negrita); una redacción distinta no cuenta, y una frase citada en medio de una respuesta tampoco. Son desvíos de la clase, que solo pide "dilo explícitamente". Desde el Bloque 31 `scripts/consultar_corpus.py` ya no usa `construir_prompt`: manda el mismo contenido como dos mensajes (sistema y humano), con el contexto solo en el humano (ver la sección del Bloque 31).
- `construir_prompt` recibe el índice ya abierto y nunca calcula embeddings del corpus (reindexar dentro de cada consulta es el error que la clase señala).
- El script exige `--confirmar-api-real` (sin él sale con código 2 sin hacer nada) y lleva topes forzados en código: gasto máximo USD 0.05, `--max-tokens` hasta 1024 (512 por defecto), `--k` de 1 a 5 y como máximo 10 preguntas. Guarda el JSON después de cada caso, de forma atómica, con la distancia y la fuente de cada fragmento recuperado (no su texto) y la verificación de las citas.
- Robustez (QA): un Ctrl-C guarda lo hecho, registra el caso en vuelo como interrumpido, informa el costo y sale con código 130. La corrida corta **entre** casos al llegar al tope (con una tolerancia de 1e-9 por el punto flotante), así que el gasto puede pasarse del tope como máximo en lo que cuesta un caso. `--salida` se comprueba antes de gastar: no puede ser vacía, estar en una carpeta sin permiso de escritura, quedar dentro del índice ni pisar un archivo que no sea un resultado previo del propio script. En consola cada línea de la respuesta del modelo lleva el prefijo `    | `, para que no pueda imitar las líneas del informe (el JSON conserva la respuesta tal cual).

```bash
python scripts/consultar_corpus.py --confirmar-api-real --max-tokens 512 --tope-usd 0.05
```

**Primera corrida real** (2026-11-12, `k=3`, 8 preguntas, 6 con respuesta en el corpus y 2 sin ella; registro en [docs/resultados_recuperacion.json](docs/resultados_recuperacion.json)): 8 consultas de embeddings y 8 llamadas a Claude, 3624 tokens de entrada y 1029 de salida, costo estimado USD 0.01754 (se estimó USD 0.017). No hubo cortes ni fallos.

| Pregunta | Tokens (entrada / salida) | Costo (USD) | Citas válidas | Admitió falta de contexto | Documento esperado recuperado |
|---|---|---|---|---|---|
| q1_plazo_critico_cola | 395 / 402 | 0.00481 | 2, 1 | no | no |
| q2_parafrasis_trafico | 466 / 147 | 0.00240 | 1, 2 | no | sí |
| q3_phishing_clave | 494 / 237 | 0.00336 | 1, 2, 3 | no | sí |
| q4_perdida_datos | 474 / 41 | 0.00136 | 1 | no | sí |
| q5_agente_sin_respuesta | 412 / 106 | 0.00188 | - | sí | sí |
| q6_nivel_s1 | 480 / 54 | 0.00150 | 1 | no | sí |
| q7_sin_respuesta_presupuesto | 502 / 21 | 0.00121 | - | sí | - |
| q8_sin_respuesta_moviles | 401 / 21 | 0.00101 | - | sí | - |

- Claude respondió con datos exactos y citas válidas en las preguntas con respuesta, y admitió la falta de contexto en las dos que el corpus no cubre, incluida una de tema cercano (el plazo de parcheo de teléfonos móviles). No inventó ningún dato.
- **La métrica por documento es demasiado generosa**: en `q5_agente_sin_respuesta` el documento correcto sí se recuperó, pero no el fragmento que contiene la respuesta (la frase "si nadie responde en 60 segundos… no aprobado" quedó en el fragmento 1 y se recuperaron el 0 y el 2 más un fragmento de otro documento). Claude respondió, correctamente, que el contexto no alcanzaba: fue una falla de recuperación y no una alucinación. En `q1_plazo_critico_cola` la política de parcheo no llegó, pero el propio aviso repite el plazo. Con `k=3` los fragmentos vecinos y solapados ocupan los puestos. La medición útil es por chunk (los ids `<documento>_<n>` ya lo permiten) y es lo que mide el Bloque 27 (sección siguiente); el corpus y el tamaño de chunk no se tocaron para partir de esta línea base.
- El costo es una estimación con los precios del proyecto, no contrastada con la consola.

**Límites conocidos de la recuperación** (verificados con servidores locales falsos del cliente real de Gemini y de Claude, más la corrida real de arriba):

- **Corte por falta de datos de uso**: el cliente real (`ChatAnthropic`) no deja `usage_metadata` vacío cuando la API no trae el uso: entrega `input_tokens: 0`. Por eso el script trata `input_tokens` igual a 0 (o no entero) como uso faltante: registra el caso, marca `cortado_por_falta_de_uso` y no sigue, porque sin uso no se puede controlar el gasto. Si solo falta `output_tokens` no corta y queda en 0. Probado con el cliente real contra un servidor local falso. El agente y `ejecutar_cadenas.py` no tienen esta regla (con ceros suman 0 y el costo queda subestimado). El techo natural con 8 preguntas y 512 tokens es de unos USD 0.046.
- Los reintentos del SDK de Anthropic (`max_retries=2`) no se cuentan como llamadas.
- Un 429 persistente de embeddings cuesta tiempo y no dinero: 4 intentos con esperas de 5, 10 y 20 s por pregunta (unos 35 s por pregunta).
- `acierto_recuperacion` se calcula por documento (ver arriba) y es solo informativo; desde el Bloque 31 solo cuenta los fragmentos que vio el modelo (uno descartado por el filtro de inyección no cuenta).
- La pregunta del archivo de casos se usa tal cual (es un archivo propio); lo que se neutraliza son las etiquetas de los chunks recuperados. No se cubren los homoglifos de otros alfabetos (`</соntехtо>` con letras cirílicas), las entidades HTML ni acentos (`</CONTEXTÓ>`). Aun así, la validación de citas solo comprueba que el número **existe** en el mapa: cualquier `[n]` con n ≤ k es "válido" aunque ese fragmento no respalde la frase, así que "citas verificables" significa "la fuente existe", no "respalda la afirmación" (y un chunk hostil con un modelo real no se ha probado: la instrucción de tratar el contexto como datos es la única barrera adicional).
- El guardián de las pruebas del índice compara solo la fecha del directorio `vectorstore/` y no detecta escrituras sobre un índice existente que no lo cambien.
- Hallazgos menores de la QA que se documentan y no se corrigen: el índice y las rutas relativas se resuelven contra el directorio actual (los valores por defecto cuelgan de la raíz del repo); si el índice se creó con otro modelo de embeddings, la consulta hace una petición por pregunta y falla con un mensaje rotulado "API" (indexar sí da el aviso de dimensión); abrir una colección inexistente en un índice ya creado la crea vacía antes de fallar; los errores locales (permisos de disco) se rotulan como error de la API; reindexar vuelve a calcular los embeddings de todo el corpus, sin caché; una pregunta de solo caracteres de control se acepta; los reintentos del SDK dejan un traceback por caso en stderr.
- Sin verificar: la calidad de la recuperación con otras preguntas distintas de las 8 de la corrida, y la comparación de tamaños de chunk con embeddings reales (la evaluación por chunk de la sección siguiente mide 14 preguntas con un solo tamaño).

## RAG: evaluación de la recuperación por chunk (Bloque 27)

Mide si la recuperación trae los chunks correctos, sin llamar a Claude: `src/metricas_recuperacion.py` (Python puro) calcula precisión, exhaustividad y F1 en el top-k sobre **ids de chunk** (`<documento>_<n>`, por ejemplo `politica_parcheo_0`), y `scripts/evaluar_recuperacion.py` lo ejecuta contra el índice real. Los nombres del proyecto van en español (`precision_en_k`, `exhaustividad_en_k`; la clase dice `recall_at_k`).

**Conjunto de evaluación** (`JSON/evaluacion_recuperacion.json`): 14 casos, 11 con respuesta en el corpus y 3 sin ella. Los 8 primeros son los de `casos_recuperacion.json` (mismo id y misma pregunta) con el chunk que contiene la respuesta; los 6 nuevos (q9 a q14) cubren otras zonas del corpus. Las etiquetas se hicieron leyendo el corpus y no el retriever, y una prueba comprueba que cada id etiquetado exista en el corpus. Una pregunta puede tener dos chunks relevantes (q1) o ninguno.

**Desvío respecto de la clase**: los promedios de precisión, exhaustividad y F1 se calculan **solo sobre los casos con respuesta**. En una pregunta sin respuesta la exhaustividad no está definida (`None`) y la precisión sería siempre 0 si se recupera algo, lo que arrastraría el promedio; esos casos se informan aparte (ids recuperados y distancia mínima).

**Cómo se ejecuta** (con costo solo en embeddings de las consultas; no estima dólares porque el proyecto no conoce el precio de `gemini-embedding-001`):

```
python scripts/evaluar_recuperacion.py --confirmar-api-real
python scripts/evaluar_recuperacion.py --confirmar-api-real --k 5 --repeticiones 1
```

- Sin `--confirmar-api-real` sale con código 2 sin leer ni importar nada. Topes en el código: 20 casos, 2 repeticiones y 40 consultas de embedding, que se comprueban antes de empezar y también en cada petición (los reintentos de un 429 cuentan).
- Cada pregunta se embebe una vez y recupera 5 chunks; las métricas de `k` = 1 a 5 salen de ese mismo ranking. El id del chunk se construye desde la metadata (`documento` y `fragmento`), no desde `Document.id`.
- Valida antes de gastar: estructura de los casos, ids etiquetados contra el corpus, preguntas largas, `--salida` (no pisa archivos ajenos ni cae dentro del índice) y que el índice exista y tenga los mismos 29 chunks e ids que el corpus.
- Con 2 repeticiones comprueba si la recuperación es determinista. Guarda el JSON de forma atómica tras cada caso de la primera pasada. Mientras no termina todo, el archivo lleva `incompleto: true`; si aborta por un error (no Ctrl-C) guarda lo hecho con `error` (el mismo texto de stderr) y sale con 1; con Ctrl-C sale con 130. Si falla el primer caso no escribe ningún archivo.

**Primera corrida real** (2026-11-14, 14 casos x 2 repeticiones = 28 consultas de embedding, sin llamadas a Claude; registro en [docs/resultados_evaluacion_recuperacion.json](docs/resultados_evaluacion_recuperacion.json)). Índice de 29 chunks (`chunk_size` 200, solapamiento 30). La segunda pasada dio los mismos ids en el mismo orden con diferencia de distancia 0. Promedios de los 11 casos con respuesta (recalculados a mano a partir de los rankings del JSON):

| k | Precisión | Exhaustividad | F1 |
|---|---|---|---|
| 1 | 0,82 | 0,82 | 0,82 |
| 2 | 0,45 | 0,86 | 0,59 |
| 3 | 0,30 | 0,86 | 0,45 |
| 4 | 0,25 | 0,95 | 0,39 |
| 5 | 0,20 | 0,95 | 0,33 |

- **El mejor F1 está en `k=1`**: en 9 de los 11 casos el chunk etiquetado es el primero. Subir `k` agrega ruido (los fragmentos vecinos y solapados ocupan puestos), el trade-off de la clase.
- **`q5_agente_sin_respuesta`**: el chunk con la respuesta (`agente_aprobacion_humana_1`) queda en el rango 4; los puestos 1 y 3 son sus fragmentos vecinos y el 2 es de otro documento. Con `k=3` falla (precisión 0, exhaustividad 0). La métrica por documento de la sección anterior lo marcaba como acierto.
- **`q1_plazo_critico_cola`**: `aviso_cve_2099_10002_2` queda en el rango 2, pero `politica_parcheo_0` no entra en los 5 primeros: exhaustividad 0,5. Es el único caso con relevantes parciales. Con la etiqueta actual la exhaustividad de q1 se penaliza aunque el aviso solo ya permitiría responder (el aviso repite el plazo); es una decisión de etiquetado, no una falla comprobada de la recuperación.
- **Preguntas sin respuesta** (q7, q8, q14): recuperan chunks con distancia mínima de 0,49 a 0,50, parecida a la de aciertos reales de otros casos (0,43 en q1, 0,57 en q5 para el chunk correcto). Con solo 3 casos, la distancia por sí sola no separa "hay respuesta" de "no hay"; en la corrida anterior fue Claude, al leer el contexto, quien lo admitió.

**Límites conocidos de la evaluación** (de método, salvo los dos últimos, que vienen de la verificación del script con un servidor local falso del cliente real de Gemini):

- 14 casos (11 con respuesta) son pocos: la clase recomienda 15 a 30 para un conjunto realista, y el corpus de 29 chunks hace que `k=5` ya recupere el 17 % del índice. Una diferencia de un caso mueve los promedios cerca de 9 puntos.
- El etiquetado lo hizo una sola persona con ayuda de Antares, y al etiquetar q1 y q5 ya se conocía su resultado por documento (README, sección anterior): no son una prueba ciega. Las etiquetas de q9 a q14 sí se hicieron sin ver el retriever.
- La métrica ignora el orden dentro del top-k y trata igual a todos los chunks etiquetados; la comparación de ids es exacta.
- Solo se mide la recuperación: que el chunk correcto llegue al contexto no garantiza que Claude lo use bien, y viceversa (q1 se responde con el aviso solo).
- No se compararon tamaños de chunk ni otros `k` de indexación; el índice y la configuración son los de la corrida de citas.
- El determinismo se comprobó con dos pasadas seguidas sobre el mismo índice, no entre días ni tras reindexar.
- Si el guardado de un resultado falla (`OSError`) dentro del manejador de error, esa excepción sustituye al error original y no se avisa del parcial. Dos huecos de pruebas conocidos y aceptados: el texto exacto del mensaje del tope de consultas (sin esa rama el mensaje sale genérico, pero legible) y la validación de preguntas largas en un caso posterior al primero (`recuperar` ya las rechaza; el chequeo previo solo evita gastar consultas antes de fallar).

## RAG: latencia y costo reales del flujo completo (Bloque 28)

Mide cuánto tarda y cuánto cuesta una consulta RAG de punta a punta, por etapa: `src/medicion_rag.py` instrumenta `embed_query` (red, Gemini), `busqueda_vectorial` (local, Chroma) y `generacion` (red, Claude) con `time.perf_counter`, y `scripts/medir_latencia_costo.py` corre las consultas reales y anexa un registro por llamada a `docs/registros_latencia_costo.jsonl` (JSON Lines, archivo de solo anexar). Se embebe **una sola vez** por consulta y el vector se reutiliza en la búsqueda por vector (con `recuperar` el embed ocurre dentro de la búsqueda y no se podría separar); una prueba y una verificación con el cliente real contra un servidor falso comprueban que da los mismos ids que `recuperar`.

**Desvíos respecto de la clase** (declarados): la generación es Claude y no Gemini; los precios del proyecto son USD **por token** (`api_claude_refactor`: USD 2 de entrada y 10 de salida por millón) y no por 1000 tokens; el costo de los embeddings no se estima (no se conoce el precio de `gemini-embedding-001`: se cuentan peticiones); el registro agrega `ids_recuperados`, `truncado`, `reintentos_embed`, `admite_falta_de_contexto` y la respuesta recortada a 1000 caracteres.

**Cómo se ejecuta** (con costo en embeddings y en Claude):

```
python scripts/preflight_rag.py
python scripts/medir_latencia_costo.py --confirmar-api-real
python scripts/medir_latencia_costo.py --confirmar-api-real --ks 3,6 --max-tokens 512 --tope-usd 0.08
```

- Sin `--confirmar-api-real` sale con código 2 sin importar los clientes ni leer archivos. Topes en el código: 6 casos, 2 valores de k (cada uno de 1 a 6), 12 llamadas, 20 consultas de embedding forzadas en ejecución (los reintentos de un 429 cuentan) y tope de gasto de USD 0,08 que se comprueba antes de cada llamada a Claude (el gasto puede pasarse como máximo en lo que cuesta una llamada).
- `--salida` es un archivo de solo anexar: si existe debe estar vacío o ser un JSONL de registros de este script; cualquier otro archivo se rechaza antes de gastar. Cada registro se anexa al instante, con una sola escritura.
- Un error del embedding (límite, clave, tope) detiene la corrida con `Error:` y no deja línea para esa llamada. Un error de Claude queda en el campo `error` del registro y la corrida sigue. Si Claude devuelve el uso en cero (el cliente real no lo deja vacío) se anexa la línea con `sin_datos_de_uso`, la corrida se detiene y sale con 1 (sin uso no se controla el gasto). Con Ctrl-C lo ya anexado se conserva y sale con 130.
- `scripts/preflight_rag.py` revisa sin API que el índice exista, que su conteo y sus ids coincidan con los chunks del corpus y que el directorio esté ignorado por git (lee el `.gitignore` por líneas: gana la última coincidencia y una negación la anula; la búsqueda de subcadena de la clase daría por bueno `vectorstore_viejo/`).

**Primera corrida real** (2026-11-16, 5 preguntas de `JSON/casos_latencia_costo.json` x k = 3 y 6 = 10 llamadas, `--max-tokens 512`, tope USD 0,08; registro en [docs/registros_latencia_costo.jsonl](docs/registros_latencia_costo.jsonl)): 10 consultas de embedding y 10 llamadas a Claude, sin errores, sin uso faltante y sin respuestas cortadas. Costo estimado total USD 0,02409 (recalculado con los tokens del registro; se estimó entre 0,025 y 0,03 en el caso típico y 0,069 en el peor).

| Pregunta | k | Embed (ms) | Búsqueda (ms) | Generación (ms) | Tokens (entrada / salida) | USD |
|---|---|---|---|---|---|---|
| q1_plazo_critico_cola | 3 | 1360,1 | 1,6 | 4508,7 | 395 / 428 | 0,00507 |
| q1_plazo_critico_cola | 6 | 1222,8 | 1,5 | 1842,2 | 689 / 218 | 0,00356 |
| q2_parafrasis_trafico | 3 | 409,0 | 1,3 | 9213,7 | 466 / 145 | 0,00238 |
| q2_parafrasis_trafico | 6 | 1229,4 | 1,4 | 1943,8 | 706 / 183 | 0,00324 |
| q4_perdida_datos | 3 | 1024,2 | 1,6 | 1021,4 | 474 / 41 | 0,00136 |
| q4_perdida_datos | 6 | 386,2 | 1,5 | 1727,7 | 678 / 41 | 0,00177 |
| q5_agente_sin_respuesta | 3 | 349,9 | 1,4 | 1424,0 | 412 / 21 | 0,00103 |
| q5_agente_sin_respuesta | 6 | 409,8 | 1,3 | 1636,2 | 632 / 108 | 0,00234 |
| q6_nivel_s1 | 3 | 325,9 | 1,7 | 1515,8 | 480 / 54 | 0,00150 |
| q6_nivel_s1 | 6 | 410,1 | 1,4 | 1226,3 | 650 / 54 | 0,00184 |

Promedios por k (5 registros cada uno): con k=3, embed 693,8 ms, búsqueda 1,5 ms, generación 3536,7 ms, 445,4 tokens de entrada y 137,8 de salida, USD 0,01134; con k=6, embed 731,7 ms, búsqueda 1,4 ms, generación 1675,3 ms, 671,0 de entrada y 120,8 de salida, USD 0,01275.

- **`input_tokens` crece con k** en los 5 casos (en promedio un 51 %), pero el costo total sube solo un 12 %: en esta muestra la salida pesa más que la entrada (con k=3 la salida es el 61 % del costo; con k=6, el 47 %). Es la propiedad que pide la clase, y confirma que no se debe juzgar el costo por el tamaño del prompt.
- **La generación fue la etapa más lenta en 9 de 10 registros**; la excepción es `q4` con k=3 (embed 1024,2 ms contra generación 1021,4 ms). La búsqueda local (1,3 a 1,7 ms) es de dos a tres órdenes de magnitud menor que las dos etapas de red. El embed real tarda entre 326 y 1360 ms, no los milisegundos del ejemplo de la clase; el más lento (1360,1 ms) fue el primero de la corrida, que incluye el establecimiento de la conexión.
- **No se puede concluir que k cambie la latencia.** Con k=6 la generación promedió menos que con k=3 (1675 contra 3537 ms), pero es ruido: `q2` con k=3 tardó 9,2 s con solo 145 tokens de salida, y el tiempo de generación depende sobre todo de los tokens de salida y de la red (21 a 428 tokens de salida entre registros). Hay una sola medición por combinación.
- **`q5_agente_sin_respuesta` es el caso más claro.** Con k=3 el chunk con la respuesta (`agente_aprobacion_humana_1`) no entra y Claude responde "El contexto no alcanza para responder esa pregunta." (21 tokens de salida, USD 0,00103); con k=6 entra y responde correctamente con cita (108 tokens, USD 0,00234). La respuesta barata era la equivocada: menor costo no significa mejor resultado. Es la falla por chunk que midió el Bloque 27 vista en una respuesta real. En `q1` con k=6 entran los dos chunks etiquetados, incluido `politica_parcheo_0`, que no entraba con k=5 en la evaluación.
- **La heurística `len // 4` de la clase subestima los tokens**: el conteo real fue entre 1,62 y 1,77 veces el aproximado en las 10 llamadas (por ejemplo 395 reales contra 237 aproximados en `q1` con k=3), es decir, unos 2,4 caracteres por token con texto en español y Claude. Sirve para estimar el orden de magnitud, no el costo; para el costo se usa el conteo de la API.

**Límites conocidos de la medición** (de método, salvo los marcados como verificación):

- Una sola medición por (pregunta, k): la latencia de red varía entre corridas y los valores son un orden de magnitud, no estadística. La primera llamada de la corrida incluye el establecimiento de la conexión.
- El tiempo de generación incluye los reintentos del SDK de Anthropic (`max_retries=2`) si los hubo, sin marcarlo en el registro (no hubo ninguno en la corrida real). Con un 500 persistente cada llamada del script son 3 peticiones HTTP; el tope de llamadas no las cuenta.
- El costo de Claude es una estimación con los precios del proyecto, no contrastada con la consola de Anthropic; el costo de los embeddings no se estima. La muestra son 5 preguntas del corpus propio, con respuestas de entre 21 y 428 tokens.
- Si todas las llamadas a Claude fallan, el script deja las líneas con `error` y **termina con código 1** (pasada H2, verificado con un servidor falso con 500); si falla solo una parte, termina con 0 y hay que mirar el campo `error` o el informe.
- `preflight_rag.py` no modifica datos ni el conteo del índice, pero **no es de solo lectura**: abrir la colección con Chroma reescribe unos bytes de `chroma.sqlite3` (cambia su hash; verificado en una copia y en el índice real) y las fechas de los `.bin`, igual que hace cualquier script que abre el índice. Las pruebas comparan nombres y tamaños de archivos y, desde H2, los ids y los documentos de todos los chunks antes y después; no comparan el hash del sqlite, porque cambia.
- Huecos de pruebas conocidos y aceptados (mutantes vivos de la verificación de la pasada H2): con un solo registro que falla el script sigue saliendo con 0 (la prueba usa 2 o más), el código 130 de Ctrl-C con todos los registros en error no tiene prueba propia y el informe impreso una sola vez se comprueba con un texto que un informe duplicado no altera. Los huecos de la pasada H (zona horaria del `timestamp`, `max_tokens`, `reintentos_embed`, recorte del error, mensaje del tope, empate de tokens y decimales) se cerraron en H2. Además, el `except Exception` de `_correr` es más ancho que el error de embedding: un fallo de la búsqueda local saldría con el mensaje genérico de `describir_error_api`.

## Personalización de modelos: dataset de tuning para PEFT (Bloque 29)

La clase ubica RAG junto a otras tres técnicas y pide preparar y validar un dataset de tuning sin lanzar ningún entrenamiento. Regla de decisión que se aplicó: si al modelo **le falta información** (y esa información cambia), la técnica es RAG, y es lo que construyó la Unidad 2.5; si lo que falla es el **comportamiento** (formato, estilo, clasificación fija) con muchos ejemplos disponibles, la técnica es PEFT; el fine-tuning completo hace lo mismo a mayor costo, y el soft-prompting no está en la API gestionada de Gemini (solo existen `TUNING_MODE_FULL` y `TUNING_MODE_PEFT_ADAPTER`, verificado en el SDK). Enseñarle hechos nuevos con tuning es el error típico que advierte la clase.

**La tarea elegida** es clasificar mensajes de seguridad como un JSON de formato fijo, `{"categoria": ..., "prioridad": ...}`. Las categorías son las del router de `src/enrutamiento.py` (`incidente`, `vulnerabilidad`, `consulta`) y la prioridad es `alta`, `media` o `baja`. Es un problema de comportamiento y no de información: el router de hoy le pide al modelo una palabra y debe normalizar lo que devuelve (`_normalizar_categoria`), que es la inconsistencia de formato que un modelo afinado resolvería. El dataset no se usa todavía para entrenar nada.

**El dataset** (`JSON/dataset_tuning_clasificacion.json`) tiene 24 ejemplos ficticios (dominios `.ejemplo.test`, IP de documentación y `CVE-2099-*`), 8 por categoría y 8 por prioridad, sin ningún texto igual a los casos de evaluación de los otros bloques (lo comprueba una prueba). La prioridad se decide por la urgencia del mensaje y no por su categoría: alta es impacto activo, explotación en curso o una respuesta inmediata; media, afectación acotada o plazo corto; baja, sin urgencia.

| Categoría | Alta | Media | Baja |
|---|---|---|---|
| incidente | 5 | 2 | 1 |
| vulnerabilidad | 2 | 4 | 2 |
| consulta | 1 | 2 | 5 |

Cuatro etiquetas son discutibles y el dataset las deja como están: `ej_07` (un bucket público con descargas ya registradas: incidente, podría ser vulnerabilidad), `ej_09` (una clave de producción en un repositorio público: vulnerabilidad alta, podría ser incidente si ya se usó), `ej_11` (la única consulta con prioridad alta, porque pide una respuesta en plena reunión) y `ej_19` (un portátil perdido con el disco cifrado: incidente de prioridad media). Que las etiquetas sean las correctas es un juicio humano que el validador no puede comprobar.

**Cómo se ejecuta** (sin API, sin claves, sin red, no lee `.env` y no escribe ningún archivo):

```
python scripts/preparar_dataset_tuning.py
python scripts/preparar_dataset_tuning.py --dataset ruta/al/dataset.json
python scripts/preparar_dataset_tuning.py --demostrar-validador
```

Imprime `24 ejemplos, 0 problemas encontrados` y el balance por categoría y por prioridad con su razón máx/mín. Sale con 0 si el dataset está listo, 1 si tiene problemas, no se pudo leer o le falta algún valor permitido (una categoría o una prioridad sin ningún ejemplo, con un `Error:` por clave), y 2 con argumentos inválidos; un desbalance sin valores ausentes (razón máx/mín mayor que 2,0) es solo un aviso y no cambia el código. `--demostrar-validador` es el paso de la clase que comprueba que el validador detecta errores: agrega dos ejemplos rotos en memoria y exige que salgan exactamente cuatro problemas (`ejemplo 24: text_input vacio`, `ejemplo 24: output no tiene la clave "prioridad"`, `ejemplo 25: text_input duplicado del ejemplo 0 con otra salida` y `ejemplo 25: output no es JSON valido`).

**El validador** (`src/dataset_tuning.py`) devuelve todos los problemas, no solo el primero, con mensajes que no repiten el contenido del archivo (es dato, nunca instrucciones: todo lo que llega a la consola se limpia de caracteres de control). Revisa el texto (vacío, no textual, de más de 1000 caracteres, con caracteres de control o con un surrogate suelto que no se puede codificar en UTF-8, caso en que el SDK falla al serializar la petición con `PydanticSerializationError`), los duplicados normalizados (con aviso extra si la etiqueta contradice a la del primero), y la salida: texto codificable en UTF-8, JSON válido, objeto, sin claves repetidas, faltantes ni sobrantes, con valores dentro del conjunto permitido y en el formato canónico de `json.dumps` (claves en orden, separadores por defecto y acentos sin escapar, porque el modelo aprende el formato tal cual está escrito). Exige entre 10 y 500 ejemplos y que las claves y los valores permitidos del archivo coincidan con las constantes del código.

**Desvíos respecto de la clase** (declarados): el entregable único `dataset_tuning.py` se separa en un módulo, un script y un JSON, como el resto del proyecto, con nombres en español; el validador es más estricto que el de la clase, que tiene dos defectos reproducidos (con una salida que es JSON válido pero no un objeto, como `5`, se cae con `TypeError`, y una lista pasa por válida; y el balance de la actividad se calcula antes de validar y falla con `KeyError` si falta la clave) y no valida que los valores sean de un conjunto cerrado.

**Restricción de la plataforma (verificada, no se intentó ningún tuning real).** El bloque demostrativo de la clase, con `tuning_mode=TUNING_MODE_PEFT_ADAPTER` y `adapter_size=ADAPTER_SIZE_FOUR`, **falla con una `GEMINI_API_KEY`**: el SDK (`google-genai` 3.0.0) lanza `ValueError` ("adapter_size parameter is only supported in Gemini Enterprise Agent Platform mode, not in Gemini Developer API mode") antes de salir a la red, e igual con `tuning_mode`; también `batch_size` es solo para modelos de código abierto en esa plataforma. Es decir, PEFT con tamaño de adapter solo existe en el modo Vertex, no con la clave que usa este curso. Una prueba lo fija con un cliente de clave falsa y el socket bloqueado, así que si una versión futura del SDK cambia esto la prueba lo avisa. El `base_model` de la clase (`models/gemini-2.0-flash`) no se contrastó con la API: el SDK no lo valida y no se hizo ninguna llamada.

**Límites conocidos:**

- Un dataset muy desbalanceado (razón máx/mín mayor que 2,0) sale con código 0 y solo un aviso; una categoría o prioridad sin ningún ejemplo sale con código 1 desde la pasada I2 (la regla vive en el script, no en `validar_dataset`, que sigue dando solo los problemas de los ejemplos).
- Con 24 ejemplos no se sabe si alcanza para entrenar: la clase dice que la documentación oficial no fija un número verificado y hay que consultarla antes de enviar un dataset real. El mínimo de 10 del validador es el de la actividad, no el de la API.
- Las claves repetidas se detectan solo en el objeto de nivel superior de la salida, y un ejemplo con salida vacía no cuenta como "visto" para la regla de duplicados.
- Huecos de pruebas conocidos y aceptados (mutantes vivos de la verificación de la pasada I2; los de la pasada I se cerraron): el separador de coma y espacio del mensaje `Error:` cuando faltan dos valores de la misma clave, el caso en que hay a la vez problemas de los ejemplos y valores ausentes (no se prueba que ambos mensajes salgan) y el orden entre el mensaje de caracteres de control y el de UTF-8 cuando un texto tiene los dos. Cosmético sin efecto: `normalizado =normalizar_texto(...)` sin espacio en `src/dataset_tuning.py`.

## Personalización de modelos: costo y decisión (Bloque 30)

La clase cierra la unidad con tres piezas: cuánto cuesta preparar los datos de una personalización, un método de decisión por puntos entre RAG, fine-tuning completo y PEFT, y la redacción de un documento que fundamente la técnica elegida. No usa la API en el método; la única parte con costo es la medición del clasificador, que se pide aparte.

**Las cifras de la clase se reprodujeron por código**: 8 ejemplos a 5 minutos y USD 25 por hora son 0,67 h y USD 16,67; 200 ejemplos son 16,67 h y USD 416,67; el mini-ejercicio de 500 ejemplos con una segunda revisión del 20 % da 36,67 h, USD 733,33 y 4,6 jornadas de 8 horas. El método da 0/3/3 (empate entre fine-tuning completo y PEFT) para el caso 2 y 5/1/3 (RAG) para el caso 1, con las mismas cinco variantes de sensibilidad que la clase.

**El método endurecido** (`src/decision_personalizacion.py`, Python puro) corrige tres defectos del de la clase, reproducidos antes de escribirlo: un valor que no es booleano cuenta como verdadero (con las cinco respuestas en `"no"` da 5/3/6 y gana PEFT), una clave faltante lanza `KeyError`, y con las cinco respuestas en `False` reporta un empate de tres técnicas con 0 puntos cuando en realidad significa que ninguna técnica está justificada. Aquí solo valen `True` o `False`, las cinco claves deben estar exactas, y con máximo 0 el resultado es `sin_necesidad`. Los pesos son los de la clase. También valida los costos (rechaza negativos, NaN, infinitos y booleanos) y genera el esqueleto del documento sin escribir archivos.

**Cómo se ejecuta** (sin API, sin claves, sin red; `scripts/decidir_personalizacion.py`):

```
python scripts/decidir_personalizacion.py --respuestas ruta/respuestas.json
python scripts/decidir_personalizacion.py --respuestas ruta/respuestas.json --salida docs/decision_personalizacion.md
python scripts/decidir_personalizacion.py --estimar-etiquetado 200 5 25
python scripts/decidir_personalizacion.py --estimar-etiquetado 500 4 20 --revision 0.2 2
```

El JSON de respuestas trae el problema, las cinco respuestas booleanas y una evidencia de texto por cada una (la clase exige justificarlas). El script imprime los puntajes, la recomendación o el empate, la sensibilidad de invertir una respuesta a la vez y la evidencia, y sale con 1 con un `Error:` por cada problema del archivo. Con `--salida` crea el esqueleto en modo exclusivo: **nunca pisa un archivo existente** (la clase lo hacía en silencio) y falla antes de imprimir si no puede crearlo.

**Medición del clasificador del router** (`src/medicion_clasificador.py` y `scripts/medir_clasificador_router.py`, con costo). La regla de la clase para `problema_de_formato_o_tono` es que, en una muestra de respuestas reales, el 15 % o más incumple el formato. El router pide a Claude una sola palabra y normaliza lo que vuelve, así que se midió con los 24 ejemplos del dataset de tuning y solo la plantilla de clasificación. Se reportan dos proporciones: **estricta** (la respuesta cruda no es exactamente una categoría) y **efectiva** (no queda válida tras la normalización del router). El umbral se compara con enteros, no con flotantes.

```
python scripts/medir_clasificador_router.py --confirmar-api-real --max-tokens 256 --tope-usd 0.05
```

Sin `--confirmar-api-real` sale con 2 sin importar el cliente ni leer el dataset. Los topes están en el código: 75 llamadas, de 1 a 3 repeticiones, de 1 a 256 tokens y como máximo USD 0,05; el gasto se controla antes de cada llamada y un uso de tokens faltante o en cero detiene la corrida. El JSON de `--salida` se reemplaza de forma atómica y queda marcado `incompleto` si la corrida terminó antes (tope, uso faltante o Ctrl-C).

**Corrida real (2026-11-19, `claude-sonnet-5-5`, 24 llamadas, una pasada):**

| Corrida | Fallos estrictos | Fallos efectivos | Exactitud | Costo estimado |
|---|---|---|---|---|
| `--max-tokens 64` | 4 de 24 (16,7 %) | 2 de 24 (8,3 %) | 22 de 24 | USD 0,00831 |
| `--max-tokens 256` | 2 de 24 (8,3 %) | 0 de 24 (0,0 %) | 23 de 24 | USD 0,00904 |

La primera corrida superó el umbral en estricto, pero **dos de sus fallos eran respuestas vacías con 64 de 64 tokens de salida**: el tope de la propia medición se agotó antes de que saliera la categoría (el router real usa 2048). Con 256 desaparecieron. Los dos fallos estrictos que quedan son "Incidente" con mayúscula, que el router normaliza. El único error de categoría es `ej_09` (una clave de producción en un repositorio público: el dataset dice vulnerabilidad, el modelo dijo incidente), uno de los cuatro casos de etiqueta discutible del Bloque 29, y usó 196 tokens de salida frente a los 5 a 10 de los demás. Los costos se recalcularon por token desde el JSON y coinciden con el total. El resultado de la primera corrida no se versiona; el de la segunda es `docs/resultados_clasificador_router.json`.

**La decisión** (`docs/decision_personalizacion.md`): con las cinco respuestas en `False` el método da "ninguna técnica", y la medición la respalda: no hay información que cambie, el formato falla por debajo del umbral, el dataset es de 24 ejemplos y no hay variantes. El documento lo redactó Antares con el método corrido primero; el autor lo revisó, confirmó la respuesta de presupuesto (`False`) y lo firmó el 2026-11-25 (sección 5 del documento). Dice además que con una sola respuesta cambiada a `True` el método recomienda una técnica distinta cada vez (por ejemplo RAG por presupuesto en una tarea sin información externa): es una heurística con pesos fijos, no una prueba.

**Desvíos respecto de la clase** (declarados): el script de la actividad se separa en módulo y scripts con nombres en español, y el esqueleto no pisa archivos; la muestra de la medición son 24 ejemplos y no 20, porque el dataset es de 24 y balanceado; la prioridad no se mide porque el router no la predice.

**Límites conocidos:**

- La muestra son ejemplos que escribió la orquestadora, no mensajes reales, y es una sola pasada: sirve como indicio, no como estadística. El router del SDK puede reintentar, y el costo es una estimación con los precios del proyecto.
- Corregido en la pasada J2: `decidir_personalizacion.py --salida` con un carácter surrogate suelto (`\ud800`) en el problema ahora termina con `Error: problema no se puede codificar en UTF-8` (código 1, sin crear el archivo), y `--estimar-etiquetado` con un número de ejemplos de cientos de dígitos, o con `1e308`, termina con `Error: los numeros son demasiado grandes` en vez de un traceback o de imprimir `inf`.
- Con una `GEMINI_API_KEY`, el SDK (`google-genai` 3.0.0) solo llena las estadísticas facturables del tuning (`tuning_data_stats`) para Vertex: el bloque de costo de cómputo de la clase diría siempre "sin estadísticas". Y, como en el Bloque 29, el tuning con adapter no corre con esa clave.
- Huecos de pruebas: la pasada J2 cerró los mutantes vivos de la pasada J (bordes de tamaño, `truncado`, atomicidad, caracteres no imprimibles, claves de más y abreviaturas de opciones) y los de I2; quedan dos mutantes equivalentes sin efecto observable.

## Registro estructurado y defensa contra inyección de prompt (Bloque 31)

La clase pide dos cosas para un RAG: que cada consulta deje un rastro que se pueda seguir sin guardar su contenido, y que un texto recuperado no pueda darle órdenes al modelo. El código de este bloque no usa la API (modelos y servidores falsos); la corrida real se pide aparte con `--registro-json`. Se hizo en dos pasadas (K y K2) con una revisión de QA a ciegas y una revisión de seguridad en medio.

**Registro estructurado** (`src/registro_estructurado.py`). Cada evento es **una línea JSON ASCII** con `ts` (UTC, milisegundos), `level`, `msg`, `trace_id` y los datos del evento. Una clave de datos que coincide con una reservada no la pisa (sale como `datos_level`, `datos_msg`...), un valor que JSON no sabe escribir sale como el nombre de su tipo, los caracteres no ASCII salen escapados (un ESC o un separador de línea Unicode nunca llega crudo a la terminal) y de una excepción solo se escribe el **nombre del tipo**. El `trace_id` se fija con `contextvars`: un hilo nuevo no lo hereda, pero `copy_context().run`, `asyncio` y las ramas de LangChain sí. **Regla de gobernanza:** ningún log lleva el texto de una pregunta, de un fragmento, de una respuesta ni el mensaje de una excepción; solo longitud, huella (`sha256`, 12 caracteres hexadecimales), ids y tipos. `registrar_consumo` lee el uso de tokens y marca `uso_faltante` con el mismo criterio del script (`input_tokens` en 0 o no entero), y `HandlerDeTrazas` registra el ciclo de una cadena de LangChain (cadena, retriever, modelo y sus errores) sin contenido.

**Defensa contra inyección** (`src/defensa_inyeccion.py`). El prompt se divide en dos mensajes: las instrucciones van en el de **sistema** y el contexto recuperado solo en el **humano**, entre `<contexto>` y `</contexto>`, declarado como datos (la neutralización de etiquetas de la sección de recuperación se mantiene). Además, un detector de 8 patrones (español e inglés) descarta los fragmentos que parecen órdenes. Mira el texto en **dos vistas**: una donde los caracteres invisibles se borran (`ig` + espacio de ancho cero + `nora`) y otra donde se cambian por un espacio, que es lo que recibe el modelo; entre palabras acepta cualquier signo, guion bajo o markdown. La política ante un fragmento sospechoso es descartarlo, responder con el resto y marcar la consulta con `requiere_revision`, sin frenar la respuesta. El detector es **una señal, no una frontera**: el control principal son los roles y la regla de tratar el contexto como datos.

**Consulta instrumentada** (`src/consulta_rag.py`): `preparar_consulta` recupera, filtra y arma los mensajes; `responder_consulta` llama al modelo, registra el consumo y verifica las citas; `procesar_consulta` hace ambas bajo un `trace_id` propio y valida los precios antes de pagar la llamada. Los eventos son `consulta_recibida`, `documentos_recuperados`, `documento_descartado_por_inyeccion`, `contexto_filtrado`, `llamada_modelo`, `consumo_registrado` (o `consumo_sin_datos_de_uso`) y `respuesta_generada`, más `consulta_rechazada`, `recuperacion_fallo` y `modelo_fallo` en los errores; el script agrega `consumo_fallo` y `consulta_interrumpida`.

**En el script de consulta** (`scripts/consultar_corpus.py`): el modelo recibe los dos mensajes y cada caso corre bajo su propio `trace_id`. El JSON de resultados agrega `huella_prompt`, y por caso `trace_id`, `descartados` (índice, fuente, patrones, longitud y huella, nunca el contenido) y `requiere_revision`; cada recuperado lleva `descartado` y `rango_en_contexto` (la posición que tiene en el contexto, porque las citas `[n]` se numeran solo entre los que se conservan), y `acierto_recuperacion` cuenta únicamente lo que vio el modelo. En consola un caso con descartados imprime `REQUIERE REVISION` y la fuente y los patrones de cada uno, y el recuperado descartado termina con `[DESCARTADO]`.

```
python scripts/consultar_corpus.py --confirmar-api-real --registro-json --tope-usd 0.03 2> log.jsonl
```

`--registro-json` está apagada por defecto: sin ella stdout, stderr, código de salida y JSON de resultados quedan como antes (salvo las claves nuevas). Con ella, stderr lleva un JSON por evento; ese mismo stderr puede traer además los mensajes finales de la herramienta (`Error: ...`, `Interrumpido por la persona`), que no son logs. Con un fallo del modelo el log solo dice `modelo_fallo` y el tipo, pero el mensaje de la excepción sigue yendo al JSON de resultados y a la consola, como antes del bloque.

**Primera corrida real con registro** (2026-11-21, `k=3`, las mismas 8 preguntas, con `--registro-json`; resultados en [docs/resultados_recuperacion_registro.json](docs/resultados_recuperacion_registro.json) y log completo en [docs/registro_consulta_real.jsonl](docs/registro_consulta_real.jsonl)): terminó con código 0, 8 consultas de embeddings y 8 llamadas a Claude, 4064 tokens de entrada y 1511 de salida, costo estimado USD 0,02324 (se había estimado unos USD 0,02, con tope de 0,03). El log tiene 48 líneas, todas JSON válido y ASCII: seis eventos por caso (`consulta_recibida`, `documentos_recuperados`, `contexto_filtrado`, `llamada_modelo`, `consumo_registrado`, `respuesta_generada`), 8 `trace_id` distintos que coinciden con los del JSON de resultados y ningún WARNING ni ERROR. **No contiene ninguna pregunta, fragmento ni respuesta** (se comprobó buscando esos textos en el log; tampoco aparece ninguna clave del `.env`). El costo del log, el calculado por token desde el JSON y el del script coinciden. Con datos reales el filtro no descartó nada (`requiere_revision` falso en los 8 casos).

Contra la corrida del Bloque 26 (más arriba): mismos documentos recuperados, mismo `acierto_recuperacion`, mismas citas válidas (en `q3` en otro orden) y las tres preguntas sin respuesta siguen admitiendo la falta de contexto con la frase fija. El costo subió de USD 0,01754 a 0,02324 (+32 %): la entrada crece exactamente 55 tokens por caso (el prompt de sistema es más largo) y la salida sube de 1029 a 1511 tokens, sobre todo en casos donde la respuesta visible no cambió (en `q8` es la misma frase de 51 caracteres con 21 tokens de salida antes y 131 ahora). Probablemente son tokens de razonamiento que no se ven en el texto, pero no se verificó (el resultado no guarda los bloques de la respuesta) y con una sola corrida por pregunta no se puede atribuir al cambio de rol.

**Qué encontró la revisión independiente y se corrigió en K2** (la QA a ciegas reprodujo 1 hallazgo alto, 3 medios y 5 bajos, y la revisión de seguridad dio el veredicto "apto con condiciones"):

- Un carácter invisible **entre palabras** anulaba el filtro: el detector los borraba sin dejar espacio (`ignoreprevious`) pero el contexto que ve el modelo los cambia por un espacio, así que `ignore previous instructions` llegaba completo. Por eso el detector ahora mira las dos vistas.
- Faltaban frases directas comunes (`ignorar las instrucciones anteriores`, `ignore the previous instructions`) y los patrones no toleraban puntuación ni markdown entre palabras.
- Tras descartar un documento las citas se renumeraban, pero `recuperados` y `acierto_recuperacion` seguían contando el descartado.
- Con `--registro-json`, un error del modelo dejaba en stderr un traceback con el mensaje de la excepción, y el script no emitía `llamada_modelo` ni `modelo_fallo`.
- Los precios inválidos se descubrían después de pagar la llamada.

Verificación propia de K2: 1171, 1215, 1230 y 1241 pruebas por commit, cliente real de Anthropic contra un servidor local falso (nueve escenarios sin fugas), el script sin la opción comparado con la versión anterior, y 77 mutantes de los que mueren 63; los vivos son 8 huecos de pruebas, 5 equivalentes o cosméticos y 1 casi equivalente. El detector no marca ninguno de los 10 documentos del corpus ni de los 323 párrafos del README y de `docs/` que existían antes de esta sección (esta sección, que cita los ataques, sí se marcaría), y analiza 5 MB de texto en unos 1,5 s.

**Límites conocidos:**

- El detector no cubre homoglifos de otros alfabetos (una `о` cirílica dentro de `ignora`), sinónimos, letras separadas por espacios, palabras en otro orden, entidades HTML ni una marca combinante que se compone con la letra anterior (a + U+0301 pasa a á): NFKC la funde con la letra y el modelo ve la palabra pegada. Desde K3 una marca que no se compone (selector de variación U+FE0F, U+034F, U+0338) **entre** palabras sí se detecta.
- Falsos positivos medidos: `nuevo sistema:`, `system prompt:`, `olvida todo lo anterior`, `ignorar las indicaciones previas` en un procedimiento, `forget the previous context` y `revela tu prompt` en un manual, además de cualquier documento que **cite** el ataque. El fragmento legítimo se descarta y la consulta queda marcada para revisión.
- `huella` es un `sha256` sin sal: un texto corto o predecible se confirma comparando huellas, así que el log de preguntas y respuestas se trata como sensible. Un valor NaN o infinito, o una clave que no es texto, deja ilegibles todos los datos de ese evento.
- Con las funciones de consulta anteriores al Gate 2 (`src/consulta_rag.py` y `scripts/consultar_corpus.py`), si todos los fragmentos se descartan igual se llama al modelo con `(sin contexto recuperado)` y cuesta tokens; el servicio del Gate 2 (`src/servicio_rag.py` y `scripts/consultar_servicio.py`) corta antes y no invoca al modelo (estado `sin_contexto_confiable`, probado con un modelo falso). `procesar_consulta` sigue sin llamador productivo y el script de corpus duplica su lógica. El filtro no tiene tope de tamaño (es lineal). `chromadb` 1.5.9 tiene 4 avisos de seguridad sin arreglo disponible, de alcance probablemente no aplicable (sin verificar).
- Huecos de pruebas de K2: cerrados en K3 (un invisible dentro de una palabra con las dos vistas, el rango con el descartado en el medio, un error de cadena hija y cinco ejemplos de patrones: indicaciones, prompts, revela con separador, forget the previous prompt y el relleno U+115F). Nota: el README de K2 decía cuatro ejemplos; eran cinco.
- Sin verificar: el comportamiento con el modelo real ante un fragmento hostil que **evada** el detector, y los eventos de error con la API real: la corrida real no tuvo descartes ni fallos, y esos casos solo se probaron con el cliente real contra un servidor falso.

## Versionado de prompts, supervisión y riesgos (Bloque 32)

La Unidad 2.7 cierra con tres piezas: un registro de versiones del prompt, un punto de supervisión humana para desplegar una versión nueva y una matriz de riesgos. Esta sección documenta las tres (pasadas L1, L2 y L3: el registro, las guardas de la suite, la supervisión del despliegue con langgraph y la matriz de riesgos). Nada de este bloque llama a la API ni usa red.

**Registro de versiones del prompt** (`src/registro_prompts.py`, `scripts/registrar_prompt.py`, `docs/registro_prompts.jsonl`). Es un JSONL de solo anexar: cada versión es una línea con nombre, versión, huella, huella anterior (encadena las versiones), texto, autor, fecha UTC, nota, modelo y parámetros. Reutiliza la huella de `registro_estructurado` (sha256 de 12 caracteres). Corrige tres defectos del código de la clase: el registro solo vivía en memoria, la fecha era la local y sin zona, y aceptaba la misma versión dos veces. La línea se escribe en ASCII, así que un carácter invisible del texto queda visible en el diff de git. La lectura es estricta (claves y su orden, tipos, huella contra texto y fecha) y su error cita solo el número de línea, nunca el contenido.

```bash
python scripts/registrar_prompt.py --listar
python scripts/registrar_prompt.py --verificar    # sale con 1 si el prompt del código no es el último registrado
python scripts/registrar_prompt.py --diff
python scripts/registrar_prompt.py --registrar --origen actual --autor "nombre" --nota "qué cambió y por qué"
```

El historial inicial (sembrado el 2026-11-22) tiene dos versiones de `prompt_rag`: la v1 (`72be37be6604`) es la plantilla de un solo mensaje de los Bloques 24 a 26 (`recuperacion.PROMPT_TEMPLATE`) y la v2 (`96f508a5ab49`) es la de roles separados del Bloque 31, el prompt de la corrida real de K2. La v1 se reconstruye del código que se conserva y su fecha es la del registro, no la de su primer uso. Una prueba (`test_l2_al`) compara el prompt del código con el registro commiteado: si cambias el prompt sin registrar la versión nueva, la suite falla.

**Guardas de la suite** (`test/unit/conftest.py`). Un sexto guardián falla toda prueba que escriba en `docs/registro_prompts.jsonl`. Una guarda de red (hallazgo K-04 de la revisión de seguridad) falla toda prueba que llame a `connect`, `connect_ex`, `create_connection`, `getaddrinfo` o `gethostbyname` con `RuntimeError("RED BLOQUEADA...")`; no tiene escape ni marcador y no impide `socketpair`, que usa asyncio.

**Supervisión del despliegue con langgraph** (`src/supervision_despliegue.py`, `scripts/comparar_supervision.py`). Un grafo de tres nodos que envuelve la aprobación humana de `src/supervision.py` sin modificarla: `proponer` valida la cadena del registro y fija la huella pendiente (no escribe), `supervisar` detiene el grafo con `interrupt` y `desplegar` anexa la versión con `registrar_version` solo si la decisión fue aprobada. El aprobador (`esperar_aprobacion`, `denegar_siempre`) sigue decidiendo; el grafo solo guarda el estado entre la propuesta y la decisión.

```python
from supervision import esperar_aprobacion
from supervision_despliegue import SupervisionDespliegue

supervision = SupervisionDespliegue("docs/registro_prompts.jsonl")   # InMemorySaver
propuesta = supervision.iniciar("prompt_rag", texto, "autor", "nota")  # {"thread_id": ..., "pendiente": {...}}
supervision.reanudar(propuesta["thread_id"], True)                    # {"estado": "desplegado", ...}
# o todo junto: supervision.ejecutar("prompt_rag", texto, "autor", "nota", aprobador)
```

Reglas:

- Solo un `True` exacto despliega; cualquier otro valor (`"s"`, `1`, `None`) rechaza en silencio y no escribe.
- Cada propuesta recibe un `thread_id` único (uuid4). Un `thread_id` desconocido o ya resuelto lanza `ValueError` y no puede desplegar dos veces.
- La huella pendiente se comprueba al decidir: si otra versión del mismo prompt se anexó (o el registro desapareció) entre la propuesta y la aprobación, el resultado es `desactualizado` y no se escribe.
- `proponer` rechaza una cadena rota y un texto idéntico al de la última versión, antes de preguntar.
- El payload de la interrupción lleva nombre, versión propuesta y huellas, no el texto del prompt.
- Con `SupervisionDespliegue(ruta_registro, ruta_sqlite)` el estado vive en SQLite (`langgraph-checkpoint-sqlite`) y la decisión puede tomarse desde otra instancia o proceso. El checkpoint SÍ guarda el texto del prompt: trátalo como el registro. Una base que no se puede abrir lanza `ValueError` sin la ruta.

`python scripts/comparar_supervision.py` da las mismas 10 respuestas (s, S, " s ", n, vacío, "si", EOF, línea parcial, línea de más de 1024 bytes, plazo vencido) por una tubería real a `esperar_aprobacion` y al grafo, imprime la tabla de decisiones y sale con 1 si alguna difiere. No usa API ni red.

Límites: no hay bloqueo de archivo entre la comprobación de la huella y la anexión (ventana teórica, igual que el registro); langgraph agrega una nota (`__notes__`) a las excepciones que salen de un nodo.

**Matriz de riesgos** (`src/matriz_riesgos.py`, `scripts/matriz_riesgos.py`, `docs/matriz_riesgos.json`). Una fila por riesgo, con probabilidad e impacto de 1 a 5 (estimaciones, no mediciones), mitigación, evidencias en el código, punto de supervisión asociado y una nota. El puntaje es probabilidad por impacto y la categoría usa cortes fijados de antemano: CRITICO desde 15, ALTO desde 9, MEDIO desde 4 y BAJO por debajo. `esta_en_el_limite` marca la fila si mover probabilidad o impacto un punto cambia su categoría.

```bash
python scripts/matriz_riesgos.py    # tabla ordenada; sale con 1 si la matriz es inválida o una evidencia no existe en el código
```

Cada evidencia es `ruta.py::simbolo` y se comprueba leyendo el archivo con `ast` (no se importa ni se ejecuta nada): el archivo debe estar dentro del proyecto y el símbolo debe ser una definición de primer nivel (función, clase o constante). Si una mitigación cita código que ya no existe, la suite falla. La lectura es estricta (claves y su orden, enteros de 1 a 5 sin `bool`, riesgos sin repetir) y sus errores citan solo el número de fila, nunca el contenido ni la ruta. Las 11 filas iniciales las prellenó Antares y salen todas `[AL LIMITE]`, como en la clase: los cortes están juntos frente al rango de 1 a 25, así que la marca no distingue entre ellas; trata la categoría como aproximada y revísala con otra persona. Incluye el consumo descontrolado de llamadas y costo (OWASP LLM10:2025, referencia de memoria sin contrastar).

Límites de la matriz: que el símbolo exista no prueba que la mitigación funcione ni que el texto de la fila describa bien el código; el texto de mitigaciones y notas no se contrastó contra el código más allá de la existencia de los símbolos.

**Límites conocidos del registro:**

- No hay bloqueo de archivo: dos escrituras simultáneas pueden mezclarse.
- La huella no impide que alguien edite el texto y la huella a mano; la cadena de huellas y el control de versiones de git lo muestran, no lo evitan.
- Si `--registro` apunta a una carpeta o a un archivo ilegible o no escribible, el script sale con 1 y un mensaje que nombra solo el tipo del error; y `registrar_version` se niega a anexar si la cadena de cualquier prompt ya está rota (L1b).
- `diff_prompts` compara por palabras, como el código de la clase, así que no conserva los saltos de línea.

**Límites conocidos de la supervisión y de las trazas (QA final, 2026-11-23):**

- `reanudar` escribe en la `ruta_registro` guardada en el checkpoint, no en la de la instancia que reanuda: con una base SQLite compartida, quien reanuda no elige el destino, y quien pueda escribir en la base lo elige. Trata la base como parte del registro.
- Un turno del agente cortado por un tope o por un error no escribe traza, aunque ya se haya aprobado y ejecutado una acción de riesgo alto: la decisión queda solo en el JSON de salida y `scripts/revisar_trazas.py` no la ve.
- `scripts/registrar_prompt.py --registrar` anexa una versión sin pasar por `SupervisionDespliegue`: el grafo solo lo usa `scripts/comparar_supervision.py`.

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

No usan la API real ni la red (todo está mockeado; una guarda de `test/unit/conftest.py` hace fallar cualquier prueba que intente abrir una conexión o resolver un nombre) y la suite completa tarda unos 95 segundos. `pytest.ini` limita la suite a `test/unit`; la prueba de consistencia sigue siendo el script aparte con `PYTHONPATH=src` y con costo (9 llamadas).

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
README2.md                 guía de revisión del Gate 2: instalación, indexado, consulta, pruebas y evidencia por criterio
test_gate2.py              las cuatro pruebas de los criterios no compensables del Gate 2 (fuera de test/unit: se corre aparte)
verificar_entorno.py       comprueba Python, paquetes, claves (sin mostrarlas) e índice antes de la corrida final
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
  supervision_despliegue.py  grafo de langgraph (proponer, supervisar, desplegar) que envuelve la aprobación para desplegar versiones de prompts; InMemorySaver o SqliteSaver
  matriz_riesgos.py          matriz de riesgos: puntaje, categoría, marca de límite y validación de evidencias contra el código con ast
  trazas.py                trazas JSONL por turno y marcas para su revisión
  embeddings_gemini.py     crea los embeddings de Gemini a pedido y clasifica los errores de la API (límite, clave u otro)
  indexacion.py            carga y valida el corpus, lo parte en chunks con ids deterministas e indexa en Chroma de forma idempotente
  recuperacion.py          recuperación con distancia, contexto numerado, prompt que trata el contexto como datos y verificación de citas
  metricas_recuperacion.py precisión, exhaustividad y F1 de la recuperación por chunk y validación de los casos de evaluación
  medicion_rag.py          mide la latencia por etapa (embed, búsqueda, generación) y el costo de una consulta RAG y resume los registros
  dataset_tuning.py        valida de forma estricta el dataset de tuning (texto, salida JSON, formato canónico, duplicados) y calcula su balance, sin API
  decision_personalizacion.py  método de decisión entre RAG, fine-tuning y PEFT con booleanos estrictos, costo de etiquetado y esqueleto del documento, sin API
  medicion_clasificador.py     clasifica la respuesta cruda del router (exacta, válida, correcta) y resume el formato contra el umbral de la clase
  registro_estructurado.py     logs JSON de una línea sin contenido, `trace_id` por contexto, huellas, consumo y handler de callbacks de LangChain
  defensa_inyeccion.py         detector de inyección con dos vistas del texto, filtro de documentos y mensajes de sistema y humano
  registro_prompts.py          registro de versiones de prompts en JSONL de solo anexar: huella, cadena de versiones, lectura estricta y comparación con el código
  consulta_rag.py              consulta RAG instrumentada: prepara (recupera y filtra), responde (modelo, consumo y citas) y procesa con un `trace_id` por consulta
  servicio_rag.py              servicio RAG del Gate 2: ejecutar_servicio y consultar; corta antes del modelo sin contexto confiable y exige al menos una cita válida
scripts/
  ensayo_rate_limit_simulado.py  ensayo de rate limit simulado con mocks
  ejecutar_pipeline.py           pipeline de texto sobre respuestas reales de Claude (con costo)
  ejecutar_metricas.py           precisión, cobertura y F1 sobre respuestas reales de Claude (con costo)
  ensayo_rate_limit_real.py      ensayo de rate limit real contra Claude (con costo)
  ejecutar_cadenas.py            chain de tres pasos y enrutamiento sobre respuestas reales de Claude o Gemini (con costo)
  ejecutar_agente.py             agente con herramientas y memoria sobre respuestas reales de Claude, con topes de llamadas y de gasto, aprobación y trazas (con costo)
  revisar_trazas.py              revisa las trazas del agente y marca candidatos a revisar (sin costo)
  indexar_corpus.py              revisa (sin API) o indexa el corpus en Chroma con embeddings de Gemini
  consultar_corpus.py            recuperación semántica con embeddings de Gemini y respuestas de Claude con citas verificadas, filtro de inyección y logs JSON opcionales (con costo)
  consultar_servicio.py          consulta de punta a punta con el servicio del Gate 2 (llamada real a Gemini y Claude; exige --confirmar-api-real)
  evaluar_recuperacion.py        evalúa la recuperación por chunk con embeddings de Gemini, sin llamar a Claude (con costo en embeddings)
  medir_latencia_costo.py        mide la latencia por etapa y el costo del flujo RAG con Gemini y Claude y anexa un registro por llamada (con costo)
  preflight_rag.py               revisa sin API el índice (conteo e ids contra el corpus) y que el directorio esté ignorado por git
  registrar_prompt.py            lista, verifica contra el código, registra y compara versiones del prompt del RAG (sin API ni red)
  comparar_supervision.py        compara la aprobación previa con el grafo de supervisión con las mismas 10 respuestas (sin API ni red)
  matriz_riesgos.py              tabla de la matriz de riesgos; sale con 1 si es inválida o una evidencia no existe (sin API ni red)
  preparar_dataset_tuning.py     prepara y valida el dataset de tuning y muestra su balance, sin API ni claves
  decidir_personalizacion.py     aplica el método de decisión a un JSON de respuestas o estima el costo de etiquetar, sin API ni claves
  medir_clasificador_router.py   mide el formato y la exactitud del clasificador del router sobre el dataset de tuning (con costo)
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
  unit/conftest.py                     guardianes automáticos: fallan una prueba que escriba en trazas/trazas_agente.jsonl, en vectorstore/, en docs/resultados_evaluacion_recuperacion.json, en docs/registros_latencia_costo.jsonl, en docs/resultados_clasificador_router.json o en docs/registro_prompts.jsonl; y una guarda de red que falla cualquier prueba que intente conectarse
  unit/test_conftest_guardian.py       pruebas de las funciones que miden el estado de trazas/ y vectorstore/
  unit/test_guardianes.py              prueba, en un pytest aparte, que los seis guardianes fallan de verdad
  unit/embeddings_falsos.py            embeddings falsos por tema para probar el orden de recuperación (sin pruebas propias)
  unit/test_indexacion.py              pruebas de la indexación con Chroma real y embeddings falsos
  unit/test_indexar_corpus.py          pruebas del script de indexación (modo revisión, errores de la API)
  unit/test_recuperacion.py            pruebas de la recuperación, el contexto numerado, el prompt y las citas
  unit/test_consultar_corpus.py        pruebas del script de consulta, con un modelo falso
  unit/test_servicio_rag.py            pruebas del servicio del Gate 2: estados, citas, corte antes del modelo y logs sin contenido (13)
  unit/test_consultar_servicio.py      pruebas del script de consulta del servicio: topes, códigos de salida y registro JSON (16)
  unit/test_verificar_entorno.py       pruebas de verificar_entorno.py: versión mínima de Python, claves sin mostrar valores y lectura del .env solo del directorio actual (9)
  unit/test_metricas_recuperacion.py   pruebas de las métricas por chunk, los casos de evaluación y su coherencia con el corpus
  unit/test_evaluar_recuperacion.py    pruebas del script de evaluación (Chroma real, embeddings falsos, topes, fallos y resultados parciales)
  unit/test_medicion_rag.py            pruebas de la medición por etapa, los registros, el costo, el uso faltante y el resumen
  unit/test_medir_latencia_costo.py    pruebas del script de medición (Chroma real, embeddings y modelo falsos, topes, fallos y archivo de solo anexar)
  unit/test_preflight_rag.py           pruebas del chequeo previo del flujo RAG, sin API
  unit/test_dataset_tuning.py          pruebas del validador y del balance del dataset de tuning, con la restricción de la API de desarrolladores fijada sin red
  unit/test_preparar_dataset_tuning.py pruebas del script del dataset de tuning (salidas, códigos de salida, texto hostil y que no usa API)
  unit/test_decision_personalizacion.py  pruebas del método de decisión endurecido, la sensibilidad, los costos y el esqueleto del documento
  unit/test_decidir_personalizacion.py   pruebas del script de decisión (archivo de respuestas, salida exclusiva, costos, texto hostil y que no usa API)
  unit/test_medicion_clasificador.py     pruebas de la clasificación de la respuesta cruda, el resumen, el umbral y el costo
  unit/test_medir_clasificador_router.py pruebas del script de medición (bandera, topes, uso faltante, Ctrl-C, salida atómica y modelo falso)
  unit/test_registro_estructurado.py     pruebas del formateador JSON, el `trace_id`, las huellas, el consumo y el handler de callbacks
  unit/test_defensa_inyeccion.py         pruebas del detector (evasiones, límites y falsos positivos), el filtro de documentos y los mensajes
  unit/test_registro_prompts.py          pruebas del registro de versiones: validaciones, cadena de huellas, lectura estricta y diff
  unit/test_registrar_prompt.py          pruebas del script del registro (listar, verificar, registrar y diff) y del registro commiteado
  unit/test_supervision_despliegue.py    pruebas del grafo de supervisión (aprobar, rechazar, huella pendiente, thread_id, SQLite) y de la comparación previo vs grafo
  unit/test_matriz_riesgos.py            pruebas de la matriz: puntaje, cortes, límite, validación de filas, evidencias con ast y script
  unit/test_guarda_de_red.py             pruebas de la guarda de red de la suite
  unit/test_consulta_rag.py              pruebas de la consulta instrumentada con un vectorstore y un modelo falsos
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
  corpus_miniapp.json              10 documentos ficticios de Taller Aurora para el RAG
  casos_recuperacion.json          8 preguntas del RAG (6 con respuesta en el corpus y 2 sin ella)
  evaluacion_recuperacion.json     14 casos de evaluación con los chunks relevantes etiquetados (11 con respuesta y 3 sin ella)
  casos_latencia_costo.json        5 preguntas de la medición de latencia y costo del RAG
  dataset_tuning_clasificacion.json  24 ejemplos ficticios para PEFT: mensajes de seguridad con su categoría y prioridad en JSON (8 por categoría y por prioridad)
docs/                      decisión de arquitectura, diagrama y evidencias
  registro_rate_limit.md   registro del ensayo de rate limit (simulado)
  registro_pipeline.md     registro del pipeline de texto (se genera al correr el script)
  registro_rate_limit_real.md  registro del ensayo de rate limit real (se genera al correr el ensayo)
  registro_rate_limit_real_concurrente.md  registro del ensayo concurrente (se genera al correr el ensayo con --concurrencia mayor que 1)
  resultados_metricas.json  resultados de la corrida real de métricas (se genera al correr ejecutar_metricas.py)
  resultados_cadenas.json  resultados de la corrida real de las cadenas (se genera al correr ejecutar_cadenas.py)
  resultados_agente.json   resultados de la primera corrida real del agente, 8 turnos (se genera al correr ejecutar_agente.py)
  resultados_agente_v2.json  resultados de la segunda corrida real, 10 turnos y una aprobación interactiva
  resultados_agente_gemini.json  resultados de la corrida real del agente con Gemini, 10 turnos (2026-11-23)
  resultados_agente_gemini_conversacion.json  repetición de los 2 turnos de conversación con Gemini tras el 503 de conversacion_1 (2026-11-23)
  resultados_recuperacion.json  resultados de la primera corrida real de la recuperación con citas
  resultados_recuperacion_registro.json  resultados de la corrida real con `--registro-json` y filtro de inyección (Bloque 31)
  registro_consulta_real.jsonl  log JSON de esa corrida, 48 líneas sin contenido de preguntas, fragmentos ni respuestas
  registro_prompts.jsonl  historial de versiones del prompt del RAG (JSONL de solo anexar; lo escribe scripts/registrar_prompt.py)
  matriz_riesgos.json     matriz de riesgos del proyecto (11 filas; la valida scripts/matriz_riesgos.py)
  resultados_evaluacion_recuperacion.json  resultados de la primera evaluación real de la recuperación por chunk (se genera al correr evaluar_recuperacion.py)
  registros_latencia_costo.jsonl  registros de latencia y costo del flujo RAG, una línea por llamada (se anexa al correr medir_latencia_costo.py)
  limites_autonomia_agente.md  límites de autonomía del agente: riesgo por herramienta, aprobación, trazas y límites conocidos
  revision_manual.json     revisión manual de 10 casos con la rúbrica de tres preguntas
  limites_metrica.md       límites conocidos de la métrica y recomendación de uso
  resultados_clasificador_router.json  resultados de la medición real del clasificador del router, 24 llamadas con 256 tokens (se genera al correr medir_clasificador_router.py)
  decision_personalizacion.md  decisión de personalización del clasificador del router (borrador pendiente de revisión del autor)
```

## Problemas comunes

| Error | Causa probable |
|---|---|
| `Credencial inválida o sin permiso` | La API key (`CLAUDE_API_KEY` o `GEMINI_API_KEY`) falta o es inválida en `.env`, o el modelo no está disponible para tu clave (401/403) |
| `Ocurrio un error de conexión: 400` | Petición mal formada, revisa el `output_schema` del prompt |
| `ModuleNotFoundError: No module named 'api_claude_refactor'` | Falta `PYTHONPATH=src` al ejecutar la prueba de consistencia |
