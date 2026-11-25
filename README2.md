# README2: guía de revisión del Gate 2

Esta guía permite instalar el proyecto desde cero, indexar el corpus, hacer una consulta de punta a punta y comprobar los cuatro criterios no compensables y las seis dimensiones de la rúbrica del Gate 2. El detalle técnico de cada bloque está en [README.md](README.md); aquí solo se dice dónde está la evidencia y cómo comprobarla.

Estado de lo medido: 2026-11-25, sobre el commit a2567d6 (el contenido del repositorio sin este archivo), suite de pruebas 1567 passed y `test_gate2.py` 4 passed, medidos con el `.venv` del repositorio. La instalación limpia con un entorno virtual nuevo se probó antes de agregar la validación de supervisión de la matriz de riesgos (1556 passed en esa copia); ver "Qué no está verificado".

## 1. Qué modelo usa el proyecto

- **Generación de respuestas: Claude Sonnet 5.5** (`claude-sonnet-5-5`, clave `CLAUDE_API_KEY`). Es el modelo del proyecto: el servicio, el CLI de consulta y las corridas reales documentadas usan Claude.
- **Embeddings: Gemini** (`gemini-embedding-001`, clave `GEMINI_API_KEY`). Solo para indexar y para convertir la pregunta en vector.
- Gemini como modelo de generación existe solo en comparaciones (cadenas y agente, ver README.md). **No forma parte del servicio del Gate 2** y `scripts/consultar_servicio.py` no lo soporta.
- Consecuencia práctica: para una consulta real hacen falta las dos claves. Sin `CLAUDE_API_KEY` se puede indexar pero no consultar.

## 2. Instalación desde cero

Requiere Python 3.10 o superior (probado con 3.14.9) y Git.

```bash
git clone <url-del-repositorio> curso_ia && cd curso_ia
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
cp .env.example .env               # completa CLAUDE_API_KEY y GEMINI_API_KEY
git check-ignore .env              # debe imprimir ".env"
python verificar_entorno.py
```

Antes de indexar, `verificar_entorno.py` termina con código 1 y una línea `Pendiente:` que dice qué falta (claves o índice). Es lo esperado. Imprime la versión de cada paquete, dice solo "definida" o "FALTA" para cada clave y nunca muestra su valor. Lee el `.env` únicamente del directorio actual.

## 3. Indexar el corpus

El índice (`vectorstore/`) **no está en el repositorio** (está en `.gitignore`), así que se construye con tu propia clave de Gemini. El corpus es `JSON/corpus_miniapp.json` (documentos de una empresa ficticia).

```bash
python scripts/indexar_corpus.py                       # modo revisión: sin API, valida el corpus
python scripts/indexar_corpus.py --confirmar-api-real  # indexa con embeddings de Gemini (llamada real)
python scripts/preflight_rag.py                        # comprueba índice, corpus y .gitignore, sin API
python verificar_entorno.py                            # ahora debe terminar con código 0
```

- Los ids de los chunks son deterministas (`<documento>_<n>`): reindexar el mismo corpus no duplica, un documento editado se reemplaza y uno quitado se borra después de agregar (actualización incremental, README.md sección del Bloque 24 a 26).
- Dos indexaciones simultáneas sobre el mismo directorio se bloquean con un candado.
- La primera indexación real documentada creó 29 chunks con una sola petición de embeddings (README.md).

## 4. Hacer una consulta de punta a punta

```bash
python scripts/consultar_servicio.py --confirmar-api-real --pregunta "¿Cuál es el plazo de parcheo?"
echo $?
```

- Sin `--confirmar-api-real` el script sale con código 2 sin cargar ninguna librería de modelos ni leer claves (comprobado).
- Hace una petición de embeddings (Gemini) y una llamada a Claude, con `--max-tokens` 512 por defecto (máximo 1024) y `--k` 3 por defecto (máximo 10). La pregunta admite hasta 500 caracteres.
- Imprime la respuesta y, debajo, una sección `Fuentes:` con el número y el documento de cada cita válida. Código 0 si hubo respuesta; 1 con un mensaje corto si falla algo; 130 con Ctrl-C.
- `--registro-json` escribe en stderr un log JSON por evento (sin contenido). Ejemplo para el criterio 3: `python scripts/consultar_servicio.py ... --registro-json 2> logs.jsonl` y luego `grep -i ana.perez logs.jsonl` no debe imprimir nada.

**Qué decide el servicio** (`src/servicio_rag.py`, función `consultar`; `ejecutar_servicio(pregunta, vectorstore, modelo)` devuelve `(respuesta, fuentes)`):

| Situación | Estado | Qué devuelve |
|---|---|---|
| Hay contexto limpio y el modelo cita al menos una fuente válida | `respondida` | La respuesta más la lista de fuentes |
| Todos los fragmentos recuperados se descartan por sospecha de inyección | `sin_contexto_confiable` | "No tengo informacion suficiente y verificada para responder esto." El modelo **no se invoca** |
| El modelo responde sin ninguna cita, o cita un número que no existe | `sin_cita_valida` | El mismo mensaje seguro, sin fuentes |
| El modelo responde que el contexto no alcanza | `sin_respuesta` | Esa frase, sin fuentes |

Así, ninguna respuesta con texto del modelo sale sin al menos una cita real.

## 5. Pruebas

```bash
python -m pytest test_gate2.py -v      # las 4 pruebas de los criterios no compensables
python -m pytest -q                    # suite completa (1567 pruebas en a2567d6)
```

Todas corren sin red ni claves (modelo y vector store falsos o en carpetas temporales). `test_gate2.py` está fuera de `test/unit/` a propósito y se ejecuta aparte, como pide el curso.

## 6. Los cuatro criterios no compensables

| Criterio | Cómo se comprueba | Estado |
|---|---|---|
| 1. El pipeline corre de punta a punta sin excepciones | `python -m pytest test_gate2.py -k test_1` y, con claves, el CLI de la sección 4 (`echo $?` = 0) | Prueba automática: pasa. CLI con APIs reales: una consulta, código 0 (2026-11-25) |
| 2. Cada respuesta cita su fuente | `test_2`; en `servicio_rag.consultar` la lista de fuentes sale solo de citas válidas y hay un mensaje seguro si no hay ninguna | Prueba automática y 13 pruebas unitarias del servicio: pasan |
| 3. Los logs no exponen PII ni contenido crudo | `test_3` (correo de prueba en la pregunta; ni el correo ni la pregunta completa aparecen en los logs); los eventos llevan longitud y huella, no texto | Pasa. Se comprobó que la prueba detecta el fallo: al volcar la pregunta a un log, `test_3` falla. En la consulta real con un correo de prueba en la pregunta, el correo no aparece en el log JSON ni en la salida |
| 4. Supervisión humana real | `test_4` con `SupervisionDespliegue` (grafo de LangGraph con `interrupt()` y `Command`): se pausa, no escribe el registro hasta la decisión y reanuda con ella; `python scripts/comparar_supervision.py` compara con la vía previa | Pasa; `comparar_supervision.py` termina con código 0 |

## 7. Las seis dimensiones y dónde está la evidencia

Las cifras de precisión, exhaustividad, latencia y costo no se repiten aquí: están medidas en las secciones indicadas de README.md, con sus corridas reales.

| Dimensión (peso) | Evidencia | Observación |
|---|---|---|
| Pipeline RAG funcional (25 %) | `scripts/indexar_corpus.py`, `src/indexacion.py`, `src/recuperacion.py`, `src/servicio_rag.py`, índice persistido en `vectorstore/` | Índice persistido, no reconstruido en cada consulta; `preflight_rag.py` comprueba que el índice coincide con el corpus (probado sobre una copia: OK) |
| Recuperación y citación (15 %) | `src/metricas_recuperacion.py`, `scripts/evaluar_recuperacion.py`, `docs/resultados_evaluacion_recuperacion.json`; README.md "RAG: evaluación de la recuperación por chunk (Bloque 27)" | Métricas medidas sobre el corpus propio (14 preguntas). Citas validadas contra el contexto real |
| Costo, latencia y checklist operativo (10 %) | `src/medicion_rag.py`, `scripts/medir_latencia_costo.py`, `docs/registros_latencia_costo.jsonl`; README.md "RAG: latencia y costo reales del flujo completo (Bloque 28)" | Corridas reales documentadas. El costo de Claude es una estimación con los precios del proyecto; los embeddings se cuentan en peticiones, no en dólares |
| Personalización justificada (15 %) | `docs/decision_personalizacion.md`, `scripts/decidir_personalizacion.py`, `src/decision_personalizacion.py` | Ver "Limitaciones": el resultado es "ninguna técnica por ahora" para el clasificador del router, y una respuesta está marcada como pendiente de confirmación del autor |
| Observabilidad y trazabilidad (15 %) | `src/registro_estructurado.py` (log JSON, `trace_id`, consumo por consulta), `src/trazas.py`, `docs/registro_consulta_real.jsonl` | `trace_id` distinto por consulta y sin valor activo fuera de ella (probado en `test_servicio_rag.py`) |
| Seguridad, versionado y gobernanza (20 %) | `src/defensa_inyeccion.py`, `src/registro_prompts.py`, `docs/registro_prompts.jsonl`, `src/supervision_despliegue.py`, `docs/matriz_riesgos.json` | `registrar_prompt.py --verificar`: el prompt del código es la versión 2, OK. `matriz_riesgos.py`: 11 riesgos, todas las evidencias existen en el código. Los siete riesgos ALTO o CRÍTICO tienen supervisión y el validador lo exige; cuatro son procesos humanos (ver "Limitaciones") |

## 8. Archivos nuevos del Gate 2

- `src/servicio_rag.py`: punto de entrada único del servicio (fachada sobre `consulta_rag.py`).
- `scripts/consultar_servicio.py`: consulta de punta a punta desde la terminal.
- `test_gate2.py`: las cuatro pruebas del curso, adaptadas a este proyecto.
- `verificar_entorno.py`: comprobación del entorno antes de la corrida final.
- `test/unit/test_servicio_rag.py`, `test_consultar_servicio.py`, `test_verificar_entorno.py`: 35 pruebas nuevas (13, 16 y 6).

## 9. Limitaciones conocidas y pendientes (declaradas, no ocultas)

- **Matriz de riesgos y supervisión.** Los siete riesgos ALTO o CRÍTICO de `docs/matriz_riesgos.json` tienen un punto de supervisión, y `python scripts/matriz_riesgos.py` sale con 1 si uno de ellos no lo tiene. Tres de esos puntos ya existían (revisión semanal de trazas, `esperar_aprobacion`, `interrupt` del grafo antes de desplegar). Los cuatro últimos (inyección de prompt, datos personales en los logs, prompt cambiado sin registrar la versión y dataset de tuning desbalanceado) son **procesos humanos, no automatizados**; la matriz lo dice en cada fila. Para la inyección, lo que el código aporta es el marcado `requiere_revision` y el log con la fuente y los patrones del documento descartado; no hay una herramienta que liste esas consultas, se buscan en el log JSON. Además, las probabilidades e impactos son estimaciones y las 11 filas quedan en el límite de otra categoría; no tienen revisión externa.
- **Respuesta cortada por el límite de tokens.** `consultar_servicio.py` imprime la respuesta tal como llega aunque el modelo la haya cortado por `--max-tokens` (el log lo registra como `truncado`, pero la consola no avisa). En la consulta real documentada el valor por defecto (512) cortó la respuesta a mitad de frase. Con `--max-tokens 1024` hay más margen.
- **Decisión de personalización.** El método da cero puntos a RAG, fine-tuning completo y PEFT para el caso evaluado (el clasificador del router, no el RAG), así que la decisión es mantener el clasificador actual. La respuesta sobre presupuesto está marcada `[A CONFIRMAR POR EL AUTOR]` y la firma del documento está pendiente.
- **Falsos positivos del detector de inyección.** Un fragmento legítimo que cite el ataque (por ejemplo "nuevo sistema:") se descarta. El detector no cubre homoglifos de otros alfabetos ni sinónimos (README.md, sección del Bloque 31).
- **Índice no versionado.** Quien revise debe reindexar con su propia clave de Gemini (sección 3).
- **Dos claves.** La consulta real necesita `CLAUDE_API_KEY` y `GEMINI_API_KEY`.
- **Dependencias.** `chromadb` 1.5.9 tiene avisos de seguridad sin arreglo disponible, de alcance probablemente no aplicable (sin verificar, README.md).

## 10. Qué no está verificado

- La consulta real de punta a punta se ejecutó una vez (2026-11-25, una pregunta, código 0, respuesta con tres citas válidas, sin el correo de prueba en los logs) sobre una copia del índice y con una sola pregunta. No hay varianza medida ni se contrastó el costo con la consola de Anthropic. Las pruebas automáticas usan modelos y vector stores falsos.
- La instalación con Python 3.11, que es la versión de los ejemplos del curso: se probó con 3.14.9, con el que están fijadas las versiones de `requirements.txt`.
- `verificar_entorno.py` contra un `.env` y un `vectorstore/` reales.
