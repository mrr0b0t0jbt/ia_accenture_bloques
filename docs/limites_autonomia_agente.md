# Límites de autonomía: agente de apoyo en ciberseguridad

Este documento describe el agente tal como está implementado hoy, no como debería ser. La única herramienta de riesgo alto, `aislar_host_de_laboratorio`, está simulada: no toca redes, procesos ni archivos, y la acción real no existe. Lo que no está implementado se dice como tal.

## Herramientas y nivel de riesgo

Los niveles son los de `RIESGO` en `src/herramientas_agente.py`.

- buscar_en_documentacion: bajo (solo lee textos fijos del proyecto)
- calcular: bajo (solo calcula con un evaluador de ast, no ejecuta código ni toca archivos ni red)
- consultar_cve_de_laboratorio: bajo (solo lee una tabla ficticia en memoria)
- aislar_host_de_laboratorio: alto (en un entorno real, aislar un host corta su conectividad y puede afectar servicios; aquí está simulada)

Nota: el nivel "medio" (escritura reversible) no se usa todavía; ninguna herramienta cae en él.

## Herramientas que requieren aprobación humana

Solo `aislar_host_de_laboratorio`. Aislar un host es una acción con impacto y, en un entorno real, difícil de deshacer a tiempo; por eso una persona debe aprobarla antes de que se ejecute.

Cómo funciona:

- Pregunta por consola con un tiempo límite: 60 s por defecto y 300 s como máximo (el script rechaza otros valores).
- Sin respuesta, fin de entrada (EOF) o error cuenta como no aprobado. Solo la respuesta "s" aprueba; cualquier otra, incluido "sí", niega.
- El motivo lo escribe el modelo: se sanea (una sola línea, solo caracteres imprimibles, máximo 200 caracteres) y se muestra entre comillas como texto no verificado.
- Solo se acepta un host `*.ejemplo.test` (máximo 100 caracteres). Un host fuera de laboratorio se rechaza sin pedir aprobación y sin registro de aprobación.
- Cada decisión, aprobada o no, queda registrada en `aprobaciones` (herramienta, argumentos, aprobado y momento en UTC), por turno y para toda la corrida.
- La espera usa `select` sobre la entrada estándar, que solo funciona en Linux y macOS.

Modo del script `scripts/ejecutar_agente.py`:

- `--aprobacion denegar` (por defecto): nadie aprueba, así que una corrida desatendida nunca ejecuta una acción de riesgo alto. El intento queda registrado como no aprobado.
- `--aprobacion interactiva`: pregunta por consola; el tiempo se fija con `--tiempo-aprobacion` (segundos, mayor que 0 y como máximo 300).

Las herramientas de riesgo bajo no piden aprobación a propósito: si se pidiera siempre, quien aprueba dejaría de leer lo que aprueba (fatiga) y la aprobación perdería valor justo donde importa.

## Revisión de trazas

Dónde se guardan: `trazas/trazas_agente.jsonl` (se cambia con `--trazas`; `--sin-trazas` no escribe nada). El directorio `trazas/` está fuera de git porque contiene las entradas de las consultas.

Cada línea trae: timestamp en UTC, `session_id`, entrada, acciones (herramienta, argumentos y observación), respuesta final y aprobaciones del turno.

Cómo revisarlas: `python scripts/revisar_trazas.py` (o con `--trazas <ruta>`). Marca dos cosas:

- La misma herramienta usada dos veces seguidas en un turno.
- Una herramienta de riesgo alto sin ningún registro de aprobación en esa traza.

Frecuencia propuesta: una muestra pequeña cada semana y TODAS las trazas donde se usó una herramienta de riesgo alto. Es un proceso que debe seguir una persona; no está automatizado (nada lo ejecuta ni avisa solo). El script solo marca candidatos a revisar, no concluye nada.

Lo que NO se revisa automáticamente: que la respuesta final contradiga las observaciones de las herramientas. Eso lo tiene que leer una persona al revisar la muestra.

## Qué hacer ante un comportamiento inesperado

1. Detener la corrida con Ctrl-C. El script también se corta solo al llegar al tope de gasto, al máximo de llamadas o si un turno no devuelve datos de uso.
2. Desactivar la herramienta involucrada: quitarla de la lista de `construir_herramientas` o dejar el aprobador `denegar_siempre`, que es el valor por defecto. Si se quita, los turnos de `JSON/casos_agente.json` que la esperan dejan de validar y hay que ajustarlos.
3. Revisar las trazas y el JSON de resultados de esa corrida (`docs/resultados_agente.json`) con `scripts/revisar_trazas.py`.
4. Revisar la descripción de la herramienta y el prompt del agente (`TEXTO_SISTEMA` en `src/agente.py`), porque son lo que el modelo lee para decidir qué usar.
5. Reproducir el caso con el modelo falso de `test/unit/modelo_falso_herramientas.py` y una prueba nueva que fije el comportamiento correcto.
6. Reactivar la herramienta solo con esa prueba en verde.
7. Anotar lo ocurrido y la decisión tomada.

## Controles automáticos del script

Lo que hace `scripts/ejecutar_agente.py` sin que nadie intervenga:

- Exige `--confirmar-api-real`; sin él termina con código 2 sin cargar langchain ni leer la clave.
- `--tope-usd` máximo 0.10 (estimado con los tokens informados; no es el cobro real).
- Máximo 30 llamadas al modelo por corrida.
- Máximo 4 iteraciones del agente por turno.
- Máximo 10 turnos y 2000 caracteres por mensaje.
- El tope de gasto y el de llamadas se comprueban ANTES de cada llamada al modelo.
- Si un turno con llamadas no trae datos de uso, la corrida se corta, porque sin ellos el tope no se puede controlar.
- El JSON de resultados se guarda después de cada turno con escritura atómica (archivo temporal y reemplazo).

## Límites conocidos

- Los reintentos internos del SDK no se cuentan como llamadas.
- Un error 401 no corta la corrida: cada turno falla por separado.
- Argumentos de herramienta inválidos matan el turno.
- El texto de `max_iterations` ("Agent stopped due to max iterations.") queda en el historial de la sesión.
- El `session_id` no se normaliza.
- La aprobación interactiva solo se probó con pipes y un terminal simulado, no con una persona en una sesión real y larga.
- Un turno que termina con error (por ejemplo, por un límite) no deja traza; su aprobación, si la hubo, solo queda en el JSON de resultados.
- La revisión de trazas busca el registro de aprobación por nombre de herramienta dentro de la misma traza: una sola aprobación cubre varias acciones de esa herramienta en el turno.
- La acción real de aislamiento no existe; todo lo dicho sobre aprobación se probó solo contra la simulación.
