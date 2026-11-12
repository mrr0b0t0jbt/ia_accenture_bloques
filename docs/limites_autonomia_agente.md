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

- Pregunta por consola con un plazo: 60 s por defecto. El plazo cubre TODA la lectura (`os.read` con `select`, buffer propio, línea máxima de 1024 bytes), no solo el inicio de la espera. El prompt muestra el plazo y, si vence, avisa "Tiempo agotado: no aprobado."
- Cuenta como no aprobado: plazo vencido, fin de entrada (EOF) sin salto de línea, una línea parcial sin salto, una línea de más de 1024 bytes, un tiempo inválido (no numérico, no finito, menor o igual a 0 o mayor a 3600), un descriptor cerrado o fuera de rango, `sys.stdin` ausente y cualquier otro error. Solo `KeyboardInterrupt` se propaga.
- Aprueba únicamente la respuesta "s" terminada en salto de línea (se ignoran las mayúsculas y los espacios alrededor, así que "S" y " s " también aprueban); cualquier otra, incluido "sí", niega.
- En una terminal se descarta lo tecleado antes de preguntar (`tcflush`): una tecla suelta no puede aprobar una pregunta que aún no se vio.
- La función pública acepta hasta 3600 s; el script limita `--tiempo-aprobacion` a mayor que 0 y como máximo 300 s.
- El motivo lo escribe el modelo: se sanea (una sola línea, solo caracteres imprimibles, máximo 200 caracteres) y se muestra entre comillas como texto no verificado.
- Solo se acepta un host con la forma `*.ejemplo.test` (máximo 100 caracteres), sin importar si es válido o no según DNS. Se normaliza con `strip` y minúsculas, y lo registrado es el host normalizado. Un host fuera de laboratorio se rechaza sin pedir aprobación, pero el intento queda registrado como no aprobado (con el rechazo "host no permitido"), porque es una señal de seguridad; en ese registro el host y el motivo se limpian (una línea, imprimibles, hasta 100 y 200 caracteres). Por eso el conteo "Aprobaciones de riesgo alto" que imprime el script incluye estos rechazos automáticos.
- Un aprobador personalizado solo aprueba si devuelve exactamente `True`; cualquier otro valor ("n", 1, `[False]`) se trata como no aprobado y se registra `False`.
- Topes: máximo 3 aprobaciones solicitadas por turno y 10 por corrida (`MAX_APROBACIONES_POR_TURNO` y `MAX_APROBACIONES_POR_CORRIDA` en `src/supervision.py`). Un host ya negado en el mismo turno no se vuelve a preguntar. Ambos casos se registran con el rechazo "tope de aprobaciones" o "ya denegado en este turno" y cuentan como denegadas en el resumen. Se aplican también en el modo `denegar`.
- Peor caso desatendido con `--tiempo-aprobacion 300`: 10 aprobaciones x 300 s = 50 minutos esperando a una persona que no responde.
- Cada decisión, aprobada o no, queda registrada en `aprobaciones` (herramienta, argumentos, aprobado y momento en UTC), por turno y para toda la corrida.
- La espera usa `select` y `termios`, que solo funcionan en Linux y macOS.

Modo del script `scripts/ejecutar_agente.py`:

- `--aprobacion denegar` (por defecto): nadie aprueba, así que una corrida desatendida nunca ejecuta una acción de riesgo alto. El intento queda registrado como no aprobado.
- `--aprobacion interactiva`: pregunta por consola; el tiempo se fija con `--tiempo-aprobacion` (segundos, mayor que 0 y como máximo 300).

Las herramientas de riesgo bajo no piden aprobación a propósito: si se pidiera siempre, quien aprueba dejaría de leer lo que aprueba (fatiga) y la aprobación perdería valor justo donde importa.

## Revisión de trazas

Dónde se guardan: `trazas/trazas_agente.jsonl` (se cambia con `--trazas`; `--sin-trazas` no escribe nada). Quedan fuera de git: `trazas/` y cualquier ruta que case con `*trazas*.jsonl` o `trazas*/` (ver `.gitignore`). El archivo se crea con permisos 0600 y el directorio con 0700; si el archivo ya existía, sus permisos no se tocan.

Contienen las entradas completas de los turnos, es decir, lo que escribió quien consultó (privacidad). No hay rotación ni retención: el borrado es manual.

`--trazas` se rechaza si es el mismo archivo que `--casos` o `--salida`, o si existe y sus primeras líneas no vacías (hasta 5) no son objetos JSON (o no se puede leer). Un archivo inexistente o vacío se acepta.

Cada línea trae: timestamp en UTC, `session_id`, entrada, acciones (herramienta, argumentos y observación), respuesta final y aprobaciones del turno. Un turno interrumpido con Ctrl-C trae además `"interrumpido": true`.

Cómo revisarlas: `python scripts/revisar_trazas.py` (o con `--trazas <ruta>`). Marca dos cosas:

- La misma herramienta usada dos veces seguidas en un turno.
- Una herramienta de riesgo alto sin ningún registro de aprobación en esa traza.

Un intento rechazado por host no permitido cuenta como registro, así que no se marca.

Frecuencia propuesta: una muestra pequeña cada semana y TODAS las trazas donde se usó una herramienta de riesgo alto. Es un proceso que debe seguir una persona; no está automatizado (nada lo ejecuta ni avisa solo). El script solo marca candidatos a revisar, no concluye nada.

Lo que NO se revisa automáticamente: que la respuesta final contradiga las observaciones de las herramientas. Eso lo tiene que leer una persona al revisar la muestra.

## Qué hacer ante un comportamiento inesperado

1. Detener la corrida con Ctrl-C. Las decisiones ya tomadas en el turno interrumpido quedan en el JSON de resultados y en la traza con `"interrumpido": true`; el script sale con código 130 y el mensaje "Interrumpido por la persona. Resultados parciales guardados." El script también se corta solo al llegar al tope de gasto, al máximo de llamadas o si un turno termina sin error y no devuelve datos de uso (un turno con error no activa este corte).
2. Neutralizar la ejecución de la herramienta involucrada: el modelo seguirá viendo y llamando la herramienta (se rechaza y queda registrada). `denegar_siempre` ya es el valor por defecto. Quitarla de la lista de `construir_herramientas` exige quitarla también de `RIESGO`, y los turnos de `JSON/casos_agente.json` que la esperan dejan de validar y hay que ajustarlos.
3. Revisar las trazas con `scripts/revisar_trazas.py` (solo lee el JSONL de trazas) y, aparte y a mano, el JSON de resultados de esa corrida (por ejemplo `docs/resultados_agente.json`).
4. Revisar la descripción de la herramienta y el prompt del agente (`TEXTO_SISTEMA` en `src/agente.py`), porque son lo que el modelo lee para decidir qué usar.
5. Reproducir el caso con el modelo falso de `test/unit/modelo_falso_herramientas.py` y una prueba nueva que fije el comportamiento correcto.
6. Reactivar la herramienta solo con esa prueba en verde.
7. Anotar lo ocurrido y la decisión tomada.

## Controles automáticos del script

Lo que hace `scripts/ejecutar_agente.py` sin que nadie intervenga:

- Exige `--confirmar-api-real`; sin él termina con código 2 sin cargar langchain ni leer la clave.
- `--tope-usd` máximo 0.10 (estimado con los tokens informados; no es el cobro real).
- Máximo 30 llamadas al modelo por corrida.
- Máximo 4 iteraciones del agente por turno. Una iteración puede traer varias llamadas de herramienta (por ejemplo, varios `aislar_host_de_laboratorio` a la vez), por eso existe el tope de aprobaciones.
- Máximo 10 turnos y 2000 caracteres por mensaje.
- El tope de gasto y el de llamadas se comprueban ANTES de cada llamada al modelo.
- Si un turno con llamadas termina sin error y no trae datos de uso (un turno con error no activa este corte), la corrida se corta, porque sin ellos el tope no se puede controlar.
- El JSON de resultados se guarda después de cada turno con escritura atómica (archivo temporal y reemplazo).

## Límites conocidos

- Los reintentos internos del SDK no se cuentan como llamadas.
- Un error 401 no corta la corrida: cada turno falla por separado.
- Argumentos de herramienta inválidos matan el turno.
- El texto de `max_iterations` ("Agent stopped due to max iterations.") queda en el historial de la sesión.
- El `session_id` no se normaliza.
- La aprobación interactiva solo se probó con pipes y un terminal simulado, no con una persona en una sesión real y larga.
- Un turno que termina con error (por ejemplo, por un límite) no deja traza; su aprobación, si la hubo, solo queda en el JSON de resultados.
- La revisión de trazas busca el registro de aprobación por nombre de herramienta, no por host, dentro de la misma traza: una sola aprobación cubre varias acciones de esa herramienta en el turno.
- Si un aprobador personalizado lanza una excepción, la herramienta falla: no ejecuta la acción (queda sin efecto) pero la decisión no se registra y el turno termina con error.
- No se verificó con la API real cómo se comporta Claude tras un "no" ni con llamadas de herramienta paralelas; el tope de aprobaciones cubre ambos casos.
- El motivo es texto del modelo: un texto imprimible que imite la interfaz dentro de las comillas no se puede filtrar. Se mitiga con la advertencia "no verificado".
- El guardián de las pruebas (`test/unit/conftest.py`) solo vigila `trazas/` relativo al directorio de trabajo.
- La concurrencia de escrituras a las trazas solo se probó en un sistema de archivos local de Linux.
- La acción real de aislamiento no existe; todo lo dicho sobre aprobación se probó solo contra la simulación.
