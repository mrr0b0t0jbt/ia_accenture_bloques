# Decision de personalizacion

## 1. El problema
El clasificador del router (src/enrutamiento.py) pide a Claude Sonnet 5.5 una sola palabra entre incidente, vulnerabilidad o consulta y normaliza la respuesta: se evalua si hace falta personalizar esa tarea con RAG, fine-tuning completo o PEFT.

## 2. Resultado del metodo de decision
- RAG: 0 puntos
- Fine-tuning completo: 0 puntos
- PEFT: 0 puntos

**Ninguna tecnica obtuvo puntos**

Respuestas y evidencia (las cinco son False):

- informacion_cambia_seguido: la tarea no consulta informacion externa; las tres categorias son fijas (CATEGORIAS en src/enrutamiento.py).
- problema_de_formato_o_tono: medido con la API real sobre 24 ejemplos (max_tokens 256, una pasada, docs/resultados_clasificador_router.json). 2 de 24 respuestas no eran exactamente la categoria (8,3 %; ambas "Incidente" con mayuscula) y 0 de 24 quedaron invalidas tras la normalizacion del router. El umbral de la clase es 15 %.
- dataset_grande_y_de_calidad: hay 24 ejemplos balanceados escritos por la orquestadora, no mensajes reales de un usuario.
- presupuesto_limitado_o_iterar_rapido: no hay restriccion de presupuesto declarada; la clasificacion actual cuesta unos USD 0,0004 por llamada y ajustar el prompt ya es una iteracion inmediata.
- necesita_multiples_variantes: hay un solo clasificador con un solo conjunto de categorias.

Sensibilidad (se invierte una respuesta a la vez; todas cambian el resultado):

- informacion_cambia_seguido en True: RAG.
- problema_de_formato_o_tono en True: PEFT.
- dataset_grande_y_de_calidad en True: Fine-tuning completo.
- presupuesto_limitado_o_iterar_rapido en True: RAG.
- necesita_multiples_variantes en True: PEFT.

## 3. Tecnica elegida y por que
Ninguna por ahora: se mantiene el clasificador actual (prompt de una palabra mas la normalizacion del router).

La unica respuesta que podria justificar personalizar es problema_de_formato_o_tono, y la medicion la contradice: el fallo de formato efectivo fue 0 de 24, por debajo del 15 % de la clase, y los dos fallos estrictos los corrige el router al normalizar. Con 64 tokens de salida la primera corrida dio 16,7 % estricto, pero dos de esos fallos eran respuestas vacias por el tope de tokens de la propia medicion, no un defecto del modelo; con 256 desaparecieron.

Limites de esta decision: una sola pasada, 24 ejemplos escritos por la orquestadora (no mensajes reales) y una exactitud de 23 de 24. El unico error de categoria fue un caso limite (un secreto de produccion en un repositorio publico: el dataset dice vulnerabilidad y el modelo dijo incidente). Si cambia la respuesta a presupuesto_limitado_o_iterar_rapido, el metodo sugeriria RAG, pero RAG resuelve informacion que cambia y aqui no hay informacion externa: el puntaje es una heuristica, no una prueba.

## 4. Que necesitarias para ejecutarla
No hace falta nada para mantener la decision. Habria que reabrirla si: (a) una muestra de mensajes reales muestra 15 % o mas de respuestas fuera de formato tras normalizar; (b) se agregan categorias o variantes por cliente, idioma o tono; o (c) la exactitud sobre casos reales no alcanza lo que el uso exige.

En ese caso faltaria un conjunto de ejemplos reales etiquetados. Con los supuestos de la clase (5 minutos por ejemplo y USD 25 por hora), 200 ejemplos serian unas 16,7 horas y unos USD 416,67 de etiquetado, antes de cualquier costo de computo. Ademas, con una GEMINI_API_KEY el tuning solo corre en modo de API de desarrolladores sin PEFT: el tuning con adapter requiere Vertex (verificado con google-genai 3.0.0).

## 5. Aprobacion del autor

Decision revisada y confirmada por el autor: mr_r0b0t0.
Fecha: 25/11/2026.
Respuesta confirmada: presupuesto_limitado_o_iterar_rapido = False (no hay restriccion de presupuesto declarada).
