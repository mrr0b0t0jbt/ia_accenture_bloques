# Límites de la métrica de evaluación

## Qué mide

Coincidencia de palabras entre una respuesta generada y una referencia, contando repeticiones: precisión, cobertura y F1 sobre tokens (`src/metricas_texto.py`, después de limpiar y tokenizar con `src/pipeline_texto.py`). No mide si la respuesta es correcta ni si dice lo mismo, solo cuánto vocabulario comparte con la referencia.

## Modos de falla conocidos

Cada modo remite a un caso de `docs/revision_manual.json` y a los resultados de `docs/resultados_metricas.json` (20 respuestas reales de Claude, 2026-10-30).

**Observados en la corrida real**

- **La longitud castiga respuestas correctas.** La referencia es una frase corta y el modelo contesta con frases más largas, aunque se le pide una sola frase. En 19 de 20 casos la precisión queda por debajo de la cobertura (promedios: precisión 0,40, cobertura 0,80, F1 0,52). Ejemplos de respuestas correctas con F1 bajo: Log4Shell (0,26), antivirus como defensa de un endpoint (0,31) y SIEM (0,31).
- **Sin lematización ni sinónimos.** «Engaña» y «engañar» cuentan como tokens distintos (phishing, 0,32), igual que «parches al día» frente a «mantener todo actualizado» (antivirus, 0,31).
- **Texto adicional que no es contenido.** Los nombres en inglés entre paréntesis (STRIDE, 0,65) agregan tokens y bajan la precisión, aunque las seis categorías son correctas. Los asteriscos de markdown no afectan, porque la limpieza los elimina.
- **Una consulta ambigua rompe la referencia.** «¿Qué es una VPN?» produjo una respuesta sobre navegar por internet con la IP oculta; la referencia solo cubría el acceso a una red privada. Se enfocó la consulta a «VPN de acceso a una red privada, como una red empresarial» y la respuesta pasó a coincidir en significado, pero el F1 se mantuvo en 0,35 (0,353 antes y 0,353 después). La ambigüedad no explica el puntaje: la longitud sí.
- **En esta muestra no hubo ningún error factual.** Las 20 respuestas son correctas, así que los modos de falla de «dato equivocado» y «negación» no se observaron en la corrida real. Se ilustran con los casos construidos de abajo.

**Construidos a partir de respuestas reales (NO observados en la corrida)**

Cada uno cambia mínimamente una respuesta real para mostrar qué pasa cuando el error sí existe. Cálculos con `calcular_metricas`, sin llamar a la API.

| Cambio | F1 de la respuesta real | F1 con el error | Qué muestra |
|---|---|---|---|
| SSH: puerto 22 → 23 | 0,93 | 0,80 | Un puerto equivocado puntúa más que 18 de las 20 respuestas reales y correctas. |
| Heartbleed: 2014 → 2012 | 0,60 | 0,50 | Un año equivocado puntúa más que 9 de las 20 respuestas reales y correctas. |
| MD5: se quitan «No,» y «no» (inversión) | 0,46 | 0,41 | La respuesta que afirma lo contrario puntúa más que 5 respuestas reales y correctas. |

Salvedad del caso MD5: la edición es mecánica y la respuesta queda incoherente (dice que MD5 es seguro y luego recomienda bcrypt). No es una salida natural del modelo; solo muestra que quitar la negación casi no mueve el número.

**Conocidos por el código, sin impacto en esta corrida**

- La limpieza elimina los guiones (`CVE-2021-44228` → `cve202144228`, `AES-256` → `aes256`). Un identificador con un dígito distinto cuenta como un token fallado, y «AES 256» no coincide con «AES-256» (F1 0,71 en el ejemplo del README). En la corrida no pesó: el modelo escribió los identificadores igual que la referencia.

**Límites del script de corrida**

- El tope de gasto es blando: se comprueba antes de cada llamada y puede excederse en una (unos US$0,01, estimado por lectura de código, no medido).
- Si fallan todas las llamadas, el script termina con código 0 y no toca la salida previa.
- No valida la estructura de los casos (clave `referencia` ausente): falla después de pagar la llamada.
- Casos exóticos de `--salida` que pasan la validación y fallan tras pagar una llamada: hardlink de `--casos`, directorio padre con modo `0600`, ruta con barra final y symlink colgante a un directorio inexistente (QA de Mira, `reportes-qa/qa-metricas-pasada2.md`).

## Muestra revisada

10 casos de 20 respuestas reales, generadas el 2026-10-30: los 5 de menor F1 y los 5 de mayor F1, elegidos con `seleccionar_muestra_revision`. Cada caso se calificó con la rúbrica de tres preguntas (corrección factual, completitud, contradicción). Resultado: 10 de 10 «sí, sí, no»; ninguna respuesta tenía un error. Los extras que la respuesta agrega y la referencia no menciona (por ejemplo EDR, criptomonedas) se calificaron «sí» cuando no contradicen ni cambian la referencia.

Los casos construidos de arriba no forman parte de la muestra.

La consulta de la VPN se modificó después de ver su primer resultado (se repitió solo ese caso, con una llamada adicional de US$0,00125; el resultado nuevo reemplazó al original en `docs/resultados_metricas.json` y la consulta nueva está en `JSON/casos_metricas.json`). La respuesta original, que ya no queda en el archivo de resultados, decía: «Una VPN (red privada virtual) es un servicio que crea una conexión cifrada entre tu dispositivo e internet, ocultando tu dirección IP y protegiendo tus datos para que navegues con mayor privacidad y seguridad.» (F1 0,353). Gasto total de la corrida: unos US$0,018 según el estimador del cliente.

## Recomendación de uso

Usar F1 para comparar en volumen entre versiones del cliente o del prompt con las mismas referencias, nunca como criterio de «la respuesta es correcta»: en esta muestra un puerto equivocado (0,80) puntúa más que casi todas las respuestas correctas. Revisar siempre a mano cualquier respuesta con cifras, puertos, años, identificadores o negaciones, y cualquier F1 bajo en una respuesta que suena razonable.
