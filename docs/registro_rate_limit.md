# Registro del ensayo de rate limit (Bloque 13)

> **Aviso:** este ensayo es **SIMULADO** con mocks: `requests.post` y `time.sleep` se reemplazan por versiones falsas. **No es un 429 real del proveedor**, no usó la red ni la API key, y por eso el costo es el estimado a partir de tokens ficticios. Un ensayo real contra la API queda **PENDIENTE** de autorización del usuario.

## Registro del ensayo de rate limit

- Fecha: 2026-10-28
- Peticiones enviadas: 10
- Petición donde apareció el primer 429: la número 6 (simulado)
- Esperas observadas: 2.0s, 4.3s (backoff exponencial con jitter: bases de 2 s y 4 s más hasta 10 %; time.sleep estaba mockeado, no se esperó de verdad)
- Costo total del ensayo: $0.00561 (estimado con tokens ficticios; las dos respuestas 429 simuladas no suman costo)
- Resultado: las 10 peticiones terminaron con éxito; la número 6 necesitó 3 intentos (12 llamadas a `requests.post` en total)

## Cómo se generó

El script [scripts/ensayo_rate_limit_simulado.py](../scripts/ensayo_rate_limit_simulado.py) envía 10 peticiones a `get_claude_response` con `requests.post` y `time.sleep` parcheados. En la petición 6 el falso `requests.post` devuelve dos 429 seguidos y luego un 200, lo que activa el reintento con backoff exponencial de `with_retry`; el resto responde 200 de inmediato. Al final muestra la petición del primer 429, las esperas pedidas, el total de llamadas a `requests.post` y el costo acumulado de la sesión.

Se ejecuta desde la raíz del proyecto:

```bash
.venv/bin/python scripts/ensayo_rate_limit_simulado.py
```

Los valores de espera cambian entre corridas por el jitter (hasta 10 % sobre la espera base).

## Salida real de la corrida

```
Operación: 63 tokens, costo estimado: $0.00046 (acumulado: $0.00046)
Respuesta 1: Respuesta simulada número 1...
Operación: 66 tokens, costo estimado: $0.00048 (acumulado: $0.00095)
Respuesta 2: Respuesta simulada número 2...
Operación: 69 tokens, costo estimado: $0.00051 (acumulado: $0.00145)
Respuesta 3: Respuesta simulada número 3...
Operación: 72 tokens, costo estimado: $0.00053 (acumulado: $0.00198)
Respuesta 4: Respuesta simulada número 4...
Operación: 75 tokens, costo estimado: $0.00055 (acumulado: $0.00253)
Respuesta 5: Respuesta simulada número 5...
Intento 1 de 3 falló. Reintentando en 2.0s...
Intento 2 de 3 falló. Reintentando en 4.3s...
Operación: 78 tokens, costo estimado: $0.00057 (acumulado: $0.00310)
Respuesta 6: Respuesta simulada número 6...
Operación: 81 tokens, costo estimado: $0.00059 (acumulado: $0.00370)
Respuesta 7: Respuesta simulada número 7...
Operación: 84 tokens, costo estimado: $0.00062 (acumulado: $0.00431)
Respuesta 8: Respuesta simulada número 8...
Operación: 87 tokens, costo estimado: $0.00064 (acumulado: $0.00495)
Respuesta 9: Respuesta simulada número 9...
Operación: 90 tokens, costo estimado: $0.00066 (acumulado: $0.00561)
Respuesta 10: Respuesta simulada número 10...
Primer 429 en la petición: 6
Esperas pedidas a time.sleep: 2.0s, 4.3s
Llamadas totales a requests.post: 12
Costo total del ensayo: $0.00561
```

## Límites del ensayo

- No prueba el comportamiento real del proveedor (cabecera `retry-after`, cuota real).
- No mide latencia real.
- El ensayo real contra la API queda pendiente.
