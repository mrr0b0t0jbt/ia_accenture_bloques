# Registro del ensayo REAL de rate limit

> **Aviso:** este es un ensayo **REAL** contra la API de Claude: usó la red y la API key, y tuvo costo.

> **Aviso (concurrencia):** esto es una ráfaga concurrente deliberada para superar el límite por minuto de la cuenta; puede provocar 429 y reintentos.

## Registro del ensayo de rate limit

- Fecha: 2026-10-28
- Peticiones enviadas: 100
- Concurrencia: 20 (olas: 5)
- Petición donde apareció el primer 429: no apareció ningún 429
- Esperas observadas: ninguna
- Costo total del ensayo: $0.01060
- Resultado: no se alcanzó ningún 429 en 100 peticiones
- Llamadas HTTP: 100
- Respuestas 429 totales: 0
- Fallos definitivos: 0
- Duración: 10.7s

## Límites del ensayo

- Puede NO aparecer ningún 429: depende de la cuota real de la cuenta en ese momento.
- No mide latencia fina: la duración es el total del ensayo, incluidas las esperas.

## Cómo correrlo

```bash
python scripts/ensayo_rate_limit_real.py --confirmar-api-real
```
