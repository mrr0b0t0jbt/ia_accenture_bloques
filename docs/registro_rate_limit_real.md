# Registro del ensayo REAL de rate limit

> **Aviso:** este es un ensayo **REAL** contra la API de Claude: usó la red y la API key, y tuvo costo.

## Registro del ensayo de rate limit

- Fecha: 2026-10-28
- Peticiones enviadas: 30
- Petición donde apareció el primer 429: no apareció ningún 429
- Esperas observadas: ninguna
- Costo total del ensayo: $0.00318
- Resultado: no se alcanzó ningún 429 en 30 peticiones
- Llamadas HTTP: 30
- Respuestas 429 totales: 0
- Fallos definitivos: 0
- Duración: 33.6s

## Límites del ensayo

- Puede NO aparecer ningún 429: depende de la cuota real de la cuenta en ese momento.
- No mide latencia fina: la duración es el total del ensayo, incluidas las esperas.

## Cómo correrlo

```bash
python scripts/ensayo_rate_limit_real.py --confirmar-api-real
```
