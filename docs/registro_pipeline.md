# Registro del pipeline de texto (Bloque 15)

> **Aviso:** las respuestas de este registro vienen de llamadas **REALES** a la API de Claude, con costo.

- Fecha: 2026-10-28

## Caso 1

- Consulta: ¿Qué es una inyección SQL?
- Referencia: Una inyección SQL es un ataque que inserta código SQL malicioso en una consulta de la aplicación para leer o modificar la base de datos.
- Respuesta: Una inyección SQL es un ataque en el que una persona malintencionada introduce código SQL en un campo de entrada de una aplicación (como un formulario o una URL) para manipular la base de datos y así acceder, modificar o borrar datos sin autorización.
- [Caso 1] Coincidencia texto crudo: 77.3%
- [Caso 1] Coincidencia texto procesado: 81.0%
- Costo del caso: $0.00095

## Caso 2

- Consulta: ¿Para qué sirve la autenticación de múltiples factores?
- Referencia: La autenticación multifactor exige más de una prueba de identidad, de modo que una contraseña robada no basta para acceder a la cuenta.
- Respuesta: La autenticación de múltiples factores sirve para proteger cuentas y sistemas exigiendo dos o más pruebas de identidad independientes (como una contraseña, un código enviado al móvil o una huella dactilar), de modo que, aunque un atacante robe una contraseña, no pueda acceder sin los demás factores.
- [Caso 2] Coincidencia texto crudo: 45.0%
- [Caso 2] Coincidencia texto procesado: 63.2%
- Costo del caso: $0.00119

## Caso 3

- Consulta: ¿Qué es una vulnerabilidad crítica según CVSS?
- Referencia: Una vulnerabilidad crítica en CVSS tiene una puntuación de 9.0 a 10.0 y suele permitir comprometer el sistema con facilidad y gran impacto.
- Respuesta: Según CVSS (v3.x y v4.0), una vulnerabilidad crítica es aquella con una puntuación base de entre 9.0 y 10.0, lo que indica que es fácil de explotar y puede tener un impacto muy grave en la confidencialidad, integridad o disponibilidad del sistema, a menudo sin necesidad de privilegios ni interacción del usuario.
- [Caso 3] Coincidencia texto crudo: 50.0%
- [Caso 3] Coincidencia texto procesado: 66.7%
- Costo del caso: $0.00141

**Costo total:** $0.00354

## Errores

Ninguno

## Límites del método

- La métrica no distingue el orden de las palabras: "el perro persiguió al gato" y "el gato persiguió al perro" coinciden al 100 %.
- La métrica no distingue las repeticiones: compara conjuntos de palabras distintas.
- Los signos de apertura (¿ ¡) se eliminan junto con el resto de la puntuación.
- Un decimal pierde el punto: 9.0 pasa a 90.
