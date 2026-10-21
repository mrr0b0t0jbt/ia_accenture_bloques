# Decisión de arquitectura: elección de modelo

## Modelo elegido

Claude sonnet 5.5 para la funcion principal de la miniaplicacion

## Diagrama de arquitectura

El diagrama de arquitectura se encuentra en la ruta docs/diagrama_arquitectura.png
 
## Comparación de candidatos

 | Criterio | Claude Sonnet 5.5 | Gemini 3.1 Pro|
 |---|---|---|
 |Costo 1 millon tokens entrada|2 USD|2 USD/4 USD|
 |Costo 1 millon tokens salida|10 USD|12 USD/18 USD|
 |Latencia promedio|60|120|
 |Calidad de respuesta para el caso|Buena: La respuesta aputaba bastante bien tanto al entendimiento de la vulnerabilidad como a las remediaciones para esta misma, asi como citar las fuentes de las que otuvo la informacion entregada|Aceptable: logro describir cual era la vulnerabilidad y una remediacion posible y un par de remediaciones opcionales|
 |Fecha medicion|20/10/2026|20/10/2026|

 ## Justificación

 Con respecto a el objetivo de mi miniaplicacion, pensando en que esto pueda ayudar a equipos tecnicos de pequeñas empresas a el entendimiento, toma de decisiones y remediaciones de vulnerabilidades, me parecio mucho mas precisa para este objetivo las respuestas entregadas por Claude Sonnet 5.5, en ambas probe el mismo promp letra por letra y compare ambas respuestas y su coherencia, la forma en que abordo lo solicitado sonnet y las multiples alternativas que indico para la remediacion aperturan mucho mas a tomar decisiones a un equipo tecnico que las respuestas que entrego Gemini 3.1 Pro, puede ser por que ambos modelos son mejores para distintas tareas y areas, pero aunque los precios son similares entre una y la otra, la calidad de la respuesta esperada para el contexto de ciberseguridad es superior en Claude sonnet

## Alternativa descartada y por qué
Gemini 3.1 Pro no es un mal modelo, pero para el objetivo de la miniaplicacion no cumple con lo esperado, ya que al medir el mismo promp con otro modelo la respuesta no llega a ser tan satisfactoria o lo esperado vs el modelo  que se eligio finalmente para avanzar, en un futuro si esta miniaplicacion se expande podria utilizar este modelo para otras tareas que se le pueden dar bastante bien, pero para el objetivo core de la miniaplicacion no lo alcanza.