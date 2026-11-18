# Comparación de Two-Model en Hillstrom Conversion

| Indicador o parámetro | Artículo original | Réplica en prueba | Lectura y comparabilidad |
| --- | --- | --- | --- |
| UQC en prueba | 2,0011 (DE 0,23) | 1,3686 | Promedio publicado entre modelos de folds frente a un solo par final de la réplica. |
| Referencia aleatoria del UQC | 1,0000 | 1,0000 | Un valor mayor que 1 indica mayor área que la diagonal en esta evaluación. |
| Ganancia acumulada al seleccionar el 20 %, en p.p. | Aprox. 0,410 | 0,240 | Normalizada respecto del total tratado de prueba, no del segmento seleccionado. |
| Ganancia acumulada al seleccionar el 100 %, en p.p. | Aprox. 0,370 | 0,370 | Es la diferencia agregada de tasas al incluir a todos, independiente del ranking. |
| AUUC relativa de la curva de diez deciles | No reportada para TM en tabla 5 | 0,0025250 | Integral de la curva normalizada, sin restar la diagonal. |
| Área Qini por encima de la diagonal | No reportada para TM en tabla 5 | 0,0006750 | Área adicional sobre q*g(1). No es el coeficiente normalizado por el óptimo. |
| Árboles por bosque | 100 | 200 | Parámetros publicados en tabla A1 frente a los elegidos en nuestra validación. |
| Profundidad máxima | 25 | 25 | Coincide el valor seleccionado. |
| Mínimo de registros por hoja | 30 | 50 | Los parámetros de la réplica se congelaron antes de evaluar prueba. |

Los valores del artículo corresponden a las tablas 5 y A1 y al apéndice B. No se trata de una reproducción numérica exacta.

Rößler, J., & Schoder, D. (2022). Bridging the gap: A systematic benchmarking of uplift modeling and heterogeneous treatment effects methods. *Journal of Interactive Marketing, 57*(4), 629–650. https://doi.org/10.1177/10949968221111083
