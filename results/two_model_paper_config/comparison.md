# Two-Model con configuración exacta del artículo

| Variante | Árboles | Profundidad | Mínimo por hoja | UQC medio de validación | UQC en prueba |
| --- | ---: | ---: | ---: | ---: | ---: |
| Modelo anterior, reentrenado con todo entrenamiento | 200 | 25 | 50 | 1,04416 | 1,3686 |
| Parámetros del artículo, reentrenado con todo entrenamiento | 100 | 25 | 30 | 0,89191 | **1,6450** |
| Parámetros del artículo, promedio de diez modelos de folds | 100 | 25 | 30 | 0,89191 | **1,3984** (DE muestral 0,2488) |
| Resultado publicado, tabla 5 | 100 | 25 | 30 | No informado en la tabla | 2,0011 (DE 0,23) |

Los dos resultados finales usan toda la muestra de entrenamiento. La media de folds promedia las diez métricas, no las puntuaciones de los usuarios. Comparación exploratoria posterior a la primera inspección de prueba.
