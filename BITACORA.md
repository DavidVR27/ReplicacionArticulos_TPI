# Bitácora de replicación de Hillstrom

## 17 de noviembre de 2026 · Preparación de datos

### Alcance acordado

Reproducir la preparación del experimento **Hillstrom Conversion** para el correo de productos femeninos frente al grupo que no recibió correo. Esta etapa comprende la inspección del CSV, la definición de variables, la codificación categórica y el tratamiento de duplicados. El entrenamiento y la evaluación de modelos corresponden a una etapa posterior.

La implementación propia se desarrolla en `src` y los datos permanecen en `data`. `original-paper` se consulta como referencia y no se modifica. El archivo `data/hillstrom-email/hillstrom.csv` se conserva intacto.

### Fuente metodológica

- Rößler, J., y Schoder, D. (2022). *Bridging the gap: A systematic benchmarking of uplift modeling and heterogeneous treatment effects methods*. https://doi.org/10.1177/10949968221111083
- Código local de los autores: `original-paper/autoum-intmar/src/preparation/hillstrom.py`.
- Versión pública de referencia: https://github.com/jroessler/autoum/tree/intmar
- Configuración que se reproduce: `prep(only_women=True, only_men=False, visit_T=False)`.

### Decisiones previas a la ejecución

1. Cargar el archivo original y comprobar su estructura, sus grupos y los valores faltantes.
2. Conservar `Womens E-Mail` y `No E-Mail`; excluir `Mens E-Mail`.
3. Definir `treatment = 1` para `Womens E-Mail` y `treatment = 0` para `No E-Mail`.
4. Codificar `history_segment`, `zip_code` y `channel` por el orden de sus categorías, reproduciendo los códigos que generan `OrdinalEncoder` y `LabelEncoder` en la referencia. Guardar el diccionario de códigos.
5. Definir `response = 1` si `conversion > 0` y `response = 0` en otro caso.
6. Excluir `segment`, `visit`, `conversion` y `spend` del conjunto preparado. `visit`, `conversion` y `spend` son resultados posteriores a la campaña; no se utilizan como predictores.
7. Eliminar duplicados sobre **todas las columnas preparadas**, incluyendo `treatment` y `response`, conservando la primera aparición y el orden original. No eliminar registros solo por compartir características predictoras.
8. Guardar el CSV preparado, los registros excluidos y un informe de auditoría. Comprobar tamaños, tasas de respuesta, codificación, ausencia de duplicados remanentes e integridad del archivo original.

### Interpretación de los duplicados

El dataset no incluye un identificador individual de cliente. Por ello, dos filas idénticas no prueban que se trate de la misma persona. Además, al retirar las otras variables de resultado, algunas filas que antes eran distintas pasan a ser idénticas. La eliminación reproduce una decisión de los autores; no se presenta como evidencia de un error en los datos.

Se conservarán las filas descartadas y su número de fila de origen para permitir rastrear la operación. Dos perfiles iguales con distinto tratamiento o respuesta se mantienen cuando sus filas completas son distintas.

### Observaciones sobre la referencia

- El comentario inicial del método menciona balancear grupos, pero la función no ejecuta un remuestreo. Se sigue el código ejecutable y no se realiza balanceo ni SMOTE.
- La función conserva ocho predictores y añade `treatment` y `response`, para un total de diez columnas. Esto debe distinguirse de la mención de diez covariables en el artículo.
- `channel` describe el canal de compra anterior del cliente, no el canal de envío de esta campaña.
- La codificación categórica de la referencia se ajusta antes de separar entrenamiento y prueba. Se mantiene para reproducir esta preparación; no utiliza las etiquetas de respuesta.
- No se imputan valores ni se estandarizan variables. Se verificará que no existan faltantes y no se introducirán cambios silenciosos respecto de la preparación de referencia.

### Estado

Bitácora iniciada antes de implementar y ejecutar el tratamiento. Los resultados observados y la comprobación de correspondencia con el código original se incorporarán después de la ejecución.


### Resultados observados y verificaciones

Se ejecutó la preparación el 17 de noviembre de 2026 con Python 3.11.0 y pandas 2.0.1. Se empleó el Python instalado en el equipo del alumno. No fue necesario instalar paquetes ni modificar el entorno del repositorio original.

| Paso | Registros |
| --- | ---: |
| CSV original | 64.000 |
| Correo masculino excluido | 21.307 |
| Correo femenino y control antes de deduplicar | 42.693 |
| Filas repetidas descartadas sobre las columnas preparadas | 4.769 |
| Muestra final | 37.924 |

| Grupo | Filas antes | Filas después | Compras antes | Compras después | Conversión antes | Conversión después |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sin correo | 21,306 | 18,926 | 122 | 122 | 0,5726 % | 0,6446 % |
| Correo femenino | 21,387 | 18,998 | 189 | 187 | 0,8837 % | 0,9843 % |

Los 18.998 registros tratados y 18.926 de control coinciden con los tamaños de Hillstrom Conversion reportados en la Tabla 3 del artículo. Las tasas finales son aproximadamente 0,98 % y 0,64 %, respectivamente. La diferencia descriptiva entre ambas tasas no sustituye una evaluación de modelos ni una prueba de significancia.

En los dos grupos seleccionados ya había 4.438 filas repetidas al considerar todas las columnas originales. Al retirar las otras variables de resultado, el criterio de los autores descarta 331 filas adicionales, para un total de 4.769. La operación también retira dos registros con conversión positiva del grupo tratado; por eso se documentan las compras y las tasas antes y después. No se identifica a estas filas como clientes duplicados demostrados.

Se verificó lo siguiente:

- El resultado coincide fila por fila con una reconstrucción independiente de la secuencia del código original, utilizando `pandas.factorize(sort=True)` para contrastar los códigos categóricos.
- Los 4.769 registros excluidos corresponden a una fila anterior idéntica en las ocho características, tratamiento y respuesta. Se verificaron todas las correspondencias, no solo una muestra.
- El CSV final tiene diez columnas: ocho predictores, `treatment` y `response`. No contiene faltantes ni filas completas repetidas.
- El CSV guardado se volvió a cargar y coincide con el resultado en memoria.
- Los hashes del CSV original y del archivo de preparación de referencia se mantienen intactos.
- Las tres celdas de código de `main.ipynb` se ejecutaron secuencialmente con Python y completaron la preparación. No se usó un motor Jupyter para esta verificación; el notebook queda guardado sin salidas ni contadores de ejecución.
- Una segunda ejecución produjo el mismo hash del CSV preparado.

### Codificación y variables

Los predictores son `recency`, `history_segment`, `history`, `mens`, `womens`, `zip_code`, `newbie` y `channel`.

`history_segment` se codifica de 0 a 6 siguiendo el orden de los tramos de gasto originales. Para `zip_code` se conserva la grafía original `Surburban`, sin corregirla silenciosamente: `Rural = 0`, `Surburban = 1`, `Urban = 2`. Para `channel`, los códigos son `Multichannel = 0`, `Phone = 1`, `Web = 2`. El diccionario completo está en la auditoría JSON.

La implementación propia reproduce la codificación ordenada de los autores usando pandas. No se ejecutó el repositorio completo ni se instalaron sus versiones antiguas de scikit-learn, EconML o CausalML. Esta sustitución de implementación conserva los códigos observados, la secuencia de operaciones, el orden de filas y el dataset final de esta preparación.

### Archivos y forma de repetir la preparación

- Bitácora: `BITACORA.md`.
- Implementación: `src/pipeline/prepare_hillstrom.py`.
- Notebook de trabajo: `src/notebooks/main.ipynb`.
- Datos preparados: `data/hillstrom-email/processed/Hillstrom_Email_w_conversion.csv`.
- Filas excluidas y sus referencias al original: `data/hillstrom-email/processed/Hillstrom_Email_w_conversion_duplicados.csv`.
- Recuentos, tasas, mapeos y hashes: `data/hillstrom-email/processed/Hillstrom_Email_w_conversion_auditoria.json`.

Desde la raíz de `Replicacion`, ejecutar:

```powershell
python src/pipeline/prepare_hillstrom.py
```

También puede ejecutarse `src/notebooks/main.ipynb` de arriba hacia abajo con un kernel que tenga pandas instalado. La preparación usa pandas 2.0.1; no requiere scikit-learn. Las salidas de `processed` se regeneran al ejecutar y el CSV original permanece intacto.

SHA-256 del original: `0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`.

SHA-256 del preparado: `f817371dbdc848b757265691b0cb20e17ec52fd191c39ad3c08fab43d063411b`.

### Estado al cierre de esta etapa

Preparación y tratamiento de duplicados completados y verificados. La siguiente etapa será separar entrenamiento y prueba siguiendo el protocolo del artículo. No se han entrenado modelos ni calculado métricas Qini.


## 17 de noviembre de 2026 · Separación de entrenamiento y prueba

### Protocolo previo a la ejecución

Se empleará la base preparada, con 37.924 filas, sin nueva deduplicación, imputación ni balanceo. Se reproducirá `HelperPipeline.get_dataframe` de la referencia con `test_size=0.2`, `random_state=123`, `shuffle=True` y estratificación conjunta por `response` y `treatment`. El valor 123 procede del parámetro por defecto de `PipelineRW`.

Se guardarán entrenamiento, prueba, asignaciones de filas y auditoría en `data/hillstrom-email/processed/splits`. El conjunto de prueba se reserva para la evaluación final. Los ajustes de modelos y la validación cruzada se realizarán después, exclusivamente sobre entrenamiento.

Se creará `.venv` en este proyecto con acceso a los paquetes existentes de Python 3.11 y scikit-learn 1.3.2 para ejecutar la misma función de partición que la referencia. No se instalarán paquetes en el Python global ni se modificará `original-paper`.

### Resultados de la partición

Se ejecutó la división con scikit-learn 1.3.2. Se reservaron **30.339 filas para entrenamiento** y **7.585 para prueba**. La diferencia mínima respecto de 80/20 exacto corresponde al redondeo de una cantidad entera de filas.

| Respuesta | Tratamiento | Total | Entrenamiento | Prueba |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 18804 | 15043 | 3761 |
| 0 | 1 | 18811 | 15049 | 3762 |
| 1 | 0 | 122 | 98 | 24 |
| 1 | 1 | 187 | 149 | 38 |

Se verificó ausencia de solapamiento de índices, cobertura de todas las filas, conservación de los recuentos por estrato, lectura correcta de los CSV y preservación del hash del dataset preparado. Una segunda ejecución generó los mismos hashes de entrenamiento, prueba y asignaciones. Las nuevas celdas del notebook se ejecutaron secuencialmente con Python, sin motor Jupyter.

Cada conjunto conserva ocho predictores, `treatment` y `response`. El archivo `partition_assignments.csv` conserva el índice de fila preparado, la fila lógica del CSV y la posición dentro de cada partición. Estos identificadores quedan fuera de los predictores. El dataset no contiene un ID de persona, de modo que la verificación de no solapamiento corresponde a filas.

Archivos nuevos: `src/pipeline/split_hillstrom.py`, `requirements-split.txt` y los cuatro archivos de `data/hillstrom-email/processed/splits`. Se amplió `src/notebooks/main.ipynb` conservando sus celdas anteriores.

Para repetir la división desde la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe src/pipeline/split_hillstrom.py
```

El entorno `.venv` utiliza `--system-site-packages`, por lo que reutiliza pandas, NumPy y otras dependencias existentes; no constituye un entorno totalmente aislado. Las versiones directas utilizadas se registran en `requirements-split.txt` y las versiones de ejecución en `split_audit.json`.

### Estado al cierre de la división

Entrenamiento y prueba separados y verificados. No se ajustaron modelos, no se crearon aún las diez particiones de validación cruzada y no se evaluó el desempeño predictivo ni causal. La siguiente etapa será entrenar el primer modelo y ajustar sus parámetros usando únicamente entrenamiento.


## 17 de noviembre de 2026 · Entrenamiento de Two-Model

### Protocolo previo a la ejecución

Se entrenarán dos `RandomForestClassifier`, como en `original-paper/autoum-intmar/src/approaches/two_model.py`. Un bosque aprenderá de quienes recibieron el correo femenino y otro del grupo sin correo. Para cada perfil, el uplift estimado será la probabilidad de compra del primer modelo menos la del segundo. Es una estimación condicional del efecto, no una observación del efecto individual real.

Los ocho predictores y las particiones guardadas se conservan. `treatment`, `response` y los índices de trazabilidad se excluyen de los predictores. No se utilizarán ponderaciones de clase, SMOTE, estandarización ni información del conjunto de prueba para ajustar parámetros.

Se probarán las 16 combinaciones del apéndice A del artículo, con los mismos parámetros para los dos bosques.

| Parámetro | Valores |
| --- | --- |
| `n_estimators` | 100, 200 |
| `max_depth` | 25, 50 |
| `min_samples_leaf` | 10, 30, 50, 70 |
| `max_features` | `sqrt` |
| `random_state` | 123 |

Dentro de las 30.339 filas de entrenamiento se aplicará validación cruzada de diez particiones con `shuffle=True`, semilla 123 y estratificación conjunta por tratamiento y respuesta. Cada ajuste utiliza nueve particiones y se valida en la restante. Se seleccionará el mayor promedio de UQC de validación, con desempate determinista por el orden de `ParameterGrid`. Esto supone 320 ajustes de bosques para seleccionar parámetros y dos ajustes finales sobre todo entrenamiento. Los folds se ejecutarán en paralelo, con un solo hilo por bosque.

### Correspondencias y adaptaciones de la replicación

- `max_features='sqrt'` sustituye a `auto` de la versión antigua del clasificador y conserva su significado.
- La curva se calcula por deciles separados de tratamiento y control, con los redondeos de la referencia. UQC es el cociente entre el área de esa curva y la de selección aleatoria. Su referencia aleatoria es 1 y no debe confundirse con el coeficiente Qini normalizado por una curva óptima.
- El código original desempata puntuaciones usando la respuesta observada. Se sustituye por ordenación estable únicamente por puntuación para que la respuesta no intervenga en el ranking. Esta es una adaptación explícita, por lo que no se promete reproducir exactamente los valores de sus tablas.
- La selección usa el promedio aritmético de UQC de los diez folds de entrenamiento. El puntaje ganador sirve para seleccionar parámetros y no constituye una estimación final imparcial del desempeño.

Solo después de congelar los parámetros y reentrenar ambos bosques se generarán predicciones en las 7.585 filas de prueba. Se guardarán el modelo, las predicciones fuera de fold, los resultados de validación, las asignaciones de folds y un informe de ejecución en `results/two_model`. La evaluación final en prueba corresponde a una etapa posterior.

### Estado antes de entrenar

Implementación y protocolo registrados. Pendientes la ejecución, la comprobación de los artefactos y el registro de resultados observados.


### Resultados del entrenamiento y comprobaciones

Se completó la búsqueda de 16 configuraciones y diez folds por configuración. Los parámetros seleccionados fueron `n_estimators=200`, `max_depth=25` y `min_samples_leaf=50`. Ambos modelos conservan `max_features='sqrt'` y semilla 123.

El UQC medio de validación de la configuración ganadora fue **1.04416**. Es un resultado de selección dentro de entrenamiento, no el resultado final en prueba. La referencia aleatoria de esta definición es 1. La variabilidad de los folds puede consultarse en `cv_fold_metrics.csv` y no debe interpretarse este promedio como una mejora garantizada en otros datos.

El bosque de control se entrenó con **15.141 filas**, de las cuales 98 tienen compra positiva. El bosque tratado se entrenó con **15.198 filas**, de las cuales 149 tienen compra positiva. Se guardaron 30.339 predicciones fuera de fold y 7.585 predicciones de prueba. La respuesta se conserva en los CSV de predicciones para su evaluación posterior, pero no entra en el cálculo del score.

La selección y los ajustes finales tardaron aproximadamente 117.4 segundos. Se usaron las versiones registradas en `training_report.json` y las dependencias directas de `requirements-training.txt`. La carpeta `.venv` sigue reutilizando paquetes del Python instalado.

Se verificaron los siguientes puntos.

- Los 160 registros de validación corresponden a las 16 configuraciones y diez folds. El ganador coincide con el mayor promedio registrado y los folds reproducen la estratificación prevista.
- Cada fila de entrenamiento aparece exactamente una vez como validación. Los cuatro estratos están presentes en cada fold. Los índices de trazabilidad no se utilizan como predictores.
- Se reconstruyeron los dos bosques de un fold ganador y se reprodujeron sus predicciones fuera de fold.
- El modelo guardado se volvió a cargar. Sus probabilidades reproducen las 7.585 filas del archivo de predicciones de prueba y el uplift coincide con su diferencia.
- Se ejecutó el método de entrenamiento de la clase `TwoModel` original con los mismos parámetros y datos. Sus probabilidades coincidieron exactamente con las del modelo propio sobre 100 perfiles de entrenamiento. Esto comprueba el ajuste de los bosques, no la equivalencia de toda la evaluación publicada.
- La fórmula UQC se contrastó con los métodos originales en 20 casos sintéticos con scores distintos y efectos agregados de distinto signo. El caso de denominador cero queda explícitamente indefinido.
- Los hashes de datos originales, preparados, particiones y fuentes de referencia se conservaron. La selección de parámetros no utilizó el contenido de prueba.
- Se conservaron todas las celdas anteriores de `main.ipynb` y se añadieron tres celdas para explicar, cargar y reutilizar el entrenamiento. Se comprobó la sintaxis de todas las celdas y se ejecutó con Python la nueva celda de lectura, sin motor Jupyter. El notebook queda sin salidas ni contadores de ejecución.

### Archivos y ejecución

- Implementación propia en `src/pipeline/train_two_model.py`.
- Ambos bosques y metadatos en `results/two_model/two_model.joblib`.
- Parámetros y resultados de validación en `cv_results.csv` y `cv_fold_metrics.csv`.
- Folds y predicciones fuera de fold en `cv_assignments.csv` y `oof_predictions.csv`.
- Probabilidades con correo, sin correo y uplift de prueba en `test_predictions.csv`.
- Protocolo ejecutado, versiones y verificaciones en `training_report.json` y `verification_report.json`.

Los archivos de resultados mencionados están en `results/two_model`. Para repetir el entrenamiento desde la raíz de `Replicacion` se puede ejecutar el comando siguiente. Regenera los resultados de esta etapa.

```powershell
.\.venv\Scripts\python.exe src/pipeline/train_two_model.py --jobs 4
```

En el notebook los resultados se cargan por defecto sin repetir la búsqueda. Para regenerarlos desde allí, cambiar `REENTRENAR_TWO_MODEL` a `True`.

### Estado al cierre del entrenamiento

Two-Model entrenado, guardado y verificado. La evaluación final en prueba todavía no se ha realizado. La selección y el entrenamiento están completos; la siguiente etapa es evaluar las predicciones congeladas en prueba sin reajustar con esos resultados.


## 17 de noviembre de 2026 · Evaluación final de Two-Model en prueba

### Protocolo de evaluación

Se evalúa el par de bosques ya guardado, con los parámetros congelados antes de observar el desempeño en prueba. No se entrena ningún modelo adicional ni se seleccionan parámetros o umbrales a partir de estos resultados. Se conservan los archivos y reportes del entrenamiento como registros históricos.

Se volverán a calcular sus probabilidades para verificar que reproduzcan `test_predictions.csv`. La evaluación principal utilizará las 7.585 filas de prueba y el ranking estable por score, sin desempatar con `response`. Se calcularán la curva de diez deciles, UQC, AUUC relativa y el área por encima de la diagonal. Las dos áreas se identificarán con sus definiciones para evitar confundirlas con otras métricas llamadas AUUC o Qini.

Se contrastarán los resultados con las tablas 5 y A1 y el apéndice B de Rößler y Schoder (2022), consultados en la versión editorial. La comparación distinguirá el promedio publicado entre modelos de folds de nuestro único par reentrenado. El desempate por respuesta del código original se calculará aparte como diagnóstico, nunca como política aplicable ni para escoger el resultado principal.


### Resultados observados en prueba

Se evaluaron **7.585 filas**, con 3.800 del grupo tratado y 3.785 de control. Se observaron 38 compras en tratados y 24 en control. Sus tasas son 1,0000 % y 0,6341 %, respectivamente. La diferencia agregada es 0,365918 p.p., que se presenta como 0,370 p.p. al aplicar el redondeo de la curva de referencia. Esta diferencia de tasas no depende de que el modelo ordene bien a los usuarios.

El modelo final obtuvo **UQC = 1,3686**, **AUUC relativa = 0,0025250** y **área Qini por encima de la diagonal = 0,0006750**. El UQC de validación de entrenamiento, 1,04416, permanece registrado como criterio de selección y no sustituye este resultado de prueba.

### Tabla comparativa con el artículo original

Los valores publicados se verificaron en la versión editorial de Rößler y Schoder (2022), tablas 5 y A1 y apéndice B. La comparación corresponde exclusivamente a **TM sobre Hillstrom Conversion**.

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

**Nota sobre las unidades.** La ganancia acumulada se calcula como `G(q) = rT(q) - rC(q)*nT(q)/nC(q)` y se normaliza mediante `g(q) = G(q)/N_T`, donde `N_T` es el total tratado de prueba. Se selecciona la fracción `q` superior de cada grupo por separado. Los valores en p.p. son `100*g(q)`. No son porcentajes de mejora en ventas ni tasas de compra del segmento seleccionado. Por ejemplo, al 20 % se seleccionan 760 tratados y 757 controles, con 14 y 5 compras. La ganancia estimada es 8,9802 respuestas, equivalente a 0,236321 p.p. antes del redondeo, o 0,240 p.p. al aplicar los cuatro decimales de `g(q)` usados por la referencia.

**Nota sobre las áreas.** AUUC integra la curva `g(q)` de los deciles redondeados. Para UQC, la diagonal también se redondea a cuatro decimales como en el código original. La métrica suplementaria de área Qini resta la diagonal lineal sin redondear `q*g(1)`. Por ello, UQC y esa área suplementaria no deben intercambiarse ni calcularse una a partir de otra ignorando los redondeos. No se atribuyen al artículo valores de AUUC o área Qini que no presenta para TM en la tabla 5.

### Interpretación y alcance de la comparación

El UQC supera la referencia aleatoria de 1 en esta prueba, pero queda **0,6325 unidades por debajo** del promedio publicado. Esto indica una mayor área que la diagonal aleatoria para esta evaluación puntual y una puntuación menor que la referencia publicada. No demuestra significancia estadística ni una mejora de ese porcentaje en conversiones. La curva queda por debajo de la diagonal en los deciles del 40 %, 50 % y 60 %, por lo que el modelo no domina a la selección aleatoria en todos los presupuestos.

La comparación es descriptiva y no una reproducción numérica exacta. El código de referencia evalúa modelos de los folds sobre el mismo conjunto de prueba y agrega sus métricas. Aquí se evalúa un único par reentrenado con las 30.339 filas de entrenamiento. La desviación estándar publicada de 0,23 corresponde a sus modelos de folds; no es un intervalo de confianza de nuestro resultado y no se estima una desviación entre modelos a partir de una sola pareja final. Además, los parámetros seleccionados y las versiones de Python y scikit-learn difieren. Son diferencias verificadas del protocolo, no causas cuantificadas de la discrepancia.

El diagnóstico con el desempate por respuesta del original produjo **UQC = 1,3686** y los mismos once puntos de la curva. En esta evaluación, esa adaptación no cambia el resultado y no explica la diferencia con la cifra publicada. No se ha identificado una causa única ni se han reajustado parámetros para acercar el resultado al del artículo.

El máximo de la curva observado en prueba es descriptivo. No se utiliza para elegir un presupuesto de campaña ni se presenta como un umbral óptimo validado. Tampoco se calcula rentabilidad neta, porque esta evaluación utiliza una respuesta binaria y no incorpora costos de campaña ni márgenes.

### Verificaciones y archivos de evaluación

Se reprodujeron las probabilidades del modelo guardado y las 7.585 predicciones de prueba. La curva y el UQC se contrastaron con las fórmulas ejecutables del original, preservando el ranking estable sin respuesta. Se comprobó que el punto final coincide con la diferencia observada de tasas, que los recuentos acumulados son consistentes y que los hashes de modelos, predicciones, datos, fuentes y reportes previos se conservan. No se volvió a entrenar ningún bosque.

Se añadieron tres celdas al notebook conservando todas las anteriores. Se comprobó su sintaxis y se ejecutó con Python la celda nueva que carga la evaluación. El notebook queda sin salidas ni contadores de ejecución. La celda comprueba que el modelo y la prueba coincidan con los hashes de la evaluación para evitar mostrar resultados desactualizados. La figura se revisó visualmente.

Los artefactos se guardaron en `results/two_model/evaluation`.

- `test_metrics.json` contiene métricas, definiciones, referencia y verificaciones de integridad.
- `test_decile_curve.csv` contiene los once puntos, recuentos, ganancias y diagnóstico del desempate original.
- `paper_comparison.csv` y `paper_comparison.md` contienen la comparación reutilizable.
- `test_qini_curve.png` muestra el modelo y la diagonal aleatoria.
- `evaluation_verification.json` registra las comprobaciones y hashes de las salidas.

La implementación está en `src/pipeline/evaluate_two_model.py` y las dependencias directas en `requirements-evaluation.txt`. Para repetir únicamente la evaluación desde la raíz de `Replicacion` se puede ejecutar el comando siguiente.

```powershell
.\.venv\Scripts\python.exe src/pipeline/evaluate_two_model.py
```

En el notebook, `REEVALUAR_TWO_MODEL = True` permite repetir la evaluación sin entrenar. Los indicadores `test_performance_evaluated=False` de los reportes de entrenamiento previos describen el estado histórico de esa etapa. El estado actualizado de evaluación se registra como `True` en `test_metrics.json`.

### Referencia de la comparación

Rößler, J., & Schoder, D. (2022). Bridging the gap: A systematic benchmarking of uplift modeling and heterogeneous treatment effects methods. *Journal of Interactive Marketing, 57*(4), 629–650. https://doi.org/10.1177/10949968221111083

Fuente consultada: https://journals.sagepub.com/doi/10.1177/10949968221111083. Consulta realizada el 17 de noviembre de 2026.

### Estado al cierre de la evaluación

Evaluación de prueba y comparación con el artículo completadas. Se conserva el resultado obtenido, sin cambiar el modelo después de observar prueba. La tabla documenta tanto las coincidencias como las diferencias del protocolo y de desempeño.


## 17 de noviembre de 2026 · Two-Model con configuración exacta del artículo

### Protocolo registrado antes de ejecutar esta variante

Se desarrolla una variante para mejorar la fidelidad de la replicación. Se fijan los parámetros publicados en la tabla A1 para TM en Hillstrom Conversion: `n_estimators=100`, `max_depth=25`, `min_samples_leaf=30`. Se mantienen `max_features='sqrt'`, semilla 123 y ausencia de ponderaciones, como en la implementación documentada. No se inicia una nueva búsqueda de parámetros.

La configuración publicada ya formó parte de nuestra búsqueda original y obtuvo UQC medio de validación 0,89191, frente a 1,04416 del ganador previo. Por tanto, se adopta para reproducir el artículo, no porque haya ganado la selección de entrenamiento.

Se entrenarán los diez pares de bosques con las particiones de validación originales y se promediarán las diez métricas de prueba, reproduciendo la agregación del pipeline de referencia. Se conservarán todos los folds; no se elegirá el de mejor puntuación. Además, se ajustará un único par con toda la muestra de entrenamiento para compararlo bajo la misma unidad que el modelo anterior. Estos dos resultados se reportarán por separado.

Las particiones, las métricas y el ranking estable se conservan. El desempate por respuesta del original se calculará solo como diagnóstico. La variante quedará en `results/two_model_paper_config` y no sustituirá los archivos del experimento previo.

El conjunto de prueba ya fue observado en la etapa anterior. Esta comparación adicional se registra como exploratoria y no constituye una nueva evaluación confirmatoria independiente. Se reportará la puntuación que resulte sin utilizar una cifra deseada como criterio para ajustar el experimento.


### Resultados de la configuración publicada

Se entrenaron 22 bosques, correspondientes a diez pares de folds y un par final con todo entrenamiento. Se guardaron todos los modelos, sus predicciones de prueba, las predicciones fuera de fold y los recuentos de las curvas. Los resultados son los siguientes.

| Variante | Árboles | Profundidad | Mínimo por hoja | UQC medio de validación | UQC en prueba |
| --- | ---: | ---: | ---: | ---: | ---: |
| Modelo anterior, reentrenado con todo entrenamiento | 200 | 25 | 50 | 1,04416 | 1,3686 |
| Parámetros del artículo, reentrenado con todo entrenamiento | 100 | 25 | 30 | 0,89191 | **1,6450** |
| Parámetros del artículo, promedio de diez modelos de folds | 100 | 25 | 30 | 0,89191 | **1,3984** (DE muestral 0,2488) |
| Resultado publicado, tabla 5 | 100 | 25 | 30 | No informado en la tabla | 2,0011 (DE 0,23) |

El par final con los parámetros publicados obtuvo **1,6450**, una diferencia de **+0,2764 unidades de UQC** frente al par final anterior. Es una mejora de puntuación en la prueba observada, no un porcentaje de aumento en ventas ni una prueba de mejora generalizable. Su ganancia acumulada normalizada al 20 % es 0,320 p.p., frente a 0,240 p.p. del modelo anterior, con la definición y el redondeo ya documentados.

Al reproducir la agregación de diez modelos de folds, el UQC medio es **1,3984**. Este resultado es el de la comparación más cercana al procedimiento del pipeline publicado. No debe reemplazarse por 1,6450 como si ambos fuesen promedios equivalentes. La DE muestral entre modelos es 0,2488 con `ddof=1`; con `ddof=0` es 0,2360. Estas desviaciones describen la variabilidad entre los diez ajustes que comparten prueba, no un intervalo de confianza de observaciones independientes.

Los resultados individuales de los folds abarcan desde 0,9133 hasta 1,7696. Conservar solo el fold más alto no representa el desempeño del procedimiento. Se guardaron los diez resultados y se calculó su promedio completo.

La configuración exacta no ganó la validación de entrenamiento: mantiene UQC medio 0,89191 frente a 1,04416 del ganador anterior. Esto ilustra variabilidad entre particiones y evita presentar los parámetros publicados como una nueva selección exitosa de entrenamiento. El resultado final de 1,6450 tampoco alcanza el rango solicitado de 1,70–1,80. Se conserva la cifra obtenida y no se inicia una búsqueda sobre prueba para acercarla a un objetivo numérico.

El desempate por respuesta del original produce 1,6450 en el par final y 1,3984 como media de folds. En esta variante tampoco explica la diferencia frente al valor publicado. Se mantiene el ranking estable como evaluación principal.

### Verificaciones y reproducibilidad de esta variante

Se reprodujeron los diez UQC de validación de esta configuración obtenidos durante la búsqueda inicial. Las asignaciones de folds coinciden con las guardadas. Las curvas y el UQC de los once pares se contrastaron con las fórmulas originales, usando rangos únicos para conservar el ranking estable. Todos los archivos de modelos se recargaron y reprodujeron sus scores de prueba. Se comprobaron 7.585 predicciones del par final, 75.850 de los diez folds y 30.339 predicciones fuera de fold.

Los datos, los modelos anteriores, sus reportes y `original-paper` permanecen intactos según las comprobaciones de hashes. La configuración, las fuentes y el hecho de que prueba ya se había observado están registrados en `experiment_protocol.json` y `experiment_report.json`.

Los archivos propios de esta variante están en `results/two_model_paper_config`.

- `two_model_paper_final.joblib` guarda el par con los parámetros exactos y todo entrenamiento.
- `two_model_paper_fold_models.joblib` guarda los diez pares de los folds.
- `comparison.csv` y `comparison.md` presentan las cuatro filas de la tabla comparativa.
- `fold_test_metrics.csv` registra los diez resultados individuales y los puntajes de validación.
- `final_test_predictions.csv`, `fold_test_predictions.csv` y `oof_predictions.csv` conservan probabilidades, scores y trazabilidad.
- `final_test_curve.csv`, `fold_test_curves.csv` y `mean_fold_test_curve.csv` registran las curvas.
- `comparison_qini.png` permite comparar las curvas del modelo anterior, del nuevo par final y del promedio de folds.

Se añadieron tres celdas al notebook conservando las anteriores. Se comprobó la sintaxis de todas las celdas y se ejecutó la nueva celda de lectura con Python, sin motor Jupyter. El notebook se conserva sin salidas ni contadores de ejecución.

Para regenerar exclusivamente esta variante desde la raíz de `Replicacion` se puede ejecutar el comando siguiente. Se reutilizan las dependencias registradas en `requirements-evaluation.txt`.

```powershell
.\.venv\Scripts\python.exe src/pipeline/replicate_two_model_paper.py --overwrite
```

### Estado y uso recomendado de los resultados

La variante con parámetros exactos está entrenada y documentada. La mejora observada del par final es exploratoria porque la prueba ya se había utilizado. Para demostrar una mejora generalizable en futuros desarrollos se requiere una evaluación adicional independiente o un protocolo de validación anidada definido antes de comparar sus resultados. Esta variante sirve para analizar la fidelidad de la replicación y no reemplaza silenciosamente la selección original de entrenamiento.


## 17 de noviembre de 2026 · Cierre de la auditoría de replicación

### Protocolo

Se identificó `requirements_causalml.txt`, que fija las versiones relevantes de las bibliotecas originales. Se preparó un runtime oficial de Python 3.8.10 embebido y aislado en `.venv-paper`, con scikit-learn 1.0.2, NumPy 1.22.4, pandas 1.4.2 y SciPy 1.8.1. No es un venv convencional ni modifica el Python global. Se ejecutarán los cuerpos originales de las clases pertinentes mediante extracción AST, omitiendo únicamente imports y algoritmos ajenos a Two-Model.

Se compararán los datos originales, las filas y columnas preparadas, las particiones y los scores con los artefactos existentes. Se utilizarán los parámetros publicados y la agregación original de diez folds. Esta auditoría no buscará parámetros para alcanzar un UQC determinado. Los resultados previos se conservarán.


### Resultado del cierre técnico

Se ejecutó el núcleo de Two-Model con Python 3.8.10 y las versiones fijadas en `requirements_causalml.txt`. Se utilizaron directamente los cuerpos de las clases y métodos originales, extraídos mediante AST para omitir imports de algoritmos ajenos. No se modificó `original-paper` ni se ejecutaron los otros catorce algoritmos. Esta precisión distingue la auditoría del núcleo pertinente de una ejecución integral del repositorio.

| Componente | Estado | Evidencia |
| --- | --- | --- |
| CSV de entrada | Coincide | Hash idéntico al archivo Hillstrom incluido en la copia de referencia. |
| Preparación y duplicados | Coincide | Se ejecutó Hillstrom.prep original. Coinciden todas las filas, columnas, valores y su orden. |
| Entrenamiento y prueba | Coincide | HelperPipeline.get_dataframe original reproduce ambas particiones completas con semilla 123. |
| Ocho predictores y orden | Coincide | recency, history_segment, history, mens, womens, zip_code, newbie, channel |
| Parámetros de TM | Coincide con tabla A1 | 100 árboles, profundidad 25, mínimo por hoja 30; semilla 123, sin pesos de clase. |
| Parámetros restantes | Equivalentes | Se contrastaron los parámetros efectivos. Solo auto/sqrt cambia de nombre, conservando su significado. |
| Entorno relevante | Reproducido | Python 3.8.10, sklearn 1.0.2, NumPy 1.22.4, pandas 1.4.2, SciPy 1.8.1, joblib 1.1.0. |
| Métodos originales y diez folds | Ejecutados | Se ejecutaron DataSetsHelper, TwoModel.analyze y create_k_splits originales, sin alterar sus cuerpos. |
| Métrica y agregación | Ejecutadas como referencia | Evaluación original con desempate por respuesta y calculate_mean sobre los diez folds. |
| Scores frente a implementación propia | Equivalentes numéricamente | Máxima diferencia absoluta de aproximadamente 1e-16, atribuible a precisión de escritura/lectura; métricas iguales a cuatro decimales. |
| Resultado publicado 2,0011 | No reproducido | La ejecución de referencia da 1,3984 como media de folds. La causa de la brecha sigue sin determinarse. |

### Comparación después de igualar el entorno

| Procedimiento | Entorno moderno | Entorno original | Artículo publicado |
| --- | ---: | ---: | ---: |
| Par final con parámetros de tabla A1 | 1,6450 | 1,6450 | No es la unidad de tabla 5 |
| Media de los diez modelos de folds | 1,3984 | 1,3984 | 2,0011 |

El entorno original dio los mismos UQC que el moderno. La mayor diferencia entre los scores originales y los scores modernos guardados en CSV es aproximadamente 1e-16, dentro de la precisión de almacenamiento decimal. Las métricas de los diez folds coinciden a cuatro decimales. Por tanto, **las versiones y la implementación propia no explican la discrepancia con el 2,0011 publicado para este experimento**.

También se contrastaron los parámetros efectivos completos del Random Forest. La única diferencia nominal es `max_features='auto'` frente a `sqrt`, que conserva el significado del clasificador. No se identificó un parámetro por defecto distinto que explique el resultado.

La preparación original y la propia coinciden exactamente en filas, columnas, valores y orden. Lo mismo ocurre con entrenamiento y prueba. Se verificó además que el CSV de entrada propio es idéntico a la copia del CSV del repositorio de referencia. Esta comprobación acredita la coincidencia con el material disponible; no demuestra qué archivos concretos se utilizaron para producir la tabla del artículo.

### Discrepancia pendiente frente a la publicación

El `main.py` distribuido contiene un ejemplo configurado para Hillstrom Visit, con valores distintos de los de la tabla A1. Ejecutar sus valores de ejemplo sin adaptar el experimento no reproduce automáticamente Hillstrom Conversion. En esta auditoría se utilizó la configuración de la tabla A1 y el procedimiento de diez folds documentado, no los valores de ejemplo.

Se cierra la auditoría técnica con coincidencia entre la implementación propia y los métodos originales en ambos entornos. **No se declara reproducido el resultado numérico publicado.** La media obtenida permanece en 1,3984 frente a 2,0011. No se encontró una corrección verificable que eleve esa media al valor del artículo.

Para resolver esa diferencia harían falta artefactos de la corrida publicada: configuración concreta, revisión exacta del código, datos preparados o índices de partición y métricas por fold o modelos guardados. Sin esos elementos no es posible atribuir la brecha a una causa específica ni afirmar que exista un error en el artículo. Cambiar semillas o seleccionar folds para acercar la cifra no resolvería la discrepancia metodológica.

### Archivos de cierre y reproducción

- `src/pipeline/audit_original_two_model.py` contiene la auditoría reproducible del núcleo original.
- `requirements-paper-core.txt` fija las dependencias relevantes de ese núcleo.
- `.venv-paper/python.exe` ejecuta el runtime oficial embebido de Python 3.8.10. Es aislado y no un venv convencional. Las dependencias científicas y sus versiones coinciden con la referencia; no se instalaron todos los paquetes de los demás algoritmos.
- `results/replication_closure/closure_audit.json` guarda versiones, parámetros efectivos, hashes y resultados de comprobación.
- `audit_checklist.csv` y `CLOSURE.md` resumen el cierre en esa misma carpeta.
- `original_fold_metrics.csv`, `original_fold_predictions.csv`, `original_decile_metrics.csv` y `original_aggregated_metrics.csv` conservan los resultados originales de los folds.
- `original_prepared.csv` conserva la preparación ejecutada por el método original.
- `original_environment_final.joblib` y `original_final_predictions.csv` guardan el par final y sus predicciones en el entorno antiguo. El modelo debe cargarse con scikit-learn 1.0.2.

Para repetir la auditoría desde la raíz del proyecto:

```powershell
.\.venv-paper\python.exe src/pipeline/audit_original_two_model.py
```

Los archivos de cierre se regeneran al ejecutar, conservando los resultados de las etapas anteriores. Se añadieron dos celdas al notebook; las anteriores se preservaron. Se verificó la sintaxis y se ejecutó la nueva celda de lectura con Python, sin motor Jupyter. No se modificó el Python global ni las particiones existentes.

### Estado final

Auditoría técnica completada. Núcleo original reproducido bajo las versiones pertinentes y concordancia comprobada con la implementación propia. Reproducción de la cifra publicada: no alcanzada, con discrepancia documentada. No se realizaron ajustes dirigidos a una puntuación objetivo.


## 17 de noviembre de 2026 · Presentación del piloto

Se preparó `VILELA_REPLICACION_PILOTO_4_SLIDES.pptx` con cuatro diapositivas totales: carátula, artículo y metodología, experimentación y métricas, y resultados y conclusiones. Se conserva la tipografía Verdana y el diseño de la plantilla suministrada. La presentación incluye ejemplos reales de probabilidades y scores, las definiciones de UQC, AUUC y área Qini, y la distinción entre el par final y la media de folds. Las tablas y la curva son objetos editables de PowerPoint. Se verificaron las cuatro diapositivas renderizadas en PowerPoint, además de la edición y reapertura de una copia privada del gráfico. La plantilla original se conservó intacta.


## 17 de noviembre de 2026 · Recuperación de main.ipynb

Al revisar el archivo en disco, se encontraron únicamente las seis celdas de preparación con salidas ejecutadas. Se recuperaron las secciones posteriores desde las copias previas del proyecto y se reconstruyó la sección de cierre a partir de su auditoría guardada. El notebook queda con veinte celdas y conserva las tres celdas de código ejecutadas y sus salidas de preparación. Se actualizó el encabezado y la transición de preparación para reflejar el alcance completo.

Se guardó una copia anterior en `src/notebooks/backups/main_preparacion_recuperada_452e19caaa.ipynb`. Se verificó la sintaxis de todas las celdas de código y se ejecutaron con Python las cuatro celdas de lectura de resultados, sin motor Jupyter ni nuevo entrenamiento. Los resultados de modelos y las particiones no se regeneraron. No se pudo determinar el origen de la sustitución por la versión anterior. Una pestaña abierta puede conservar una copia previa en memoria y debe recargarse desde disco antes de seguir editando.


## 17 de noviembre de 2026 · Preparación para GitHub

Se preparó el repositorio `https://github.com/DavidVR27/ReplicacionArticulos_TPI` con el notebook completo, los scripts, la bitácora, Hillstrom y los resultados y modelos del piloto. Se excluyeron los PowerPoint, los entornos, los respaldos y los datasets ajenos al piloto. Se añadió un README con instalación, ejecución y limitaciones. El código externo de los autores se obtiene por separado siguiendo el README. `.gitattributes` evita transformar finales de línea y preserva los hashes de las auditorías.
