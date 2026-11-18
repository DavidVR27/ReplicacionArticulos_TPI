# Cierre técnico de la replicación

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

| Procedimiento | Entorno moderno | Entorno original | Artículo publicado |
| --- | ---: | ---: | ---: |
| Par final con parámetros de tabla A1 | 1,6450 | 1,6450 | No es la unidad de tabla 5 |
| Media de los diez modelos de folds | 1,3984 | 1,3984 | 2,0011 |

La ejecución de los métodos originales bajo las versiones fijadas reproduce la implementación propia, pero no la cifra publicada. La discrepancia queda abierta y no se presenta como una reproducción numérica exitosa del artículo.
