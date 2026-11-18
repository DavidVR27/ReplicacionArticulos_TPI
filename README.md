# Replicación de Two-Model para targeting promocional

Implementación piloto de **Two-Model / T-Learner con Random Forest** sobre Hillstrom Conversion. Compara el correo de productos femeninos con el grupo que no recibió correo y utiliza la compra posterior como respuesta binaria.

El piloto reproduce una parte del artículo de Rößler y Schoder (2022), no la comparación completa de sus quince métodos.

## Contenido

- `src/notebooks/main.ipynb`: notebook completo, con preparación, partición, entrenamiento, evaluación y auditoría.
- `src/pipeline/`: scripts reproducibles de cada etapa.
- `data/hillstrom-email/`: datos usados y archivos de preparación y partición.
- `results/two_model/`: búsqueda inicial y evaluación del modelo seleccionado por validación.
- `results/two_model_paper_config/`: parámetros exactos publicados y resultados de los diez folds y del par final.
- `results/replication_closure/`: contraste con los métodos y las versiones originales.
- `BITACORA.md`: decisiones, comprobaciones y limitaciones de la replicación.

Las presentaciones de PowerPoint, los entornos de Python y los datasets ajenos a Hillstrom están excluidos mediante `.gitignore`. Los modelos y resultados guardados sí forman parte de esta entrega. `.gitattributes` conserva los bytes y finales de línea para no invalidar los hashes de auditoría.

## Instalación

Desde la raíz del repositorio, con Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-evaluation.txt
```

El entorno moderno utilizado originalmente fue Python 3.11.0, NumPy 1.24.3, pandas 2.0.1 y scikit-learn 1.3.2. Se incluyen las versiones directas en los archivos `requirements-*.txt`.

## Código de referencia de los autores

La evaluación y la auditoría consultan archivos de `original-paper/autoum-intmar`. Esa carpeta externa no se publica dentro de este repositorio. Para obtenerla:

```powershell
$env:GIT_LFS_SKIP_SMUDGE = "1"
git clone --depth 1 --branch intmar https://github.com/jroessler/autoum.git original-paper/autoum-intmar
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
```

Se omite la descarga automática de datasets externos gestionados mediante Git LFS. El CSV Hillstrom necesario ya está en `data/hillstrom-email`. El repositorio externo puede evolucionar: `results/replication_closure/closure_audit.json` registra los hashes de las fuentes utilizadas en esta auditoría.

## Ejecución

```powershell
.\.venv\Scripts\python.exe src/pipeline/prepare_hillstrom.py
.\.venv\Scripts\python.exe src/pipeline/split_hillstrom.py
.\.venv\Scripts\python.exe src/pipeline/train_two_model.py --jobs 4
.\.venv\Scripts\python.exe src/pipeline/evaluate_two_model.py
.\.venv\Scripts\python.exe src/pipeline/replicate_two_model_paper.py --jobs 4 --overwrite
```

Los scripts regeneran archivos de su etapa. El notebook carga por defecto los resultados de modelos guardados y solo reentrena cuando se activan sus variables de control. Las primeras celdas de preparación y división sí ejecutan esas etapas. Los enlaces y rutas absolutas de algunos reportes y hashes corresponden a la ejecución original y se conservan como trazabilidad histórica.

### Auditoría con el entorno original

Con un entorno independiente de **Python 3.8.10**, instalar `requirements-paper-core.txt` y ejecutar:

```powershell
python src/pipeline/audit_original_two_model.py
```

La carpeta local `.venv-paper` usada en la auditoría no se incluye en Git. Los modelos de `results/replication_closure` deben cargarse con scikit-learn 1.0.2. No se deben cargar archivos pickle/joblib de procedencia desconocida.

## Datos y resultados

La preparación conserva 37.924 filas, con ocho predictores, tratamiento y respuesta. La división reserva 30.339 para entrenamiento y 7.585 para prueba. La eliminación de filas idénticas reproduce el código de los autores; sin un ID individual, no demuestra que se trate de clientes duplicados.

| Procedimiento | UQC en prueba |
| --- | ---: |
| Selección inicial mediante validación, par final | 1,3686 |
| Parámetros publicados, par final con todo entrenamiento | 1,6450 |
| Parámetros publicados, promedio de diez modelos de folds | 1,3984 |
| Promedio publicado para TM en Hillstrom Conversion | 2,0011 |

UQC compara el área de la curva con la de selección aleatoria, cuya referencia es 1. El resultado del par final y el promedio de folds corresponden a unidades distintas. Se calculan también AUUC relativa y área Qini, con sus definiciones y redondeos documentados en la bitácora.

La auditoría con Python 3.8.10 y las bibliotecas originales reprodujo las puntuaciones de nuestra implementación, pero **no reprodujo el 2,0011 publicado**. La variante con parámetros publicados se ejecutó después de revisar prueba y su comparación es exploratoria. Estas puntuaciones no establecen significancia estadística, rentabilidad ni una mejora generalizable.

## Fuentes

- Rößler, J., & Schoder, D. (2022). Bridging the gap: A systematic benchmarking of uplift modeling and heterogeneous treatment effects methods. *Journal of Interactive Marketing, 57*(4), 629–650. https://doi.org/10.1177/10949968221111083
- Código original: https://github.com/jroessler/autoum/tree/intmar
- Dataset Hillstrom: https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html
