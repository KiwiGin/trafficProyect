## UTD19 Traffic Big Data & Deep Learning Project

Este proyecto implementa un pipeline extremo a extremo para la gestión inteligente de tráfico utilizando el dataset UTD19 (datos masivos de aforo, velocidad y ocupación de detectores). Integra procesamiento distribuido con Apache Spark, ingeniería de características, modelado secuencial profundo (LSTM espaciotemporal) y evaluación con métricas operativas y de impacto (tiempo de viaje, combustible y ROI estimado).

### Objetivos Principales
1. Ingerir y limpiar datos masivos (100M+ registros potenciales) de tráfico urbano.
2. Enriquecer con variables temporales, espaciales y de estado de congestión.
3. Entrenar un modelo secuencial capaz de predecir flujo vehicular y nivel de congestión.
4. Generar visualizaciones analíticas y un reporte ejecutivo de métricas.

---
## Integrantes

| Nombre | Contacto |
|--------|----------|
| Aldair Rivera | aldair.rivera1@unmsm.edu.pe |
| Cristhian Torres | cristhian.torres1@unmsm.edu.pe |
| Eva Bayes | eva.bayes@unmsm.edu.pe |
| José Sernaque Cobeñas | jose.sernaque2@unmsm.edu.pe |

---
## Arquitectura de la Solución

La arquitectura lógica se compone de los siguientes módulos / capas:

1. Ingesta y Almacenamiento Raw
	- Fuente: CSVs (`detectors_public.csv`, `links.csv`, `utd19_u.csv`) ubicados en `data/raw/`.
	- Formato inicial: CSV sin inferencia automática de tipos (se fuerza casting controlado).

2. Procesamiento Distribuido (Spark ETL)
	- `UTD19_ETL_Pipeline` realiza: extracción, validación de reglas de dominio, imputación, tratamiento de outliers y creación de variables derivadas.
	- Optimización: reparticionamiento por `city` y `detid`; winsorización percentil 1–99 para `flow` y `speed`.
	- Persistencia de datos limpios en `data/processed/` (CSV coalesced). (Versión Parquet comentada para futura optimización.)

3. Feature Engineering & Dataset ML
	- Variables temporales: hora, día de semana, flag fin de semana, hora punta.
	- Variables espaciales: carriles, proximidad a intersección (`near_intersection`), posición relativa en link, límite de velocidad.
	- Variables derivadas: `speed_ratio`, `congestion_level` categorizado (0–4).
	- Flags de imputación para trazabilidad (`*_was_imputed`).

4. Preparación Secuencial para Deep Learning
	- Conversión a Pandas de una muestra configurable (limit para evitar OOM).
	- Escalado estándar de features numéricas y mapeo de detectores a índices para embeddings.
	- Construcción de secuencias por detector (ventana deslizante) para LSTM.

5. Modelo Espacio-Temporal (Keras / TensorFlow)
	- Embedding de detector (32 dimensiones) + concatenación con features temporales.
	- Dos capas LSTM (128 y 64 unidades) con dropout.
	- Cabezas múltiples: regresión de flujo (MSE) + clasificación de congestión (Softmax 5 clases).

6. Evaluación y Métricas de Negocio
	- Métricas técnicas: MAE flujo, accuracy congestión, F1 por clase, matriz de confusión.
	- Métricas operativas: reducción de tiempo de viaje (estimada), ahorro de combustible, satisfacción simulada, ROI económico.

7. Visualización
	- Distribuciones de variables, correlaciones, curvas de entrenamiento, residuales, matriz de confusión y comparación predicho vs real.

8. Persistencia de Resultados
	- Pesos del modelo (`utd19_spatiotemporal_model.h5`).
	- Artefactos gráficos (`*.png`).
	- Métricas consolidadas (`case_study_metrics.json`).

### Flujo de Datos (Resumen)
Raw CSV -> Spark ETL (limpieza + enriquecimiento) -> Datos procesados -> Muestra a Pandas -> Secuencias -> Entrenamiento LSTM -> Evaluación/Reportes.

---
## Tecnologías y Herramientas

| Componente | Tecnología | Justificación |
|------------|------------|---------------|
| Procesamiento distribuido | Apache Spark (PySpark) | Manejo de decenas / cientos de millones de filas y transformaciones escalables. |
| Almacenamiento intermedio | CSV (actual) / Parquet (sugerido) | CSV por compatibilidad Windows; Parquet recomendado para compresión y pushdown. |
| Ciencia de Datos | Pandas / NumPy | Manipulación tabular en memoria para muestra de entrenamiento. |
| Modelado Secuencial | TensorFlow / Keras | Flexibilidad para modelo multi-salida y embeddings. |
| Ingeniería de Features | Spark SQL & funciones | Escalabilidad y expresividad para derivar columnas. |
| Evaluación ML | scikit-learn | Métricas estándar (MAE, accuracy, classification report). |
| Visualización | Matplotlib / Seaborn / Plotly | Gráficos estáticos y potencial interactivo. |
| Gestión de Dependencias | (Agregar `requirements.txt`) | Reproducibilidad del entorno. |
| Sistema Operativo Destino | Windows (ajustes Hadoop desactivados) | Alineado a entorno del desarrollador. |

### Dependencias Principales (Sugeridas)
```
pyspark>=3.3
tensorflow>=2.10,<2.13
pandas>=1.3
numpy>=1.21
scikit-learn>=1.0
matplotlib>=3.4
seaborn>=0.11
plotly>=5.0
```

> Ajustar versiones tras validación local (CPU/GPU disponible).

---
## Próximas Mejores Prácticas / Roadmap
1. Añadir `requirements.txt` o `environment.yml` con versiones fijadas.
2. Migrar persistencia a Parquet particionado por `city` y/o rango temporal.
3. Implementar generación de features lag (`flow_lag_1`, `speed_trend`) dentro del pipeline Spark para evitar fallos.
4. Evitar fuga de datos moviendo el escalado posterior a un split train/test.
5. Incorporar `tf.data` para streaming de batches sin convertir grandes volúmenes a Pandas.
6. Añadir semillas de reproducibilidad y logging estructurado.
7. Contener ejecución en script modular (`src/`) y notebook solo para exploración.
8. Implementar validación incremental (pruebas unitarias de funciones clave).

---
## Ejecución Rápida (Esquema)
1. Colocar archivos raw en `data/raw/`.
2. (Opcional) Crear entorno virtual.
3. Ejecutar notebook o script principal para generar datos procesados y entrenar modelo.
4. Revisar artefactos generados y métricas JSON.

### Ejemplo de creación de entorno (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install pyspark tensorflow pandas numpy scikit-learn matplotlib seaborn plotly
```


