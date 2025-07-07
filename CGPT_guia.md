# Prompt
Quiero comenzar un proyecto que engloba todo lo visto en un curso universitario de Data Science. Este proyecto va a seguir estrictamente el siguiente flujo:
- Exploración de Datos (EDA) 
- Limpieza, en caso de ser necesario, y preprocesamiento 
- Modelado y evaluación 
- Conclusiones claras y justificadas 

Este proyecto buscar resolver un problema de la siguiente variedad 
- Clasificación (binaria o multiclase)
- Regresión
- Clustering
- NLP / Computer Vision 


sobre el siguiente conjunto de datos  platadisponible en la plataforma Kaggle: https://www.kaggle.com/datasets/akrashnoor/brain-tumor. 
Para el cual esperamos implementar 2 modelos de aprendizaje, ya sea supervizado o no, donde se pueden incluir LogisticRegression, SVM, kNN, Arboles, Random Forest, MLP, Autoencoder, DBSCAN, etc. Incluso otros como EBM, Transformers, CNN, LSTM.
Como equipo necesitamos que nos proporciones una guia detallada en donde

1. Nos proporciones una recomendacion sobre qué problema resolver para este dataset. Basado en esa respuesta, quiero que proporciones el tipo de modelos que se deberan implementar y cuales son los mas adecuados para implementar para el tipo de problema a resolver. Como comentario adicional, nos interesa mucho la implementacion de redes neuronales.
2. Proporciona pasos a seguir detallados y puntos a considerar en el momento que llevemos a cabo la exploracion de datos.
3. Responde lo siguiente ¿es necesaria la limpieza de datos, considerando que el tipo de datos de este conjunto de datos son imagenes? Luego, segun los modelos que se recomendaron en el paso 1, quiero que nos des pasos detallados para llevar a cabo un preprocesamiento adecuado para cada modelo
4. Proporciona una guia detallada para la implementacion de cada modelo recomendado 
5. Considerando las instrucciones del paso 3 y 4, quiero que proporciones todas las metricas de evaluacion que sean necesarias con el fin de hacer lo mas critico posible nuestro analisis. 
6. Considera las metricas del paso 5 y proporciona una serie de herramientas de visualizacion para contrastar nuestros analisis, tales como matriz de confusion, clusterin 2D, curvas, etc.

Esta guia debe ser lo mas detallada posible, puesto que, se busca utilizarla para la confeccion de un proyecto final de un curso de data science universitario, por lo tanto, los criterios criticos y formales son clave para esta tarea

---

## 1. Selección del problema

**Recomendación:**
Dado que el conjunto de datos contiene imágenes de resonancia magnética (MRI) categorizadas en varios tipos de tumores (por ejemplo: glioma, meningioma, pituitario, y “no tumor”), el problema más natural y valioso es **clasificación multiclase** de tumores cerebrales.

* **Modelos recomendados:**

  * **Redes neuronales convolucionales (CNNs)** (p. ej. arquitectura propia simple o **transfer learning** con VGG16, ResNet50 o EfficientNet).
  * **SVM** con kernel RBF (sobre features extraídas).
  * **Random Forest** (sobre features extraídas o embeddings de red).

> *Comentario:* Si quieres profundizar en redes profundas, combina un **CNN from scratch** con **transfer learning** para comparar resultados.

---

## 2. Exploración de Datos (EDA)

1. **Carga y conteo de clases**

   * Número de imágenes por categoría → identificar desbalance.
   * Visualizar carpetas y renombrar si es necesario.

2. **Visualización de ejemplos**

   * Muestra 5–10 imágenes representativas de cada clase (tamaño, contraste, artefactos).

3. **Análisis estadístico**

   * Distribución de tamaños (ancho × alto).
   * Histograma de intensidades de píxeles (por clase).

4. **Chequeo de calidad**

   * Detección de imágenes corruptas o duplicadas.
   * Verificar canales (gris vs RGB).

5. **Resumen de hallazgos**

   * Anotar si hay clases muy minoritarias.
   * Documentar rango de valores de píxel.

---

## 3. Limpieza y Preprocesamiento

### 3.1 ¿Es necesaria limpieza?

* **Básica:** asegurar que todas las imágenes cargan correctamente y que no hay duplicados.
* **Avanzada:** opcionalmente recortar fondo negro excesivo o normalizar tamaños de ROI (región de interés) si hay mucho espacio vacío.

### 3.2 Preprocesamiento por tipo de modelo

| Etapa                      | CNN / Transfer Learning                     | SVM / Random Forest                           |
| -------------------------- | ------------------------------------------- | --------------------------------------------- |
| **Redimensionar**          | 224×224 (o el tamaño de la arquitectura TL) | 64×64 o 128×128 (menor, para acelerar)        |
| **Normalización**          | Escalar \[0,1] o z‑score (media/desv)       | Escalar \[0,1] + opcionalmente PCA (n=50–100) |
| **Aumento de datos**       | Rotaciones ±15°, flips horizontales, zoom   | No suele aplicarse directamente               |
| **Extracción de features** | —                                           | HOG, LBP o embeddings intermedios de CNN      |
| **Conversión a array**     | Tensor 4D `(batch, h, w, canales)`          | Vector 1D por imagen `(n_features,)`          |

---

## 4. Implementación de Modelos

### 4.1 CNN from Scratch (Keras/TensorFlow)

1. Definir `Sequential` con bloques conv → ReLU → MaxPool.
2. Finalizar con Flatten → Dense (128) → Dropout(0.5) → Dense(n\_clases, softmax).
3. Compilar con `Adam(lr=1e‑4)`, `loss='categorical_crossentropy'`.
4. Entrenar con `fit`, validación 20%, `EarlyStopping(patience=5)`.

### 4.2 Transfer Learning (p. ej. ResNet50)

1. Cargar base `ResNet50(weights='imagenet', include_top=False)`.
2. Congelar capas base y añadir top como en 4.1.
3. Fine‑tuning: descongelar últimas 10 capas tras entrenamiento inicial.

### 4.3 SVM / Random Forest (scikit‑learn)

1. **Pipeline:**

   ```python
   Pipeline([
     ('resize', FunctionTransformer(resize_to_fixed)),
     ('hog', HOGTransformer()),
     ('scale', StandardScaler()),
     ('pca', PCA(n_components=50)),
     ('clf', SVC(kernel='rbf', probability=True))  # o RandomForestClassifier(n_estimators=100)
   ])
   ```
2. **GridSearchCV** para hiperparámetros (`C`, `gamma` para SVM; `max_depth`, `n_estimators` para RF).

---

## 5. Métricas de Evaluación

* **Clasificación multiclase:**

  * **Accuracy** global
  * **Precision, Recall, F1‑score** (macro y weighted)
  * **ROC‑AUC** (one‑vs‑rest)
  * **Matriz de confusión**

* **Entrenamiento (deep learning):**

  * Curvas de `loss` y `accuracy` por época en entrenamiento/validación.

* **Comparación de modelos:**

  * Tabla resumen de métricas
  * Prueba estadística (p.ej. McNemar) para comparar dos clasificadores.

---

## 6. Visualización de Resultados

* **Matriz de confusión** (heatmap con seaborn).
* **Curvas ROC** y **Precision‑Recall** para cada clase.
* **Curvas de entrenamiento** (loss & accuracy vs epochs).
* **Embedding 2D** (PCA o t-SNE) de features extraídos, coloreado por etiqueta y por predicción.
* **Ejemplos de aciertos/errores**: mostrar imágenes mal clasificadas con etiquetas reales vs predichas.

---

Con esta guía podrás estructurar tu trabajo desde la EDA inicial hasta la presentación de resultados, con un enfoque crítico y formal acorde a un proyecto final universitario. ¡Éxito en la implementación y análisis!
