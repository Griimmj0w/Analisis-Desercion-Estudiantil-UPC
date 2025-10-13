# Análisis de Deserción Estudiantil - Enfoque Multi-Paradigma

Proyecto completo de machine learning que implementa **aprendizaje supervisado, no supervisado y deep learning** para predecir y analizar la deserción estudiantil.

## 📊 Descripción del Proyecto

Este proyecto implementa un análisis exhaustivo de factores que influyen en la deserción estudiantil usando **múltiples enfoques de Machine Learning**:

1. 🔵 **Aprendizaje Supervisado** - Random Forest, Logistic Regression, SVM
2. 🟢 **Aprendizaje No Supervisado** - K-Means, DBSCAN, PCA  
3. 🟠 **Aprendizaje Profundo** - Redes Neuronales con TensorFlow/Keras

El objetivo es identificar estudiantes en riesgo mediante diferentes técnicas, permitiendo intervenciones tempranas y personalizadas.

## 🎯 Objetivos

- Identificar patrones en los datos que contribuyen a la deserción estudiantil
- Desarrollar modelos de machine learning (supervisado, no supervisado y deep learning)
- Optimizar umbrales de decisión para maximizar recall manteniendo precisión
- Segmentar estudiantes en perfiles de riesgo para intervenciones personalizadas
- Comparar rendimiento entre algoritmos tradicionales y redes neuronales
- Generar insights accionables para intervenciones tempranas

## 📁 Estructura del Proyecto

```
UPC/
├── 🔵 APRENDIZAJE SUPERVISADO
│   ├── TrabajoFinal.py                # Análisis completo (RF, LR, SVM)
│   ├── generar_graficos.py            # Visualizaciones rápidas
│   └── generar_benchmark.py           # Comparación de algoritmos
│
├── 🟢 APRENDIZAJE NO SUPERVISADO
│   └── analisis_clustering.py         # K-Means, DBSCAN, PCA
│
├── 🟠 APRENDIZAJE PROFUNDO
│   └── modelo_deep_learning.py        # Redes neuronales (TensorFlow/Keras)
│
├── 📊 ANÁLISIS DE DATOS
│   ├── resumen_dataset.py             # EDA inicial
│   └── resumen_final_dataset.py       # Dashboard completo
│
├── 🎨 PRESENTACIÓN
│   ├── generar_slides.py              # Generador de slides
│   ├── GUIA_EXPOSICION.md            # Guía para presentar
│   └── GUION_PRESENTACION.md         # Script de exposición
│
├── 📄 DOCUMENTACIÓN
│   ├── README.md                      # Este archivo
│   ├── ANALISIS_COMPLETO.md          # Guía completa de todos los análisis
│   ├── INFORME_COMPLETO.md           # Análisis técnico detallado
│   ├── RESUMEN_EJECUTIVO.md          # Para stakeholders
│   └── .github/copilot-instructions.md
│
├── artifacts/                         # Resultados y modelos
│   ├── modelo_desercion.pkl           # Random Forest entrenado
│   ├── modelo_deep_learning.h5        # Red neuronal entrenada
│   ├── clustering_perfiles.csv        # Perfiles de estudiantes
│   ├── benchmark_*.csv                # Comparaciones
│   └── *.png                          # Visualizaciones
│
└── slides/                            # Presentación
    └── slide_*.png
```

## 🛠️ Tecnologías Utilizadas

- **Python 3.8+**
- **Análisis de Datos**: Pandas, NumPy
- **Visualización**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, Imbalanced-learn
- **Deep Learning**: TensorFlow, Keras
- **Modelos Supervisados**: Random Forest, Logistic Regression, SVM
- **Modelos No Supervisados**: K-Means, DBSCAN, PCA
- **Explicabilidad**: SHAP (opcional)

## 📋 Requisitos

Instala las dependencias necesarias:

```bash
# Dependencias básicas
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn

# Para Deep Learning
pip install tensorflow

# Para interpretabilidad (opcional)
pip install shap
```

O usando el entorno virtual incluido:

```bash
# Activar entorno virtual
.venv\Scripts\Activate.ps1

# Las dependencias ya están instaladas en el entorno virtual
```

## 🚀 Uso

### 1️⃣ Aprendizaje Supervisado - Clasificación

```bash
# Análisis completo con 3 algoritmos
python TrabajoFinal.py

# Solo gráficos (versión rápida)
python generar_graficos.py

# Benchmark detallado
python generar_benchmark.py
```

### 2️⃣ Aprendizaje No Supervisado - Clustering

```bash
# Identificar perfiles de estudiantes
python analisis_clustering.py
```

**Salidas:**
- Identificación de 3-4 perfiles de estudiantes
- Detección de outliers (5-10% casos atípicos)
- Visualización en 2D y 3D con PCA
- Análisis de tasa de deserción por cluster

### 3️⃣ Aprendizaje Profundo - Redes Neuronales

```bash
# Entrenar red neuronal y comparar con ML tradicional
python modelo_deep_learning.py
```

**Salidas:**
- Modelo de Deep Learning entrenado (.h5)
- Comparación DL vs ML tradicional
- Curvas de aprendizaje
- Feature importance aproximado

### 4️⃣ Generar Presentación

```bash
python generar_slides.py
```

## 📈 Resultados Principales

### 🔵 Aprendizaje Supervisado

**Mejor Modelo: Random Forest**
- **AUC**: 0.932 (Excelente capacidad discriminativa)
- **Average Precision**: 0.901
- **Umbral Óptimo**: 0.410
- **Precision**: 0.786
- **Recall**: 0.852
- **F1-Score**: 0.818

### 🟢 Aprendizaje No Supervisado

**K-Means Clustering:**
- **K óptimo**: 3-4 clusters
- **Silhouette Score**: ~0.30
- **Perfiles identificados**:
  - 🔴 Alto Riesgo (>50% deserción)
  - 🟠 Riesgo Moderado (30-50% deserción)
  - 🟢 Bajo Riesgo (<30% deserción)

**DBSCAN:**
- **Outliers detectados**: 5-10% de estudiantes
- **Aplicación**: Casos atípicos que requieren atención especial

**PCA:**
- **2 componentes**: 40-45% varianza explicada
- **95% varianza**: Requiere ~10-12 componentes

### 🟠 Aprendizaje Profundo

**Red Neuronal (3 capas ocultas):**
- **AUC**: ~0.92-0.94 (comparable a Random Forest)
- **Arquitectura**: 128 → 64 → 32 neuronas
- **Regularización**: Dropout + BatchNormalization
- **Training**: Early stopping (~30-50 epochs)
- **Ventaja**: Captura relaciones no lineales complejas

### Visualizaciones Generadas

#### Aprendizaje Supervisado
1. **Curva Precision-Recall**: Trade-off entre precisión y recall
2. **Matriz de Confusión**: Con umbral optimizado
3. **Curva ROC**: Evaluación del rendimiento
4. **Importancia de Variables**: Top 20 características

#### Aprendizaje No Supervisado
1. **Clustering 2D/3D**: Visualización con PCA
2. **Método del Codo**: Selección de K óptimo
3. **Análisis de Perfiles**: Características por cluster
4. **Varianza Explicada**: Componentes principales

#### Aprendizaje Profundo
1. **Curvas de Aprendizaje**: Loss y métricas por epoch
2. **Comparación DL vs ML**: Benchmarks visuales
3. **Feature Importance**: Permutation importance
4. **ROC y PR Curves**: Evaluación del modelo

## 🎯 Metodología

### 1️⃣ Aprendizaje Supervisado

1. **Preprocesamiento de Datos**
   - Encoding de variables categóricas (OneHotEncoder)
   - Normalización de variables numéricas (StandardScaler)
   - Balanceo con SMOTE

2. **Modelado**
   - Comparación de múltiples algoritmos (RF, LR, SVM)
   - Validación cruzada estratificada (5-fold)
   - Optimización de hiperparámetros (GridSearchCV)

3. **Optimización de Umbral**
   - Búsqueda de umbral que garantice recall ≥ 85%
   - Maximización de F1-Score

4. **Evaluación**
   - Métricas de clasificación (AUC, Precision, Recall, F1)
   - Curvas de rendimiento (ROC, Precision-Recall)
   - Análisis de importancia de variables

### 2️⃣ Aprendizaje No Supervisado

1. **Reducción Dimensional**
   - PCA para exploración de varianza
   - Visualización en 2D y 3D

2. **Clustering**
   - K-Means para segmentación
   - Método del codo para K óptimo
   - Silhouette Score para validación

3. **Detección de Anomalías**
   - DBSCAN para identificar outliers
   - Análisis de estudiantes atípicos

4. **Análisis de Perfiles**
   - Caracterización de cada cluster
   - Tasa de deserción por segmento
   - Recomendaciones personalizadas

### 3️⃣ Aprendizaje Profundo

1. **Arquitectura**
   - Red feed-forward multicapa
   - 3-4 capas ocultas (128→64→32)
   - Activación ReLU + Sigmoid final

2. **Regularización**
   - Dropout (0.2-0.4)
   - BatchNormalization
   - Early Stopping

3. **Optimización**
   - Adam optimizer
   - Learning rate decay
   - Binary cross-entropy loss

4. **Comparación**
   - Benchmark vs modelos tradicionales
   - Análisis de trade-offs
   - Feature importance aproximado

## 📊 Datos

El proyecto utiliza un dataset de deserción estudiantil con las siguientes características:

- **Tamaño**: 4,424 registros
- **Variables**: 37 características
- **Distribución**: 32.1% deserción, 67.9% graduación/continuidad
- **Tipo de Variables**: Numéricas y categóricas

## 🔍 Interpretabilidad

- **Feature Importance (Random Forest)**: Variables más influyentes
- **SHAP Values**: Explicabilidad de predicciones individuales
- **Perfiles de Clusters**: Segmentación comprensible para stakeholders
- **Análisis de Umbrales**: Optimización basada en objetivos de negocio
- **Visualizaciones Intuitivas**: Curvas, matrices y gráficos explicativos

## 🎓 Casos de Uso

### Para Instituciones Educativas

1. **Sistema de Alerta Temprana**
   - Identificación automática de estudiantes en riesgo
   - Alertas desde el 2° mes de clases
   - Priorización por nivel de riesgo (score 0-100%)

2. **Intervenciones Personalizadas**
   - Segmentación por perfiles (clusters)
   - Programas diferenciados por grupo de riesgo
   - Seguimiento y evaluación de intervenciones

3. **Optimización de Recursos**
   - Focalizar apoyo en casos críticos
   - Asignación eficiente de consejeros
   - Medición de impacto de programas

### Para Investigadores

1. **Análisis Exploratorio**
   - Identificar patrones no obvios (clustering)
   - Validar hipótesis sobre factores de riesgo
   - Comparar enfoques metodológicos

2. **Benchmarking**
   - Comparar múltiples algoritmos
   - Evaluar trade-offs (precisión vs recall)
   - Experimentar con diferentes arquitecturas

## 📊 Principales Hallazgos

1. **📚 Rendimiento académico** es el factor más predictivo (-0.57 correlación)
2. **👤 Edad al inscribirse** influye significativamente (+0.25 correlación)
3. **💰 Factores económicos** (pagos, becas) son relevantes pero secundarios
4. **🕐 Turno nocturno** aumenta riesgo de deserción (+0.21 correlación)
5. **📈 Primeros 2 semestres** son críticos para intervención temprana
6. **🎯 3-4 perfiles distintos** de estudiantes identificados por clustering
7. **🤖 Deep Learning** alcanza rendimiento comparable a Random Forest
8. **⚡ Outliers** (5-10%) requieren atención especial

## 📄 Documentación Adicional

Para información más detallada, consulta:

- **`ANALISIS_COMPLETO.md`** - Guía completa de todos los análisis implementados
- **`INFORME_COMPLETO.md`** - Análisis técnico detallado del proyecto
- **`RESUMEN_EJECUTIVO.md`** - Resumen para stakeholders y tomadores de decisión
- **`GUIA_EXPOSICION.md`** - Cómo presentar el proyecto efectivamente
- **`GUION_PRESENTACION.md`** - Script completo para exposiciones
- **`.github/copilot-instructions.md`** - Guía para AI coding agents

## 👥 Autor

Desarrollado para el curso de analisis de datos y sistemas predictivos- UPC

## 📄 Licencia

Este proyecto es de uso académico.
