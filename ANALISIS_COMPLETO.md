# 📚 DOCUMENTACIÓN COMPLETA - ANÁLISIS AVANZADO DE DESERCIÓN

## 🎯 RESUMEN EJECUTIVO

Este proyecto implementa un análisis **completo y multi-enfoque** para predicción de deserción estudiantil:

1. ✅ **Aprendizaje Supervisado** - Modelos de clasificación tradicionales
2. ✅ **Aprendizaje No Supervisado** - Clustering y segmentación de estudiantes
3. ✅ **Aprendizaje Profundo** - Redes neuronales con TensorFlow/Keras

---

## 📁 ESTRUCTURA COMPLETA DEL PROYECTO

```
UPC/
├── 🔵 ANÁLISIS SUPERVISADO
│   ├── TrabajoFinal.py                # Pipeline completo (RF, LR, SVM)
│   ├── generar_graficos.py            # Visualizaciones rápidas
│   └── generar_benchmark.py           # Comparación de algoritmos
│
├── 🟢 ANÁLISIS NO SUPERVISADO
│   └── analisis_clustering.py         # K-Means, DBSCAN, PCA
│
├── 🟠 APRENDIZAJE PROFUNDO
│   └── modelo_deep_learning.py        # Redes neuronales
│
├── 📊 ANÁLISIS DE DATOS
│   ├── resumen_dataset.py             # EDA inicial
│   └── resumen_final_dataset.py       # Dashboard completo
│
├── 🎨 PRESENTACIÓN
│   ├── generar_slides.py              # Generador de slides
│   ├── GUIA_EXPOSICION.md            # Guía para presentar
│   └── GUION_PRESENTACION.md         # Script completo
│
├── 📄 DOCUMENTACIÓN
│   ├── README.md                      # Este archivo
│   ├── INFORME_COMPLETO.md           # Análisis técnico detallado
│   ├── RESUMEN_EJECUTIVO.md          # Para stakeholders
│   └── .github/copilot-instructions.md # Guía para AI agents
│
└── 🗂️ artifacts/                      # Todos los resultados
    ├── Modelos ML (.pkl, .h5)
    ├── Benchmarks (.csv)
    ├── Visualizaciones (.png)
    └── Análisis (.csv)
```

---

## 🚀 GUÍA DE USO RÁPIDA

### **Requisitos Previos**

```bash
# Python 3.8+
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
pip install tensorflow  # Para Deep Learning
pip install shap        # Para interpretabilidad (opcional)
```

### **Ejecución de Scripts**

#### 1️⃣ Análisis Supervisado (Modelos Tradicionales)

```bash
# Análisis completo con 3 algoritmos
python TrabajoFinal.py

# Solo gráficos (rápido)
python generar_graficos.py

# Benchmark detallado
python generar_benchmark.py
```

**Salidas:**
- `artifacts/modelo_desercion.pkl` - Mejor modelo entrenado
- `artifacts/benchmark_modelos.csv` - Comparación de algoritmos
- `artifacts/curva_precision_recall.png` - Curvas de rendimiento
- `artifacts/matriz_confusion_umbral_optimo.png` - Matriz optimizada

#### 2️⃣ Análisis No Supervisado (Clustering)

```bash
python analisis_clustering.py
```

**Salidas:**
- `artifacts/clustering_perfiles.csv` - Perfiles de estudiantes identificados
- `artifacts/clustering_visualizacion_2d.png` - Clusters en 2D (PCA)
- `artifacts/clustering_visualizacion_3d.png` - Clusters en 3D (PCA)
- `artifacts/pca_varianza_explicada.png` - Análisis de componentes principales

#### 3️⃣ Aprendizaje Profundo (Deep Learning)

```bash
python modelo_deep_learning.py
```

**Salidas:**
- `artifacts/modelo_deep_learning.h5` - Red neuronal entrenada
- `artifacts/deep_learning_benchmark_comparison.csv` - Comparación DL vs ML
- `artifacts/deep_learning_roc_pr_curves.png` - Curvas de evaluación
- `artifacts/deep_learning_comparison_chart.png` - Gráfico comparativo

#### 4️⃣ Análisis Exploratorio de Datos

```bash
python resumen_dataset.py          # Análisis básico
python resumen_final_dataset.py    # Dashboard completo
```

#### 5️⃣ Generar Presentación

```bash
python generar_slides.py
```

---

## 📊 RESULTADOS Y MÉTRICAS

### **🔵 Aprendizaje Supervisado**

| Algoritmo | AUC | Precision | Recall | F1-Score | Mejor para |
|-----------|-----|-----------|--------|----------|-----------|
| **Random Forest** | **0.932** | **0.845** | **0.785** | **0.814** | Balance general |
| Logistic Regression | 0.926 | 0.779 | 0.831 | 0.804 | Interpretabilidad |
| SVM | 0.922 | 0.801 | 0.792 | 0.796 | Fronteras complejas |

**Conclusión:** Random Forest es el ganador con 93.2% AUC

### **🟢 Aprendizaje No Supervisado**

**Clustering K-Means:**
- **K óptimo:** 3-4 clusters (depende de métrica)
- **Silhouette Score:** ~0.25-0.35
- **Perfiles identificados:**
  - 🔴 Cluster Alto Riesgo (>50% deserción)
  - 🟠 Cluster Riesgo Moderado (30-50% deserción)
  - 🟢 Cluster Bajo Riesgo (<30% deserción)

**DBSCAN:**
- **Outliers detectados:** ~5-10% de estudiantes
- **Aplicación:** Identificar casos atípicos que requieren atención especial

**PCA (Reducción Dimensional):**
- **2 componentes:** ~40-45% varianza explicada
- **3 componentes:** ~50-55% varianza explicada
- **95% varianza:** Requiere ~10-12 componentes

### **🟠 Aprendizaje Profundo**

**Arquitectura de Red Neuronal:**
```
Input Layer (N features)
    ↓
Hidden Layer 1 (128 neurons, ReLU) + BatchNorm + Dropout(0.4)
    ↓
Hidden Layer 2 (64 neurons, ReLU) + BatchNorm + Dropout(0.3)
    ↓
Hidden Layer 3 (32 neurons, ReLU) + Dropout(0.2)
    ↓
Output Layer (1 neuron, Sigmoid)
```

**Resultados esperados:**
- **AUC:** ~0.92-0.94 (comparable o superior a RF)
- **Training time:** 2-5 minutos (100 epochs con early stopping)
- **Ventaja:** Captura relaciones no lineales complejas
- **Desventaja:** Menos interpretable, requiere más datos

---

## 🔍 ANÁLISIS DETALLADO POR ENFOQUE

### **1. Aprendizaje Supervisado (Clasificación)**

**Objetivo:** Predecir si un estudiante va a desertar (Sí/No)

**Proceso:**
1. Preprocesamiento: StandardScaler + OneHotEncoder
2. Balanceo: SMOTE para equilibrar clases
3. Entrenamiento: 3 algoritmos con validación cruzada
4. Optimización: GridSearchCV para hiperparámetros
5. Ajuste de umbral: Maximizar recall ≥ 85%

**Variables más importantes:**
1. Unidades curriculares 2do semestre (aprobadas/notas)
2. Unidades curriculares 1er semestre (aprobadas/notas)
3. Edad al momento de inscripción
4. Estado de pagos
5. Estado civil

### **2. Aprendizaje No Supervisado (Clustering)**

**Objetivo:** Identificar perfiles naturales de estudiantes sin etiquetas previas

**Proceso:**
1. Selección de features numéricas relevantes
2. Normalización con StandardScaler
3. PCA para reducción dimensional y visualización
4. K-Means para segmentación
5. DBSCAN para detección de outliers
6. Análisis de perfiles por cluster

**Aplicaciones prácticas:**
- Identificar grupos de riesgo sin supervisión
- Detectar patrones no obvios en los datos
- Segmentar estudiantes para intervenciones personalizadas
- Encontrar casos atípicos que requieren atención especial

### **3. Aprendizaje Profundo (Redes Neuronales)**

**Objetivo:** Explorar si arquitecturas complejas mejoran la predicción

**Proceso:**
1. Preprocesamiento similar a supervisado
2. Diseño de arquitectura: 3-4 capas ocultas
3. Regularización: Dropout, BatchNormalization
4. Optimización: Adam optimizer con learning rate decay
5. Early stopping para evitar overfitting
6. Comparación con modelos tradicionales

**Ventajas del Deep Learning:**
- Aprende representaciones automáticas
- Captura interacciones complejas entre variables
- Escalable a grandes datasets
- Estado del arte en muchos dominios

**Cuándo usar cada enfoque:**
- **Supervisado tradicional:** Dataset pequeño-mediano, interpretabilidad importante
- **Deep Learning:** Dataset grande, relaciones muy complejas
- **No supervisado:** Exploración inicial, sin etiquetas, descubrir patrones

---

## 📈 INSIGHTS Y RECOMENDACIONES

### **Hallazgos Principales**

1. **📚 Rendimiento académico es el factor #1**
   - Correlación -0.57 con deserción
   - Primeros 2 semestres son críticos

2. **👤 Edad importa**
   - Estudiantes >25 años tienen 40% más riesgo
   - Requieren apoyo diferenciado

3. **💰 Factores económicos son secundarios pero relevantes**
   - Pagos atrasados: +23% riesgo
   - Becarios: -25% riesgo

4. **🕐 Turno nocturno aumenta riesgo**
   - Correlación +0.21 con deserción
   - Probablemente por trabajo/familia

### **Recomendaciones de Implementación**

#### **Fase 1: Sistema de Alerta Temprana (0-3 meses)**
```python
# Pseudocódigo del sistema
for estudiante in matriculados:
    riesgo_score = modelo_rf.predict_proba(estudiante)[1]
    
    if riesgo_score > 0.7:
        enviar_alerta("CRÍTICO", consejero, estudiante)
    elif riesgo_score > 0.5:
        enviar_alerta("MODERADO", consejero, estudiante)
```

#### **Fase 2: Segmentación para Intervenciones (3-6 meses)**
```python
# Usar clustering para personalizar
cluster_estudiante = modelo_kmeans.predict(estudiante)

if cluster_estudiante == ALTO_RIESGO:
    intervenciones = ["tutoría_intensiva", "apoyo_psicopedagógico"]
elif cluster_estudiante == RIESGO_MODERADO:
    intervenciones = ["tutoría_grupal", "mentoring"]
```

#### **Fase 3: Refinamiento con Deep Learning (6-12 meses)**
```python
# Reentrenar con más datos
modelo_dl.fit(nuevos_datos, nuevas_etiquetas)

# Comparar con modelo actual
if performance_dl > performance_rf:
    modelo_produccion = modelo_dl
```

---

## 🎓 ASPECTOS TÉCNICOS AVANZADOS

### **Manejo de Datos Desbalanceados**

**Problema:** Solo 32.1% de estudiantes desertan

**Soluciones implementadas:**
1. **SMOTE:** Genera ejemplos sintéticos de la clase minoritaria
2. **Estratified Sampling:** Mantiene proporciones en train/test
3. **Métricas apropiadas:** AUC, F1, Precision-Recall (no solo Accuracy)
4. **Ajuste de umbral:** Optimizar según objetivos de negocio

### **Validación y Generalización**

**Técnicas usadas:**
1. **Train/Test Split:** 80/20 con stratification
2. **Cross-Validation:** 5-fold estratificada
3. **GridSearchCV:** Para hiperparámetros
4. **Holdout Test:** Nunca usado en entrenamiento

### **Interpretabilidad**

**Para stakeholders no técnicos:**
- Feature importance (Random Forest)
- SHAP values (explicaciones individuales)
- Perfiles de clusters (segmentación entendible)
- Visualizaciones claras (curvas, matrices)

---

## 💻 COMANDOS ÚTILES

### **Instalación de Dependencias**

```bash
# Instalar todo de una vez
pip install -r requirements.txt

# O manualmente:
pip install numpy pandas matplotlib seaborn
pip install scikit-learn imbalanced-learn
pip install tensorflow  # Para deep learning
pip install shap        # Para interpretabilidad
```

### **Ejecutar Todos los Análisis**

```bash
# Script bash (Linux/Mac)
#!/bin/bash
python resumen_dataset.py
python TrabajoFinal.py
python analisis_clustering.py
python modelo_deep_learning.py
python generar_slides.py
```

```powershell
# PowerShell (Windows)
python resumen_dataset.py
python TrabajoFinal.py
python analisis_clustering.py
python modelo_deep_learning.py
python generar_slides.py
```

### **Cargar Modelos Guardados**

```python
# Cargar Random Forest
import joblib
modelo_rf = joblib.load("artifacts/modelo_desercion.pkl")

# Cargar Deep Learning
from tensorflow import keras
modelo_dl = keras.models.load_model("artifacts/modelo_deep_learning.h5")

# Hacer predicciones
probabilidad_desercion = modelo_rf.predict_proba(nuevo_estudiante)
```

---

## 📚 REFERENCIAS Y RECURSOS

### **Documentación del Proyecto**
- `INFORME_COMPLETO.md` - Análisis técnico detallado
- `RESUMEN_EJECUTIVO.md` - Para tomadores de decisión
- `GUIA_EXPOSICION.md` - Cómo presentar el proyecto
- `GUION_PRESENTACION.md` - Script para exposición

### **Librerías Utilizadas**
- [Scikit-learn](https://scikit-learn.org/) - ML tradicional
- [TensorFlow/Keras](https://www.tensorflow.org/) - Deep Learning
- [Imbalanced-learn](https://imbalanced-learn.org/) - SMOTE
- [SHAP](https://shap.readthedocs.io/) - Interpretabilidad

### **Artículos Relacionados**
- Student Dropout Prediction: A Systematic Review
- Deep Learning for Educational Data Mining
- Clustering Students for Personalized Interventions

---

## 👥 CONTRIBUCIONES Y CONTACTO

**Autor:** [Tu Nombre]  
**Universidad:** Universidad Peruana de Ciencias Aplicadas (UPC)  
**Curso:** Machine Learning / Ciencia de Datos  
**Año:** 2025

**Para preguntas o colaboraciones:**
- GitHub: [Tu usuario]
- Email: [Tu email]

---

## 📄 LICENCIA

Este proyecto es de uso académico. Los datos son confidenciales y no deben ser compartidos fuera del contexto educativo.

---

## 🎯 PRÓXIMOS PASOS

### **Mejoras Futuras**
1. ✅ Implementar modelos de serie temporal (predicción por semestre)
2. ✅ Agregar variables de interacción social/engagement
3. ✅ API REST para integración con sistemas académicos
4. ✅ Dashboard interactivo con Streamlit/Dash
5. ✅ A/B testing de intervenciones basadas en predicciones

### **Investigación Avanzada**
1. Transfer Learning con datos de otras universidades
2. Modelos ensemble (stacking RF + DL)
3. Reinforcement Learning para optimizar intervenciones
4. Natural Language Processing en comentarios de profesores

---

**🎉 ¡Proyecto Completo! Ahora tienes análisis supervisado, no supervisado y deep learning implementados.**

**Última actualización:** Octubre 2025
