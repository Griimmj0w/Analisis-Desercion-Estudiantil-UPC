# Análisis de Deserción Estudiantil

Proyecto de machine learning para predecir la deserción estudiantil utilizando técnicas de clasificación y análisis de datos.

## 📊 Descripción del Proyecto

Este proyecto analiza factores que influyen en la deserción estudiantil y construye modelos predictivos para identificar estudiantes en riesgo. Utiliza técnicas de machine learning para clasificar estudiantes según su probabilidad de abandono académico.

## 🎯 Objetivos

- Identificar patrones en los datos que contribuyen a la deserción estudiantil
- Desarrollar modelos de machine learning para predecir deserción
- Optimizar umbrales de decisión para maximizar recall manteniendo precisión
- Generar insights accionables para intervenciones tempranas

## 📁 Estructura del Proyecto

```
UPC/
├── TrabajoParcial.py          # Script principal de análisis
├── generar_graficos.py        # Script simplificado para generar gráficos
├── artifacts/                 # Resultados y visualizaciones
│   ├── curva_precision_recall.png
│   ├── matriz_confusion_umbral_optimo.png
│   ├── curva_roc.png
│   ├── matriz_confusion.png
│   ├── top20_importancias.png
│   ├── modelo_desercion.pkl
│   ├── benchmark_modelos.csv
│   └── resumen_variables.csv
└── README.md
```

## 🛠️ Tecnologías Utilizadas

- **Python 3.13**
- **Análisis de Datos**: Pandas, NumPy
- **Visualización**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, Imbalanced-learn
- **Modelos**: Random Forest, Logistic Regression, SVM
- **Explicabilidad**: SHAP

## 📋 Requisitos

Instala las dependencias necesarias:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn shap
```

O usando el entorno virtual incluido:

```bash
# Activar entorno virtual
.venv\Scripts\Activate.ps1

# Las dependencias ya están instaladas en el entorno virtual
```

## 🚀 Uso

### Ejecutar Análisis Completo

```bash
python TrabajoParcial.py
```

### Generar Solo Gráficos (Versión Rápida)

```bash
python generar_graficos.py
```

## 📈 Resultados Principales

### Métricas del Modelo Óptimo

- **Average Precision**: 0.901
- **Umbral Óptimo**: 0.410
- **Precision**: 0.786
- **Recall**: 0.852
- **F1-Score**: 0.818

### Visualizaciones Generadas

1. **Curva Precision-Recall**: Muestra el trade-off entre precisión y recall
2. **Matriz de Confusión**: Con umbral optimizado para maximizar recall
3. **Curva ROC**: Evaluación del rendimiento del clasificador
4. **Importancia de Variables**: Top 20 características más relevantes

## 🎯 Metodología

1. **Preprocesamiento de Datos**
   - Encoding de variables categóricas
   - Normalización de variables numéricas
   - Balanceo con SMOTE

2. **Modelado**
   - Comparación de múltiples algoritmos
   - Validación cruzada estratificada
   - Optimización de hiperparámetros

3. **Optimización de Umbral**
   - Búsqueda de umbral que garantice recall ≥ 85%
   - Maximización de F1-Score

4. **Evaluación**
   - Métricas de clasificación
   - Curvas de rendimiento
   - Análisis de importancia de variables

## 📊 Datos

El proyecto utiliza un dataset de deserción estudiantil con las siguientes características:

- **Tamaño**: 4,424 registros
- **Variables**: 37 características
- **Distribución**: 32.1% deserción, 67.9% graduación/continuidad
- **Tipo de Variables**: Numéricas y categóricas

## 🔍 Interpretabilidad

- **SHAP Values**: Explicabilidad de predicciones individuales
- **Feature Importance**: Ranking de variables más influyentes
- **Análisis de Umbrales**: Optimización basada en objetivos de negocio

## 👥 Autor

Desarrollado para el curso de analisis de datos y sistemas predictivos- UPC

## 📄 Licencia

Este proyecto es de uso académico.
