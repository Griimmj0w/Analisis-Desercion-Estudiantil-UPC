# AI Coding Instructions - Student Dropout Prediction ML Project

## Project Overview
This is a **comprehensive machine learning project** analyzing student dropout patterns using 4,424 student records with 37 variables. The project implements **three major ML paradigms**:

1. **Supervised Learning** - Classification models (Random Forest, Logistic Regression, SVM)
2. **Unsupervised Learning** - Clustering and dimensionality reduction (K-Means, DBSCAN, PCA)
3. **Deep Learning** - Neural networks with TensorFlow/Keras

The main goal is to build predictive models that identify at-risk students for early intervention using multiple analytical approaches.

## Core Architecture & Data Flow

### Data Pipeline Pattern
```
Raw CSV → df["target_bin"] = (df["Target"] == "Dropout").astype(int) → Train/Test Split → Preprocessing → SMOTE → ML Models
```

**Key files follow this pattern:**
- `TrabajoFinal.py` - Complete supervised learning pipeline with 19 numbered sections
- `generar_graficos.py` - Simplified version focusing on visualization
- `resumen_final_dataset.py` - Dataset analysis and summary generation
- `analisis_clustering.py` - Unsupervised learning (K-Means, DBSCAN, PCA)
- `modelo_deep_learning.py` - Deep learning with neural networks

### Standard Preprocessing Pipeline
Always use this `ColumnTransformer` pattern for consistency:
```python
num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

preproc = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])
```

### Imbalanced Pipeline Convention
Use `imblearn.pipeline.Pipeline as ImbPipeline` with SMOTE:
```python
def make_pipe(estimator):
    return ImbPipeline([
        ("prep", preproc),
        ("smote", SMOTE(random_state=42)),
        ("clf", estimator),
    ])
```

## File Organization Patterns

### Script Categories
- **Supervised learning**: `TrabajoFinal.py`, `generar_benchmark.py` - Classification workflows
- **Unsupervised learning**: `analisis_clustering.py` - Clustering and PCA
- **Deep learning**: `modelo_deep_learning.py` - Neural networks with TensorFlow/Keras
- **Visualization scripts**: `generar_graficos.py`, `generar_slides.py` - Chart generation
- **Data analysis**: `resumen_dataset.py`, `resumen_final_dataset.py` - EDA
- **Documentation**: Comprehensive `.md` files for reports and presentation guides

### Artifacts Directory Structure
All outputs go to `artifacts/` with predictable naming:
- **Supervised models**: `modelo_desercion.pkl`
- **Deep learning models**: `modelo_deep_learning.h5`, `modelo_deep_learning_savedmodel/`
- **Benchmarks**: `benchmark_modelos.csv`, `benchmark_completo_algoritmos.csv`, `deep_learning_benchmark_comparison.csv`
- **Visualizations**: `curva_precision_recall.png`, `matriz_confusion_umbral_optimo.png`, `deep_learning_*.png`, `clustering_*.png`
- **Data summaries**: `resumen_variables.csv`, `correlaciones_desercion.csv`, `clustering_perfiles.csv`

### Critical Data Path
**Hardcoded CSV path**: `r'C:\\Users\\SISTEMAS\\Documents\\PYTHON\\DATA\\data.csv'`
- When modifying scripts, preserve this exact path format
- Data loading always uses `sep=';'` parameter

## ML Model Standards

### Algorithm Benchmark Pattern
Always compare these three models in this order:
1. **RandomForestClassifier** (winner: n_estimators=300, n_jobs=1)
2. **LogisticRegression** (max_iter=1000)  
3. **SVC** (kernel="rbf", probability=True)

### Evaluation Metrics Priority
1. **AUC-ROC** - Primary metric (current best: 0.932)
2. **Precision-Recall optimization** - Target recall ≥ 85%
3. **Threshold optimization** - Business logic over 0.5 default

### Cross-Validation Standard
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Use n_splits=3 for faster iterations in benchmark scripts
```

## Domain-Specific Conventions

### Target Variable Pattern
Always create binary target: `df["target_bin"] = (df["Target"] == "Dropout").astype(int)`
- Never work directly with original "Target" column (Graduate/Dropout/Enrolled)

### Feature Importance Analysis
Key variables to highlight in analysis:
- `Curricular units 2nd sem (grade)` - Top predictor (-0.572 correlation)
- `Curricular units 1st sem (approved)` - Early warning signal
- `Age at enrollment` - Demographic risk factor
- `Tuition fees up to date` - Economic indicator

### Visualization Standards
- Use `plt.savefig()` with `dpi=300, bbox_inches="tight"` for publication quality
- Color palette: Blues for confusion matrices, standard seaborn for others
- Always include grid: `plt.grid(True)` or `plt.grid(alpha=0.3)`

## Development Workflow

### Script Execution Order
1. Run `resumen_dataset.py` first for data understanding
2. Use `generar_graficos.py` for quick iterations  
3. Execute `TrabajoFinal.py` for complete analysis
4. Generate presentations with `generar_slides.py`

### Common Debugging Points
- **SMOTE memory issues**: Reduce n_splits to 3 in CV
- **OneHotEncoder errors**: Always use `handle_unknown="ignore"`
- **Joblib warnings**: Filter with `warnings.filterwarnings("ignore")`
- **Parallel processing**: Keep `n_jobs=1` to avoid Windows multiprocessing issues

### Output Validation Checklist
- [ ] All artifacts saved to `artifacts/` directory
- [ ] Model achieves AUC > 0.90  
- [ ] Recall optimization shows improvement over 0.5 threshold
- [ ] Feature importance plot shows academic variables as top predictors

## Business Context Integration

### Educational Domain Knowledge
- **First two semesters are critical** - Focus predictive features here
- **Age is a major risk factor** - Students >25 have higher dropout rates  
- **Economic factors matter** - Payment status, scholarship status are predictive
- **Academic performance dominates** - Grades and approved courses are strongest signals

### Intervention-Focused Modeling
- **Optimize for recall** - Missing at-risk students is costlier than false positives
- **Early detection priority** - Model should work with partial semester data
- **Actionable insights** - Feature importance must translate to intervention strategies

When extending this codebase, maintain the established patterns for data processing, model comparison, and artifact generation. The goal is always practical application for student retention programs.