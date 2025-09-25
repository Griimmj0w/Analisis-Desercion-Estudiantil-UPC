import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import os

# Crear directorio si no existe
os.makedirs('artifacts', exist_ok=True)

print("="*80)
print("                    BENCHMARK DE ALGORITMOS - ML")
print("="*80)

# Cargar y preparar datos
df = pd.read_csv(r'C:\\Users\\SISTEMAS\\Documents\\PYTHON\\DATA\\data.csv', sep=';')
df["target_bin"] = (df["Target"] == "Dropout").astype(int)

# Separar variables
id_cols = [c for c in df.columns if c.lower() in ["id","student_id"]]
X = df.drop(columns=["Target","target_bin"] + id_cols, errors="ignore")
y = df["target_bin"]

# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Preprocesamiento
num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

preproc = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# Función para crear pipeline
def make_pipe(estimator):
    return ImbPipeline([
        ("prep", preproc),
        ("smote", SMOTE(random_state=42)),
        ("clf", estimator),
    ])

# Modelos a evaluar (sin paralelización para evitar errores)
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reducido a 3 folds

print("Ejecutando validación cruzada para cada algoritmo...")
print("(Esto puede tomar varios minutos...)\n")

# Evaluar modelos
benchmark_results = []
detailed_results = {}

for name, estimator in models.items():
    print(f"Evaluando {name}...")
    pipe = make_pipe(estimator)
    
    try:
        # Validación cruzada
        auc_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
        f1_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=1)
        precision_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="precision", n_jobs=1)
        recall_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="recall", n_jobs=1)
        
        # Entrenar en todo el conjunto para evaluación final
        pipe.fit(X_train, y_train)
        test_proba = pipe.predict_proba(X_test)[:, 1]
        test_pred = pipe.predict(X_test)
        
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
        
        test_auc = roc_auc_score(y_test, test_proba)
        test_precision = precision_score(y_test, test_pred)
        test_recall = recall_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred)
        
        # Guardar resultados
        result = {
            "Algoritmo": name,
            "AUC_CV_Mean": auc_scores.mean(),
            "AUC_CV_Std": auc_scores.std(),
            "AUC_Test": test_auc,
            "Precision_CV_Mean": precision_scores.mean(),
            "Precision_Test": test_precision,
            "Recall_CV_Mean": recall_scores.mean(),
            "Recall_Test": test_recall,
            "F1_CV_Mean": f1_scores.mean(),
            "F1_Test": test_f1
        }
        
        benchmark_results.append(result)
        detailed_results[name] = {
            'auc_scores': auc_scores,
            'f1_scores': f1_scores,
            'precision_scores': precision_scores,
            'recall_scores': recall_scores,
            'test_proba': test_proba
        }
        
        print(f"  ✓ {name} completado - AUC: {test_auc:.3f}, F1: {test_f1:.3f}")
        
    except Exception as e:
        print(f"  ❌ Error en {name}: {e}")
        # Agregar resultado con errores
        result = {
            "Algoritmo": name,
            "AUC_CV_Mean": np.nan,
            "AUC_CV_Std": np.nan,
            "AUC_Test": np.nan,
            "Precision_CV_Mean": np.nan,
            "Precision_Test": np.nan,
            "Recall_CV_Mean": np.nan,
            "Recall_Test": np.nan,
            "F1_CV_Mean": np.nan,
            "F1_Test": np.nan
        }
        benchmark_results.append(result)

# Crear DataFrame de resultados
benchmark_df = pd.DataFrame(benchmark_results)
benchmark_df = benchmark_df.round(3)

print("\n" + "="*80)
print("                    RESULTADOS DEL BENCHMARK")
print("="*80)
print(benchmark_df.to_string(index=False))

# Guardar benchmark
benchmark_df.to_csv('artifacts/benchmark_completo_algoritmos.csv', index=False)
print(f"\n✅ Benchmark guardado en: artifacts/benchmark_completo_algoritmos.csv")

# Crear visualización del benchmark
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# 1. Comparación AUC
valid_data = benchmark_df.dropna()
if not valid_data.empty:
    ax1.bar(valid_data['Algoritmo'], valid_data['AUC_Test'], color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    ax1.set_title('Comparación AUC en Test Set', fontweight='bold')
    ax1.set_ylabel('AUC Score')
    ax1.set_ylim(0, 1)
    for i, v in enumerate(valid_data['AUC_Test']):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # 2. Comparación F1-Score
    ax2.bar(valid_data['Algoritmo'], valid_data['F1_Test'], color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    ax2.set_title('Comparación F1-Score en Test Set', fontweight='bold')
    ax2.set_ylabel('F1 Score')
    ax2.set_ylim(0, 1)
    for i, v in enumerate(valid_data['F1_Test']):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # 3. Precision vs Recall
    ax3.scatter(valid_data['Precision_Test'], valid_data['Recall_Test'], 
               s=200, alpha=0.7, c=['#1f77b4', '#ff7f0e', '#2ca02c'])
    for i, algo in enumerate(valid_data['Algoritmo']):
        ax3.annotate(algo, (valid_data['Precision_Test'].iloc[i], valid_data['Recall_Test'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax3.set_xlabel('Precision')
    ax3.set_ylabel('Recall')
    ax3.set_title('Precision vs Recall Trade-off', fontweight='bold')
    ax3.grid(alpha=0.3)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    # 4. Resumen de métricas
    metrics = ['AUC_Test', 'Precision_Test', 'Recall_Test', 'F1_Test']
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, algo in enumerate(valid_data['Algoritmo']):
        values = [valid_data[metric].iloc[i] for metric in metrics]
        ax4.bar(x + i*width, values, width, label=algo, alpha=0.7)
    
    ax4.set_xlabel('Métricas')
    ax4.set_ylabel('Score')
    ax4.set_title('Comparación Completa de Métricas', fontweight='bold')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(['AUC', 'Precision', 'Recall', 'F1'])
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('artifacts/benchmark_visualizacion.png', dpi=300, bbox_inches='tight')
plt.show()

# Crear tabla de ranking
print("\n🏆 RANKING DE ALGORITMOS (por AUC):")
print("="*50)
if not valid_data.empty:
    ranking = valid_data.sort_values('AUC_Test', ascending=False)
    for i, (idx, row) in enumerate(ranking.iterrows(), 1):
        print(f"{i}. {row['Algoritmo']:<20} AUC: {row['AUC_Test']:.3f} | F1: {row['F1_Test']:.3f}")

print(f"\n✅ Archivos generados:")
print(f"  - artifacts/benchmark_completo_algoritmos.csv")
print(f"  - artifacts/benchmark_visualizacion.png")
print("\n" + "="*80)