import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import os

# Configuraciones
RANDOM_STATE = 42
plt.style.use('default')
os.makedirs("artifacts", exist_ok=True)

print("Cargando datos y generando modelo simplificado...")

# Cargar datos
df = pd.read_csv('DATA/data.csv', sep=';')
print(f"Datos cargados: {df.shape}")

# Preparar variable objetivo binaria (Dropout=1, otros=0)
df["target_bin"] = (df["Target"] == "Dropout").astype(int)

# Separar variables predictoras y objetivo
id_cols = [c for c in df.columns if c.lower() in ["id","student_id"]]
X = df.drop(columns=["Target","target_bin"] + id_cols, errors="ignore")
y = df["target_bin"]

# Separar datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print(f"Proporción de deserción - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")

# Preprocesamiento simplificado
num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

preproc = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# Pipeline simplificado con RandomForest básico
pipe = ImbPipeline([
    ("prep", preproc),
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("clf", RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=RANDOM_STATE)),
])

print("Entrenando modelo...")
pipe.fit(X_train, y_train)

print("Generando predicciones...")
proba = pipe.predict_proba(X_test)[:, 1]

# === GENERAR CURVA PRECISION-RECALL ===
print("Generando curva Precision-Recall...")
prec, rec, th = precision_recall_curve(y_test, proba)
ap = average_precision_score(y_test, proba)

# Buscar umbral óptimo (recall >= 0.85 y F1 máximo)
RECALL_MIN = 0.85
rec_th = rec[:-1]
prec_th = prec[:-1]
mask = rec_th >= RECALL_MIN

if mask.sum() > 0:
    f1_mask = 2 * (prec_th[mask] * rec_th[mask]) / (prec_th[mask] + rec_th[mask] + 1e-12)
    i = f1_mask.argmax()
    th_opt = th[mask][i] if i < len(th[mask]) else th[mask][0]
else:
    f1_all = 2 * (prec_th * rec_th) / (prec_th + rec_th + 1e-12)
    i = f1_all.argmax()
    th_opt = th[i] if i < len(th) else th[0]

# Calcular métricas para ambos umbrales
pred_50 = (proba >= 0.5).astype(int)
pred_opt = (proba >= th_opt).astype(int)

# Métricas umbral 0.5
from sklearn.metrics import precision_recall_fscore_support
P50, R50, F150, _ = precision_recall_fscore_support(y_test, pred_50, average="binary", zero_division=0)
cm50 = confusion_matrix(y_test, pred_50)

# Métricas umbral óptimo
Popt, Ropt, F1opt, _ = precision_recall_fscore_support(y_test, pred_opt, average="binary", zero_division=0)
cmopt = confusion_matrix(y_test, pred_opt)

print(f"\nResultados:")
print(f"Average Precision (área bajo PR): {ap:.3f}")
print(f"Umbral óptimo seleccionado: {th_opt:.3f}")
print(f"[0.50]  precision={P50:.3f}  recall={R50:.3f}  F1={F150:.3f}")
print(f"[{th_opt:.3f}] precision={Popt:.3f}  recall={Ropt:.3f}  F1={F1opt:.3f}")

# === GRÁFICO PRECISION-RECALL ===
plt.figure(figsize=(8, 6))
plt.plot(rec, prec, label=f"PR curve (AP={ap:.3f})", linewidth=2, color='blue')
plt.scatter([R50, Ropt], [P50, Popt], c=["gray","orange"], s=100, 
           label=f"Umbrales: 0.50 y {th_opt:.3f}", zorder=5)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precision-Recall")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("artifacts/curva_precision_recall.png", dpi=300, bbox_inches="tight")
print("✓ Gráfico guardado: artifacts/curva_precision_recall.png")
plt.show()

# === MATRIZ DE CONFUSIÓN CON UMBRAL ÓPTIMO ===
plt.figure(figsize=(6, 5))
cm_display = ConfusionMatrixDisplay(cmopt, display_labels=['No Dropout', 'Dropout'])
cm_display.plot(cmap="Blues", ax=plt.gca())
plt.title(f"Matriz de confusión (umbral={th_opt:.3f})")
plt.tight_layout()
plt.savefig("artifacts/matriz_confusion_umbral_optimo.png", dpi=300, bbox_inches="tight")
print("✓ Gráfico guardado: artifacts/matriz_confusion_umbral_optimo.png")
plt.show()

# === RESUMEN DE ARCHIVOS GENERADOS ===
print("\n=== ARCHIVOS GUARDADOS EN 'artifacts/' ===")
files_generated = [
    "curva_precision_recall.png - Curva PR con umbrales",
    "matriz_confusion_umbral_optimo.png - Matriz confusión (umbral óptimo)"
]

for file_info in files_generated:
    print(f"✓ {file_info}")

print(f"\n🎯 Proceso completado exitosamente!")
print(f"📊 Métricas finales con umbral óptimo ({th_opt:.3f}): P={Popt:.3f}, R={Ropt:.3f}, F1={F1opt:.3f}")