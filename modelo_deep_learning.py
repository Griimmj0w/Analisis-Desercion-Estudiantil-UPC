"""
APRENDIZAJE PROFUNDO - RED NEURONAL PARA PREDICCIÓN DE DESERCIÓN
==================================================================
Script para implementar y comparar modelos de Deep Learning con algoritmos tradicionales.
Usa TensorFlow/Keras para crear una red neuronal feed-forward.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, confusion_matrix, 
                             ConfusionMatrixDisplay, classification_report,
                             precision_recall_fscore_support)
from imblearn.over_sampling import SMOTE
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("artifacts", exist_ok=True)

# Importar TensorFlow/Keras (con manejo de errores si no está instalado)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    print(f"✓ TensorFlow versión: {tf.__version__}")
    TF_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow no está instalado. Instalando...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'tensorflow', '-q'])
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    print(f"✓ TensorFlow instalado - versión: {tf.__version__}")
    TF_AVAILABLE = True

# Configurar seed para reproducibilidad
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

print("="*80)
print("     APRENDIZAJE PROFUNDO - PREDICCIÓN DE DESERCIÓN ESTUDIANTIL")
print("="*80)

# ====================================================================
# 1. CARGAR Y PREPARAR DATOS
# ====================================================================
print("\n📊 1. Cargando y preparando datos...")
df = pd.read_csv(r'C:\\Users\\SISTEMAS\\Documents\\PYTHON\\DATA\\data.csv', sep=';')
df["target_bin"] = (df["Target"] == "Dropout").astype(int)

print(f"   Datos cargados: {df.shape[0]:,} estudiantes × {df.shape[1]} variables")
print(f"   Tasa de deserción: {df['target_bin'].mean()*100:.1f}%")

# Separar variables predictoras y objetivo
id_cols = [c for c in df.columns if c.lower() in ["id","student_id"]]
X = df.drop(columns=["Target","target_bin"] + id_cols, errors="ignore")
y = df["target_bin"].values

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print(f"   Train set: {len(X_train)} estudiantes")
print(f"   Test set: {len(X_test)} estudiantes")

# ====================================================================
# 2. PREPROCESAMIENTO AVANZADO
# ====================================================================
print("\n🔧 2. Preprocesamiento de datos...")

# Identificar tipos de variables
num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

print(f"   Variables numéricas: {len(num_cols)}")
print(f"   Variables categóricas: {len(cat_cols)}")

# Preprocesador
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
])

# Transformar datos
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"   Dimensión después de encoding: {X_train_processed.shape[1]} features")

# Aplicar SMOTE para balancear clases
print("\n⚖️  Aplicando SMOTE para balanceo de clases...")
print(f"   Antes de SMOTE - Clase 0: {sum(y_train==0)}, Clase 1: {sum(y_train==1)}")

smote = SMOTE(random_state=RANDOM_STATE)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)

print(f"   Después de SMOTE - Clase 0: {sum(y_train_balanced==0)}, Clase 1: {sum(y_train_balanced==1)}")

# ====================================================================
# 3. ARQUITECTURA DE LA RED NEURONAL
# ====================================================================
print("\n🧠 3. Construyendo arquitectura de Red Neuronal...")

def create_neural_network(input_dim, architecture='medium'):
    """
    Crea diferentes arquitecturas de redes neuronales
    
    Parámetros:
    - input_dim: dimensión de entrada
    - architecture: 'simple', 'medium', 'complex'
    """
    model = keras.Sequential(name=f'NN_{architecture}')
    
    if architecture == 'simple':
        # Arquitectura simple: 2 capas ocultas
        model.add(layers.Dense(64, activation='relu', input_shape=(input_dim,), name='hidden_1'))
        model.add(layers.Dropout(0.3, name='dropout_1'))
        model.add(layers.Dense(32, activation='relu', name='hidden_2'))
        model.add(layers.Dropout(0.2, name='dropout_2'))
        model.add(layers.Dense(1, activation='sigmoid', name='output'))
        
    elif architecture == 'medium':
        # Arquitectura media: 3 capas ocultas con batch normalization
        model.add(layers.Dense(128, activation='relu', input_shape=(input_dim,), name='hidden_1'))
        model.add(layers.BatchNormalization(name='bn_1'))
        model.add(layers.Dropout(0.4, name='dropout_1'))
        
        model.add(layers.Dense(64, activation='relu', name='hidden_2'))
        model.add(layers.BatchNormalization(name='bn_2'))
        model.add(layers.Dropout(0.3, name='dropout_2'))
        
        model.add(layers.Dense(32, activation='relu', name='hidden_3'))
        model.add(layers.Dropout(0.2, name='dropout_3'))
        
        model.add(layers.Dense(1, activation='sigmoid', name='output'))
        
    elif architecture == 'complex':
        # Arquitectura compleja: 4 capas ocultas
        model.add(layers.Dense(256, activation='relu', input_shape=(input_dim,), name='hidden_1'))
        model.add(layers.BatchNormalization(name='bn_1'))
        model.add(layers.Dropout(0.5, name='dropout_1'))
        
        model.add(layers.Dense(128, activation='relu', name='hidden_2'))
        model.add(layers.BatchNormalization(name='bn_2'))
        model.add(layers.Dropout(0.4, name='dropout_2'))
        
        model.add(layers.Dense(64, activation='relu', name='hidden_3'))
        model.add(layers.BatchNormalization(name='bn_3'))
        model.add(layers.Dropout(0.3, name='dropout_3'))
        
        model.add(layers.Dense(32, activation='relu', name='hidden_4'))
        model.add(layers.Dropout(0.2, name='dropout_4'))
        
        model.add(layers.Dense(1, activation='sigmoid', name='output'))
    
    return model

# Crear modelos con diferentes arquitecturas
input_dimension = X_train_balanced.shape[1]
models_nn = {
    'NN_Simple': create_neural_network(input_dimension, 'simple'),
    'NN_Medium': create_neural_network(input_dimension, 'medium'),
    'NN_Complex': create_neural_network(input_dimension, 'complex')
}

# Mostrar arquitectura del modelo medium
print("\n   📋 Arquitectura del modelo 'Medium' (seleccionado):")
models_nn['NN_Medium'].summary()

# ====================================================================
# 4. CONFIGURACIÓN Y ENTRENAMIENTO
# ====================================================================
print("\n🎓 4. Entrenando modelos de Deep Learning...")

# Configurar callbacks
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=0
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-7,
    verbose=0
)

# Compilar y entrenar cada modelo
trained_models = {}
histories = {}

for name, model in models_nn.items():
    print(f"\n   🔄 Entrenando {name}...")
    
    # Compilar modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc'), 
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')]
    )
    
    # Entrenar
    history = model.fit(
        X_train_balanced, y_train_balanced,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    trained_models[name] = model
    histories[name] = history
    
    # Evaluar en test set
    test_loss, test_acc, test_auc, test_prec, test_rec = model.evaluate(
        X_test_processed, y_test, verbose=0
    )
    
    print(f"      ✓ Test AUC: {test_auc:.4f}")
    print(f"      ✓ Test Accuracy: {test_acc:.4f}")
    print(f"      ✓ Test Precision: {test_prec:.4f}")
    print(f"      ✓ Test Recall: {test_rec:.4f}")
    print(f"      ✓ Epochs completados: {len(history.history['loss'])}")

# ====================================================================
# 5. VISUALIZAR CURVAS DE APRENDIZAJE
# ====================================================================
print("\n📊 5. Generando curvas de aprendizaje...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Curvas de Aprendizaje - Modelos de Deep Learning', fontsize=16, fontweight='bold')

for idx, (name, history) in enumerate(histories.items()):
    col = idx
    
    # Loss
    axes[0, col].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, col].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, col].set_title(f'{name} - Loss')
    axes[0, col].set_xlabel('Epoch')
    axes[0, col].set_ylabel('Loss')
    axes[0, col].legend()
    axes[0, col].grid(alpha=0.3)
    
    # AUC
    axes[1, col].plot(history.history['auc'], label='Train AUC', linewidth=2)
    axes[1, col].plot(history.history['val_auc'], label='Val AUC', linewidth=2)
    axes[1, col].set_title(f'{name} - AUC')
    axes[1, col].set_xlabel('Epoch')
    axes[1, col].set_ylabel('AUC')
    axes[1, col].legend()
    axes[1, col].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("artifacts/deep_learning_curvas_aprendizaje.png", dpi=300, bbox_inches="tight")
print("   ✓ Gráfico guardado: artifacts/deep_learning_curvas_aprendizaje.png")
plt.show()

# ====================================================================
# 6. EVALUACIÓN DETALLADA DEL MEJOR MODELO
# ====================================================================
print("\n🏆 6. Evaluación detallada del mejor modelo...")

# Seleccionar modelo medium como principal
best_model = trained_models['NN_Medium']
best_name = 'NN_Medium'

# Predicciones
y_pred_proba = best_model.predict(X_test_processed, verbose=0).flatten()
y_pred = (y_pred_proba >= 0.5).astype(int)

# Métricas completas
auc_score = roc_auc_score(y_test, y_pred_proba)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
ap_score = average_precision_score(y_test, y_pred_proba)

print(f"\n   📊 Métricas del modelo {best_name}:")
print(f"      • AUC-ROC: {auc_score:.4f}")
print(f"      • Average Precision: {ap_score:.4f}")
print(f"      • Precision: {precision:.4f}")
print(f"      • Recall: {recall:.4f}")
print(f"      • F1-Score: {f1:.4f}")

print(f"\n   📋 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# ====================================================================
# 7. CURVAS ROC Y PRECISION-RECALL
# ====================================================================
print("\n📈 7. Generando curvas ROC y Precision-Recall...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0].plot(fpr, tpr, linewidth=2, label=f'{best_name} (AUC={auc_score:.4f})')
axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('Curva ROC - Deep Learning')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Curva Precision-Recall
prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_pred_proba)
axes[1].plot(rec_curve, prec_curve, linewidth=2, 
            label=f'{best_name} (AP={ap_score:.4f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Curva Precision-Recall - Deep Learning')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("artifacts/deep_learning_roc_pr_curves.png", dpi=300, bbox_inches="tight")
print("   ✓ Gráfico guardado: artifacts/deep_learning_roc_pr_curves.png")
plt.show()

# ====================================================================
# 8. MATRIZ DE CONFUSIÓN
# ====================================================================
print("\n📊 8. Generando matriz de confusión...")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(cm, display_labels=['No Deserción', 'Deserción'])
disp.plot(cmap='Blues', ax=ax, values_format='d')
ax.set_title(f'Matriz de Confusión - {best_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("artifacts/deep_learning_confusion_matrix.png", dpi=300, bbox_inches="tight")
print("   ✓ Gráfico guardado: artifacts/deep_learning_confusion_matrix.png")
plt.show()

# ====================================================================
# 9. COMPARACIÓN CON MODELOS TRADICIONALES (BENCHMARK)
# ====================================================================
print("\n⚖️  9. Comparación con modelos tradicionales de ML...")

# Importar modelos tradicionales para comparación
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline

# Entrenar modelos tradicionales
print("   🔄 Entrenando modelos tradicionales para comparación...")

models_traditional = {
    'Random Forest': RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
}

results_comparison = []

# Evaluar modelos tradicionales
for name, model in models_traditional.items():
    # Entrenar
    model.fit(X_train_balanced, y_train_balanced)
    
    # Predicciones
    y_pred_trad = model.predict(X_test_processed)
    y_pred_proba_trad = model.predict_proba(X_test_processed)[:, 1]
    
    # Métricas
    auc_trad = roc_auc_score(y_test, y_pred_proba_trad)
    prec_trad, rec_trad, f1_trad, _ = precision_recall_fscore_support(
        y_test, y_pred_trad, average='binary'
    )
    ap_trad = average_precision_score(y_test, y_pred_proba_trad)
    
    results_comparison.append({
        'Modelo': name,
        'Tipo': 'ML Tradicional',
        'AUC': auc_trad,
        'Precision': prec_trad,
        'Recall': rec_trad,
        'F1-Score': f1_trad,
        'Avg Precision': ap_trad
    })
    
    print(f"      ✓ {name}: AUC={auc_trad:.4f}, F1={f1_trad:.4f}")

# Agregar resultados de Deep Learning
for name, model in trained_models.items():
    y_pred_proba_dl = model.predict(X_test_processed, verbose=0).flatten()
    y_pred_dl = (y_pred_proba_dl >= 0.5).astype(int)
    
    auc_dl = roc_auc_score(y_test, y_pred_proba_dl)
    prec_dl, rec_dl, f1_dl, _ = precision_recall_fscore_support(
        y_test, y_pred_dl, average='binary'
    )
    ap_dl = average_precision_score(y_test, y_pred_proba_dl)
    
    results_comparison.append({
        'Modelo': name,
        'Tipo': 'Deep Learning',
        'AUC': auc_dl,
        'Precision': prec_dl,
        'Recall': rec_dl,
        'F1-Score': f1_dl,
        'Avg Precision': ap_dl
    })

# Crear DataFrame de comparación
df_comparison = pd.DataFrame(results_comparison)
df_comparison = df_comparison.sort_values('AUC', ascending=False)

print("\n   📊 Tabla de Comparación:")
print(df_comparison.to_string(index=False))

# Guardar resultados
df_comparison.to_csv("artifacts/deep_learning_benchmark_comparison.csv", index=False)
print("\n   ✓ Tabla guardada: artifacts/deep_learning_benchmark_comparison.csv")

# ====================================================================
# 10. VISUALIZACIÓN DE COMPARACIÓN
# ====================================================================
print("\n📊 10. Generando gráficos comparativos...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Comparación: Deep Learning vs ML Tradicional', fontsize=16, fontweight='bold')

metrics = ['AUC', 'Precision', 'Recall', 'F1-Score']
colors_map = {'Deep Learning': 'coral', 'ML Tradicional': 'steelblue'}

for idx, metric in enumerate(metrics):
    row = idx // 2
    col = idx % 2
    
    # Crear gráfico de barras agrupadas
    df_plot = df_comparison.copy()
    df_plot = df_plot.sort_values(metric, ascending=False)
    
    x = np.arange(len(df_plot))
    colors = [colors_map[tipo] for tipo in df_plot['Tipo']]
    
    bars = axes[row, col].bar(x, df_plot[metric], color=colors, alpha=0.7)
    axes[row, col].set_xlabel('Modelo')
    axes[row, col].set_ylabel(metric)
    axes[row, col].set_title(f'Comparación de {metric}')
    axes[row, col].set_xticks(x)
    axes[row, col].set_xticklabels(df_plot['Modelo'], rotation=45, ha='right')
    axes[row, col].grid(alpha=0.3, axis='y')
    
    # Agregar valores en las barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[row, col].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.3f}',
                          ha='center', va='bottom', fontsize=9)

# Crear leyenda personalizada
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='coral', alpha=0.7, label='Deep Learning'),
                   Patch(facecolor='steelblue', alpha=0.7, label='ML Tradicional')]
fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

plt.tight_layout()
plt.savefig("artifacts/deep_learning_comparison_chart.png", dpi=300, bbox_inches="tight")
print("   ✓ Gráfico guardado: artifacts/deep_learning_comparison_chart.png")
plt.show()

# ====================================================================
# 11. GUARDAR MODELO ENTRENADO
# ====================================================================
print("\n💾 11. Guardando modelo de Deep Learning...")

best_model.save("artifacts/modelo_deep_learning.h5")
print("   ✓ Modelo guardado: artifacts/modelo_deep_learning.h5")

# Guardar también en formato Keras nativo
best_model.save("artifacts/modelo_deep_learning.keras")
print("   ✓ Modelo guardado: artifacts/modelo_deep_learning.keras")

# ====================================================================
# 12. ANÁLISIS DE FEATURE IMPORTANCE (APROXIMADO)
# ====================================================================
print("\n🔍 12. Análisis de importancia de features (aproximado)...")

# Calcular importancia basada en gradientes (método alternativo)
print("   Calculando importancia mediante análisis de varianza...")

# Obtener nombres de features
feature_names = num_cols.copy()
if len(cat_cols) > 0:
    ohe = preprocessor.named_transformers_['cat']
    cat_feature_names = ohe.get_feature_names_out(cat_cols)
    feature_names.extend(cat_feature_names)

# Calcular importancia usando variación de predicciones
feature_importance_scores = []

for i in range(X_test_processed.shape[1]):
    # Permutar cada feature y medir cambio en predicciones
    X_permuted = X_test_processed.copy()
    np.random.shuffle(X_permuted[:, i])
    
    # Predicciones con feature permutada
    pred_original = best_model.predict(X_test_processed, verbose=0).flatten()
    pred_permuted = best_model.predict(X_permuted, verbose=0).flatten()
    
    # Calcular diferencia (importancia)
    importance = np.abs(pred_original - pred_permuted).mean()
    feature_importance_scores.append(importance)

feature_importance_scores = np.array(feature_importance_scores)

# Top 20 features más importantes
indices_top = np.argsort(feature_importance_scores)[-20:][::-1]
top_features = [feature_names[i] for i in indices_top]
top_importance = feature_importance_scores[indices_top]

print(f"   Top 10 features más importantes:")
for i, (feat, imp) in enumerate(zip(top_features[:10], top_importance[:10]), 1):
    print(f"      {i}. {feat}: {imp:.4f}")

# Visualizar
plt.figure(figsize=(10, 8))
plt.barh(range(len(top_features)), top_importance, color='coral', alpha=0.7)
plt.yticks(range(len(top_features)), top_features)
plt.xlabel('Importancia (Variación de Predicción)')
plt.title('Top 20 Features - Deep Learning Model')
plt.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig("artifacts/deep_learning_feature_importance.png", dpi=300, bbox_inches="tight")
print("   ✓ Gráfico guardado: artifacts/deep_learning_feature_importance.png")
plt.close()

# ====================================================================
# 13. RESUMEN FINAL
# ====================================================================
print("\n" + "="*80)
print("                    RESUMEN FINAL - DEEP LEARNING")
print("="*80)

print(f"\n🧠 MODELOS ENTRENADOS:")
for name in trained_models.keys():
    print(f"   ✓ {name}")

print(f"\n🏆 MEJOR MODELO: {best_name}")
print(f"   • AUC-ROC: {auc_score:.4f}")
print(f"   • Average Precision: {ap_score:.4f}")
print(f"   • Precision: {precision:.4f}")
print(f"   • Recall: {recall:.4f}")
print(f"   • F1-Score: {f1:.4f}")

print(f"\n📊 COMPARACIÓN CON ML TRADICIONAL:")
best_traditional = df_comparison[df_comparison['Tipo'] == 'ML Tradicional'].iloc[0]
print(f"   Mejor Tradicional: {best_traditional['Modelo']}")
print(f"      AUC: {best_traditional['AUC']:.4f}")
print(f"      F1: {best_traditional['F1-Score']:.4f}")

if auc_score > best_traditional['AUC']:
    print(f"   ✓ Deep Learning SUPERA a modelos tradicionales")
    print(f"   Mejora en AUC: +{(auc_score - best_traditional['AUC'])*100:.2f}%")
else:
    print(f"   ℹ️  Modelos tradicionales aún competitivos")
    print(f"   Diferencia en AUC: {(auc_score - best_traditional['AUC'])*100:.2f}%")

print(f"\n💾 ARCHIVOS GENERADOS:")
print(f"   ✓ artifacts/deep_learning_curvas_aprendizaje.png")
print(f"   ✓ artifacts/deep_learning_roc_pr_curves.png")
print(f"   ✓ artifacts/deep_learning_confusion_matrix.png")
print(f"   ✓ artifacts/deep_learning_comparison_chart.png")
print(f"   ✓ artifacts/deep_learning_feature_importance.png")
print(f"   ✓ artifacts/deep_learning_benchmark_comparison.csv")
print(f"   ✓ artifacts/modelo_deep_learning.h5")
print(f"   ✓ artifacts/modelo_deep_learning_savedmodel/")

print("\n" + "="*80)
print("              ANÁLISIS DE DEEP LEARNING COMPLETADO ✓")
print("="*80)
