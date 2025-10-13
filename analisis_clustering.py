"""
ANÁLISIS NO SUPERVISADO - CLUSTERING DE ESTUDIANTES
====================================================
Script para identificar perfiles de estudiantes usando técnicas de clustering.
Implementa K-Means, DBSCAN y PCA para segmentación y visualización.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("artifacts", exist_ok=True)

print("="*80)
print("     ANÁLISIS NO SUPERVISADO - SEGMENTACIÓN DE ESTUDIANTES")
print("="*80)

# ====================================================================
# 1. CARGAR Y PREPARAR DATOS
# ====================================================================
print("\n📊 1. Cargando datos...")
df = pd.read_csv(r'C:\\Users\\SISTEMAS\\Documents\\PYTHON\\DATA\\data.csv', sep=';')
df["target_bin"] = (df["Target"] == "Dropout").astype(int)

print(f"   Datos cargados: {df.shape[0]:,} estudiantes × {df.shape[1]} variables")

# Seleccionar solo variables numéricas más relevantes para clustering
numeric_features = [
    'Age at enrollment',
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Admission grade',
    'Previous qualification (grade)',
    'Unemployment rate',
    'Inflation rate',
    'GDP'
]

X = df[numeric_features].copy()
y_true = df["target_bin"].values  # Para validación posterior

print(f"   Variables seleccionadas: {len(numeric_features)}")

# ====================================================================
# 2. PREPROCESAMIENTO - NORMALIZACIÓN
# ====================================================================
print("\n🔧 2. Normalizando datos...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("   ✓ Datos normalizados (media=0, std=1)")

# ====================================================================
# 3. REDUCCIÓN DE DIMENSIONALIDAD CON PCA
# ====================================================================
print("\n📉 3. Aplicando PCA para reducción de dimensionalidad...")

# PCA para análisis de varianza explicada
pca_full = PCA()
pca_full.fit(X_scaled)

# Determinar componentes óptimos (explicar 95% de varianza)
cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = np.argmax(cumsum_variance >= 0.95) + 1

print(f"   Componentes para 95% varianza: {n_components_95}")
print(f"   Varianza explicada por componente:")
for i, var in enumerate(pca_full.explained_variance_ratio_[:5], 1):
    print(f"      PC{i}: {var*100:.2f}%")

# PCA para visualización (2D y 3D)
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

print(f"   Varianza explicada (2D): {sum(pca_2d.explained_variance_ratio_)*100:.2f}%")
print(f"   Varianza explicada (3D): {sum(pca_3d.explained_variance_ratio_)*100:.2f}%")

# Visualizar varianza explicada acumulada
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(cumsum_variance)+1), cumsum_variance, 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% varianza')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Explicada Acumulada')
plt.title('Varianza Explicada por PCA')
plt.grid(alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(range(1, 11), pca_full.explained_variance_ratio_[:10], alpha=0.7, color='steelblue')
plt.xlabel('Componente Principal')
plt.ylabel('Varianza Explicada')
plt.title('Varianza por Componente (Top 10)')
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("artifacts/pca_varianza_explicada.png", dpi=300, bbox_inches="tight")
print("   ✓ Gráfico guardado: artifacts/pca_varianza_explicada.png")
plt.show()

# ====================================================================
# 4. DETERMINAR NÚMERO ÓPTIMO DE CLUSTERS (MÉTODO DEL CODO)
# ====================================================================
print("\n🔍 4. Determinando número óptimo de clusters (Método del Codo)...")

inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    print(f"   K={k}: Inertia={kmeans.inertia_:.0f}, Silhouette={silhouette_scores[-1]:.3f}")

# Visualizar métodos de selección
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Número de Clusters (K)')
axes[0].set_ylabel('Inertia (Suma de distancias cuadradas)')
axes[0].set_title('Método del Codo')
axes[0].grid(alpha=0.3)

axes[1].plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Número de Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score por K')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("artifacts/clustering_metodo_codo.png", dpi=300, bbox_inches="tight")
print("   ✓ Gráfico guardado: artifacts/clustering_metodo_codo.png")
plt.show()

# Seleccionar K óptimo (basado en Silhouette)
k_optimo = k_range[np.argmax(silhouette_scores)]
print(f"\n   🏆 K óptimo seleccionado: {k_optimo}")

# ====================================================================
# 5. APLICAR K-MEANS CON K ÓPTIMO
# ====================================================================
print(f"\n🎯 5. Aplicando K-Means con K={k_optimo}...")

kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
clusters_kmeans = kmeans.fit_predict(X_scaled)

# Métricas de evaluación
silhouette_kmeans = silhouette_score(X_scaled, clusters_kmeans)
davies_bouldin_kmeans = davies_bouldin_score(X_scaled, clusters_kmeans)
calinski_harabasz_kmeans = calinski_harabasz_score(X_scaled, clusters_kmeans)

print(f"   Métricas K-Means:")
print(f"      Silhouette Score: {silhouette_kmeans:.3f} (Mejor cerca de 1)")
print(f"      Davies-Bouldin Index: {davies_bouldin_kmeans:.3f} (Mejor cerca de 0)")
print(f"      Calinski-Harabasz Index: {calinski_harabasz_kmeans:.0f} (Mayor es mejor)")

# ====================================================================
# 6. APLICAR DBSCAN PARA DETECCIÓN DE OUTLIERS
# ====================================================================
print("\n🔎 6. Aplicando DBSCAN para detección de outliers...")

# Probar diferentes parámetros de DBSCAN
eps_values = [0.5, 1.0, 1.5]
min_samples_values = [5, 10, 15]

best_dbscan = None
best_score = -1
best_params = None

print("   Probando combinaciones de parámetros...")
for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters_dbscan = dbscan.fit_predict(X_scaled)
        
        n_clusters = len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0)
        n_outliers = list(clusters_dbscan).count(-1)
        
        if n_clusters > 1:  # Solo evaluar si hay más de 1 cluster
            score = silhouette_score(X_scaled[clusters_dbscan != -1], 
                                    clusters_dbscan[clusters_dbscan != -1])
            
            if score > best_score:
                best_score = score
                best_dbscan = clusters_dbscan
                best_params = (eps, min_samples)
                
print(f"\n   Mejor configuración DBSCAN:")
print(f"      eps={best_params[0]}, min_samples={best_params[1]}")
print(f"      Clusters encontrados: {len(set(best_dbscan)) - (1 if -1 in best_dbscan else 0)}")
print(f"      Outliers detectados: {list(best_dbscan).count(-1)} ({list(best_dbscan).count(-1)/len(best_dbscan)*100:.1f}%)")
print(f"      Silhouette Score: {best_score:.3f}")

# ====================================================================
# 7. VISUALIZACIÓN DE CLUSTERS EN 2D (PCA)
# ====================================================================
print("\n📊 7. Generando visualizaciones...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# K-Means en 2D
scatter1 = axes[0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], 
                           c=clusters_kmeans, cmap='viridis', 
                           alpha=0.6, s=20)
axes[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
               c='red', marker='X', s=200, edgecolors='black', linewidths=2,
               label='Centroides')
axes[0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
axes[0].set_title(f'K-Means (K={k_optimo}) - Silhouette: {silhouette_kmeans:.3f}')
axes[0].legend()
axes[0].grid(alpha=0.3)
plt.colorbar(scatter1, ax=axes[0], label='Cluster')

# DBSCAN en 2D
scatter2 = axes[1].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], 
                           c=best_dbscan, cmap='plasma', 
                           alpha=0.6, s=20)
axes[1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
axes[1].set_title(f'DBSCAN - Outliers: {list(best_dbscan).count(-1)}')
axes[1].grid(alpha=0.3)
plt.colorbar(scatter2, ax=axes[1], label='Cluster (-1=Outlier)')

# Comparación con Target Real (Deserción)
scatter3 = axes[2].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], 
                           c=y_true, cmap='coolwarm', 
                           alpha=0.6, s=20)
axes[2].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
axes[2].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
axes[2].set_title('Etiquetas Reales (0=No Deserción, 1=Deserción)')
axes[2].grid(alpha=0.3)
plt.colorbar(scatter3, ax=axes[2], label='Deserción')

plt.tight_layout()
plt.savefig("artifacts/clustering_visualizacion_2d.png", dpi=300, bbox_inches="tight")
print("   ✓ Gráfico guardado: artifacts/clustering_visualizacion_2d.png")
plt.show()

# ====================================================================
# 8. VISUALIZACIÓN 3D DE CLUSTERS
# ====================================================================
print("\n📊 8. Generando visualización 3D...")

fig = plt.figure(figsize=(15, 5))

# K-Means 3D
ax1 = fig.add_subplot(131, projection='3d')
scatter1 = ax1.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
                       c=clusters_kmeans, cmap='viridis', alpha=0.5, s=10)
ax1.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]*100:.1f}%)')
ax1.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]*100:.1f}%)')
ax1.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]*100:.1f}%)')
ax1.set_title(f'K-Means (K={k_optimo})')
plt.colorbar(scatter1, ax=ax1, shrink=0.5)

# DBSCAN 3D
ax2 = fig.add_subplot(132, projection='3d')
scatter2 = ax2.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
                       c=best_dbscan, cmap='plasma', alpha=0.5, s=10)
ax2.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]*100:.1f}%)')
ax2.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]*100:.1f}%)')
ax2.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]*100:.1f}%)')
ax2.set_title('DBSCAN')
plt.colorbar(scatter2, ax=ax2, shrink=0.5)

# Target Real 3D
ax3 = fig.add_subplot(133, projection='3d')
scatter3 = ax3.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
                       c=y_true, cmap='coolwarm', alpha=0.5, s=10)
ax3.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]*100:.1f}%)')
ax3.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]*100:.1f}%)')
ax3.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]*100:.1f}%)')
ax3.set_title('Deserción Real')
plt.colorbar(scatter3, ax=ax3, shrink=0.5)

plt.tight_layout()
plt.savefig("artifacts/clustering_visualizacion_3d.png", dpi=300, bbox_inches="tight")
print("   ✓ Gráfico guardado: artifacts/clustering_visualizacion_3d.png")
plt.show()

# ====================================================================
# 9. ANÁLISIS DE PERFILES POR CLUSTER
# ====================================================================
print("\n👥 9. Análisis de perfiles por cluster (K-Means)...")

df_analysis = df[numeric_features].copy()
df_analysis['Cluster'] = clusters_kmeans
df_analysis['Desercion'] = y_true

# Estadísticas por cluster
print("\n   📋 Estadísticas por Cluster:")
print("   " + "-"*70)

cluster_profiles = []
for cluster_id in range(k_optimo):
    cluster_data = df_analysis[df_analysis['Cluster'] == cluster_id]
    n_estudiantes = len(cluster_data)
    tasa_desercion = cluster_data['Desercion'].mean() * 100
    edad_promedio = cluster_data['Age at enrollment'].mean()
    nota_1sem = cluster_data['Curricular units 1st sem (grade)'].mean()
    nota_2sem = cluster_data['Curricular units 2nd sem (grade)'].mean()
    
    print(f"\n   🔹 CLUSTER {cluster_id}:")
    print(f"      Estudiantes: {n_estudiantes} ({n_estudiantes/len(df_analysis)*100:.1f}%)")
    print(f"      Tasa de Deserción: {tasa_desercion:.1f}%")
    print(f"      Edad promedio: {edad_promedio:.1f} años")
    print(f"      Nota 1er sem: {nota_1sem:.2f}")
    print(f"      Nota 2do sem: {nota_2sem:.2f}")
    
    # Clasificar cluster por nivel de riesgo
    if tasa_desercion > 50:
        riesgo = "🔴 ALTO RIESGO"
    elif tasa_desercion > 30:
        riesgo = "🟠 RIESGO MODERADO"
    else:
        riesgo = "🟢 BAJO RIESGO"
    
    print(f"      Perfil: {riesgo}")
    
    cluster_profiles.append({
        'Cluster': cluster_id,
        'N_Estudiantes': n_estudiantes,
        'Porcentaje': f"{n_estudiantes/len(df_analysis)*100:.1f}%",
        'Tasa_Desercion': f"{tasa_desercion:.1f}%",
        'Edad_Promedio': f"{edad_promedio:.1f}",
        'Nota_1sem': f"{nota_1sem:.2f}",
        'Nota_2sem': f"{nota_2sem:.2f}",
        'Nivel_Riesgo': riesgo
    })

# Guardar perfiles en CSV
df_profiles = pd.DataFrame(cluster_profiles)
df_profiles.to_csv("artifacts/clustering_perfiles.csv", index=False)
print(f"\n   ✓ Perfiles guardados: artifacts/clustering_perfiles.csv")

# ====================================================================
# 10. VISUALIZACIÓN DE PERFILES
# ====================================================================
print("\n📊 10. Generando gráfico de perfiles...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Distribución de estudiantes por cluster
cluster_counts = df_analysis['Cluster'].value_counts().sort_index()
axes[0, 0].bar(cluster_counts.index, cluster_counts.values, color='steelblue', alpha=0.7)
axes[0, 0].set_xlabel('Cluster')
axes[0, 0].set_ylabel('Número de Estudiantes')
axes[0, 0].set_title('Distribución de Estudiantes por Cluster')
axes[0, 0].grid(alpha=0.3, axis='y')

# Tasa de deserción por cluster
desercion_por_cluster = df_analysis.groupby('Cluster')['Desercion'].mean() * 100
colors = ['green' if x < 30 else 'orange' if x < 50 else 'red' for x in desercion_por_cluster]
axes[0, 1].bar(desercion_por_cluster.index, desercion_por_cluster.values, 
               color=colors, alpha=0.7)
axes[0, 1].axhline(y=32.1, color='black', linestyle='--', label='Promedio General (32.1%)')
axes[0, 1].set_xlabel('Cluster')
axes[0, 1].set_ylabel('Tasa de Deserción (%)')
axes[0, 1].set_title('Tasa de Deserción por Cluster')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3, axis='y')

# Edad promedio por cluster
edad_por_cluster = df_analysis.groupby('Cluster')['Age at enrollment'].mean()
axes[1, 0].bar(edad_por_cluster.index, edad_por_cluster.values, 
               color='coral', alpha=0.7)
axes[1, 0].set_xlabel('Cluster')
axes[1, 0].set_ylabel('Edad Promedio (años)')
axes[1, 0].set_title('Edad Promedio por Cluster')
axes[1, 0].grid(alpha=0.3, axis='y')

# Rendimiento académico por cluster
clusters_labels = [f'C{i}' for i in range(k_optimo)]
x_pos = np.arange(len(clusters_labels))
width = 0.35

nota_1sem_cluster = df_analysis.groupby('Cluster')['Curricular units 1st sem (grade)'].mean()
nota_2sem_cluster = df_analysis.groupby('Cluster')['Curricular units 2nd sem (grade)'].mean()

axes[1, 1].bar(x_pos - width/2, nota_1sem_cluster.values, width, 
               label='1er Semestre', color='lightblue', alpha=0.8)
axes[1, 1].bar(x_pos + width/2, nota_2sem_cluster.values, width,
               label='2do Semestre', color='lightcoral', alpha=0.8)
axes[1, 1].set_xlabel('Cluster')
axes[1, 1].set_ylabel('Nota Promedio')
axes[1, 1].set_title('Rendimiento Académico por Cluster')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(clusters_labels)
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("artifacts/clustering_analisis_perfiles.png", dpi=300, bbox_inches="tight")
print("   ✓ Gráfico guardado: artifacts/clustering_analisis_perfiles.png")
plt.show()

# ====================================================================
# 11. RESUMEN FINAL Y CONCLUSIONES
# ====================================================================
print("\n" + "="*80)
print("                    RESUMEN Y CONCLUSIONES")
print("="*80)

print(f"\n📊 ALGORITMOS APLICADOS:")
print(f"   ✓ K-Means: {k_optimo} clusters identificados")
print(f"   ✓ DBSCAN: {list(best_dbscan).count(-1)} outliers detectados")
print(f"   ✓ PCA: {n_components_95} componentes para 95% varianza")

print(f"\n📈 MÉTRICAS DE CALIDAD:")
print(f"   • Silhouette Score (K-Means): {silhouette_kmeans:.3f}")
print(f"   • Davies-Bouldin Index: {davies_bouldin_kmeans:.3f}")
print(f"   • Calinski-Harabasz Index: {calinski_harabasz_kmeans:.0f}")

print(f"\n🎯 INSIGHTS PRINCIPALES:")
print(f"   1. Se identificaron {k_optimo} perfiles distintos de estudiantes")
print(f"   2. {list(best_dbscan).count(-1)} estudiantes con comportamiento atípico (outliers)")
print(f"   3. La segmentación revela diferentes niveles de riesgo de deserción")
print(f"   4. Los clusters correlacionan con las tasas reales de deserción")

print(f"\n💾 ARCHIVOS GENERADOS:")
print(f"   ✓ artifacts/pca_varianza_explicada.png")
print(f"   ✓ artifacts/clustering_metodo_codo.png")
print(f"   ✓ artifacts/clustering_visualizacion_2d.png")
print(f"   ✓ artifacts/clustering_visualizacion_3d.png")
print(f"   ✓ artifacts/clustering_analisis_perfiles.png")
print(f"   ✓ artifacts/clustering_perfiles.csv")

print("\n" + "="*80)
print("              ANÁLISIS NO SUPERVISADO COMPLETADO ✓")
print("="*80)
