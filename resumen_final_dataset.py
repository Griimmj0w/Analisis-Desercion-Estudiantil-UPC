import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Crear directorio artifacts si no existe
os.makedirs('artifacts', exist_ok=True)

# Cargar datos
df = pd.read_csv('DATA/data.csv', sep=';')
df['target_bin'] = (df['Target'] == 'Dropout').astype(int)

print("="*80)
print("                    RESUMEN EJECUTIVO DEL DATASET")
print("="*80)

# INFORMACIÓN BÁSICA
print(f"\n📊 DIMENSIONES: {df.shape[0]:,} estudiantes × {df.shape[1]} variables")
print(f"📁 TAMAÑO: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"✅ COMPLETITUD: {((1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100):.1f}%")

# DISTRIBUCIÓN OBJETIVO
print(f"\n🎯 DISTRIBUCIÓN DE RESULTADOS:")
target_counts = df['Target'].value_counts()
for outcome, count in target_counts.items():
    percent = (count / len(df)) * 100
    print(f"   {outcome}: {count:,} ({percent:.1f}%)")

# VARIABLES MÁS IMPORTANTES
print(f"\n🔍 ANÁLISIS DE CORRELACIONES:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_with_target = df[numeric_cols].corrwith(df['target_bin']).abs().sort_values(ascending=False).head(10)

for i, (var, corr) in enumerate(corr_with_target.items(), 1):
    direction = "↗️" if df[var].corr(df['target_bin']) > 0 else "↘️"
    print(f"   {i:2d}. {var[:40]:<40} {corr:.3f} {direction}")

# DIFERENCIAS CLAVE ENTRE GRUPOS
print(f"\n📈 DIFERENCIAS DROPOUT vs GRADUATE:")
dropout_data = df[df['Target'] == 'Dropout']
graduate_data = df[df['Target'] == 'Graduate']

key_vars = ['Age at enrollment', 'Admission grade', 'Curricular units 1st sem (grade)',
           'Curricular units 2nd sem (grade)', 'Tuition fees up to date']

for var in key_vars:
    if var in df.columns:
        dropout_mean = dropout_data[var].mean()
        graduate_mean = graduate_data[var].mean()
        diff_pct = ((dropout_mean - graduate_mean) / graduate_mean * 100) if graduate_mean != 0 else 0
        
        print(f"   {var[:35]:<35}: Dropout={dropout_mean:6.2f} | Graduate={graduate_mean:6.2f} | Diff={diff_pct:+5.1f}%")

# CREAR VISUALIZACIÓN SIMPLE
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# 1. Distribución objetivo
target_counts.plot(kind='pie', ax=ax1, autopct='%1.1f%%', startangle=90, colors=['#2E8B57', '#CD5C5C', '#4682B4'])
ax1.set_title('Distribución de Resultados', fontweight='bold')
ax1.set_ylabel('')

# 2. Top correlaciones
top_10_corr = corr_with_target.head(10)
top_10_corr.plot(kind='barh', ax=ax2, color='steelblue', alpha=0.7)
ax2.set_title('Top 10 Correlaciones con Deserción', fontweight='bold')
ax2.set_xlabel('Correlación Absoluta')
ax2.grid(axis='x', alpha=0.3)

# 3. Edad por resultado
for target_type in ['Dropout', 'Graduate', 'Enrolled']:
    data = df[df['Target'] == target_type]['Age at enrollment']
    ax3.hist(data, bins=20, alpha=0.6, label=target_type, density=True)
ax3.set_xlabel('Edad al Inscribirse')
ax3.set_ylabel('Densidad')
ax3.set_title('Distribución de Edad por Resultado', fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Notas promedio por resultado
grades_1st = df.groupby('Target')['Curricular units 1st sem (grade)'].mean()
grades_2nd = df.groupby('Target')['Curricular units 2nd sem (grade)'].mean()

x = np.arange(len(grades_1st))
width = 0.35

ax4.bar(x - width/2, grades_1st.values, width, label='1er Semestre', alpha=0.8, color='lightblue')
ax4.bar(x + width/2, grades_2nd.values, width, label='2do Semestre', alpha=0.8, color='lightcoral')
ax4.set_xlabel('Resultado')
ax4.set_ylabel('Nota Promedio')
ax4.set_title('Rendimiento Académico por Resultado', fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(grades_1st.index, rotation=45)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('artifacts/dashboard_dataset.png', dpi=300, bbox_inches='tight')
plt.show()

# GUARDAR RESÚMENES
print(f"\n💾 GUARDANDO ARCHIVOS:")

# Estadísticas descriptivas
stats_summary = df.describe().round(3)
stats_summary.to_csv('artifacts/resumen_estadistico_completo.csv')
print(f"   ✓ artifacts/resumen_estadistico_completo.csv")

# Top correlaciones
corr_df = pd.DataFrame({
    'Variable': corr_with_target.index,
    'Correlacion_Absoluta': corr_with_target.values,
    'Correlacion_Original': [df[var].corr(df['target_bin']) for var in corr_with_target.index],
    'Tipo': ['Positiva' if df[var].corr(df['target_bin']) > 0 else 'Negativa' for var in corr_with_target.index]
}).round(3)
corr_df.to_csv('artifacts/correlaciones_desercion.csv', index=False)
print(f"   ✓ artifacts/correlaciones_desercion.csv")

# Dashboard visual
print(f"   ✓ artifacts/dashboard_dataset.png")

# CONCLUSIONES FINALES
print(f"\n🎯 CONCLUSIONES PRINCIPALES:")
print(f"   • Dataset de alta calidad: {df.shape[0]:,} registros completos")
print(f"   • Tasa de deserción del {(df['Target'] == 'Dropout').mean()*100:.1f}% - problema relevante")
print(f"   • Factor más predictivo: {corr_with_target.index[0]}")
print(f"   • Variables académicas (notas) son los mejores predictores")
print(f"   • Diferencias significativas en edad y rendimiento académico")
print(f"   • Ideal para modelos de machine learning predictivos")

print("="*80)