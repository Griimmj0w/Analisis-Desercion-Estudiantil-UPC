import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
df = pd.read_csv(r'C:\\Users\\SISTEMAS\\Documents\\PYTHON\\DATA\\data.csv', sep=';')

print("="*80)
print("                    RESUMEN COMPLETO DEL DATASET")
print("="*80)

# 1. INFORMACIÓN BÁSICA
print("\n📊 1. INFORMACIÓN BÁSICA DEL DATASET")
print("-" * 50)
print(f"Dimensiones: {df.shape[0]:,} filas x {df.shape[1]} columnas")
print(f"Tamaño en memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# 2. VARIABLE OBJETIVO
print("\n🎯 2. DISTRIBUCIÓN DE LA VARIABLE OBJETIVO")
print("-" * 50)
target_counts = df['Target'].value_counts()
target_percent = df['Target'].value_counts(normalize=True) * 100

for outcome, count in target_counts.items():
    percent = target_percent[outcome]
    print(f"{outcome}: {count:,} ({percent:.1f}%)")

# 3. TIPOS DE VARIABLES
print("\n🔍 3. TIPOS DE VARIABLES")
print("-" * 50)
print(f"Variables numéricas: {df.select_dtypes(include=[np.number]).shape[1]}")
print(f"Variables categóricas: {df.select_dtypes(exclude=[np.number]).shape[1]}")

# 4. ANÁLISIS DE VALORES FALTANTES
print("\n❌ 4. VALORES FALTANTES")
print("-" * 50)
missing_data = df.isnull().sum()
if missing_data.sum() == 0:
    print("✅ No hay valores faltantes en el dataset")
else:
    print("Variables con valores faltantes:")
    for col in missing_data[missing_data > 0].index:
        count = missing_data[col]
        percent = (count / len(df)) * 100
        print(f"  {col}: {count} ({percent:.1f}%)")

# 5. ESTADÍSTICAS DESCRIPTIVAS DE VARIABLES NUMÉRICAS
print("\n📈 5. RESUMEN ESTADÍSTICO - VARIABLES NUMÉRICAS")
print("-" * 50)
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"Variables numéricas encontradas: {len(numeric_cols)}")

if len(numeric_cols) > 0:
    numeric_summary = df[numeric_cols].describe()
    print("\nPrimeras 10 variables numéricas:")
    print(numeric_summary.iloc[:, :10].round(2))

# 6. VARIABLES CATEGÓRICAS MÁS IMPORTANTES
print("\n🏷️ 6. VARIABLES CATEGÓRICAS PRINCIPALES")
print("-" * 50)
categorical_cols = df.select_dtypes(exclude=[np.number]).columns
categorical_cols = [col for col in categorical_cols if col != 'Target']  # Excluir target

print(f"Variables categóricas (excluyendo Target): {len(categorical_cols)}")

for col in categorical_cols[:5]:  # Mostrar las primeras 5
    unique_values = df[col].nunique()
    most_common = df[col].mode()[0] if len(df[col].mode()) > 0 else "N/A"
    print(f"  {col}: {unique_values} valores únicos (más común: {most_common})")

# 7. ANÁLISIS POR VARIABLE OBJETIVO
print("\n🎯 7. ANÁLISIS POR DESERCIÓN vs NO DESERCIÓN")
print("-" * 50)

# Crear variable binaria para análisis
df['target_bin'] = (df['Target'] == 'Dropout').astype(int)

# Variables numéricas más relevantes (ejemplo con algunas columnas clave)
key_numeric = []
for col in numeric_cols:
    if any(keyword in col.lower() for keyword in ['grade', 'score', 'unit', 'age']):
        key_numeric.append(col)

if key_numeric:
    print("Diferencias en variables numéricas clave:")
    for col in key_numeric[:5]:  # Primeras 5 más relevantes
        dropout_mean = df[df['target_bin'] == 1][col].mean()
        continue_mean = df[df['target_bin'] == 0][col].mean()
        print(f"  {col}:")
        print(f"    Deserción: {dropout_mean:.2f}")
        print(f"    Continuidad: {continue_mean:.2f}")
        print(f"    Diferencia: {abs(dropout_mean - continue_mean):.2f}")

# 8. CORRELACIONES MÁS ALTAS CON DESERCIÓN
print("\n📊 8. VARIABLES MÁS CORRELACIONADAS CON DESERCIÓN")
print("-" * 50)

# Calcular correlaciones solo con variables numéricas
df_corr = df[list(numeric_cols) + ['target_bin']].copy()
correlations = df_corr.corr()['target_bin'].abs().sort_values(ascending=False)

print("Top 10 correlaciones más altas:")
count = 1
for var, corr in correlations.head(11).items():  # 11 porque incluye target_bin
    if var != 'target_bin':  # Excluir la variable objetivo
        direction = "📈 Positiva" if df_corr.corr()['target_bin'][var] > 0 else "📉 Negativa"
        print(f"  {count}. {var}: {corr:.3f} ({direction})")
        count += 1
        if count > 10:  # Solo mostrar top 10
            break

# 9. RESUMEN DE CALIDAD DE DATOS
print("\n✅ 9. RESUMEN DE CALIDAD DE DATOS")
print("-" * 50)
print(f"✓ Completitud: {((1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100):.1f}%")
print(f"✓ Variables balanceadas: {'Sí' if target_percent.min() > 20 else 'No'}")
print(f"✓ Variedad de variables: {df.shape[1]} características")
print(f"✓ Tamaño de muestra: {'Adecuado' if df.shape[0] > 1000 else 'Pequeño'} ({df.shape[0]:,} registros)")

# 10. RECOMENDACIONES
print("\n💡 10. RECOMENDACIONES PARA EL ANÁLISIS")
print("-" * 50)
dropout_rate = (df['Target'] == 'Dropout').mean()

if dropout_rate < 0.1:
    print("• Dataset muy desbalanceado - considerar técnicas de balanceo (SMOTE)")
elif dropout_rate > 0.4:
    print("• Alta tasa de deserción - enfocarse en factores de retención")
else:
    print("• Balance razonable entre clases")

print("• Variables numéricas pueden beneficiarse de normalización")
print("• Variables categóricas requieren encoding (One-Hot, Label)")
print("• Considerar análisis de importancia de variables con Random Forest")
print("• Validación cruzada estratificada recomendada")

print("\n" + "="*80)
print("                        ANÁLISIS COMPLETADO")
print("="*80)