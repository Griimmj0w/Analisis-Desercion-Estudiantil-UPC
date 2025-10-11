import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import os

# Crear directorio para slides
os.makedirs('slides', exist_ok=True)

print("🎨 Generando slides para la presentación...")

# Configuración global para las slides
plt.style.use('seaborn-v0_8-whitegrid')
colors = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'success': '#2ca02c',
    'warning': '#d62728',
    'info': '#9467bd',
    'dark': '#2f2f2f'
}

# SLIDE 1: Título
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Título principal
ax.text(5, 7, '📊 ANÁLISIS PREDICTIVO DE DESERCIÓN ESTUDIANTIL', 
        fontsize=28, fontweight='bold', ha='center', color=colors['dark'])

ax.text(5, 6.2, 'Machine Learning aplicado a la retención universitaria', 
        fontsize=18, ha='center', color=colors['primary'])

# Información del proyecto
ax.text(5, 4.5, 'Universidad Peruana de Ciencias Aplicadas (UPC)', 
        fontsize=16, ha='center', color=colors['dark'])

ax.text(5, 3.8, 'Análisis de 4,424 estudiantes • 37 variables • 3 algoritmos ML', 
        fontsize=14, ha='center', color=colors['info'])

# Resultados destacados
resultado_box = FancyBboxPatch((1, 1.5), 8, 1.5, boxstyle="round,pad=0.1", 
                              facecolor=colors['success'], alpha=0.1, 
                              edgecolor=colors['success'], linewidth=2)
ax.add_patch(resultado_box)

ax.text(5, 2.7, '🏆 RESULTADO PRINCIPAL', fontsize=16, fontweight='bold', 
        ha='center', color=colors['success'])
ax.text(5, 2.2, 'Random Forest: 93.2% AUC • 84.5% Precisión • 78.5% Recall', 
        fontsize=14, ha='center', color=colors['dark'])
ax.text(5, 1.8, 'Capacidad de identificar 8 de cada 10 casos de deserción', 
        fontsize=12, ha='center', color=colors['dark'])

plt.tight_layout()
plt.savefig('slides/slide_01_titulo.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# SLIDE 2: Problemática
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9, '🚨 PROBLEMÁTICA', fontsize=32, fontweight='bold', 
        ha='center', color=colors['warning'])

# Estadística principal
circle = plt.Circle((2.5, 6.5), 1.2, color=colors['warning'], alpha=0.2)
ax.add_patch(circle)
ax.text(2.5, 6.8, '32.1%', fontsize=36, fontweight='bold', 
        ha='center', color=colors['warning'])
ax.text(2.5, 6.2, 'DESERCIÓN', fontsize=14, fontweight='bold', 
        ha='center', color=colors['dark'])
ax.text(2.5, 5.9, '1 de cada 3', fontsize=12, 
        ha='center', color=colors['dark'])

# Impactos
impacts = [
    '💰 Pérdida económica institucional',
    '📚 Truncamiento de proyectos académicos', 
    '👥 Desperdicio de recursos humanos',
    '📊 Afectación de indicadores de calidad'
]

for i, impact in enumerate(impacts):
    ax.text(6, 8 - i*0.8, impact, fontsize=16, 
            ha='left', color=colors['dark'])

# Solución propuesta
solution_box = FancyBboxPatch((1, 1), 8, 2, boxstyle="round,pad=0.2", 
                             facecolor=colors['primary'], alpha=0.1, 
                             edgecolor=colors['primary'], linewidth=3)
ax.add_patch(solution_box)

ax.text(5, 2.5, '💡 SOLUCIÓN PROPUESTA', fontsize=18, fontweight='bold', 
        ha='center', color=colors['primary'])
ax.text(5, 1.8, 'Sistema predictivo con Machine Learning', fontsize=16, 
        ha='center', color=colors['dark'])
ax.text(5, 1.4, 'Intervención temprana • Alertas automáticas • Apoyo personalizado', 
        fontsize=14, ha='center', color=colors['dark'])

plt.tight_layout()
plt.savefig('slides/slide_02_problematica.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# SLIDE 3: Dataset
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, '📊 CARACTERÍSTICAS DEL DATASET', fontsize=28, fontweight='bold', 
        ha='center', color=colors['primary'])

# Métricas principales en cajas
metrics = [
    ('4,424', 'Estudiantes', colors['primary']),
    ('37', 'Variables', colors['secondary']),
    ('100%', 'Completitud', colors['success']),
    ('3', 'Estados', colors['info'])
]

for i, (value, label, color) in enumerate(metrics):
    x_pos = 1.5 + i * 2
    
    # Caja para métrica
    metric_box = FancyBboxPatch((x_pos-0.7, 6.5), 1.4, 1.5, boxstyle="round,pad=0.1", 
                               facecolor=color, alpha=0.1, 
                               edgecolor=color, linewidth=2)
    ax.add_patch(metric_box)
    
    ax.text(x_pos, 7.6, value, fontsize=24, fontweight='bold', 
            ha='center', color=color)
    ax.text(x_pos, 7, label, fontsize=12, 
            ha='center', color=colors['dark'])

# Distribución de estados
ax.text(5, 5.5, 'DISTRIBUCIÓN DE ESTUDIANTES', fontsize=18, fontweight='bold', 
        ha='center', color=colors['dark'])

# Gráfico de barras simple
estados = ['Graduados\n49.9%', 'Deserción\n32.1%', 'Matriculados\n17.9%']
valores = [49.9, 32.1, 17.9]
colores_estados = [colors['success'], colors['warning'], colors['info']]

bar_width = 0.8
x_positions = [2, 5, 8]

for i, (estado, valor, color) in enumerate(zip(estados, valores, colores_estados)):
    # Barra
    rect = FancyBboxPatch((x_positions[i]-bar_width/2, 2), bar_width, valor/100*2, 
                         boxstyle="round,pad=0.05", 
                         facecolor=color, alpha=0.7)
    ax.add_patch(rect)
    
    # Etiqueta
    ax.text(x_positions[i], 1.5, estado, fontsize=12, fontweight='bold', 
            ha='center', color=colors['dark'])

plt.tight_layout()
plt.savefig('slides/slide_03_dataset.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# SLIDE 4: Benchmark de Algoritmos
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, '🏆 BENCHMARK DE ALGORITMOS', fontsize=28, fontweight='bold', 
        ha='center', color=colors['primary'])

# Tabla de resultados
algorithms = ['Random Forest', 'Logistic Regression', 'SVM']
aucs = [0.932, 0.926, 0.922]
f1s = [0.814, 0.804, 0.796]
medals = ['🥇', '🥈', '🥉']
colors_rank = [colors['success'], colors['secondary'], colors['info']]

# Encabezados
ax.text(2, 8.2, 'Algoritmo', fontsize=16, fontweight='bold', ha='center', color=colors['dark'])
ax.text(5, 8.2, 'AUC', fontsize=16, fontweight='bold', ha='center', color=colors['dark'])
ax.text(7, 8.2, 'F1-Score', fontsize=16, fontweight='bold', ha='center', color=colors['dark'])
ax.text(8.5, 8.2, 'Ranking', fontsize=16, fontweight='bold', ha='center', color=colors['dark'])

# Línea de separación
ax.axhline(y=7.9, xmin=0.1, xmax=0.9, color=colors['dark'], linewidth=2)

# Datos de la tabla
for i, (algo, auc, f1, medal, color) in enumerate(zip(algorithms, aucs, f1s, medals, colors_rank)):
    y_pos = 7.5 - i * 0.6
    
    # Fila con color de fondo
    if i == 0:  # Destacar ganador
        highlight_box = FancyBboxPatch((0.5, y_pos-0.2), 9, 0.4, boxstyle="round,pad=0.05", 
                                      facecolor=colors['success'], alpha=0.1)
        ax.add_patch(highlight_box)
    
    ax.text(2, y_pos, algo, fontsize=14, ha='center', color=colors['dark'])
    ax.text(5, y_pos, f'{auc:.3f}', fontsize=14, fontweight='bold', ha='center', color=color)
    ax.text(7, y_pos, f'{f1:.3f}', fontsize=14, fontweight='bold', ha='center', color=color)
    ax.text(8.5, y_pos, medal, fontsize=18, ha='center')

# Interpretación
interp_box = FancyBboxPatch((1, 2), 8, 2.5, boxstyle="round,pad=0.2", 
                           facecolor=colors['primary'], alpha=0.1, 
                           edgecolor=colors['primary'], linewidth=2)
ax.add_patch(interp_box)

ax.text(5, 4, '🎯 INTERPRETACIÓN DEL GANADOR', fontsize=18, fontweight='bold', 
        ha='center', color=colors['primary'])

interpretations = [
    '• AUC 93.2%: Excelente capacidad discriminativa',
    '• Precisión 84.5%: 85 de cada 100 predicciones correctas', 
    '• Recall 78.5%: Identifica 79 de cada 100 casos reales',
    '• F1-Score 81.4%: Balance óptimo entre precisión y recall'
]

for i, interp in enumerate(interpretations):
    ax.text(1.5, 3.4 - i*0.3, interp, fontsize=13, ha='left', color=colors['dark'])

plt.tight_layout()
plt.savefig('slides/slide_04_benchmark.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# SLIDE 5: Impacto y Recomendaciones
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, '📈 IMPACTO ESPERADO Y RECOMENDACIONES', fontsize=24, fontweight='bold', 
        ha='center', color=colors['primary'])

# Impacto esperado (lado izquierdo)
impact_box = FancyBboxPatch((0.5, 5.5), 4, 3, boxstyle="round,pad=0.2", 
                           facecolor=colors['success'], alpha=0.1, 
                           edgecolor=colors['success'], linewidth=2)
ax.add_patch(impact_box)

ax.text(2.5, 8, '📊 IMPACTO PROYECTADO', fontsize=16, fontweight='bold', 
        ha='center', color=colors['success'])

impacts = [
    'Deserción: 32.1% → 24-27%',
    'Estudiantes salvados: ~300/año',
    'ROI: Mejora sostenibilidad',
    'Rankings: Mejor retención'
]

for i, impact in enumerate(impacts):
    ax.text(0.8, 7.4 - i*0.4, f'✓ {impact}', fontsize=12, ha='left', color=colors['dark'])

# Recomendaciones (lado derecho)
recommend_box = FancyBboxPatch((5.5, 5.5), 4, 3, boxstyle="round,pad=0.2", 
                              facecolor=colors['info'], alpha=0.1, 
                              edgecolor=colors['info'], linewidth=2)
ax.add_patch(recommend_box)

ax.text(7.5, 8, '🚀 PRÓXIMOS PASOS', fontsize=16, fontweight='bold', 
        ha='center', color=colors['info'])

steps = [
    'Piloto con 100 estudiantes',
    'Validación con casos reales',
    'Integración sistemas UPC',
    'Escalamiento completo'
]

for i, step in enumerate(steps):
    ax.text(5.8, 7.4 - i*0.4, f'{i+1}. {step}', fontsize=12, ha='left', color=colors['dark'])

# Sistema de alerta (parte inferior)
alert_box = FancyBboxPatch((1, 1), 8, 3.5, boxstyle="round,pad=0.2", 
                          facecolor=colors['warning'], alpha=0.1, 
                          edgecolor=colors['warning'], linewidth=2)
ax.add_patch(alert_box)

ax.text(5, 4, '🔔 SISTEMA DE ALERTA TEMPRANA', fontsize=18, fontweight='bold', 
        ha='center', color=colors['warning'])

ax.text(5, 3.4, 'Identificación automática desde el 2° mes de clases', fontsize=14, 
        ha='center', color=colors['dark'])

features = [
    '📱 Dashboard en tiempo real para consejeros académicos',
    '📊 Scoring individual de riesgo (0-100%)',
    '📧 Alertas automáticas para casos críticos (>70%)',
    '📈 Seguimiento de intervenciones y resultados'
]

for i, feature in enumerate(features):
    ax.text(1.5, 2.8 - i*0.3, feature, fontsize=11, ha='left', color=colors['dark'])

plt.tight_layout()
plt.savefig('slides/slide_05_impacto.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Slides generadas exitosamente en la carpeta 'slides/':")
print("  - slide_01_titulo.png")
print("  - slide_02_problematica.png") 
print("  - slide_03_dataset.png")
print("  - slide_04_benchmark.png")
print("  - slide_05_impacto.png")
print("\n🎯 Usa estas imágenes como base para tu presentación PowerPoint/Google Slides")
print("🎨 Combínalas con los gráficos reales de la carpeta 'artifacts/'")