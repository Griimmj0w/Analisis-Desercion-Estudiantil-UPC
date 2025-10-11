# 🎤 GUIÓN COMPLETO PARA LA EXPOSICIÓN

## 🎯 **TIMING TOTAL: 18-20 MINUTOS**
*Distribución: Introducción (4 min) + Metodología (5 min) + Resultados (6 min) + Impacto (4 min) + Preguntas (3-5 min)*

---

## 📋 **SLIDE 1: TÍTULO Y PRESENTACIÓN** ⏱️ *2-3 minutos*

### **🎤 GUIÓN:**

> *"Buenos días/tardes. Mi nombre es [TU NOMBRE] y hoy les voy a presentar un proyecto de Machine Learning que desarrollé para abordar uno de los problemas más críticos en educación superior: la deserción estudiantil."*

> *"El título de mi proyecto es 'Análisis Predictivo de Deserción Estudiantil', donde apliqué técnicas de Machine Learning para crear un sistema que puede predecir qué estudiantes están en riesgo de abandonar sus estudios."*

> *"Para este análisis trabajé con datos reales de 4,424 estudiantes de la Universidad Peruana de Ciencias Aplicadas, analizando 37 variables diferentes y comparando 3 algoritmos de Machine Learning."*

> *"El resultado principal que quiero destacar desde el inicio es que logré desarrollar un modelo Random Forest con una precisión del 93.2% AUC, capaz de identificar correctamente 8 de cada 10 estudiantes que van a desertar."*

### **💡 Puntos clave a enfatizar:**
- Problema real con impacto social
- Datos significativos (4,424 estudiantes)
- Resultado concreto y medible (93.2% AUC)

---

## 📋 **SLIDE 2: PROBLEMÁTICA** ⏱️ *3-4 minutos*

### **🎤 GUIÓN:**

> *"Pero empecemos por el problema. ¿Por qué es importante la deserción estudiantil?"*

> *"En nuestro dataset encontré que el 32.1% de los estudiantes desertan. Esto significa que 1 de cada 3 estudiantes que ingresan a la universidad no logran completar sus estudios. Esta no es solo una estadística, representa proyectos de vida truncados."*

> *"Esta deserción tiene múltiples impactos: Primero, un impacto económico para la institución, que pierde ingresos y desperdicia recursos invertidos en esos estudiantes. Segundo, un impacto académico, donde se truncan proyectos de formación profesional. Tercero, un desperdicio de recursos humanos, tanto del estudiante como de los profesores. Y finalmente, afecta los indicadores de calidad educativa de la universidad."*

> *"La pregunta es: ¿Podríamos identificar tempranamente qué estudiantes están en riesgo para intervenir a tiempo?"*

> *"Mi propuesta es un sistema predictivo usando Machine Learning que permita identificar estudiantes en riesgo desde los primeros meses, genere alertas automáticas para los consejeros académicos, y facilite intervenciones personalizadas según el perfil de cada estudiante."*

### **💡 Puntos clave a enfatizar:**
- Impacto humano real (no solo números)
- Múltiples niveles de afectación
- Solución proactiva vs reactiva

---

## 📋 **SLIDE 3: DATASET Y METODOLOGÍA** ⏱️ *4-5 minutos*

### **🎤 GUIÓN:**

> *"Para abordar este problema, trabajé con un dataset robusto de 4,424 estudiantes reales, con 37 variables que incluyen información académica, demográfica y socioeconómica."*

> *"Una característica importante de este dataset es que tiene 100% de completitud - no hay datos faltantes, lo que me permitió hacer un análisis más confiable."*

> *"Los estudiantes se clasifican en 3 estados: 49.9% son graduados exitosos, 32.1% han desertado, y 17.9% están actualmente matriculados."*

> *"La metodología que seguí fue la siguiente: Primero, hice un análisis exploratorio para identificar patrones en los datos. Segundo, preparé los datos usando técnicas de preprocesamiento como escalado de variables numéricas y codificación de variables categóricas. Tercero, apliqué SMOTE para balancear las clases, ya que teníamos más graduados que desertores. Cuarto, entrené tres algoritmos diferentes: Random Forest, Regresión Logística y SVM. Finalmente, validé los modelos usando validación cruzada y un conjunto de prueba independiente."*

> *"Una decisión importante fue usar AUC como métrica principal porque nos interesa más la capacidad de ranking - identificar quién tiene más probabilidad de desertar - que la clasificación exacta."*

### **💡 Puntos clave a enfatizar:**
- Robustez de los datos (sin faltantes)
- Metodología rigurosa y científica
- Justificación de decisiones técnicas

---

## 📋 **SLIDE 4: RESULTADOS - BENCHMARK** ⏱️ *5-6 minutos*

### **🎤 GUIÓN:**

> *"Ahora vamos a los resultados del benchmark de algoritmos."*

> *"Evalué tres algoritmos usando las mismas condiciones: Random Forest, Regresión Logística y SVM. Los resultados muestran que Random Forest es el claro ganador."*

> *"Random Forest obtuvo un AUC de 0.932, que significa excelente capacidad discriminativa. Para ponerlo en perspectiva: un AUC de 0.5 sería equivalente a adivinar al azar, un AUC de 0.7-0.8 se considera bueno, y 0.9+ se considera excelente. Nuestro 93.2% está en la categoría de excelente."*

> *"Además, Random Forest tiene la mejor precisión con 84.5%, lo que significa que cuando el modelo predice que un estudiante va a desertar, acierta 85 de cada 100 veces. Esto es crucial para evitar falsas alarmas que podrían saturar a los consejeros académicos."*

> *"El recall de 78.5% indica que el modelo identifica correctamente 79 de cada 100 estudiantes que realmente van a desertar. Esto significa que podríamos intervenir a tiempo con 8 de cada 10 casos críticos."*

> *"El F1-Score de 81.4% confirma que tenemos un buen balance entre no generar demasiadas falsas alarmas y no perder casos importantes."*

### **💡 Puntos clave a enfatizar:**
- Traducir métricas técnicas a lenguaje cotidiano
- Explicar por qué cada métrica es importante
- Conectar con aplicación práctica

---

## 📋 **SLIDE 5: FACTORES PREDICTIVOS** ⏱️ *3-4 minutos*
*Usa aquí: `top20_importancias.png` de artifacts*

### **🎤 GUIÓN:**

> *"Pero ¿qué factores son los más importantes para predecir la deserción?"*

> *"El análisis revela algo muy interesante: los factores más predictivos son el número de unidades curriculares aprobadas en el primer y segundo semestre, con correlaciones de -0.63 y -0.59 respectivamente. Esto significa que el rendimiento académico temprano es el predictor más fuerte."*

> *"También son importantes las calificaciones obtenidas en estos primeros semestres. Esto confirma algo que intuíamos: los primeros dos semestres son críticos para la permanencia estudiantil."*

> *"Otros factores importantes incluyen la edad al momento de inscripción - estudiantes mayores tienen más riesgo - el estado civil, donde estudiantes casados o con pareja tienen mayor probabilidad de desertar, y el turno de estudio, donde el turno nocturno presenta mayor riesgo."*

> *"Esto nos da insights muy valiosos para diseñar intervenciones: debemos enfocar nuestros esfuerzos de retención en los primeros dos semestres, prestar especial atención a estudiantes mayores y del turno nocturno, y desarrollar programas de apoyo para estudiantes con responsabilidades familiares."*

---

## 📋 **SLIDE 6: IMPACTO Y RECOMENDACIONES** ⏱️ *4-5 minutos*

### **🎤 GUIÓN:**

> *"Finalmente, hablemos del impacto esperado y las recomendaciones."*

> *"Con este sistema predictivo, proyectamos reducir la tasa de deserción del 32.1% actual a aproximadamente 24-27%. Esto significaría salvar aproximadamente 300 estudiantes adicionales por año - 300 proyectos de vida que podrían completar su formación profesional."*

> *"El retorno de inversión para la universidad sería significativo: mayor retención significa mayor sostenibilidad financiera, mejores indicadores en rankings educativos, y un uso más eficiente de los recursos académicos."*

> *"Para la implementación, recomiendo empezar con un piloto de 100 estudiantes para validar las predicciones contra resultados reales. Luego, integrar gradualmente el sistema con las plataformas académicas existentes de la UPC."*

> *"El sistema de alerta temprana funcionaría así: desde el segundo mes de clases, cada estudiante tendría un score de riesgo del 0 al 100%. Los casos con más del 70% de riesgo generarían alertas automáticas para consejeros académicos, quienes podrían implementar intervenciones personalizadas: tutorías adicionales, flexibilidad horaria, apoyo psicopedagógico, o programas de nivelación académica."*

> *"Lo más importante es que este no sería un sistema punitivo, sino preventivo y de apoyo. El objetivo es identificar tempranamente para ayudar, no para juzgar."*

### **💡 Puntos clave a enfatizar:**
- Impacto cuantificable y humanizado
- Implementación práctica y gradual
- Enfoque de apoyo, no punitivo

---

## 🎤 **CIERRE Y TRANSICIÓN A PREGUNTAS** ⏱️ *1-2 minutos*

### **🎤 GUIÓN DE CIERRE:**

> *"En resumen, he desarrollado un sistema de Machine Learning que puede identificar con 93.2% de precisión qué estudiantes están en riesgo de desertar, con tiempo suficiente para implementar intervenciones efectivas."*

> *"Este proyecto no solo demuestra la aplicación práctica de Machine Learning en problemas reales, sino que puede tener un impacto directo en las vidas de cientos de estudiantes cada año."*

> *"La deserción estudiantil es un problema complejo, pero con las herramientas correctas y un enfoque basado en datos, podemos hacer una diferencia significativa."*

> *"Ahora estaré encantado de responder cualquier pregunta que tengan sobre la metodología, los resultados, o la implementación del sistema."*

---

## ❓ **RESPUESTAS A PREGUNTAS FRECUENTES**

### **P: "¿Cómo garantizan que el modelo no tenga sesgos?"**
> *"Excelente pregunta. Evalué el rendimiento del modelo en diferentes subgrupos demográficos y usé SMOTE para balancear las clases. También utilicé múltiples métricas, no solo accuracy, para detectar posibles sesgos. En una implementación real, sería crucial monitorear continuamente el rendimiento por género, edad, y otros grupos."*

### **P: "¿Qué pasa con la privacidad de los datos?"**
> *"La privacidad es fundamental. El sistema trabajaría con datos ya recolectados por la universidad para fines académicos. Las predicciones serían accesibles solo para consejeros académicos autorizados, y se implementarían todas las medidas de seguridad necesarias según las políticas de la UPC."*

### **P: "¿Cuánto tiempo toma entrenar el modelo?"**
> *"Con los datos actuales, el entrenamiento toma aproximadamente 2-3 minutos en una computadora estándar. Para uso en producción, el modelo se re-entrenaría cada semestre con los nuevos datos, manteniendo siempre la precisión actualizada."*

### **P: "¿Funciona para otras universidades?"**
> *"El marco metodológico es transferible, pero cada universidad tendría que entrenar el modelo con sus propios datos, ya que los factores de deserción pueden variar según el contexto institucional, socioeconómico y geográfico."*

---

## 📊 **MATERIAL DE APOYO PARA MOSTRAR**

### **Durante la presentación, ten listos:**
- `benchmark_visualizacion.png` - Para slide de algoritmos
- `resumen_dataset_completo.png` - Para explicar el dataset  
- `curva_roc.png` - Para explicar el rendimiento del modelo
- `matriz_confusion_umbral_optimo.png` - Para mostrar precisión
- `top20_importancias.png` - Para factores predictivos

### **Como backup:**
- `RESUMEN_EJECUTIVO.md` - Para datos específicos
- `benchmark_completo_algoritmos.csv` - Para métricas exactas
- Laptop con scripts ejecutables para demo en vivo (opcional)

---

## ⏱️ **GESTIÓN DEL TIEMPO**

### **Si te quedas sin tiempo:**
- Prioriza: Problemática → Resultados principales → Impacto
- Omite: Detalles técnicos de preprocesamiento
- Resume: Metodología en 2 minutos máximo

### **Si tienes tiempo extra:**
- Muestra demo en vivo del código
- Profundiza en casos de uso específicos
- Discute limitaciones y trabajo futuro

---

**¡Éxito en tu presentación! 🚀**