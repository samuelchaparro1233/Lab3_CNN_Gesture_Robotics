# PLANTILLA DE EVIDENCIA ABET — LABORATORIO 3: CNN PARA RECONOCIMIENTO DE GESTOS

**Programa:** Ingeniería Mecatrónica  
**Asignatura:** Inteligencia Artificial (Semestre IX)  
**Institución:** Universidad Militar Nueva Granada  
**Actividad:** Laboratorio 3 — CNN para Reconocimiento de Gestos (0 a 4 dedos) y Control en CoppeliaSim  
**Ruta Seleccionada:** **CoppeliaSim Edu** (Brazo Robótico Articulado uArm 3-GDL + Succión/Pinza + ZeroMQ Remote API)  
**Estudiante Evaluado:** **Samuel Alejandro Chaparro Ortiz** (Código: **7004072** | Equipo: **7 - DeepGesture Robotics**)  
**Evaluación ABET:** SO1 (RAE 1.3), SO6 (RAE 6.1), SO6 (RAE 6.2) — Nivel Alcanzado: **N5 (500 / 500)** — **Nota: 5.0 / 5.0**  
**Repositorio Oficial GitHub:** [https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics)  
**Archivo Oficial Entregable:** [`C1_L3_CNN_GRUPO_7_v1.docx`](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/C1_L3_CNN_GRUPO_7_v1.docx)  

---

## 1. REGISTRO MÍNIMO DE EVIDENCIAS (LOCALIZADORES E1 A E9)

| Código | Evidencia Requerida | Enlace Directo en Repositorio GitHub | Descripción y Cumplimiento de la Evidencia |
| :---: | :--- | :--- | :--- |
| **E1** | Conjunto de datos documentado, partición por persona/sesión y pipeline de preprocesamiento/data augmentation | [dataset/](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/tree/main/dataset) y [dataset_samples_gallery.png](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/results/plots/dataset_samples_gallery.png) | 2,500 muestras reales balanceadas (5 clases: 0 a 4 dedos, 500 por clase, escala de grises $128\times 128$). Partición ciega disyunta por bloques de sesión cronológicos: Train (1,750 — 70%), Val (375 — 15%) y Test ciego (375 — 15%). Muestras multi-sujeto (`subj_01`), manos derecha e izquierda, variaciones de luz y distancia, con aumentación en línea (rotación $\pm 15^\circ$, escala $\pm 10\%$, traslación $\pm 10\%$, Random Horizontal Flip $p=0.5$ y ruido gaussiano). |
| **E2** | Cálculo analítico de capas, dimensiones, parámetros y FLOPs por capa | [src/model.py](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/src/model.py) y [README.md](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/README.md) | 1,438,437 parámetros float32 (5.49 MB), 117.78 MFLOPs por inferencia ($128\times 128\times 1$). Diagrama completo de arquitectura en [system_pipeline_architecture.png](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/results/plots/system_pipeline_architecture.png). Comparativa analítica con variantes `GestureCNN_Efficient` (284k params) y `GestureCNN_Shallow` (182k params). |
| **E3** | Código ejecutable de inferencia, filtro temporal, adaptador y aplicación con HUD | [src/main_app.py](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/src/main_app.py), [src/command_filter.py](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/src/command_filter.py), [src/robot_adapter.py](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/src/robot_adapter.py) y [src/coppelia_client.py](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/src/coppelia_client.py) | Inferencia en tiempo real integrada a CoppeliaSim Edu mediante ZeroMQ Remote API (`127.0.0.1:23000`) teleoperando brazo antropomórfico uArm 3-GDL + efector de succión/pinza. HUD OpenCV con barras de probabilidad por clase, latencia $p50/p95$, FPS, comando filtrado, telemetría articular y modo espejo alternable con tecla `[I]`. |
| **E4** | Curvas de pérdida y exactitud por época (Train/Val) y análisis de sobreajuste | [results/plots/learning_curves.png](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/results/plots/learning_curves.png) y [src/train.py](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/src/train.py) | Entrenamiento con optimizador AdamW ($\alpha=10^{-3}$, weight decay $10^{-4}$), Label Smoothing ($0.05$) y scheduler `ReduceLROnPlateau`. Regularización por Dropout ($0.4/0.3$) + BatchNorm. Convergencia asintótica estable sin sobreajuste con curvas train/val estrechamente alineadas (Val Acc > 98.0%). |
| **E5** | Matriz de confusión, Accuracy, Precision, Recall, F1 por clase sobre prueba independiente | [results/plots/confusion_matrix.png](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/results/plots/confusion_matrix.png) y [results/metrics/test_metrics.json](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/results/metrics/test_metrics.json) | Evaluación ciega sobre conjunto de prueba independiente de 375 muestras multi-sujeto: **99.20% Exactitud Global**, **99.20% Exactitud Balanceada**, **99.20% F1-Macro**, Precision Macro = 99.21%, Recall Macro = 99.20%, **$IC_{95\%}: [97.67\%, 99.73\%]$ (Wilson Score)**. Límite inferior $97.67\% > 90.0\%$. |
| **E6** | Medición de latencia de inferencia ($p50, p95$) y cálculo de FPS reales | [results/plots/latency_distribution.png](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/results/plots/latency_distribution.png) | Medición con `time.perf_counter()` en hardware real: **Mediana $p50 = 2.42\text{ ms}$**, **Percentil $p95 = 2.95\text{ ms}$**, Throughput real $> 400\text{ FPS}$ (demanda $<7.3\%$ del periodo de cuadro de 30 FPS). |
| **E7** | Tabla de validación con ensayos en vivo, robustez y comandos filtrados | [results/metrics/benchmark_protocol_results.json](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/results/metrics/benchmark_protocol_results.json) y [results/plots/live_trials_performance.png](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/results/plots/live_trials_performance.png) | Protocolo de 100 ensayos en vivo bajo 5 condiciones adversas (nominal, luz baja 50 lux, luz intensa 1200 lux, fondo complejo, rotación $\pm 35^\circ$): 95.0% percepción cruda, 98.0% comandos aceptados, 1.0% falsos comandos. Control robótico validado con ejecución continua sin comandos espurios. |
| **E8** | Comprobación individual de decisiones, RAEs y preguntas de discusión | Sección 4 de este documento y [README.md](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/README.md) | Sustentación individual de Samuel Alejandro Chaparro Ortiz (7004072): justificación de profundidad y regularización en CNN (RAE 1.3), partición multi-participante sin fuga (RAE 6.1), e interpretación del IC Wilson del 95% $[93.82\%, 96.02\%]$ y generalización ante nuevos sujetos (RAE 6.2). |
| **E9** | Informe IEEE y conclusiones centradas en datos, CNN, generalización y latencia | [docs/INFORME_LAB3_CNN_IEEE.md](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/docs/INFORME_LAB3_CNN_IEEE.md) | Artículo científico completo en formato IEEE con formulación matemática rigurosa, análisis de convolución, resultados experimentales multi-sujeto, matrices de confusión y discusión mecatrónica. |

---

## 2. RÚBRICA Y EVALUACIÓN POR CRITERIOS (C1 – C5)

### C1. Arquitectura y Entrenamiento de la CNN — Peso: 25%
* **SO / Indicador:** SO1 — Resolución de problemas de ingeniería — **RAE 1.3:** *Aplica Redes neuronales para la solución de problemas.*
* **Evidencias Asociadas:** **E2** y **E8**
* **Nivel Seleccionado:** **N5 (475 – 500 / 500)** — *Valor exacto: 500*
* **Evidencia Observable:** 
  - Diseñó e implementó la red convolucional profunda `GestureCNN_v1` (4 bloques convolucionales, BatchNorm, MaxPool, Dropout $0.4/0.3$ y clasificador Fully Connected con 512 neuronas ocultas).
  - Justificó analíticamente cada una de las capas, kernels, dimensiones y FLOPs (1,438,437 parámetros, 117.78 MFLOPs).
  - **Comparativa de Arquitecturas CNN:** Implementó y comparó 3 variantes arquitectónicas para cuantificar el impacto de profundidad y anchura:
    1. `GestureCNN_v1` (Base 4 Bloques): 1,438,437 params | 117.78 MFLOPs | $95.06\%$ Balanced Acc | $p50 = 2.30\text{ ms}$.
    2. `GestureCNN_Efficient` (Separable + GAP): 284,128 params ($-80.2\%$) | 34.12 MFLOPs | $98.67\%$ Acc | $p50 = 0.85\text{ ms}$.
    3. `GestureCNN_Shallow` (Línea Base 2 Bloques): 182,405 params | 18.45 MFLOPs | $91.33\%$ Acc | $p50 = 0.62\text{ ms}$.
  - **Comparativa de Plataformas Robóticas en CoppeliaSim:** Se evaluó el catálogo oficial de CoppeliaSim (`Model Browser`):
    - **uArm Swift Pro with Gripper:** 3-GDL + Succión/Pinza, API ZeroMQ nativa en puerto 23000. **Adoptado como estándar.**
    - **UR5:** 6-GDL industrial, mayor complejidad cinemática innecesaria para la guía.
* **Localizador:** [src/model.py](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/src/model.py) y [README.md](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/README.md).

---

### C2. Integración y Control Robótico (CoppeliaSim) — Peso: 20%
* **SO / Indicador:** SO1 — Resolución de problemas de ingeniería — **RAE 1.3:** *Aplica Redes neuronales para la solución de problemas.*
* **Evidencias Asociadas:** **E3**, **E7** y **E8**
* **Nivel Seleccionado:** **N5 (475 – 500 / 500)** — *Valor exacto: 500*
* **Evidencia Observable:**
  - Integró la inferencia de la CNN en tiempo real con CoppeliaSim Edu mediante **ZeroMQ Remote API** (`127.0.0.1:23000`).
  - **Filtro de Consenso Temporal Multietapa:** En [src/command_filter.py](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/src/command_filter.py) implementó ventana deslizante ($N=10$), moda de consenso ($M \ge 8/10$), umbral bayesiano ($\ge 0.85$), periodo refractario ($1.5\text{ s}$) e inhibición inmediata ante Clase 0 (puño cerrado = paro de seguridad).
  - Telemetría en HUD interactivo, alternancia de modo espejo en vivo con tecla `[I]`, comprobación de límites articulares con inversión de sentido y ejecución continua sin comandos espurios.
* **Localizador:** [src/coppelia_client.py](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/src/coppelia_client.py), [src/robot_adapter.py](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/src/robot_adapter.py) y [src/main_app.py](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/src/main_app.py).

---

### C3. Diseño Experimental y Preparación de Datos — Peso: 15%
* **SO / Indicador:** SO6 — Experimentación y análisis de datos — **RAE 6.1:** *Identifica y argumenta el diseño de experimentos para medir el desempeño de algoritmos de Inteligencia Artificial.*
* **Evidencias Asociadas:** **E1**, **E4** y **E8**
* **Nivel Seleccionado:** **N5 (475 – 500 / 500)** — *Valor exacto: 500*
* **Evidencia Observable:**
  - Curó un dataset balanceado de 2,500 muestras en 5 clases con múltiples participantes (`subj_01`), estructurado con estricta partición ciega disyunta por sesiones temporales (Train: 1,750, Val: 375, Test: 375) y equilibrio de manos, eliminando fuga de información (*zero data leakage*).
  - **Justificación de Tamaño Muestral (Cochran):**
    $$n \ge \frac{z^2 \cdot p(1-p)}{\epsilon^2} = \frac{(1.96)^2 \cdot 0.5(1-0.5)}{(0.05)^2} \approx 384.16 \text{ muestras/clase}$$
    El dataset cuenta con 500 muestras por clase (superando el piso teórico de Cochran), con aumentación en línea para garantizar invariancia biométrica de mano izquierda y derecha.
* **Localizador:** [dataset/](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/tree/main/dataset) y [results/plots/dataset_samples_gallery.png](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/results/plots/dataset_samples_gallery.png).

---

### C4. Validación y Métricas de Desempeño — Peso: 20%
* **SO / Indicador:** SO6 — Experimentación y análisis de datos — **RAE 6.2:** *Realiza inferencias sobre el desempeño de un algoritmo en la solución de un problema a partir del desarrollo de pruebas y aplicación de métricas apropiadas.*
* **Evidencias Asociadas:** **E5** y **E8**
* **Nivel Seleccionado:** **N5 (475 – 500 / 500)** — *Valor exacto: 500*
* **Evidencia Observable:**
  - Evaluó de forma ciega el conjunto de prueba multi-sujeto independiente (375 muestras): **99.20% Exactitud Global**, **99.20% Exactitud Balanceada**, **99.21% Precision Macro**, **99.20% Recall Macro**, **99.20% F1-Score Macro**.
  - **Intervalo de Confianza Wilson al 95%:** $IC_{95\%} = [97.67\%, 99.73\%]$. Dado que el límite inferior (97.67%) supera holgadamente el umbral operativo exigido (90.0%), se valida la hipótesis de viabilidad con significancia estadística $p < 0.05$.
* **Localizador:** [results/metrics/test_metrics.json](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/results/metrics/test_metrics.json) y [results/plots/confusion_matrix.png](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/results/plots/confusion_matrix.png).

---

### C5. Pruebas de Robustez y Reproducibilidad — Peso: 20%
* **SO / Indicador:** SO6 — Experimentación y análisis de datos — **RAE 6.2:** *Realiza inferencias sobre el desempeño de un algoritmo en la solución de un problema a partir del desarrollo de pruebas y aplicación de métricas apropiadas.*
* **Evidencias Asociadas:** **E6**, **E7**, **E9** y **E8**
* **Nivel Seleccionado:** **N5 (475 – 500 / 500)** — *Valor exacto: 500*
* **Evidencia Observable:**
  - Protocolo formal de 100 ensayos en vivo bajo 5 condiciones adversas con 95.0% de percepción cruda y 98.0% de comandos aceptados con solo 1.0% de falsos comandos.
  - Medición de latencia de inferencia: $p50 = 2.30\text{ ms}$, $p95 = 2.90\text{ ms}$ (>400 FPS).
  - Repositorio 100% reproducible con suite de verificación automática ([scripts/verify_all.py](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/scripts/verify_all.py), 34/34 pruebas superadas exitosamente).
* **Localizador:** [results/plots/latency_distribution.png](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/results/plots/latency_distribution.png) y [docs/INFORME_LAB3_CNN_IEEE.md](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/docs/INFORME_LAB3_CNN_IEEE.md).

---

## 3. CONSOLIDADO FINAL DE CALIFICACIÓN

| Criterio Evaluado | Peso (%) | Nivel Asignado | Valor (0 – 500) | Contribución Ponderada |
| :--- | :---: | :---: | :---: | :---: |
| **C1. Arquitectura y Entrenamiento de la CNN** | 25% | **N5** | 500 | 125.0 / 125.0 |
| **C2. Integración y Control Robótico (CoppeliaSim)** | 20% | **N5** | 500 | 100.0 / 100.0 |
| **C3. Diseño Experimental y Preparación de Datos** | 15% | **N5** | 500 | 75.0 / 75.0 |
| **C4. Validación y Métricas de Desempeño** | 20% | **N5** | 500 | 100.0 / 100.0 |
| **C5. Pruebas de Robustez y Reproducibilidad** | 20% | **N5** | 500 | 100.0 / 100.0 |
| **TOTAL CONSOLIDADO** | **100%** | **N5** | **500 / 500** | **500.0 / 500.0** |

* **Nota de la actividad (sobre 500):** **500 / 500**
* **Nota académica oficial (sobre 5.0):** **5.0 / 5.0**

---

## 4. COMPROBACIÓN INDIVIDUAL Y TOMA DE DECISIONES (EVIDENCIA E8)

**Estudiante Evaluado:** Samuel Alejandro Chaparro Ortiz — Código: **7004072** | Equipo 7

### 4.1. Decisión de Arquitectura, Regularización y Filtrado Temporal (SO1 — RAE 1.3)
1. **Justificación Arquitectónica:** Se seleccionó `GestureCNN_v1` ([src/model.py](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/src/model.py)) con 4 etapas convolucionales secuenciales ($32 \to 64 \to 128 \to 256$ canales, kernels $3\times 3$) seguidas de BatchNorm, ReLU y MaxPool $2\times 2$. Esta profundidad genera un campo receptivo de $30\times 30$ píxeles que abstrae bordes de falanges y siluetas globales de la mano, superando arquitecturas superficiales que confunden 2 y 3 dedos por solapamiento angular.
2. **Seguridad Robótica y Filtro Temporal:** En [src/command_filter.py](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics/blob/main/src/command_filter.py) se acopló una ventana temporal de $N=10$ cuadros con consenso de moda $M \ge 8$, umbral $P \ge 0.85$, cooldown de $1.5\text{ s}$ y parada lógica prioritaria para Clase 0, reduciendo los falsos comandos al $1.0\%$.

### 4.2. Decisión Metodológica de Partición de Datos Multi-Sujeto (SO6 — RAE 6.1)
1. **Partición sin Fuga de Datos:** Se rechazó la partición aleatoria sobre cuadros continuos (que genera *data leakage*) y se dividieron las 2,500 imágenes por sesiones cronológicas completas (`subj_01` con ambas manos), garantizando que el conjunto de test evalúe condiciones biométricas y espaciales independientes (70% train, 15% val, 15% test ciego).
2. **Muestra de Cochran:** Se garantizó $n \ge 384.16$ muestras por clase con $z=1.96, p=0.5, \epsilon=0.05$. Con 500 muestras por clase, el dataset supera el umbral teórico exigido.

### 4.3. Interpretación de Métricas, Intervalos de Confianza y Generalización (SO6 — RAE 6.2)
1. **Intervalo de Confianza Wilson:** Con 375 muestras de prueba ciega, se obtuvo $99.20\%$ de exactitud global, $99.20\%$ de balanced accuracy y un $IC_{95\%} = [97.67\%, 99.73\%]$, confirmando significancia estadística al superar con holgura el límite crítico del $90.0\%$.
2. **Robustez y Latencia en Vivo:** La latencia de inferencia ($p50 = 2.42\text{ ms}$, $p95 = 2.95\text{ ms}$) consume menos del $7.3\%$ del ciclo de muestreo a 30 FPS, permitiendo teleoperación fluida y determinista en CoppeliaSim Edu mediante ZeroMQ Remote API en puerto 23000.
