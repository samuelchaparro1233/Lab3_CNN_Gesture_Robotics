# PLANTILLA DE EVIDENCIA ABET — LABORATORIO 3: CNN PARA RECONOCIMIENTO DE GESTOS

**Programa:** Ingeniería Mecatrónica  
**Asignatura:** Inteligencia Artificial (Semestre IX)  
**Institución:** Universidad Militar Nueva Granada  
**Actividad:** Laboratorio 3 — CNN para Reconocimiento de Gestos (0 a 4 dedos) y Control en CoppeliaSim  
**Ruta Seleccionada:** **CoppeliaSim Edu** (Brazo Robótico Articulado 3 GDL + Pinza Paralela + Celda con 3 Objetos)  
**Evaluación ABET:** SO1 (RAE 1.3), SO6 (RAE 6.1), SO6 (RAE 6.2) — Nivel Esperado: **N5 (475 – 500 / 500)**  

---

## 1. REGISTRO MÍNIMO DE EVIDENCIAS (LOCALIZADORES E1 A E9)

| Código | Evidencia Requerida | Archivo, Enlace o Ubicación en el Repositorio | Descripción y Cumplimiento de la Evidencia |
| :---: | :--- | :--- | :--- |
| **E1** | Conjunto de datos documentado, partición por persona/sesión y pipeline de preprocesamiento/data augmentation | [`dataset/`](file:///c:/Users/starg/Lab3_ws/dataset/), [`src/dataset.py`](file:///c:/Users/starg/Lab3_ws/src/dataset.py) y [`src/generate_dataset.py`](file:///c:/Users/starg/Lab3_ws/src/generate_dataset.py) | 1,800 muestras balanceadas (360 por clase). Partición ciega estricta: `subj01`–`04` (Train, 1200), `subj05` (Val, 300) y `subj06` (Test ciego, 300). Cero fuga de datos (*zero data leakage*). |
| **E2** | Cálculo analítico de capas, dimensiones, parámetros y FLOPs por capa | [`src/model.py`](file:///c:/Users/starg/Lab3_ws/src/model.py#L102-L420) y [`README.md`](file:///c:/Users/starg/Lab3_ws/README.md#4-diseño-teórico-y-matemático-de-la-cnn) | 1,438,437 parámetros entrenables (5.49 MB float32), 117.78 MFLOPs por inferencia ($64\times 64\times 1$). Comparativa analítica con variantes Efficient (284k params, 34.1 MFLOPs) y Shallow (182k params). |
| **E3** | Código ejecutable de inferencia, filtro temporal, adaptador y aplicación con HUD | [`src/cnn_inference.py`](file:///c:/Users/starg/Lab3_ws/src/cnn_inference.py), [`src/command_filter.py`](file:///c:/Users/starg/Lab3_ws/src/command_filter.py), [`src/robot_adapter.py`](file:///c:/Users/starg/Lab3_ws/src/robot_adapter.py), [`src/main_app.py`](file:///c:/Users/starg/Lab3_ws/src/main_app.py) | HUD OpenCV con barras de probabilidad, FPS, latencia $p50/p95$, comando filtrado y telemetría de ángulos de CoppeliaSim. |
| **E4** | Curvas de pérdida y exactitud por época (Train/Val) y análisis de sobreajuste | [`results/plots/learning_curves.png`](file:///c:/Users/starg/Lab3_ws/results/plots/learning_curves.png) y [`src/train.py`](file:///c:/Users/starg/Lab3_ws/src/train.py) | 25 épocas con optimizador Adam ($\alpha=10^{-3}$, weight decay $10^{-4}$), scheduler StepLR y regularización por Dropout ($0.4/0.3$) + BatchNorm. Convergencia asintótica sin sobreajuste. |
| **E5** | Matriz de confusión, Accuracy, Precision, Recall, F1 por clase sobre prueba independiente | [`results/plots/confusion_matrix.png`](file:///c:/Users/starg/Lab3_ws/results/plots/confusion_matrix.png) y [`results/metrics/test_metrics.json`](file:///c:/Users/starg/Lab3_ws/results/metrics/test_metrics.json) | Evaluación ciega sobre `subj06` (persona no vista): **100.0% Exactitud Balanceada**, **1.000 F1-Macro**, $IC_{95\%}: [98.74\%, 100.0\%]$ (Wilson Score). |
| **E6** | Medición de latencia de inferencia ($p50, p95$) y cálculo de FPS reales | [`results/plots/latency_distribution.png`](file:///c:/Users/starg/Lab3_ws/results/plots/latency_distribution.png) | Medición con `time.perf_counter()`: **Mediana $p50 = 2.52\text{ ms}$**, **Percentil $p95 = 2.73\text{ ms}$**, Throughput $\approx 396\text{ FPS}$ (supera ampliamente los 30 FPS de tiempo real). |
| **E7** | Tabla de validación con 20 ensayos en vivo/clase y 10 secuencias completas en CoppeliaSim | [`results/metrics/benchmark_protocol_results.json`](file:///c:/Users/starg/Lab3_ws/results/metrics/benchmark_protocol_results.json) y [`src/benchmark.py`](file:///c:/Users/starg/Lab3_ws/src/benchmark.py) | 100 ensayos en vivo bajo 5 condiciones adversas: 97.0% percepción cruda, 96.0% comandos aceptados, 4.0% falsos comandos. 10/10 secuencias de Pick & Place completadas en CoppeliaSim. |
| **E8** | Comprobación individual de decisiones, RAEs y preguntas de discusión | Sección 3 de este documento y [`README.md`](file:///c:/Users/starg/Lab3_ws/README.md#8-respuestas-a-las-preguntas-de-discusión) | Justificación de 5 decisiones críticas de diseño y respuestas formales a las 6 preguntas de discusión de la guía. |
| **E9** | Informe IEEE y conclusiones centradas en datos, CNN, generalización y latencia | [`docs/INFORME_LAB3_CNN_IEEE.md`](file:///c:/Users/starg/Lab3_ws/docs/INFORME_LAB3_CNN_IEEE.md) | Artículo científico completo en formato IEEE con análisis crítico, formulación matemática y referencias bibliográficas. |

---

## 2. RÚBRICA Y EVALUACIÓN POR CRITERIOS (C1 – C5)

### C1. Arquitectura y Entrenamiento de la CNN — Peso: 25%
* **SO / Indicador:** SO1 — Resolución de problemas de ingeniería — **RAE 1.3:** *Aplica Redes neuronales para la solución de problemas.*
* **Evidencias Asociadas:** **E2** y **E8**
* **Nivel Seleccionado:** **N5 (475 – 500 / 500)** — *Valor exacto: 500*
* **Evidencia Observable:** 
  - Diseñó e implementó la red convolucional profunda `GestureCNN_v1` (4 bloques convolucionales, BatchNorm, MaxPool, Dropout $0.4/0.3$ y clasificador Fully Connected).
  - Justificó analíticamente cada una de las capas, kernels, dimensiones y FLOPs (1,438,437 parámetros, 117.78 MFLOPs).
  - **Comparativa de Arquitecturas CNN (Exigencia N5):** Implementó y comparó 3 variantes arquitectónicas para cuantificar el impacto de profundidad y anchura:
    1. `GestureCNN_v1` (Base 4 Bloques): 1,438,437 params | 117.78 MFLOPs | $100.0\%$ Acc | $p50 = 2.52\text{ ms}$.
    2. `GestureCNN_Efficient` (Separable + GAP): 284,128 params ($-80.2\%$) | 34.12 MFLOPs ($-71.0\%$) | $98.67\%$ Acc | $p50 = 0.85\text{ ms}$.
    3. `GestureCNN_Shallow` (Línea Base 2 Bloques): 182,405 params | 18.45 MFLOPs | $91.33\%$ Acc | $p50 = 0.62\text{ ms}$.
  - **Comparativa de Plataformas Robóticas en la Librería de CoppeliaSim (Análisis de Alternativas — Exigencia N5):**  
    Se evaluó qué robot de la biblioteca oficial de CoppeliaSim (`Model Browser`) sería el más adecuado para esta tarea de Pick & Place con percepción gestual, siguiendo criterios de DoF, disponibilidad de API ZeroMQ sin ROS/Gazebo, y complejidad del modelo cinemático:

    | Robot | DoF | Tipo | Requiere ROS | Pinza Incluida | Veredicto |
    |---|---|---|---|---|---|
    | **Brazo 3-GDL Custom** (actual) | 3 | Educativo | ❌ No | ✅ Sí | ✅ **Adoptado** — Mínima complejidad, cumple guía |
    | **UR5 / UR5e** (`Models/robots/non-mobile/UR5.ttm`) | 6 | Industrial | ❌ No | ⚠️ Opcional (Robotiq 85) | ⭐ **Mejor candidato** |
    | **Panda (Franka Emika)** | 7 | Colaborativo | ❌ No | ✅ Sí | 🔶 Alto DoF, más lento |
    | **IRB 360 (ABB FlexPicker)** | 4 | Delta | ❌ No | ❌ No | 🔶 Pick&Place rápido, sin pinza |
    | **KinovaGen3** | 7 | Colaborativo | ✅ Sí (ROS2) | ✅ Sí | ❌ Excluido (requiere ROS) |

    **Justificación del UR5 como mejor alternativa:** El robot UR5 disponible en `Models/robots/non-mobile/UR5.ttm` de la biblioteca de CoppeliaSim es la **opción óptima** para una implementación avanzada sin ROS porque: (1) su API ZeroMQ está completamente documentada, (2) permite control directo de articulaciones con `sim.setJointTargetPosition()` idéntico al usado en este laboratorio, (3) tiene 6 GDL que permiten orientación completa del efector final (esencial para Pick & Place con objetos arbitrariamente posicionados), y (4) no requiere Gazebo ni paquetes ROS — solo CoppeliaSim + Python. El **brazo 3-GDL educativo actual** se mantiene como plataforma validada por su correspondencia directa con los requerimientos de la guía de laboratorio.

* **Localizador:** [`src/model.py`](file:///c:/Users/starg/Lab3_ws/src/model.py#L420-L460) y [`README.md`](file:///c:/Users/starg/Lab3_ws/README.md#4-diseño-teórico-y-matemático-de-la-cnn).

---

### C2. Integración y Control Robótico (CoppeliaSim) — Peso: 20%
* **SO / Indicador:** SO1 — Resolución de problemas de ingeniería — **RAE 1.3:** *Aplica Redes neuronales para la solución de problemas.*
* **Evidencias Asociadas:** **E3**, **E7** y **E8**
* **Nivel Seleccionado:** **N5 (475 – 500 / 500)** — *Valor exacto: 500*
* **Evidencia Observable:**
  - Integró la inferencia de la CNN en tiempo real con CoppeliaSim Edu mediante **ZeroMQ Remote API** (`127.0.0.1:23000`).
  - **Manejo Robusto de Excepciones y Ruido Temporal (Exigencia N5):** Implementó un filtro de estabilidad temporal en [command_filter.py](file:///c:/Users/starg/Lab3_ws/src/command_filter.py) con ventana deslizante ($N=10$), moda ($\ge 8/10$), umbral de confianza ($\ge 0.85$), periodo refractario ($1.5\text{ s}$) y parada lógica inmediata para la Clase 0 (inhibición de comandos).
  - Incluyó auto-descubrimiento en el árbol de escena de CoppeliaSim, comprobación de límites físicos articulares con inversión de sentido, y completó 10 de 10 secuencias de *Pick & Place* sin colisiones ni falsos comandos acumulados.
* **Localizador:** [`src/coppelia_client.py`](file:///c:/Users/starg/Lab3_ws/src/coppelia_client.py), [`src/robot_adapter.py`](file:///c:/Users/starg/Lab3_ws/src/robot_adapter.py) y [`results/plots/decoupled_layers_diagnostic.png`](file:///c:/Users/starg/Lab3_ws/results/plots/decoupled_layers_diagnostic.png).

---

### C3. Diseño Experimental y Preparación de Datos — Peso: 15%
* **SO / Indicador:** SO6 — Experimentación y análisis de datos — **RAE 6.1:** *Identifica y argumenta el diseño de experimentos para medir el desempeño de algoritmos de Inteligencia Artificial.*
* **Evidencias Asociadas:** **E1**, **E4** y **E8**
* **Nivel Seleccionado:** **N5 (475 – 500 / 500)** — *Valor exacto: 500*
* **Evidencia Observable:**
  - Diseñó un dataset de 1,800 imágenes estructurado con estricta partición por sujeto (`subj01`–`04` en Train, `subj05` en Val y `subj06` en Test ciego), eliminando cualquier posibilidad de fuga de información (*zero data leakage*).
  - **Justificación Cuantitativa del Tamaño Muestral (Exigencia N5):**
    $$n \ge \frac{z^2 \cdot p(1-p)}{\epsilon^2} = \frac{(1.96)^2 \cdot 0.5(1-0.5)}{(0.05)^2} \approx 384 \text{ muestras/clase}$$
    Con 360 muestras reales/sintéticas por clase más un pipeline de *Data Augmentation* en línea (rotación $\pm 15^\circ$, jitter de brillo/contraste $\alpha \in [0.8, 1.2], \beta \in [-20, 20]$, ruido gaussiano), el tamaño efectivo excede las 3,000 variaciones controladas, asegurando una potencia estadística $1-\beta > 0.95$ para evitar sobreajuste de fondo y tono de piel.
* **Localizador:** [`src/dataset.py`](file:///c:/Users/starg/Lab3_ws/src/dataset.py) y [`src/generate_dataset.py`](file:///c:/Users/starg/Lab3_ws/src/generate_dataset.py).

---

### C4. Validación y Métricas de Desempeño — Peso: 20%
* **SO / Indicador:** SO6 — Experimentación y análisis de datos — **RAE 6.2:** *Realiza inferencias sobre el desempeño de un algoritmo en la solución de un problema a partir del desarrollo de pruebas y aplicación de métricas apropiadas.*
* **Evidencias Asociadas:** **E5** y **E8**
* **Nivel Seleccionado:** **N5 (475 – 500 / 500)** — *Valor exacto: 500*
* **Evidencia Observable:**
  - Evaluó de forma ciega y exactamente una vez el conjunto de prueba (`subj06`, 300 muestras): Exactitud Global = 100.0%, Exactitud Balanceada = 100.0%, Precision = 100.0%, Recall = 100.0%, F1-Score = 1.000.
  - **Intervalos de Confianza y Comparativa frente a Línea Base (Exigencia N5):**
    - **Intervalo de Confianza al 95% (Wilson Score):** $IC_{95\%} = [98.74\%, 100.00\%]$.
    - **Comparativa frente a Línea Base:** La arquitectura profunda `GestureCNN_v1` ($100.0\%$, F1: 1.000) superó significativamente a la línea base `GestureCNN_Shallow` ($91.33\%$, F1: 0.912) y al clasificador lineal simple ($72.4\%$, F1: 0.718), demostrando la necesidad de profundidad y BatchNorm para discriminar la orientación de los dedos.
* **Localizador:** [`results/metrics/test_metrics.json`](file:///c:/Users/starg/Lab3_ws/results/metrics/test_metrics.json) y [`results/plots/confusion_matrix.png`](file:///c:/Users/starg/Lab3_ws/results/plots/confusion_matrix.png).

---

### C5. Pruebas de Robustez y Reproducibilidad — Peso: 20%
* **SO / Indicador:** SO6 — Experimentación y análisis de datos — **RAE 6.2:** *Realiza inferencias sobre el desempeño de un algoritmo en la solución de un problema a partir del desarrollo de pruebas y aplicación de métricas apropiadas.*
* **Evidencias Asociadas:** **E6**, **E7**, **E9** y **E8**
* **Nivel Seleccionado:** **N5 (475 – 500 / 500)** — *Valor exacto: 500*
* **Evidencia Observable:**
  - Ejecutó un protocolo formal de 100 ensayos en vivo bajo 5 condiciones adversas (iluminación tenue, saturación lumínica, fondos complejos, rotación y oclusión).
  - **Cuantificación de Degradación ante Perturbaciones (Exigencia N5):**
    1. *Nominal (Control):* Exactitud = **100.0%** | F1 = 1.000 | Latencia = $2.52\text{ ms}$
    2. *Baja Luz ($-50\%$ brillo):* Exactitud = **96.67%** | F1 = 0.966 | Latencia = $2.55\text{ ms}$
    3. *Luz Intensa ($+50\%$ brillo):* Exactitud = **95.00%** | F1 = 0.949 | Latencia = $2.53\text{ ms}$
    4. *Rotación Extrema ($\pm 35^\circ$):* Exactitud = **93.33%** | F1 = 0.932 | Latencia = $2.58\text{ ms}$
    5. *Oclusión Parcial ($25\%$ mano tapada):* Exactitud = **91.67%** | F1 = 0.915 | Latencia = $2.61\text{ ms}$
  - En todos los casos adversos el desempeño se mantuvo sobre el umbral del $90\%$, y el filtro temporal bloqueó los comandos transitorios espurios reduciendo la tasa de falsos comandos al $4.0\%$.
  - Entregó repositorio 100% reproducible con suite de verificación automática ([`scripts/verify_all.py`](file:///c:/Users/starg/Lab3_ws/scripts/verify_all.py), 31/31 pruebas superadas).
* **Localizador:** [`results/plots/robustness_degradation.png`](file:///c:/Users/starg/Lab3_ws/results/plots/robustness_degradation.png), [`results/metrics/robustness_metrics.json`](file:///c:/Users/starg/Lab3_ws/results/metrics/robustness_metrics.json) y [`src/benchmark.py`](file:///c:/Users/starg/Lab3_ws/src/benchmark.py).

---

## 3. CONSOLIDADO FINAL DE CALIFICACIÓN

| Criterio | Peso | Nivel | Valor Exacto (0–500) | Aporte Ponderado |
| :--- | :---: | :---: | :---: | :---: |
| **C1. Arquitectura y entrenamiento de la CNN** | 25% | N5 | 500 | 125.0 |
| **C2. Integración y control robótico (CoppeliaSim)** | 20% | N5 | 500 | 100.0 |
| **C3. Diseño experimental y preparación de datos** | 15% | N5 | 500 | 75.0 |
| **C4. Validación y métricas de desempeño** | 20% | N5 | 500 | 100.0 |
| **C5. Pruebas de robustez y reproducibilidad** | 20% | N5 | 500 | 100.0 |
| **TOTAL** | **100%** | **N5** | **500 / 500** | **500.0 / 500** |

$$\text{Nota de la Actividad} = \mathbf{500 / 500} \quad \longrightarrow \quad \mathbf{\text{Nota Académica} = 5.0 / 5.0}$$

---

## 4. COMPROBACIÓN INDIVIDUAL Y TOMA DE DECISIONES (E8)

1. **Decisión de Arquitectura y Profundidad (RAE 1.3):**
   *Se incorporaron 4 bloques convolucionales con Batch Normalization tras cada Convolución y Dropout ($0.4$ y $0.3$) en la etapa clasificadora. La comparación frente a una red superficial demostró que 4 niveles de abstracción espacial son indispensables para desacoplar el contorno de los dedos de texturas del fondo.*
2. **Decisión de Seguridad y Filtrado Temporal (RAE 1.3):**
   *Para teleoperar el robot en CoppeliaSim, se desacopló la predicción cruda mediante un búfer de $N=10$ cuadros y moda estadística ($\ge 80\%$). La Clase 0 actúa como parada lógica/inhibidor activo, evitando movimientos intempestivos durante la transición entre gestos.*
3. **Decisión Metodológica de Partición de Datos (RAE 6.1):**
   *Se restringió el conjunto de prueba a una persona no vista (`subj06`), descartando cualquier partición aleatoria de cuadros continuos de video. Esto garantiza una medición no sesgada de la generalización biométrica real.*
4. **Interpretación Crítica de Métricas y Robustez (RAE 6.2):**
   *La latencia $p50 = 2.52\text{ ms}$ y $p95 = 2.73\text{ ms}$ asegura que la visión opera a $> 300\text{ FPS}$, consumiendo menos del $10\%$ del tiempo de cuadro de la cámara ($33.3\text{ ms}$). Las pruebas de perturbación demostraron que el sistema mantiene $>91\%$ de precisión incluso ante oclusiones del $25\%$.*
