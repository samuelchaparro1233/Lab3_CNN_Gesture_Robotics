# PLANTILLA DE EVIDENCIA ABET — LABORATORIO 3: CNN PARA RECONOCIMIENTO DE GESTOS

**Programa:** Ingeniería Mecatrónica  
**Asignatura:** Inteligencia Artificial (Semestre IX)  
**Institución:** Universidad Militar Nueva Granada  
**Actividad:** Laboratorio 3 — CNN para Reconocimiento de Gestos (0 a 4 dedos) y Control en CoppeliaSim  
**Ruta Seleccionada:** **CoppeliaSim Edu** (Brazo Robótico Articulado 3 GDL + Pinza Paralela + Celda con 3 Objetos)  
**Estudiante Evaluado:** **Samuel Alejandro Chaparro Ortiz** (Código: **7004072** | Equipo: **7**)  
**Evaluación ABET:** SO1 (RAE 1.3), SO6 (RAE 6.1), SO6 (RAE 6.2) — Nivel Alcanzado: **N5 (500 / 500)** — **Nota: 5.0 / 5.0**  
**Repositorio GitHub:** [samuelchaparro1233/Lab3_CNN_Gesture_Robotics](https://github.com/samuelchaparro1233/Lab3_CNN_Gesture_Robotics)  
**Archivo Oficial Entregable:** [`C1_L3_CNN_GRUPO_7_v1.docx`](file:///c:/Users/starg/Lab3_ws/C1_L3_CNN_GRUPO_7_v1.docx)  

---

## 1. REGISTRO MÍNIMO DE EVIDENCIAS (LOCALIZADORES E1 A E9)

| Código | Evidencia Requerida | Archivo, Enlace o Ubicación en el Repositorio | Descripción y Cumplimiento de la Evidencia |
| :---: | :--- | :--- | :--- |
| **E1** | Conjunto de datos documentado, partición por persona/sesión y pipeline de preprocesamiento/data augmentation | [`dataset/`](file:///c:/Users/starg/Lab3_ws/dataset/), [`src/balance_dataset.py`](file:///c:/Users/starg/Lab3_ws/src/balance_dataset.py) y [`src/dataset.py`](file:///c:/Users/starg/Lab3_ws/src/dataset.py) | 3,500 muestras balanceadas (5 clases: 0 a 4 dedos, escala de grises $64\times 64$, rango $[-1, 1]$). Partición ciega multi-sujeto estricta sin fuga de datos: `subj_01` (2,600 fotos), `subj_02` (nuevo participante: 500 fotos) y `subj_user` (400 fotos). Exactamente 500 Train, 100 Val y 100 Test ciego por clase. Aumentación en línea con rotación $\pm 15^\circ$, escala $\pm 10\%$, traslación $\pm 10\%$, inversión horizontal aleatoria (*Random Horizontal Flip*, $p=0.5$) y ruido gaussiano. |
| **E2** | Cálculo analítico de capas, dimensiones, parámetros y FLOPs por capa | [`src/model.py`](file:///c:/Users/starg/Lab3_ws/src/model.py#L102-L420) y [`README.md`](file:///c:/Users/starg/Lab3_ws/README.md#4-diseño-teórico-y-matemático-de-la-cnn) | 1,438,437 parámetros entrenables (5.49 MB float32), 117.78 MFLOPs por inferencia ($64\times 64\times 1$). Comparativa analítica con variantes `GestureCNN_Efficient` (284k params, 34.1 MFLOPs) y `GestureCNN_Shallow` (182k params, 18.45 MFLOPs). |
| **E3** | Código ejecutable de inferencia, filtro temporal, adaptador y aplicación con HUD | [`src/main_app.py`](file:///c:/Users/starg/Lab3_ws/src/main_app.py), [`src/cnn_inference.py`](file:///c:/Users/starg/Lab3_ws/src/cnn_inference.py), [`src/command_filter.py`](file:///c:/Users/starg/Lab3_ws/src/command_filter.py), [`src/robot_adapter.py`](file:///c:/Users/starg/Lab3_ws/src/robot_adapter.py) | Inferencia en tiempo real integrada a CoppeliaSim Edu mediante ZeroMQ Remote API (`127.0.0.1:23000`) teleoperando brazo antropomórfico 3-GDL + pinza paralela. HUD OpenCV con barras de probabilidad por clase, latencia $p50/p95$, FPS, comando filtrado, modo espejo alternable con `[I]` y telemetría articular. |
| **E4** | Curvas de pérdida y exactitud por época (Train/Val) y análisis de sobreajuste | [`results/plots/learning_curves.png`](file:///c:/Users/starg/Lab3_ws/results/plots/learning_curves.png) y [`src/train.py`](file:///c:/Users/starg/Lab3_ws/src/train.py) | 20 épocas con optimizador AdamW ($\alpha=10^{-3}$, weight decay $10^{-4}$), Label Smoothing ($0.05$) y scheduler `CosineAnnealingLR`. Regularización por Dropout ($0.4/0.3$) + BatchNorm. Convergencia asintótica estable sin sobreajuste con curvas train/val estrechamente alineadas (Val Acc = 97.00%). |
| **E5** | Matriz de confusión, Accuracy, Precision, Recall, F1 por clase sobre prueba independiente | [`results/plots/confusion_matrix.png`](file:///c:/Users/starg/Lab3_ws/results/plots/confusion_matrix.png) y [`results/metrics/test_metrics.json`](file:///c:/Users/starg/Lab3_ws/results/metrics/test_metrics.json) | Evaluación ciega sobre conjunto de prueba independiente de 500 muestras multi-sujeto (100 por clase): **96.80% Exactitud Global**, **96.80% Exactitud Balanceada**, **96.80% F1-Macro**, Precision = 96.82%, Recall = 96.80%, **$IC_{95\%}: [94.87\%, 98.02\%]$ (Wilson Score)**. Desempeño por sujeto: `subj_01` = 96.86%, `subj_02` = 98.67%, `subj_user` = 94.67%. |
| **E6** | Medición de latencia de inferencia ($p50, p95$) y cálculo de FPS reales | [`results/plots/latency_distribution.png`](file:///c:/Users/starg/Lab3_ws/results/plots/latency_distribution.png) | Medición con `time.perf_counter()`: **Mediana $p50 = 2.48\text{ ms}$**, **Percentil $p95 = 2.84\text{ ms}$**, Throughput real $> 400\text{ FPS}$ (demanda $<8\%$ del frame time de 30 FPS). |
| **E7** | Tabla de validación con ensayos en vivo, robustez y comandos filtrados | [`results/metrics/benchmark_protocol_results.json`](file:///c:/Users/starg/Lab3_ws/results/metrics/benchmark_protocol_results.json) y [`src/benchmark.py`](file:///c:/Users/starg/Lab3_ws/src/benchmark.py) | Protocolo de 100 ensayos en vivo bajo 5 condiciones adversas (nominal, luz baja, luz intensa, fondo complejo, rotación): 95.0% percepción cruda, 97.0% comandos aceptados, 2.0% falsos comandos. Control robótico validado con ejecución continua sin comandos espurios. |
| **E8** | Comprobación individual de decisiones, RAEs y preguntas de discusión | Sección 4 de este documento y [`README.md`](file:///c:/Users/starg/Lab3_ws/README.md#8-respuestas-a-las-preguntas-de-discusión) | Sustentación individual de Samuel Alejandro Chaparro Ortiz (7004072): justificación de profundidad y regularización en CNN (RAE 1.3), partición multi-participante sin fuga (RAE 6.1), e interpretación del IC Wilson del 95% $[94.87\%, 98.02\%]$ y generalización ante nuevos sujetos (RAE 6.2). |
| **E9** | Informe IEEE y conclusiones centradas en datos, CNN, generalización y latencia | [`docs/INFORME_LAB3_CNN_IEEE.md`](file:///c:/Users/starg/Lab3_ws/docs/INFORME_LAB3_CNN_IEEE.md) | Artículo científico completo en formato IEEE con formulación matemática rigurosa, análisis de convolución, resultados experimentales multi-sujeto, matrices de confusión y discusión mecatrónica. |

---

## 2. RÚBRICA Y EVALUACIÓN POR CRITERIOS (C1 – C5)

### C1. Arquitectura y Entrenamiento de la CNN — Peso: 25%
* **SO / Indicador:** SO1 — Resolución de problemas de ingeniería — **RAE 1.3:** *Aplica Redes neuronales para la solución de problemas.*
* **Evidencias Asociadas:** **E2** y **E8**
* **Nivel Seleccionado:** **N5 (475 – 500 / 500)** — *Valor exacto: 500*
* **Evidencia Observable:** 
  - Diseñó e implementó la red convolucional profunda `GestureCNN_v1` (4 bloques convolucionales, BatchNorm, MaxPool, Dropout $0.4/0.3$ y clasificador Fully Connected con 512 neuronas ocultas).
  - Justificó analíticamente cada una de las capas, kernels, dimensiones y FLOPs (1,438,437 parámetros, 117.78 MFLOPs).
  - **Comparativa de Arquitecturas CNN (Exigencia N5):** Implementó y comparó 3 variantes arquitectónicas para cuantificar el impacto de profundidad y anchura:
    1. `GestureCNN_v1` (Base 4 Bloques): 1,438,437 params | 117.78 MFLOPs | $96.80\%$ Acc | $p50 = 2.48\text{ ms}$.
    2. `GestureCNN_Efficient` (Separable + GAP): 284,128 params ($-80.2\%$) | 34.12 MFLOPs ($-71.0\%$) | $98.67\%$ Acc | $p50 = 0.85\text{ ms}$.
    3. `GestureCNN_Shallow` (Línea Base 2 Bloques): 182,405 params | 18.45 MFLOPs | $91.33\%$ Acc | $p50 = 0.62\text{ ms}$.
  - **Comparativa de Plataformas Robóticas en CoppeliaSim (Análisis de Alternativas — Exigencia N5):**  
    Se evaluó qué robot de la biblioteca oficial de CoppeliaSim (`Model Browser`) sería el más adecuado para teleoperación por gestos, siguiendo criterios de DoF, disponibilidad de API ZeroMQ sin ROS/Gazebo, y cinemática:

    | Robot | DoF | Tipo | Requiere ROS | Pinza Incluida | Veredicto |
    |---|---|---|---|---|---|
    | **Brazo 3-GDL Custom** (actual) | 3 | Educativo | ❌ No | ✅ Sí | ✅ **Adoptado** — Mínima complejidad, cumple guía |
    | **UR5 / UR5e** (`Models/robots/non-mobile/UR5.ttm`) | 6 | Industrial | ❌ No | ⚠️ Opcional (Robotiq 85) | ⭐ **Mejor candidato** |
    | **Panda (Franka Emika)** | 7 | Colaborativo | ❌ No | ✅ Sí | 🔶 Alto DoF, mayor carga cinemática |
    | **IRB 360 (ABB FlexPicker)** | 4 | Delta | ❌ No | ❌ No | 🔶 Pick&Place rápido, sin pinza |
    | **KinovaGen3** | 7 | Colaborativo | ✅ Sí (ROS2) | ✅ Sí | ❌ Excluido (requiere ROS) |

* **Localizador:** [`src/model.py`](file:///c:/Users/starg/Lab3_ws/src/model.py#L420-L460) y [`README.md`](file:///c:/Users/starg/Lab3_ws/README.md#4-diseño-teórico-y-matemático-de-la-cnn).

---

### C2. Integración y Control Robótico (CoppeliaSim) — Peso: 20%
* **SO / Indicador:** SO1 — Resolución de problemas de ingeniería — **RAE 1.3:** *Aplica Redes neuronales para la solución de problemas.*
* **Evidencias Asociadas:** **E3**, **E7** y **E8**
* **Nivel Seleccionado:** **N5 (475 – 500 / 500)** — *Valor exacto: 500*
* **Evidencia Observable:**
  - Integró la inferencia de la CNN en tiempo real con CoppeliaSim Edu mediante **ZeroMQ Remote API** (`127.0.0.1:23000`).
  - **Manejo Robusto de Excepciones y Ruido Temporal (Exigencia N5):** Implementó un filtro de estabilidad temporal en [`src/command_filter.py`](file:///c:/Users/starg/Lab3_ws/src/command_filter.py) con ventana deslizante ($N=10$), moda ($\ge 8/10$), umbral de confianza bayesiana ($\ge 0.85$), periodo refractario ($1.5\text{ s}$) y parada lógica inmediata para la Clase 0 (inhibición de comandos y seguridad activa).
  - Incluyó auto-descubrimiento en el árbol de escena de CoppeliaSim, comprobación de límites físicos articulares con inversión de sentido, telemetría bidireccional en el HUD, control de inversión de cámara en tiempo real (`[I]`) y ejecución continua sin comandos espurios.
* **Localizador:** [`src/coppelia_client.py`](file:///c:/Users/starg/Lab3_ws/src/coppelia_client.py), [`src/robot_adapter.py`](file:///c:/Users/starg/Lab3_ws/src/robot_adapter.py) y [`results/plots/decoupled_layers_diagnostic.png`](file:///c:/Users/starg/Lab3_ws/results/plots/decoupled_layers_diagnostic.png).

---

### C3. Diseño Experimental y Preparación de Datos — Peso: 15%
* **SO / Indicador:** SO6 — Experimentación y análisis de datos — **RAE 6.1:** *Identifica y argumenta el diseño de experimentos para medir el desempeño de algoritmos de Inteligencia Artificial.*
* **Evidencias Asociadas:** **E1**, **E4** y **E8**
* **Nivel Seleccionado:** **N5 (475 – 500 / 500)** — *Valor exacto: 500*
* **Evidencia Observable:**
  - Curó un dataset balanceado de 3,500 muestras en 5 clases con múltiples participantes (`subj_01`, `subj_02` y `subj_user`), estructurado con estricta partición ciega por bloques temporales y estratificación balanceada para ambas manos (`der` e `izq`), eliminando cualquier fuga de información (*zero data leakage*).
  - **Justificación Cuantitativa del Tamaño Muestral (Exigencia N5):**
    $$n \ge \frac{z^2 \cdot p(1-p)}{\epsilon^2} = \frac{(1.96)^2 \cdot 0.5(1-0.5)}{(0.05)^2} \approx 384.16 \text{ muestras/clase}$$
    El dataset cuenta con 700 muestras reales por clase (superando el piso de Cochran en un 82.2%), complementado con un pipeline de *Data Augmentation* en línea que incluye *Random Horizontal Flip* ($p=0.5$) para garantizar invariancia biométrica de mano izquierda y derecha.
* **Localizador:** [`dataset/`](file:///c:/Users/starg/Lab3_ws/dataset/), [`src/balance_dataset.py`](file:///c:/Users/starg/Lab3_ws/src/balance_dataset.py) y [`src/dataset.py`](file:///c:/Users/starg/Lab3_ws/src/dataset.py).

---

### C4. Validación y Métricas de Desempeño — Peso: 20%
* **SO / Indicador:** SO6 — Experimentación y análisis de datos — **RAE 6.2:** *Realiza inferencias sobre el desempeño de un algoritmo en la solución de un problema a partir del desarrollo de pruebas y aplicación de métricas apropiadas.*
* **Evidencias Asociadas:** **E5** y **E8**
* **Nivel Seleccionado:** **N5 (475 – 500 / 500)** — *Valor exacto: 500*
* **Evidencia Observable:**
  - Evaluó de forma ciega y exactamente una vez el conjunto de prueba multi-sujeto independiente (500 muestras, 100 por clase): Exactitud Global = **96.80%**, Exactitud Balanceada = **96.80%**, Precision Macro = **96.82%**, Recall Macro = **96.80%**, F1-Score Macro = **96.80%**.
  - **Intervalos de Confianza y Generalización Multi-Participante (Exigencia N5):**
    - **Intervalo de Confianza al 95% (Wilson Score):** $IC_{95\%} = [94.87\%, 98.02\%]$. El límite inferior (94.87%) demuestra una fiabilidad estadística sobresaliente.
    - **Desempeño por Participante:** El nuevo participante (`subj_02`) alcanzó un **98.67%** de exactitud (74/75), `subj_01` alcanzó **96.86%** (339/350) y `subj_user` alcanzó **94.67%** (71/75), demostrando robustez biométrica inter-sujeto.
* **Localizador:** [`results/metrics/test_metrics.json`](file:///c:/Users/starg/Lab3_ws/results/metrics/test_metrics.json) y [`results/plots/confusion_matrix.png`](file:///c:/Users/starg/Lab3_ws/results/plots/confusion_matrix.png).

---

### C5. Pruebas de Robustez y Reproducibilidad — Peso: 20%
* **SO / Indicador:** SO6 — Experimentación y análisis de datos — **RAE 6.2:** *Realiza inferencias sobre el desempeño de un algoritmo en la solución de un problema a partir del desarrollo de pruebas y aplicación de métricas apropiadas.*
* **Evidencias Asociadas:** **E6**, **E7**, **E9** y **E8**
* **Nivel Seleccionado:** **N5 (475 – 500 / 500)** — *Valor exacto: 500*
* **Evidencia Observable:**
  - Ejecutó un protocolo formal de 100 ensayos en vivo bajo 5 condiciones adversas con 95.0% de percepción cruda y 97.0% de comandos aceptados con solo 2.0% de falsos comandos.
  - Medición de latencia de inferencia: $p50 = 2.48\text{ ms}$, $p95 = 2.84\text{ ms}$ (>400 FPS).
  - Entregó repositorio 100% reproducible con suite de verificación automática ([`scripts/verify_all.py`](file:///c:/Users/starg/Lab3_ws/scripts/verify_all.py), 34/34 pruebas superadas exitosamente).
* **Localizador:** [`results/plots/latency_distribution.png`](file:///c:/Users/starg/Lab3_ws/results/plots/latency_distribution.png), [`results/metrics/benchmark_protocol_results.json`](file:///c:/Users/starg/Lab3_ws/results/metrics/benchmark_protocol_results.json) y [`docs/INFORME_LAB3_CNN_IEEE.md`](file:///c:/Users/starg/Lab3_ws/docs/INFORME_LAB3_CNN_IEEE.md).

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
**Estudiante:** Samuel Alejandro Chaparro Ortiz — Código Institucional: **7004072** | Equipo: **7**

1. **Decisión de Arquitectura y Regularización (RAE 1.3):**  
   *Se diseñó GestureCNN_v1 con 4 bloques convolucionales (32, 64, 128, 256 filtros), BatchNorm tras cada conv y Dropout (0.4/0.3) en la etapa FC. La jerarquía de 4 niveles es matemáticamente necesaria para expandir el campo receptivo a 30x30 píxeles, logrando desacoplar bordes de dedos frente a fondos complejos. Frente a una red superficial (Shallow: 91.33% acc), GestureCNN_v1 alcanza 96.80% acc con 1.44M de parámetros (117.8 MFLOPs) ejecutándose en apenas 2.48 ms (>400 FPS).*

2. **Decisión de Seguridad y Filtrado Temporal (RAE 1.3):**  
   *Para gobernar el brazo robótico en CoppeliaSim, se implementó en `src/command_filter.py` un filtro de búfer circular ($N=10$), moda estadística ($\ge 80\%$), umbral bayesiano ($\ge 0.85$), periodo refractario de $1.5\text{ s}$ y política de parada lógica inmediata para la Clase 0 (puño = reset/stop sin esperar consenso). Esto redujo los falsos comandos al 2.0% en pruebas en vivo.*

3. **Decisión Metodológica de Partición de Datos Multi-Sujeto (RAE 6.1):**  
   *Se incorporaron capturas de múltiples participantes (`subj_01`, `subj_02` y `subj_user`), totalizando 3,500 imágenes balanceadas. Para evitar fuga de datos temporal y asimetría de manos, se aplicó partición estratificada de mano derecha e izquierda y muestreo espaciado de ráfagas sin solapamiento (Train: 2,500, Val: 500, Test ciego: 500). El tamaño muestral satisface el criterio de Cochran ($n \ge 384$ muestras por clase al 95% de confianza).*

4. **Interpretación de Métricas y Robustez (RAE 6.2):**  
   *La evaluación ciega arrojó 96.80% de exactitud con un Intervalo de Confianza Wilson al 95% de $[94.87\%, 98.02\%]$. Dado que el límite inferior supera el 94%, el sistema garantiza fiabilidad estadística para teleoperación. La generalización inter-sujeto fue comprobada con una exactitud del 98.67% sobre el nuevo participante (`subj_02`) y 96.86% sobre `subj_01`. La latencia $p50 = 2.48\text{ ms}$ consume únicamente el 7.4% del frame time de la cámara a 30 FPS.*
