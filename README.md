# Laboratorio 3: CNN para Reconocimiento de Gestos de Manos y Control de Brazo Robótico en CoppeliaSim

![CNN Perception & Live Trials Performance](results/plots/live_trials_performance.png)

**Asignatura:** Inteligencia Artificial  
**Programa:** Ingeniería Mecatrónica — Semestre IX  
**Institución:** Universidad Militar Nueva Granada (UMNG)  
**Ruta Seleccionada:** **CoppeliaSim Edu** (Brazo Articulado 3 GDL + Pinza Paralela + Control por Gestos)  
**Evaluación ABET:** SO1 (RAE 1.3), SO6 (RAE 6.1), SO6 (RAE 6.2) — Nivel Esperado: **N5**  

---

## RESUMEN EJECUTIVO (*ABSTRACT*)

El presente proyecto diseña, implementa y valida un sistema mecatrónico de percepción visual en tiempo real basado en una **Red Neuronal Convolucional (CNN)** profunda para la clasificación del número de dedos extendidos (de cero a cuatro dedos) capturados mediante una cámara web, integrando un **filtro temporal de estabilidad** para gobernar las articulaciones y pinza de un brazo robótico de 3 grados de libertad (GDL) en el entorno de simulación **CoppeliaSim**.

El sistema desacopla estrictamente la inferencia visual del actuador robótico. A partir de más de 6,100 capturas reales, se curó un conjunto de datos balanceado de **3,490 imágenes reales** con partición independiente por sesión y entorno para garantizar **cero fuga de datos** (*zero data leakage*). La arquitectura convolucional `GestureCNN_v1` (1,438,437 parámetros, 5.49 MB float32, 117.78 MFLOPs) entrenada con **AdamW**, *Label Smoothing* y *Cosine Annealing* alcanzó un **95.60% de Exactitud Global**, **95.60% de Exactitud Balanceada** y **95.59% de F1-Score Macro** sobre un conjunto de prueba independiente de **500 muestras** (100 por clase), con un intervalo de confianza al 95% (Wilson Score) de **[93.43%, 97.08%]** y una latencia de inferencia de **$p50 = 2.65\text{ ms}$** (> 370 FPS en CPU).

En la experimentación de validación (100 ensayos bajo variaciones de baja luz, luz intensa, fondos complejos y transiciones rápidas), el filtro temporal de comandos garantizó un **98.0% de aceptación de órdenes estables** y una **tasa de falsos comandos de apenas el 1.0%**, manteniendo la clase 0 como parada lógica e inhibidor de movimiento.

---

## TABLA DE CONTENIDOS
1. [Objetivos y Alcance](#1-objetivos-y-alcance)
2. [Arquitectura del Sistema y Módulos](#2-arquitectura-del-sistema-y-módulos)
3. [Conjunto de Datos y Preprocesamiento](#3-conjunto-de-datos-y-preprocesamiento)
4. [Diseño Teórico y Matemático de la CNN](#4-diseño-teórico-y-matemático-de-la-cnn)
5. [Filtro Temporal de Comandos y Seguridad](#5-filtro-temporal-de-comandos-y-seguridad)
6. [Integración y Adaptador en CoppeliaSim (3 GDL + Pinza)](#6-integración-y-adaptador-en-coppeliasim-3-gdl--pinza)
7. [Resultados Experimentales y Métricas de Rendimiento](#7-resultados-experimentales-y-métricas-de-rendimiento)
8. [Respuestas a las Preguntas de Discusión](#8-respuestas-a-las-preguntas-de-discusión)
9. [Instrucciones de Instalación y Ejecución](#9-instrucciones-de-instalación-y-ejecución)
10. [Alineación ABET (SO1, SO6)](#10-alineación-abet-so1-so6)

---

## 1. OBJETIVOS Y ALCANCE

### Objetivo General
Diseñar, implementar y validar un sistema de percepción basado en CNN que reconozca el número de dedos (0 a 4) en tiempo real y utilice la clasificación para controlar un brazo robótico en CoppeliaSim.

### Objetivos Específicos
1. **Curaduría y Balanceo del Dataset:** Construir un conjunto de datos balanceado a partir de capturas en vivo con variaciones de entorno, iluminación, ángulo y escala, agrupado por sesiones independientes para eliminar la correlación temporal y evitar la fuga de información (*zero data leakage*).
2. **Diseño y Entrenamiento de la CNN:** Formular analíticamente las dimensiones espaciales, parámetros y FLOPs por capa de la red, entrenando con optimizador AdamW, ponderación dinámica de clases y regularización por *Dropout*, *BatchNorm* y *Data Augmentation* enriquecido (*Cutout*, jitter fotométrico y transformaciones afines).
3. **Inferencia en Tiempo Real y Filtrado:** Implementar inferencia en vivo con cámara web y HUD OpenCV con telemetría completa, acoplando un filtro temporal de ventana deslizante ($N=10$, moda $\ge 8/10$, umbral $\ge 0.85$, cooldown $1.5\text{ s}$) que inhiba transiciones ambiguas y ruidos transitorios.
4. **Teleoperación Segura:** Mapear clases discretas validadas a las articulaciones de un manipulador de 3 GDL y su pinza en CoppeliaSim, manteniendo la clase 0 como parada lógica.

---

## 2. ARQUITECTURA DEL SISTEMA Y MÓDULOS

El sistema sigue una arquitectura modular en tubería (*pipeline*) que desacopla la percepción visual del actuador robótico:

```
[ Cámara Web / Video HD 1280x720 ]
                 │
                 ▼
[ Preprocesamiento & ROI (64x64 Grayscale Normalizado [-1, 1]) ]
                 │
                 ▼
[ GestureCNN_v1 (Inferencia: Logits -> Softmax Probabilities) ]
                 │  (Clase Cruda + Confianza % + Latencia ms)
                 ▼
[ CommandFilter (Búfer N=10, Moda M=8, Umbral >=0.85, Cooldown 1.5s, Parada C0) ]
                 │  (Comando Aceptado Estable: 0, 1, 2, 3, 4)
                 ▼
[ RobotAdapter (Límites Articulares, Inversión de Sentido, Telemetría) ]
                 │  (ZeroMQ Remote API / Puerto 23000)
                 ▼
[ CoppeliaSim 3-DoF Robotic Workcell (Joint1, Joint2, Joint3, Pinza) ]
```

### Estructura del Repositorio
```
Lab3_ws/
├── config/
│   └── config.yaml                 # Configuración de clases, ROI, filtro temporal y robot
├── dataset/
│   ├── train/                      # 2,490 imágenes balanceadas (~500 por clase)
│   ├── val/                        # 500 imágenes balanceadas (100 por clase)
│   └── test/                       # 500 imágenes balanceadas (100 por clase - prueba independiente)
├── models/
│   └── best_gesture_cnn.pt         # Checkpoint PyTorch con los mejores pesos entrenados
├── results/
│   ├── metrics/
│   │   ├── test_metrics.json       # Métricas sobre test (Accuracy, F1, Wilson CI, confusión)
│   │   ├── benchmark_protocol_results.json # Protocolo de 100 ensayos en vivo
│   │   └── robustness_metrics.json # Pruebas cuantitativas ante perturbaciones (ABET C5)
│   └── plots/
│       ├── learning_curves.png     # Diagnóstico 2x2: Pérdida, exactitud, F1 por clase y latencia
│       ├── confusion_matrix.png    # Matriz de confusión en conjunto de prueba
│       ├── latency_distribution.png# Histograma de latencia de inferencia en tiempo real
│       ├── live_trials_performance.png # Comparativa de 20 ensayos en vivo por clase
│       ├── decoupled_layers_diagnostic.png # Diagnóstico desacoplado de percepción y filtro
│       └── robustness_degradation.png # Curva de degradación ante perturbaciones ópticas
├── scenes/
│   ├── lab3_robot_3dof.ttt         # Escena de CoppeliaSim con celda de manufactura
│   └── setup_scene.py              # Verificador y constructor de escena
├── scripts/
│   └── verify_all.py               # Suite de verificación integral (34/34 pruebas superadas)
├── src/
│   ├── __init__.py
│   ├── balance_dataset.py          # Balanceador y particionador sesión-independiente
│   ├── collect_data.py             # Herramienta interactiva de captura de imágenes reales
│   ├── dataset.py                  # Dataset PyTorch y pipeline de Data Augmentation
│   ├── model.py                    # Arquitectura GestureCNN y cálculo analítico de FLOPs
│   ├── train.py                    # Bucle de entrenamiento con AdamW, weights y CosineAnnealing
│   ├── command_filter.py           # Filtro temporal, moda, umbral de confianza y parada lógica
│   ├── coppelia_client.py          # Cliente ZeroMQ Remote API de CoppeliaSim
│   ├── robot_adapter.py            # Adaptador cinemático y verificación de límites articulares
│   ├── cnn_inference.py            # Motor de inferencia en vivo y HUD OpenCV enriquecido
│   ├── main_app.py                 # Aplicación interactiva principal
│   └── benchmark.py                # Protocolo experimental de 100 ensayos y perturbaciones
├── docs/
│   ├── INFORME_LAB3_CNN_IEEE.md    # Artículo formal en formato IEEE
│   └── PLANTILLA_ABET_LAB03.md     # Evidencia de evaluación de criterios ABET (SO1, SO6)
├── requirements.txt
└── README.md
```

---

## 3. CONJUNTO DE DATOS Y PREPROCESAMIENTO

### Definición de Clases y Acciones Mecatrónicas
| Clase | Gesto Manual | Significado y Acción en CoppeliaSim |
| :---: | :--- | :--- |
| **0** | Puño cerrado (0 dedos) | **PARADA LÓGICA / INHIBICIÓN:** Bloquea comandos y congela la posición del brazo. |
| **1** | 1 dedo extendido | **Articulación 1 (Base Yaw):** Paso angular discreto de $\pm 25^\circ$. |
| **2** | 2 dedos extendidos | **Articulación 2 (Hombro Pitch):** Paso angular discreto de $\pm 20^\circ$. |
| **3** | 3 dedos extendidos | **Articulación 3 (Codo Pitch):** Paso angular discreto de $\pm 20^\circ$. |
| **4** | 4 dedos extendidos | **Pinza / Ventosa:** Conmuta estado entre activado (sujeción) y desactivado (liberación). |

### Distribución del Dataset Balanceado Curado
El dataset se estructuró a partir de múltiples sesiones de captura en diferentes entornos, agrupando por ráfaga temporal para garantizar **cero fuga de datos**:

| Subconjunto | 0 Dedos | 1 Dedo | 2 Dedos | 3 Dedos | 4 Dedos | Total Muestras | Balanceo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Entrenamiento (`train`)** | 490 | 500 | 500 | 500 | 500 | **2,490 fotos** | Homogéneo |
| **Validación (`val`)** | 100 | 100 | 100 | 100 | 100 | **500 fotos** | Exactamente parejo |
| **Prueba Ciega (`test`)** | 100 | 100 | 100 | 100 | 100 | **500 fotos** | Exactamente parejo |
| **Total Curado** | **690** | **700** | **700** | **700** | **700** | **3,490 fotos** | **Supera $n \ge 384$ (ABET)** |

### Pipeline de Preprocesamiento de Imagen
1. **Recorte de ROI:** Submatriz de la mano extraída del cuadro de video ($~380 \times 380\text{ px}$).
2. **Redimensionamiento:** Escalamiento a $64 \times 64$ píxeles con interpolación bilineal.
3. **Escala de Grises:** Conversión a 1 canal para aislar la geometría de la mano del tono de piel.
4. **Normalización:** $\text{norm} = \frac{\text{gray} / 255.0 - 0.5}{0.5} \in [-1.0, 1.0]$.
5. **Detección de Recuadro Vacío:** Si la desviación estándar $\sigma < 7.0$ (pared plana o ausencia de mano), asigna baja confianza ($20\%$) y estado `NO_HAND` para rechazo seguro en el filtro.

---

## 4. DISEÑO TEÓRICO Y MATEMÁTICO DE LA CNN

### Resumen Analítico de Capas y Parámetros
La arquitectura `GestureCNN_v1` procesa tensores de entrada $X \in \mathbb{R}^{1 \times 64 \times 64}$:

| Capa | Dimensión de Salida | Kernel / Stride / Pad | Parámetros Entrenables | Operaciones MACs (FLOPs) |
| :--- | :---: | :---: | :---: | :---: |
| **Input** | $[1, 1, 64, 64]$ | - | - | - |
| **Conv2d_1 + BN1 + ReLU** | $[1, 32, 64, 64]$ | $3 \times 3, s=1, p=1$ | $320 + 64 = 384$ | $2.36\times 10^6$ |
| **MaxPool2d_1** | $[1, 32, 32, 32]$ | $2 \times 2, s=2$ | $0$ | $32,768$ |
| **Conv2d_2 + BN2 + ReLU** | $[1, 64, 32, 32]$ | $3 \times 3, s=1, p=1$ | $18,432 + 128 = 18,560$ | $37.75\times 10^6$ |
| **MaxPool2d_2** | $[1, 64, 16, 16]$ | $2 \times 2, s=2$ | $0$ | $16,384$ |
| **Conv2d_3 + BN3 + ReLU** | $[1, 128, 16, 16]$ | $3 \times 3, s=1, p=1$ | $73,728 + 256 = 73,984$ | $37.75\times 10^6$ |
| **MaxPool2d_3** | $[1, 128, 8, 8]$ | $2 \times 2, s=2$ | $0$ | $8,192$ |
| **Conv2d_4 + BN4 + ReLU** | $[1, 256, 8, 8]$ | $3 \times 3, s=1, p=1$ | $294,912 + 512 = 295,424$ | $37.75\times 10^6$ |
| **AdaptiveAvgPool2d** | $[1, 256, 4, 4]$ | $4 \times 4$ | $0$ | $4,096$ |
| **Flatten** | $[1, 4096]$ | - | $0$ | $0$ |
| **Dropout(0.4) + FC1 + ReLU** | $[1, 256]$ | $4096 \rightarrow 256$ | $1,048,832$ | $2.10\times 10^6$ |
| **Dropout(0.3) + FC2 (Logits)**| $[1, 5]$ | $256 \rightarrow 5$ | $1,285$ | $2,560$ |
| **TOTAL** | - | - | **1,438,437** | **117.78 MFLOPs** |

---

## 5. FILTRO TEMPORAL DE COMANDOS Y SEGURIDAD

Para desacoplar la predicción visual de las acciones mecánicas del robot y evitar movimientos intempestivos:
1. **Ventana Deslizante:** Búfer circular de $N = 10$ cuadros consecutivos.
2. **Moda Estadística:** Se requiere que la clase dominante aparezca al menos $M = 8$ veces en la ventana ($\ge 80\%$).
3. **Umbral de Confianza:** La probabilidad promedio de la clase debe ser $\ge 0.85$.
4. **Periodo Refractario (*Cooldown*):** Intervalo mínimo de $1.5\text{ s}$ entre comandos robóticos consecutivos.
5. **Parada Lógica Inmediata (Clase 0):** Inhibe instantáneamente el envío de nuevas órdenes articulares.

---

## 6. INTEGRACIÓN Y ADAPTADOR EN COPPELIASIM (3 GDL + PINZA)

El adaptador en [`src/robot_adapter.py`](file:///c:/Users/starg/Lab3_ws/src/robot_adapter.py) se comunica con CoppeliaSim Edu mediante la API remota ZeroMQ (`127.0.0.1:23000`):
* **Joint 1 (Base Yaw):** Límites $[-170^\circ, 170^\circ]$, paso $\pm 25^\circ$.
* **Joint 2 (Hombro Pitch):** Límites $[-30^\circ, 120^\circ]$, paso $\pm 20^\circ$.
* **Joint 3 (Codo Pitch):** Límites $[-110^\circ, 110^\circ]$, paso $\pm 20^\circ$.
* **Inversión de Sentido:** Si la articulación alcanza su límite físico, invierte automáticamente el sentido del movimiento para evitar bloqueos.
* **Pinza / Succión:** Señal booleana de activación/desactivación sobre `uArm_suction`.

---

## 7. RESULTADOS EXPERIMENTALES Y MÉTRICAS DE RENDIMIENTO

### A. Evaluación en Conjunto de Prueba Independiente (500 Muestras: 100 por Clase)
* **Exactitud Global (*Overall Accuracy*):** **95.60%**
* **Exactitud Balanceada (*Balanced Accuracy*):** **95.60%**
* **F1-Score Macro:** **95.59%**
* **Precisión Macro:** **95.65%**
* **Recall Macro:** **95.60%**
* **Intervalo de Confianza al 95% (Wilson Score):** **[93.43%, 97.08%]**
* **Latencia de Inferencia Mediana ($p50$):** **$2.65\text{ ms}$**
* **Latencia Percentil 95 ($p95$):** **$3.78\text{ ms}$**

#### Matriz de Confusión en Prueba:
```text
               Predicho:
              0     1     2     3     4
Real 0:     [100,    0,    0,    0,    0]  --> 100.0% Recall (F1: 1.00)
Real 1:     [  0,   91,    9,    0,    0]  -->  91.0% Recall (F1: 0.94)
Real 2:     [  0,    3,   93,    4,    0]  -->  93.0% Recall (F1: 0.92)
Real 3:     [  0,    0,    0,   94,    6]  -->  94.0% Recall (F1: 0.95)
Real 4:     [  0,    0,    0,    0,  100]  --> 100.0% Recall (F1: 0.97)
```

### B. Protocolo Experimental en Vivo (100 Ensayos: 20 por Clase)
* **Exactitud de Percepción Cruda:** **94.0%**
* **Aceptación de Comandos por Filtro:** **98.0%**
* **Tasa de Falsos Comandos:** **1.0%**
* **Latencia Promedio en Vivo:** **$0.02 - 2.65\text{ ms}$**

### C. Pruebas Cuantitativas de Robustez ante Perturbaciones (ABET C5)
| Condición Experimental | Exactitud (%) | F1-Score | Latencia $p50$ (ms) | Estado del Filtro |
| :--- | :---: | :---: | :---: | :--- |
| **1. Nominal (Control)** | **100.0%** | 1.000 | 2.52 | Comandos aceptados |
| **2. Baja Luz (-50% Brillo)** | **96.67%** | 0.966 | 2.55 | Estable |
| **3. Luz Intensa (+50% Brillo)** | **95.00%** | 0.949 | 2.53 | Estable |
| **4. Rotación Extrema ($\pm 35^\circ$)** | **93.33%** | 0.932 | 2.58 | Filtro amortigua transiciones |
| **5. Oclusión Parcial (25% Mano)** | **91.67%** | 0.915 | 2.61 | Bloqueo seguro ante baja confianza |

---

## 8. RESPUESTAS A LAS PREGUNTAS DE DISCUSIÓN

### Pregunta 1: ¿Qué cambio en datos o arquitectura reduciría el error de la CNN sin aumentar de forma inaceptable la latencia? ¿Cómo se comprobaría?
**Respuesta:** Implementar convoluciones separables en profundidad (*Depthwise Separable Convolutions*) como en la variante `GestureCNN_Efficient` incluida en el proyecto. Reduce los parámetros en un $80\%$ y los FLOPs en un $71\%$ (de 117.8 a 34.1 MFLOPs), manteniendo latencia $<1\text{ ms}$ y permitiendo incorporar atención espacial ligera (SE/CBAM) para discriminar la punta de los dedos. Se comprueba mediante la suite de benchmarks midiendo la matriz de confusión y el tiempo de inferencia $p95$ con `time.perf_counter()`.

### Pregunta 2: ¿Cómo evita la división por persona o sesión que la métrica de prueba sea artificialmente alta?
**Respuesta:** Cuando cuadros consecutivos de un mismo video se reparten aleatoriamente entre Train y Test, la CNN memoriza el fondo estático, la iluminación y el tono de piel (*background memorization / data leakage*), produciendo una exactitud ficticia del $99\%$. Al aislar sesiones y entornos completos exclusivamente en Test, se evalúa la **generalización real ante cambios de dominio ambiental**.

### Pregunta 3: ¿Cómo cambian precisión, falsos comandos y latencia al modificar el umbral y la ventana temporal?
**Respuesta:** 
* Aumentar el umbral ($\ge 0.90$) y la ventana temporal ($N=15$) reduce a cero los falsos positivos, pero introduce un retardo de respuesta mecatrónica de medio segundo.
* Reducir el umbral ($<0.70$) o la ventana ($N=5$) hace al sistema más reactivo, pero permeable a comandos espurios durante la transición entre gestos.

### Pregunta 4: ¿Qué condiciones visuales provocaron mayor cambio de dominio y qué aumento de datos sería pertinente?
**Respuesta:** La iluminación directa de lámparas (reflejos especulares) y sombras duras entre dedos adyacentes. Los aumentos pertinentes implementados en [`src/dataset.py`](file:///c:/Users/starg/Lab3_ws/src/dataset.py) fueron *Random Cutout* (oclusión de parches de $6\times 6$ a $14\times 14$), fluctuación de brillo/contraste y rotaciones afines con zoom.

### Pregunta 5: ¿Cómo se demuestra si una tarea falló por percepción, filtro, adaptador o robot?
**Respuesta:** Mediante el desacoplamiento y registro de telemetría por capas:
1. Si la clase predicha por la CNN difiere del gesto mostrado $\rightarrow$ **Fallo de Percepción**.
2. Si la CNN acierta pero la confianza es $<0.85$ o la moda es inestable $\rightarrow$ **Fallo de Filtro**.
3. Si el comando es aceptado pero el ángulo excede los límites articulares $\rightarrow$ **Fallo de Adaptador**.
4. Si el robot recibe el comando pero la simulación no responde $\rightarrow$ **Fallo de Comunicación ZeroMQ**.

---

## 9. INSTRUCCIONES DE INSTALACIÓN Y EJECUCIÓN

### Requisitos del Sistema
- Sistema Operativo: Windows 10/11
- Python 3.12 con PyTorch (`.venv312` preconfigurado en el workspace)
- Simulador: CoppeliaSim Edu v4.3+

### Pasos de Ejecución (PowerShell)

1. **Balancear y Particionar el Dataset:**
```powershell
.venv312\Scripts\python.exe src\balance_dataset.py
```

2. **Entrenar la Red Neuronal Convolucional:**
```powershell
.venv312\Scripts\python.exe src\train.py
```

3. **Ejecutar el Protocolo Experimental y Benchmarks:**
```powershell
.venv312\Scripts\python.exe src\benchmark.py
```

4. **Lanzar la Aplicación Principal Interactiva en Vivo:**
```powershell
.venv312\Scripts\python.exe src\main_app.py
```

5. **Ejecutar Verificación Integral de Integridad (34 Pruebas):**
```powershell
.venv312\Scripts\python.exe scripts\verify_all.py
```

*Controles de la Aplicación en Vivo:*
* `[0, 1, 2, 3, 4]`: Cambiar simulación de gestos si no hay cámara web física.
* `[Q]` o `[ESC]`: Salir de la aplicación y cerrar el visor HUD.

---

## 10. ALINEACIÓN ABET (SO1, SO6)

* **SO1 — RAE 1.3 (Nivel N5):** Diseñó, implementó y justificó analíticamente una arquitectura profunda convolucional (`GestureCNN_v1`), calculando parámetros, FLOPs y campos receptivos, integrándola en un sistema mecatrónico de control robótico desacoplado.
* **SO6 — RAE 6.1 (Nivel N5):** Diseñó y argumentó un protocolo experimental libre de fuga de datos con partición independiente por sesiones y cálculo del tamaño muestral estadístico ($n \ge 384$).
* **SO6 — RAE 6.2 (Nivel N5):** Realizó inferencias rigurosas sobre el desempeño a partir de matrices de confusión, exactitud balanceada ($95.6\%$), F1-score ($95.59\%$), intervalos de confianza de Wilson al 95% y curvas cuantitativas de degradación de robustez bajo condiciones adversas.
