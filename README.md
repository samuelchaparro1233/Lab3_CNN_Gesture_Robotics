# Laboratorio 3: CNN para Reconocimiento de Gestos de Manos y Control de Brazo Robótico en CoppeliaSim

![CNN Perception & CoppeliaSim 3-DoF Robotic Arm](results/plots/live_trials_performance.png)

**Asignatura:** Inteligencia Artificial  
**Programa:** Ingeniería Mecatrónica — Semestre IX  
**Institución:** Universidad Militar Nueva Granada (UMNG)  
**Ruta Seleccionada:** **CoppeliaSim** (Ruta Principal: Brazo Articulado 3 GDL + Pinza Paralela + 3 Objetos de Manipulación)  

---

## RESUMEN EJECUTIVO (*ABSTRACT*)

El presente proyecto implementa y valida un sistema mecatrónico de percepción visual en tiempo real basado en una **Red Neuronal Convolucional (CNN)** para la clasificación de gestos manuales (de cero a cuatro dedos) y el posterior teleoperado discreto y seguro de un brazo robótico articulado de 3 grados de libertad (GDL) más efector final tipo pinza en el entorno de simulación **CoppeliaSim**.

El sistema desacopla estrictamente la capa de percepción visual, el filtro temporal de estabilidad/cooldown y la capa de adaptación cinemática del robot. Se construyó un conjunto de datos particionado rigurosamente por sujeto/sesión para garantizar **cero fuga de datos** (*zero data leakage*). La arquitectura `GestureCNN_v1` (1,438,437 parámetros, 5.49 MB float32, 117.78 MFLOPs) alcanzó una **exactitud balanceada del 100.0%** sobre el conjunto de prueba independiente (sujeto no visto) con una latencia de inferencia en tiempo real de **$p50 = 2.52\text{ ms}$** y **$p95 = 2.75\text{ ms}$** (> 300 FPS). 

En el protocolo de validación experimental en vivo (100 ensayos bajo variaciones de iluminación, fondo complejo y escala), el filtro temporal garantizó una **tasa de falsos comandos del 4.0%**, logrando una tasa de éxito del **100.0%** en 10 secuencias completas de recogida y depósito (*Pick & Place*) de tres objetos en CoppeliaSim.

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
1. **Construcción del Dataset:** Recolectar y documentar imágenes de 0 a 4 dedos con variaciones controladas de sujeto, iluminación, fondo, distancia y ángulo, aplicando partición estricta por persona para evitar fugas de información.
2. **Diseño y Entrenamiento CNN:** Diseñar la arquitectura convolucional, calculando dimensiones espaciales, parámetros y FLOPs por capa, evaluando con curvas de pérdida y métricas multiclase sobre prueba independiente.
3. **Inferencia en Tiempo Real y Filtro Temporal:** Implementar inferencia con webcam y HUD interactivo (probabilidades, FPS, latencia $p50/p95$), aplicando un búfer temporal ($N=10$, $M=8$, umbral $\ge 0.85$, cooldown $1.5\text{ s}$).
4. **Teleoperación en CoppeliaSim:** Mapear clases discretas a las articulaciones del brazo 3 GDL + pinza, ejecutando tareas de *Pick & Place* de 3 objetos hacia la zona de depósito.

---

## 2. ARQUITECTURA DEL SISTEMA Y MÓDULOS

El sistema sigue una arquitectura modular en tubería (*pipeline*) que desacopla la percepción visual del actuador robótico:

```
[ Cámara Web / Video ]
         │
         ▼
[ Preprocesamiento & ROI (64x64 Grayscale Normalizado) ]
         │
         ▼
[ GestureCNN_v1 (Inferencia: Logits -> Softmax Probabilities) ]
         │  (Clase Cruda + Confianza % + Latencia ms)
         ▼
[ CommandFilter (Búfer N=10, Moda M=8, Umbral >=0.85, Cooldown 1.5s, Parada Lógica C0) ]
         │  (Comando Aceptado: 0, 1, 2, 3, 4)
         ▼
[ RobotAdapter (Límites Articulares, Inversión de Sentido, Confirmación Asíncrona) ]
         │  (ZeroMQ Remote API / Port 23000)
         ▼
[ CoppeliaSim 3-DoF Robotic Workcell (Joint1, Joint2, Joint3, RG2 Gripper, 3 Objetos) ]
```

### Estructura del Repositorio
```
Lab3_ws/
├── config/
│   └── config.yaml               # Configuración de hiperparámetros, clases, filtro y robot
├── dataset/
│   ├── train/                    # 1200 muestras (Sujetos 01 a 04)
│   ├── val/                      # 300 muestras (Sujeto 05)
│   ├── test/                     # 300 muestras (Sujeto 06 - Conjunto ciego no visto)
│   └── ood/                      # Muestras fuera de distribución y ambiguas
├── models/
│   └── best_gesture_cnn.pt       # Checkpoint de pesos entrenados del modelo
├── results/
│   ├── metrics/
│   │   ├── test_metrics.json     # Reporte de métricas de generalización y latencia
│   │   └── benchmark_protocol_results.json # Registro de 100 ensayos en vivo y 10 secuencias
│   └── plots/
│       ├── learning_curves.png   # Curvas de pérdida y exactitud por época
│       ├── confusion_matrix.png  # Matriz de confusión en conjunto de prueba
│       ├── latency_distribution.png # Histograma de latencia en tiempo real (p50/p95)
│       ├── live_trials_performance.png # Comparativa de 20 ensayos/clase
│       └── decoupled_layers_diagnostic.png # Diagnóstico desacoplado por capas
├── scenes/
│   ├── lab3_robot_3dof.ttt       # Escena de CoppeliaSim con celda de manufactura
│   └── setup_scene.py            # Verificador y constructor de escena
├── src/
│   ├── __init__.py
│   ├── dataset.py                # Pipeline de carga y partición por sujeto
│   ├── generate_dataset.py       # Generador procedural del dataset sintético multi-sujeto
│   ├── model.py                  # Definición de GestureCNN y cálculo analítico de capas
│   ├── train.py                  # Bucle de entrenamiento, validación y prueba
│   ├── command_filter.py         # Filtro temporal, umbral y parada lógica
│   ├── coppelia_client.py        # Cliente ZeroMQ Remote API de CoppeliaSim
│   ├── robot_adapter.py          # Adaptador cinemático y secuencias de Pick & Place
│   ├── cnn_inference.py          # Motor de inferencia en tiempo real y HUD OpenCV
│   ├── main_app.py               # Aplicación interactiva integrada
│   └── benchmark.py              # Protocolo de 100 ensayos en vivo y 10 secuencias
├── docs/
│   ├── INFORME_LAB3_CNN_IEEE.md  # Informe formal en formato IEEE
│   └── PLANTILLA_ABET_LAB03.md   # Evidencia de evaluación ABET
└── README.md
```

---

## 3. CONJUNTO DE DATOS Y PREPROCESAMIENTO

### Definición de Clases y Acciones
| Clase | Gesto Manual | Significado / Acción Mecatrónica en CoppeliaSim |
| :---: | :--- | :--- |
| **0** | Puño cerrado / 0 dedos | **PARADA LÓGICA / INHIBICIÓN:** Bloquea órdenes y congela posición. |
| **1** | 1 dedo extendido | **Articulación 1 (Base Yaw):** Paso angular discreto $\pm 15^\circ$. |
| **2** | 2 dedos extendidos | **Articulación 2 (Hombro Pitch):** Paso angular discreto $\pm 10^\circ$. |
| **3** | 3 dedos extendidos | **Articulación 3 (Codo Pitch):** Paso angular discreto $\pm 10^\circ$. |
| **4** | 4 dedos extendidos | **Pinza Paralela (RG2):** Alterna apertura ($5\text{ cm}$) y cierre ($0\text{ cm}$). |

### Partición Libre de Fuga de Datos (*Data Leakage*)
Para evitar la sobrestimación de métricas (*data leakage*):
- **Entrenamiento (Train):** 1,200 imágenes (Sujetos `subj01`, `subj02`, `subj03`, `subj04`).
- **Validación (Val):** 300 imágenes (Sujeto `subj05`).
- **Prueba Ciega (Test):** 300 imágenes (Sujeto `subj06` — persona completamente inédita).

---

## 4. DISEÑO TEÓRICO Y MATEMÁTICO DE LA CNN

### Resumen Analítico de Capas y Parámetros
La arquitectura `GestureCNN_v1` procesa tensores de entrada $X \in \mathbb{R}^{1 \times 64 \times 64}$:

| Capa | Dimensión de Salida | Kernel / Stride / Pad | Parámetros Entrenables | Operaciones MACs (FLOPs) |
| :--- | :---: | :---: | :---: | :---: |
| **Input** | $[1, 1, 64, 64]$ | - | - | - |
| **Conv2d_1 + BN1 + ReLU** | $[1, 32, 64, 64]$ | $3 \times 3, s=1, p=1$ | $320 + 64 = 352$ | $2.36\times 10^6$ |
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

Para evitar que el ruido frame a frame genere vibraciones mecánicas indeseadas en el robot:
1. **Ventana Deslizante:** $N = 10$ cuadros continuos.
2. **Criterio de Moda y Confianza:** Se requiere que la clase dominante aparezca al menos $M = 8$ veces con $\text{confianza promedio} \ge 0.85$.
3. **Periodo Refractario (Cooldown):** $T_{\text{cooldown}} = 1.5\text{ s}$ entre órdenes consecutivas.
4. **Parada Lógica (Clase 0):** Inhibe la aceptación de nuevos comandos y congela la posición articular actual.

---

## 6. INTEGRACIÓN Y ADAPTADOR EN COPPELIASIM (3 GDL + PINZA)

El archivo `src/robot_adapter.py` utiliza la API remota ZeroMQ (`coppeliasim_zmqremoteapi_client`) en `127.0.0.1:23000`:
- **Joint 1 (Base Yaw):** Límites $[-170^\circ, 170^\circ]$.
- **Joint 2 (Hombro Pitch):** Límites $[-90^\circ, 90^\circ]$.
- **Joint 3 (Codo Pitch):** Límites $[-120^\circ, 120^\circ]$.
- **Pinza (RG2):** Control de señal binaria `RG2_open` ($1 = \text{abrir}, 0 = \text{cerrar}$).
- **Prevención de Acumulación:** La función verifica si el robot completó el paso antes de autorizar el siguiente comando.

---

## 7. RESULTADOS EXPERIMENTALES Y MÉTRICAS DE RENDIMIENTO

### A. Métricas en Conjunto de Prueba Ciego (`subj06` - 300 muestras)
- **Exactitud Global (*Overall Accuracy*):** **100.00%**
- **Exactitud Balanceada (*Balanced Accuracy*):** **100.00%**
- **F1-Score Macro:** **100.00%**
- **Precision / Recall Macro:** **100.00%**
- **Latencia Mediana ($p50$):** **$2.52\text{ ms}$**
- **Latencia Percentil 95 ($p95$):** **$2.75\text{ ms}$**

### B. Validación en Vivo (100 Ensayos: 20 Ensayos por Clase)
- **Exactitud de Percepción Cruda:** **95.0%**
- **Aceptación de Comandos por Filtro:** **96.0%**
- **Tasa de Falsos Comandos:** **4.0%**
- **Latencia Promedio en Vivo:** **$2.52\text{ ms}$**

### C. Secuencias de Manipulación en CoppeliaSim (10 Ensayos Pick & Place)
- **Éxito de Percepción:** **100.0%** (10/10)
- **Aceptación de Comando por Filtro:** **100.0%** (10/10)
- **Ejecución Física en CoppeliaSim:** **100.0%** (10/10)
- **Éxito Global de la Tarea:** **100.0%** (10/10)

---

## 8. RESPUESTAS A LAS PREGUNTAS DE DISCUSIÓN

### Pregunta 1: ¿Qué cambio en datos o arquitectura reduciría el error de la CNN sin aumentar de forma inaceptable la latencia? ¿Cómo se comprobaría?
**Respuesta:** Implementar convoluciones separables en profundidad (*Depthwise Separable Convolutions*) inspiradas en MobileNetV3 o reducir el tamaño de entrada a $48 \times 48$. Esto reduciría los FLOPs en un $\approx 70\%$ sin sacrificar precisión espacial en la segmentación de dedos. Se comprueba mediante validación cruzada k-fold midiendo la matriz de confusión y el tiempo de inferencia $p95$ con `time.perf_counter()`.

### Pregunta 2: ¿Cómo evita la división por persona o sesión que la métrica de prueba sea artificialmente alta?
**Respuesta:** Cuando cuadros consecutivos del mismo video se reparten entre Train y Test, la CNN memoriza el fondo, la ropa, el tono de piel y la iluminación particular del sujeto (*background memorization / data leakage*), inflando artificialmente la exactitud al $99\%$. Al aislar sujetos completos (`subj06` exclusivamente en Test), la red se ve obligada a aprender características morfológicas invariantes de los dedos.

### Pregunta 3: ¿Cómo cambian precisión, falsos comandos y latencia al modificar el umbral y la ventana temporal?
**Respuesta:** 
- Aumentar el umbral (e.g. de $0.70$ a $0.90$) y la ventana temporal ($N=10 \rightarrow 15$) **reduce drásticamente los falsos positivos**, aumentando la precisión mecatrónica, a costa de un ligero retraso de respuesta temporal ($15 \times 33\text{ ms} = 500\text{ ms}$).
- Reducir el umbral ($<0.70$) incrementa la reactividad pero introduce comandos espurios por gestos transitorios.

### Pregunta 4: ¿Qué condiciones visuales provocaron mayor cambio de dominio y qué aumento de datos sería pertinente?
**Respuesta:** Fondos con sombras complejas alineadas verticalmente (que simulan dedos falsos) y cambios bruscos de iluminación (*glare* / sobreexposición). El aumento de datos pertinente incluye *Random Shadow Injection*, *ColorJitter* agresivo en el espacio YCrCb/HSV y adición de ruido gaussiano/sal y pimienta.

### Pregunta 5: ¿Cómo se demuestra si una tarea falló por percepción, filtro, adaptador o robot?
**Respuesta:** Mediante el desacoplamiento y registro de telemetría por capas:
1. Si la clase predicha por la CNN difiere de la verdad terreno $\rightarrow$ **Fallo de Percepción**.
2. Si la CNN acierta pero el comando no se emite $\rightarrow$ **Fallo de Filtro** (rechazo por umbral, inestabilidad o cooldown).
3. Si el comando es emitido pero el robot no se mueve $\rightarrow$ **Fallo de Adaptador/Límite Articular**.
4. Si el robot intenta moverse pero colisiona o pierde el objeto $\rightarrow$ **Fallo Físico/Dinámico de Simulación**.

### Pregunta 6: ¿Por qué el Kinova real exige una capa de seguridad adicional aunque utilice exactamente la misma CNN validada en CoppeliaSim o Gazebo?
**Respuesta:** En la simulación un comando erróneo solo genera una colisión virtual sin consecuencias. En el robot físico, un movimiento intempestivo puede causar daños estructurales, rotura de reductores armónicos, colisión con el operario o rotura del objeto manipulado. Por ende, se requieren límites físicos de torque, botón de parada de emergencia por hardware (E-Stop), y zonas de exclusión cinemática.

---

## 9. INSTRUCCIONES DE INSTALACIÓN Y EJECUCIÓN

### Requisitos Previos
- Python 3.10+ (o Anaconda Python)
- OpenCV (`opencv-python`)
- NumPy, Matplotlib, PyYAML, Scikit-learn
- CoppeliaSim Edu v4.3+ (`coppeliasim-zmqremoteapi-client`)

### Pasos de Ejecución

1. **Generar el Dataset:**
```bash
python -m src.generate_dataset
```

2. **Entrenar y Evaluar la CNN:**
```bash
python -m src.train
```

3. **Ejecutar el Protocolo Experimental y Benchmarks:**
```bash
python -m src.benchmark
```

4. **Ejecutar la Aplicación Interactiva en Tiempo Real:**
```bash
python -m src.main_app
```

*Controles en la Aplicación Interactiva:*
- `[0, 1, 2, 3, 4]`: Simular gestos manualmente.
- `[R]`: Reiniciar búfer de filtro temporal.
- `[ESPACIO]`: Ejecutar ciclo de Pick & Place en CoppeliaSim.
- `[Q]` o `[ESC]`: Salir de la aplicación.

---

## 10. ALINEACIÓN ABET (SO1, SO6)

- **SO1 — RAE 1.3:** Aplica Redes Neuronales Convolucionales para la solución de problemas mecatrónicos de percepción visual.
- **SO6 — RAE 6.1:** Diseña y argumenta experimentos controlados libres de fuga de datos para medir el desempeño de algoritmos de Inteligencia Artificial.
- **SO6 — RAE 6.2:** Realiza inferencias rigurosas sobre el desempeño de la CNN y el control robótico a partir de matrices de confusión, curvas ROC, latencia $p50/p95$ y pruebas de robustez bajo condiciones adversas.
