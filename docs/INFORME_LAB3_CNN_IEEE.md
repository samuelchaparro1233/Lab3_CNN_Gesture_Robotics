# Reconocimiento de Gestos Manuales Mediante Redes Neuronales Convolucionales para el Control Desacoplado de un Brazo Robótico en CoppeliaSim

**Autores:** Equipo de Investigación en Inteligencia Artificial y Robótica  
**Filiación:** Programa de Ingeniería Mecatrónica, Facultad de Ingeniería, Universidad Militar Nueva Granada, Bogotá D.C., Colombia  
**Asignatura:** Inteligencia Artificial (Semestre IX)  
**Ruta de Implementación:** **CoppeliaSim Edu** (Brazo Robótico Articulado 3 GDL + Pinza Paralela + Celda de Manufactura)  

---

## RESUMEN (*ABSTRACT*)
Este artículo presenta el diseño, entrenamiento, evaluación experimental y validación en tiempo real de un sistema de percepción visual mecatrónico basado en Redes Neuronales Convolucionales (CNN) para clasificar el número de dedos extendidos (0 a 4 dedos) frente a una cámara web y comandar de forma desacoplada un manipulador de 3 grados de libertad (GDL) más efector final en el simulador físico **CoppeliaSim**. 

Para asegurar validez científica, el conjunto de datos de 1,800 imágenes se particionó estrictamente por sujeto y sesión (1,200 entrenamiento en `subj01`–`04`, 300 validación en `subj05` y 300 prueba en sujeto no visto `subj06`), garantizando **cero fuga de datos** (*zero data leakage*). La arquitectura propuesta `GestureCNN_v1` (1,438,437 parámetros, 117.78 MFLOPs) integró cuatro bloques convolucionales con normalización por lotes (*Batch Normalization*), activación ReLU y regularización por Dropout. 

En la evaluación sobre el conjunto ciego independiente, el modelo alcanzó una **exactitud balanceada del 100.0%**, precisión del 100.0%, F1-Score de 1.000 con un intervalo de confianza al 95% de $[98.74\%, 100.0\%]$ (Wilson Score), y una latencia de inferencia en tiempo real de **$2.52\text{ ms}$ en la mediana ($p50$) y $2.73\text{ ms}$ en el percentil 95 ($p95$)** (> 350 FPS). Se realizó una comparativa formal con variantes arquitectónicas (Mobile/Separable y Shallow). 

Se diseñó un filtro temporal con ventana deslizante ($N=10$, $M=8$, umbral $\ge 0.85$, cooldown de $1.5\text{ s}$ y parada lógica en clase 0), el cual redujo la tasa de falsos comandos al 4.0% en 100 ensayos en vivo bajo 5 condiciones adversas (iluminación tenue, saturación, fondos complejos, rotación y oclusión). En 10 secuencias completas de recogida y depósito (*Pick & Place*) en CoppeliaSim, el sistema alcanzó una tasa de éxito del 100.0%, demostrando la solidez del desacoplamiento entre percepción, filtro y ejecución.

*Palabras clave—* Redes Neuronales Convolucionales (CNN), Reconocimiento de Gestos, Visión por Computador, CoppeliaSim, ZeroMQ Remote API, Robótica Industrial, Desacoplamiento de Percepción, Filtro Temporal de Comandos, ABET SO1/SO6.

---

## I. INTRODUCCIÓN Y CONTEXTUALIZACIÓN TEÓRICA

### A. Percepción Visual en Robótica y Redes Convolucionales
En la robótica mecatrónica moderna, la interacción persona-robot (HRI) mediante visión artificial exige interfaces sin contacto que procesen información espacial de forma determinista y con latencias inferiores a $30\text{ ms}$ ($> 30\text{ FPS}$) para garantizar control en tiempo real. 

Las Redes Neuronales Convolucionales (CNN) permiten extraer jerarquías de características visuales: desde bordes y gradientes en capas tempranas hasta geometrías complejas de falanges y palmas en capas profundas.

### B. El Problema de Fuga de Información (*Data Leakage*)
Un error crítico en visión por computador es repartir cuadros consecutivos de un mismo video entre entrenamiento y prueba. Debido a la correlación temporal y la presencia del mismo fondo e iluminación, la CNN tiende a memorizar el contexto en lugar de la morfología intrínseca de los dedos, reportando exactitudes artificialmente altas que colapsan en despliegue real. Por ende, la partición estricta por sujeto/sesión es un requisito metodológico insoslayable.

---

## II. METODOLOGÍA Y DISEÑO EXPERIMENTAL

### A. Especificación de Clases y Acciones en CoppeliaSim
| Clase | Gesto | Acción en CoppeliaSim | Rango Articular / Parámetros |
| :---: | :--- | :--- | :--- |
| **0** | Puño cerrado (0 dedos) | **Parada Lógica / Inhibición** | Congela posición y rechaza órdenes |
| **1** | 1 dedo extendido | **Paso Articulación 1 (Base)** | $\theta_1 \leftarrow \theta_1 \pm 15^\circ \in [-170^\circ, 170^\circ]$ |
| **2** | 2 dedos extendidos | **Paso Articulación 2 (Hombro)**| $\theta_2 \leftarrow \theta_2 \pm 10^\circ \in [-90^\circ, 90^\circ]$ |
| **3** | 3 dedos extendidos | **Paso Articulación 3 (Codo)** | $\theta_3 \leftarrow \theta_3 \pm 10^\circ \in [-120^\circ, 120^\circ]$ |
| **4** | 4 dedos extendidos | **Alternar Pinza (RG2)** | Apertura $5\text{ cm} \leftrightarrow$ Cierre $0\text{ cm}$ |

### B. Formulación Matemática de la Arquitectura `GestureCNN_v1`
Dado un tensor de entrada $X \in \mathbb{R}^{B \times C_{in} \times H \times W}$ con $C_{in}=1, H=64, W=64$:

1. **Convolución 2D con Batch Normalization y ReLU:**
$$Y = \text{ReLU}\left( \text{BN}\left( W * X \right) \right)$$
Donde la dimensión espacial de salida $H_{out}, W_{out}$ con kernel $K=3$, stride $S=1$, padding $P=1$ es:
$$H_{out} = \left\lfloor \frac{H + 2P - K}{S} \right\rfloor + 1 = \left\lfloor \frac{64 + 2(1) - 3}{1} \right\rfloor + 1 = 64$$

2. **Max-Pooling (Reducción $2 \times 2$):**
$$H_{pool} = \left\lfloor \frac{H_{out} - 2}{2} \right\rfloor + 1 = 32$$

3. **Función de Pérdida de Entropía Cruzada Multiclase:**
$$\mathcal{L}_{CE} = -\frac{1}{B} \sum_{i=1}^B \sum_{c=0}^4 y_{i,c} \ln\left( \frac{e^{z_{i,c}}}{\sum_{j=0}^4 e^{z_{i,j}}} \right)$$

---

## III. COMPARATIVA DE ARQUITECTURAS Y ANÁLISIS DE PROFUNDIDAD (ABET C1 N5)

Se diseñaron y evaluaron tres variantes para cuantificar el compromiso (*trade-off*) entre exactitud, capacidad representacional y latencia en el borde:

| Modelo | Profundidad (Capas) | Parámetros | Tamaño (MB) | MACs (FLOPs) | Exactitud Test (%) | Latencia $p50$ (ms) | Notas de Diseño |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`GestureCNN_v1` (Propuesta)** | **12** | **1,438,437** | **5.49** | **117.78 M** | **100.0%** | **2.52** | Máxima capacidad de abstracción; robusta ante sombras. |
| **`GestureCNN_Efficient`** | 9 | 284,128 | 1.08 | 34.12 M | 98.67% | 0.85 | Separable Conv + GAP ($-80\%$ params); ideal para microcontroladores. |
| **`GestureCNN_Shallow` (Base)** | 4 | 182,405 | 0.70 | 18.45 M | 91.33% | 0.62 | 2 bloques convolucionales; propensa a error con rotaciones. |

---

## IV. FILTRO TEMPORAL Y SEGURIDAD MECATRÓNICA

El módulo `CommandFilter` procesa la predicción cruda $(\hat{y}_t, p_t)$ según tres restricciones formales:
1. **Moda Estadística en Búfer:**
$$\text{Moda}(\{ \hat{y}_{t-N+1}, \dots, \hat{y}_t \}) = c^* \quad \text{con } \text{count}(c^*) \ge M = 8, \quad N=10$$
2. **Umbral de Confianza:**
$$\frac{1}{M} \sum_{k \in \mathcal{I}_{c^*}} p_k \ge \tau = 0.85$$
3. **Periodo Refractario (Cooldown):**
$$\Delta t = t_{\text{actual}} - t_{\text{último\_comando}} \ge T_{\text{cooldown}} = 1.5\text{ s}$$

---

## V. RESULTADOS EXPERIMENTALES Y DISCUSIÓN

### A. Métricas de Evaluación en Conjunto de Prueba Ciego
- **Exactitud Global:** $100.0\%$ ($IC_{95\%}: [98.74\%, 100.0\%]$)
- **Exactitud Balanceada:** $100.0\%$
- **F1-Score Macro:** $1.000$
- **Latencia Mediana ($p50$):** $2.52\text{ ms}$
- **Latencia Percentil 95 ($p95$):** $2.73\text{ ms}$

### B. Pruebas de Robustez Cuantitativa ante Perturbaciones (ABET C5 N5)
| Condición de Perturbación | Exactitud (%) | F1-Score | Latencia $p50$ (ms) | Diagnóstico / Comportamiento |
| :--- | :---: | :---: | :---: | :--- |
| **1. Nominal (Control)** | **100.0%** | **1.000** | **2.52** | Clasificación perfecta y respuesta inmediata. |
| **2. Baja Luz ($-50\%$ Brillo)** | **96.67%** | **0.966** | **2.55** | Excelente segmentación por escala de grises. |
| **3. Luz Intensa ($+50\%$ Brillo)** | **95.00%** | **0.949** | **2.53** | BatchNorm compensa saturación de brillo. |
| **4. Rotación Extrema ($\pm 35^\circ$)** | **93.33%** | **0.932** | **2.58** | La red conserva la topología de falanges. |
| **5. Oclusión Parcial ($25\%$)** | **91.67%** | **0.915** | **2.61** | Mantiene precisión por encima del umbral del 90%. |

### C. Secuencias de Manipulación en CoppeliaSim (Pick & Place)
En 10 ejecuciones completas de recogida de cilindro, cubo y esfera y transporte a la caja de depósito:
- Éxito de Percepción: **100%** (10/10)
- Aceptación de Comando: **100%** (10/10)
- Ejecución Física: **100%** (10/10)
- Tasa Global de Misión: **100%** (10/10)

---

## VI. CONCLUSIONES
1. Se demostró que el desacoplamiento estricto entre la capa de visión por computador (`cnn_inference.py`), el filtro de estabilidad temporal (`command_filter.py`) y el adaptador cinemático (`robot_adapter.py`) permite diagnosticar y aislar fallos con precisión mecatrónica.
2. La partición rigurosa por sujeto garantizó que el modelo generalizara eficientemente frente a usuarios no observados durante el entrenamiento.
3. La latencia de inferencia de $2.52\text{ ms}$ ($p50$) superó ampliamente los requisitos de tiempo real, operando a más de 350 cuadros por segundo.

---

## VII. BIBLIOGRAFÍA
1. I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. MIT Press, 2016.
2. R. Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed. Springer, 2022.
3. P. Corke, *Robotics, Vision and Control: Fundamental Algorithms in MATLAB*. Springer, 2017.
4. M. Sandler, A. Howard, et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks," *IEEE/CVF CVPR*, pp. 4510–4520, 2018.
5. Coppelia Robotics, *CoppeliaSim ZeroMQ Remote API Documentation*, 2024. [Online]. Available: https://www.coppeliarobotics.com/
