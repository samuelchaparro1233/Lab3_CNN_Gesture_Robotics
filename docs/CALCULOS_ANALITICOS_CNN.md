# Memoria de Cálculos Analíticos y Estadísticos
## Laboratorio 3: CNN para Reconocimiento de Gestos y Control Robótico en CoppeliaSim
**Universidad Militar Nueva Granada — Facultad de Ingeniería Mecatrónica**  
**Autor:** Samuel Alejandro Chaparro Ortiz (7004072) — Equipo 7

---

## 1. Fórmulas de Transformación Espacial en la CNN

### a) Dimensión de salida de una capa Convolucional
Para una entrada de altura H_in, ancho W_in, tamaño de kernel K, stride S y padding P:

H_out = piso((H_in + 2 * P - K) / S) + 1  
W_out = piso((W_in + 2 * P - K) / S) + 1  

*En nuestro Bloque 1 (H_in = 64, K = 3, S = 1, P = 1):*  
H_out = piso((64 + 2 * 1 - 3) / 1) + 1 = 64  
W_out = piso((64 + 2 * 1 - 3) / 1) + 1 = 64  
*(El padding = 1 preserva exactamente la resolución espacial).*

---

### b) Dimensión de salida de MaxPooling (2x2, stride 2)
Para K = 2, S = 2, P = 0:

H_out = piso((H_in - K) / S) + 1 = H_in / 2  
W_out = piso((W_in - K) / S) + 1 = W_in / 2  

*En el Bloque 1 tras MaxPool (H_in = 64):*  
H_out = 64 / 2 = 32  
W_out = 64 / 2 = 32  

---

## 2. Cálculo Analítico de Parámetros Entrenables por Capa

### a) Capas Convolucionales con Batch Normalization
* **Pesos convolucionales (sin sesgo)**:  
  Param_conv = (C_in * K * K) * C_out  
  *(No se usa sesgo en Conv2D porque la capa siguiente es BatchNorm2D, la cual ya incluye término de desplazamiento beta).*
* **Parámetros entrenables de BatchNorm2D**:  
  Param_bn = 2 * C_out  *(escalado gamma y desplazamiento beta)*.
* **Total bloque convolucional**:  
  Total_bloque = Param_conv + Param_bn  

#### Desglose por capas:
1. **Bloque 1** (C_in = 1, C_out = 32, K = 3):  
   Conv1 = (1 * 3 * 3) * 32 = 288 pesos  
   BN1 = 2 * 32 = 64 parámetros  
   **Total Bloque 1 = 288 + 64 = 352 parámetros**

2. **Bloque 2** (C_in = 32, C_out = 64, K = 3):  
   Conv2 = (32 * 3 * 3) * 64 = 18.432 pesos  
   BN2 = 2 * 64 = 128 parámetros  
   **Total Bloque 2 = 18.432 + 128 = 18.560 parámetros**

3. **Bloque 3** (C_in = 64, C_out = 128, K = 3):  
   Conv3 = (64 * 3 * 3) * 128 = 73.728 pesos  
   BN3 = 2 * 128 = 256 parámetros  
   **Total Bloque 3 = 73.728 + 256 = 73.984 parámetros**

4. **Bloque 4** (C_in = 128, C_out = 256, K = 3):  
   Conv4 = (128 * 3 * 3) * 256 = 294.912 pesos  
   BN4 = 2 * 256 = 512 parámetros  
   **Total Bloque 4 = 294.912 + 512 = 295.424 parámetros**

---

### b) Capas Totalmente Conectadas (MLP Classifier)
Tras `AdaptiveAvgPool2d(4, 4)` sobre 256 canales, el vector aplanado tiene:  
Dim_flatten = 256 * 4 * 4 = 4.096 neuronas.

1. **Capa Densa Oculta (Linear 1: 4.096 a 256)**:  
   Pesos = 4.096 * 256 = 1.048.576  
   Sesgos (bias) = 256  
   **Total Linear 1 = 1.048.576 + 256 = 1.048.832 parámetros**

2. **Capa de Salida (Linear 2: 256 a 5)**:  
   Pesos = 256 * 5 = 1.280  
   Sesgos (bias) = 5  
   **Total Linear 2 = 1.280 + 5 = 1.285 parámetros**

---

### c) Total de Parámetros de la Red
Total = 352 + 18.560 + 73.984 + 295.424 + 1.048.832 + 1.285  
**Total Parámetros Entrenables = 1.438.437 parámetros**  

* **Tamaño en memoria (Float32 = 4 bytes por parámetro)**:  
  Memoria = 1.438.437 * 4 bytes = 5.753.748 bytes  
  **Memoria = 5.487 MB (~5.49 MB)**

---

## 3. Cálculo de Operaciones: MACs y FLOPs por Inferencia

Una operación MAC (Multiply-Accumulate) consta de 1 multiplicación + 1 suma (2 FLOPs).

* **Para una convolución**:  
  FLOPs_conv = 2 * (C_in * K * K * C_out) * H_out * W_out  
* **Para una capa lineal**:  
  FLOPs_fc = 2 * (N_in * N_out) + N_out  

1. **Conv1**: 2 * 288 * 64 * 64 = 2.359.296 FLOPs (2.36 MFLOPs)  
2. **Conv2**: 2 * 18.432 * 32 * 32 = 37.748.736 FLOPs (37.75 MFLOPs)  
3. **Conv3**: 2 * 73.728 * 16 * 16 = 37.748.736 FLOPs (37.75 MFLOPs)  
4. **Conv4**: 2 * 294.912 * 8 * 8 = 37.748.736 FLOPs (37.75 MFLOPs)  
5. **Linear 1**: 2 * (4.096 * 256) = 2.097.152 FLOPs (2.10 MFLOPs)  
6. **Linear 2**: 2 * (256 * 5) = 2.560 FLOPs  
7. **Poolings + Activaciones**: ~73.728 FLOPs  

* **Total de Operaciones por Inferencia**:  
  **Total FLOPs = 117.778.944 (~117.78 MFLOPs)**  
  **Total MACs = 58.889.472 (~58.89 MMACs)**

---

## 4. Cálculo del Campo Receptivo Efectivo (Receptive Field)

RF_nuevo = RF_anterior + (K - 1) * J_anterior  
J_nuevo = J_anterior * Stride  

* Estado inicial: RF_0 = 1 píxel, J_0 = 1.
1. **Conv1 (K=3, S=1)**: RF = 1 + (3 - 1) * 1 = **3**, J = 1 * 1 = 1  
2. **Pool1 (K=2, S=2)**: RF = 3 + (2 - 1) * 1 = **4**, J = 1 * 2 = 2  
3. **Conv2 (K=3, S=1)**: RF = 4 + (3 - 1) * 2 = **8**, J = 2 * 1 = 2  
4. **Pool2 (K=2, S=2)**: RF = 8 + (2 - 1) * 2 = **10**, J = 2 * 2 = 4  
5. **Conv3 (K=3, S=1)**: RF = 10 + (3 - 1) * 4 = **18**, J = 4 * 1 = 4  
6. **Pool3 (K=2, S=2)**: RF = 18 + (2 - 1) * 4 = **22**, J = 4 * 2 = 8  
7. **Conv4 (K=3, S=1)**: RF = 22 + (3 - 1) * 8 = **38**, J = 8 * 1 = 8  

**Resultado**: En la capa Conv4, cada punto del mapa de características abarca un campo receptivo de **38x38 píxeles**, cubriendo la estructura completa de la mano dentro de la resolución normalizada.

---

## 5. Tabla Resumen Capa por Capa

| Capa | Dimensión de Entrada | Dimensión de Salida | Kernel / Stride | Parámetros | FLOPs | Campo Receptivo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Conv1 + BN + ReLU** | [1, 1, 64, 64] | [1, 32, 64, 64] | 3x3 / s=1 | 352 | 2.36 M | 3x3 |
| **MaxPool 1** | [1, 32, 64, 64] | [1, 32, 32, 32] | 2x2 / s=2 | 0 | 0.03 M | 4x4 |
| **Conv2 + BN + ReLU** | [1, 32, 32, 32] | [1, 64, 32, 32] | 3x3 / s=1 | 18.560 | 37.75 M | 8x8 |
| **MaxPool 2** | [1, 64, 32, 32] | [1, 64, 16, 16] | 2x2 / s=2 | 0 | 0.02 M | 10x10 |
| **Conv3 + BN + ReLU** | [1, 64, 16, 16] | [1, 128, 16, 16] | 3x3 / s=1 | 73.984 | 37.75 M | 18x18 |
| **MaxPool 3** | [1, 128, 16, 16] | [1, 128, 8, 8] | 2x2 / s=2 | 0 | 0.01 M | 22x22 |
| **Conv4 + BN + ReLU** | [1, 128, 8, 8] | [1, 256, 8, 8] | 3x3 / s=1 | 295.424 | 37.75 M | 38x38 |
| **AdaptiveAvgPool** | [1, 256, 8, 8] | [1, 256, 4, 4] | Adaptativo | 0 | 0.02 M | 38x38 |
| **Flatten** | [1, 256, 4, 4] | [1, 4096] | — | 0 | 0 | 38x38 |
| **Dropout(0.4) + FC1**| [1, 4096] | [1, 256] | Matriz [256, 4096] | 1.048.832 | 2.10 M | 38x38 |
| **Dropout(0.3) + FC2**| [1, 256] | [1, 5] | Matriz [5, 256] | 1.285 | 0.003 M | 38x38 |
| **TOTAL** | — | — | — | **1.438.437** | **117.78 M** | **38x38** |

---

## 6. Cálculo del Tamaño Muestral Teórico (Fórmula de Cochran)

* Nivel de confianza: 95% (z = 1.96)
* Proporción esperada de varianza máxima: p = 0.50
* Margen de error aceptado: epsilon = 0.05 (+-5%)

n_min = (z^2 * p * (1 - p)) / (epsilon^2)  
n_min = (1.96^2 * 0.5 * 0.5) / (0.05^2) = **384.16 muestras por clase**

* **Cumplimiento**: El dataset contiene **500 muestras por clase** (2.500 en total), superando el requisito teórico.

---

## 7. Cálculo del Intervalo de Confianza al 95% (Wilson Score)

Datos sobre el conjunto de prueba ciego:
* Muestras de prueba: n = 375
* Aciertos: 372 de 375
* Proporción observada: p = 372 / 375 = **0.9920 (99.20%)**
* Nivel de confianza: 95% (z = 1.96)

Centro ajustado = (p + (z^2 / (2 * n))) / (1 + (z^2 / n)) = 0.987011  
Margen de error = (z / (1 + (z^2 / n))) * raiz((p * (1 - p) / n) + (z^2 / (4 * n^2))) = 0.010264  

* Límite Inferior: 0.987011 - 0.010264 = **0.9767 (97.67%)**  
* Límite Superior: 0.987011 + 0.010264 = **0.9973 (99.73%)**  

**Intervalo de Confianza Wilson al 95%: [97.67%, 99.73%]**  
Supera holgadamente el umbral crítico operacional del 90.0% con significancia estadística p < 0.05.

---

## 8. Presupuesto Temporal en Tiempo Real (CoppeliaSim Control)

* Periodo de cuadro a 30 FPS: T_frame = 1 / 30 = **33.33 ms**
* Latencia de Inferencia Mediana (p50): **2.42 ms**
* Latencia de Inferencia Percentil 95 (p95): **2.95 ms**
* **Carga de CPU de la CNN**: (2.42 ms / 33.33 ms) * 100% = **7.26%**
* **Holgura libre de procesamiento**: **92.74%**
