"""
Script to generate visual assets for GitHub README and ABET documentation:
1. results/plots/dataset_samples_gallery.png: Grid of representative dataset photos across classes and subjects.
2. results/plots/system_pipeline_architecture.png: Visual diagram of the perception, filtering, and robotics pipeline.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs("results/plots", exist_ok=True)

def generate_dataset_gallery():
    classes = [
        ("0_dedos", "Clase 0: 0 Dedos (Puño)\nInhibición / Parada Lógica"),
        ("1_dedo", "Clase 1: 1 Dedo (Índice)\nJoint 1: Base Yaw (±25°)"),
        ("2_dedos", "Clase 2: 2 Dedos (Índice+Medio)\nJoint 2: Shoulder Pitch (±20°)"),
        ("3_dedos", "Clase 3: 3 Dedos\nJoint 3: Elbow Pitch (±20°)"),
        ("4_dedos", "Clase 4: 4 Dedos\nPinza / Succión: Alternar Open/Close")
    ]
    
    # Select 4 samples per class from train/test showing variety (subj_01, subj_02, etc.)
    fig, axes = plt.subplots(5, 4, figsize=(14, 16))
    fig.patch.set_facecolor('#0f172a') # Slate dark background
    
    for row_idx, (c_folder, label_text) in enumerate(classes):
        samples = []
        # Try both test and train to find subj_01 and subj_02
        for sdir in ['test', 'train', 'val']:
            p = os.path.join('dataset', sdir, c_folder)
            if os.path.exists(p):
                files = sorted(os.listdir(p))
                s1 = [f for f in files if 'subj_01' in f or 'subj_user' in f]
                s2 = [f for f in files if 'subj_02' in f]
                if s1 and len(samples) < 2:
                    samples.append(os.path.join(p, s1[len(s1)//4]))
                    if len(s1) > 2:
                        samples.append(os.path.join(p, s1[3*len(s1)//4]))
                if s2 and len(samples) < 4:
                    samples.append(os.path.join(p, s2[len(s2)//4]))
                    if len(s2) > 2:
                        samples.append(os.path.join(p, s2[3*len(s2)//4]))
            if len(samples) >= 4:
                break
        
        # Fallback if specific subjects not found in that exact split
        if len(samples) < 4:
            p = os.path.join('dataset', 'test', c_folder)
            all_f = sorted(os.listdir(p))
            samples = [os.path.join(p, all_f[i * (len(all_f)//4)]) for i in range(4)]
            
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            img_path = samples[col_idx]
            bgr = cv2.imread(img_path)
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                ax.imshow(rgb)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Border style
            for spine in ax.spines.values():
                spine.set_edgecolor('#38bdf8')
                spine.set_linewidth(1.5)
                
            subj_tag = "Subj 1 (Mano Der.)" if col_idx < 2 else "Subj 2 (Mano Izq./Var.)"
            ax.set_xlabel(subj_tag, color='#94a3b8', fontsize=9, fontweight='semibold', labelpad=4)
            
            if col_idx == 0:
                ax.set_ylabel(label_text, color='#f8fafc', fontsize=10, fontweight='bold', labelpad=8)
                
    plt.suptitle("Galería de Muestras del Dataset Multi-Sujeto (0 a 4 Dedos)\nVariación de Participantes, Manos, Iluminación y Distancias (128x128 px)",
                 color='#38bdf8', fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    out_path = "results/plots/dataset_samples_gallery.png"
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[OK] Generada galería: {out_path}")


def generate_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor('#0b0f19')
    ax.set_facecolor('#0b0f19')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Blocks definition
    blocks = [
        {"x": 0.5, "y": 2.2, "w": 2.2, "h": 2.6, "title": "1. Captura en Vivo", "sub": "Webcam HD (1280x720)\nModo Espejo [I]\nROI 320x320 central", "color": "#0284c7"},
        {"x": 3.2, "y": 2.2, "w": 2.2, "h": 2.6, "title": "2. Preprocesamiento", "sub": "Resize a 128x128\nEscala de Grises\nNormalización [0, 1]\nTensores PyTorch", "color": "#0d9488"},
        {"x": 5.9, "y": 1.9, "w": 2.5, "h": 3.2, "title": "3. GestureCNN_v1", "sub": "4 Bloques Conv2D + BN\nReLU + MaxPool + Dropout\nGlobal Avg Pooling\n1.44M Params (5.49 MB)\nInferencia: 2.3 ms", "color": "#6366f1"},
        {"x": 8.9, "y": 2.0, "w": 2.3, "h": 3.0, "title": "4. Filtro Temporal", "sub": "Ventana N = 10 cuadros\nConsenso M = 8 concordantes\nUmbral P >= 0.85\nCooldown = 1.5 s\nRechazo de Falsos Pos.", "color": "#d97706"},
        {"x": 11.7, "y": 2.2, "w": 2.0, "h": 2.6, "title": "5. Robot Adapter", "sub": "Traducción a Comandos\nClamping de Seguridad\nLímites Articulares\nControl de Pinza", "color": "#16a34a"},
        {"x": 14.1, "y": 2.2, "w": 1.6, "h": 2.6, "title": "6. CoppeliaSim", "sub": "ZeroMQ API\nPuerto 23000\nuArm Swift Pro\n3-GDL + Succión", "color": "#9333ea"},
    ]
    
    for b in blocks:
        # Card box
        box = patches.FancyBboxPatch((b["x"], b["y"]), b["w"], b["h"],
                                     boxstyle="round,pad=0.1,rounding_size=0.2",
                                     linewidth=2, edgecolor=b["color"], facecolor='#1e293b')
        ax.add_patch(box)
        
        # Header strip
        header_strip = patches.FancyBboxPatch((b["x"], b["y"] + b["h"] - 0.7), b["w"], 0.7,
                                              boxstyle="round,pad=0.05,rounding_size=0.15",
                                              linewidth=0, facecolor=b["color"])
        ax.add_patch(header_strip)
        
        # Text header
        ax.text(b["x"] + b["w"]/2, b["y"] + b["h"] - 0.35, b["title"],
                color="#ffffff", fontsize=10, fontweight="bold", ha="center", va="center")
        
        # Text body
        ax.text(b["x"] + b["w"]/2, b["y"] + (b["h"] - 0.7)/2, b["sub"],
                color="#e2e8f0", fontsize=8.5, ha="center", va="center", linespacing=1.35)
                
    # Arrows connecting blocks
    for i in range(len(blocks)-1):
        x_start = blocks[i]["x"] + blocks[i]["w"] + 0.05
        x_end = blocks[i+1]["x"] - 0.05
        y_pos = 3.5
        ax.annotate("", xy=(x_end, y_pos), xytext=(x_start, y_pos),
                    arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.4",
                                    color="#38bdf8", lw=2.5))
                    
    # Bottom legend and annotations
    ax.text(8.0, 6.3, "ARQUITECTURA DEL SISTEMA INTEGRAL EN TIEMPO REAL",
            color="#38bdf8", fontsize=16, fontweight="bold", ha="center")
    ax.text(8.0, 5.85, "Pipeline de Percepción CNN, Estabilización por Consenso Temporal y Teleoperación Robótica ZeroMQ",
            color="#94a3b8", fontsize=11, ha="center")
            
    # Guarantee metric banners at bottom
    badges = [
        ("Tasa de Inferencia", ">400 FPS (2.3 ms)", "#38bdf8"),
        ("Exactitud Balanceada", "95.06% (Test Ciego)", "#4ade80"),
        ("Comandos Falsos", "< 1.0% (Filtro N=10)", "#fbbf24"),
        ("Puerto de Enlace", "ZeroMQ 127.0.0.1:23000", "#c084fc")
    ]
    
    for idx, (label, val, col) in enumerate(badges):
        bx = 1.0 + idx * 3.7
        b_patch = patches.FancyBboxPatch((bx, 0.4), 3.2, 0.9,
                                         boxstyle="round,pad=0.08,rounding_size=0.15",
                                         linewidth=1, edgecolor=col, facecolor='#1e293b')
        ax.add_patch(b_patch)
        ax.text(bx + 1.6, 0.95, label, color="#94a3b8", fontsize=8, fontweight="semibold", ha="center")
        ax.text(bx + 1.6, 0.65, val, color=col, fontsize=10, fontweight="bold", ha="center")

    out_path = "results/plots/system_pipeline_architecture.png"
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[OK] Generado diagrama de pipeline: {out_path}")

if __name__ == "__main__":
    generate_dataset_gallery()
    generate_pipeline_diagram()
