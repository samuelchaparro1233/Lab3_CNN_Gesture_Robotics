"""
Interactive Real Webcam Dataset Collector
Laboratorio 3: CNN para Reconocimiento de Gestos (0 a 4 dedos)
Universidad Militar Nueva Granada - Inteligencia Artificial
"""

import os
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np


import argparse
import glob

def get_class_counts(output_dir: str, classes: list) -> dict:
    """Returns current number of saved images for each class."""
    counts = {}
    for c in classes:
        folder = os.path.join(output_dir, c)
        if os.path.exists(folder):
            counts[c] = len(glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg")))
        else:
            counts[c] = 0
    return counts


def collect_webcam_dataset(
    output_dir: str = "dataset/raw",
    subject_id: str = "subj_01",
    burst_size: int = 50
):
    """
    Captures real hand images from webcam for each class (0 to 4 fingers)
    with interactive ROI guidance box, dynamic burst sizing, continuous stream mode,
    hand switching [M] (Derecha/Izquierda), and participant switching [P].
    """
    classes = ["0_dedos", "1_dedo", "2_dedos", "3_dedos", "4_dedos"]
    hands = ["der", "izq"]
    hand_labels = {"der": "MANO DERECHA", "izq": "MANO IZQUIERDA"}
    
    active_hand_idx = 0
    subject_counter = 1
    
    # Try parsing initial subject_id to extract integer if format is subj_XX
    if "subj_" in subject_id:
        try:
            subject_counter = int(subject_id.split("subj_")[1])
        except Exception:
            pass

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] No se pudo abrir la cámara web. Verifica que esté conectada y no esté en uso por otra app.")
        return

    print("=" * 75)
    print("RECOLECTOR AVANZADO DE IMÁGENES REALES CON CÁMARA WEB - LAB 3 UMNG")
    print(f"Sujeto ID: {subject_id} | Mano Activa: {hand_labels[hands[active_hand_idx]]}")
    print(f"Carpeta destino: {output_dir}")
    print("=" * 75)
    print("Controles:")
    print(" - [0, 1, 2, 3, 4] : Seleccionar clase activa (0 a 4 dedos).")
    print(" - [M]             : Cambiar mano activa (DERECHA <-> IZQUIERDA).")
    print(" - [I] o [F]       : Invertir cámara horizontalmente (modo espejo On/Off).")
    print(" - [P]             : Cambiar participante / sujeto (subj_01 -> subj_02 ...).")
    print(" - [ESPACIO]       : Capturar ráfaga de fotos.")
    print(" - [C]             : Modo Grabación Continua (inicia/detiene captura).")
    print(" - [W] / [S] o +/- : Aumentar / Disminuir cantidad de fotos por ráfaga.")
    print(" - [Q] o [ESC]     : Guardar y salir.")
    print("=" * 75)

    current_class_idx = 0
    roi = (720, 90, 380, 380)  # (x, y, w, h)
    continuous_mode = False
    continuous_counter = 0
    flip_camera = True  # Modo espejo activado por defecto
    
    cv2.namedWindow("Captura de Dataset Real - Lab 3 UMNG", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Captura de Dataset Real - Lab 3 UMNG", 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if flip_camera:
            frame = cv2.flip(frame, 1)
            
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Dynamic ROI scaling based on resolution
        if w >= 1200:
            box_size = min(int(h * 0.70), 400)
            rx = int(w * 0.58)
            ry = int((h - box_size) / 2)
            rw, rh = box_size, box_size
        else:
            box_size = min(int(h * 0.65), 260)
            rx = int(w * 0.50)
            ry = int((h - box_size) / 2)
            rw, rh = box_size, box_size

        class_name = classes[current_class_idx]
        active_hand = hands[active_hand_idx]
        hand_text = hand_labels[active_hand]
        counts = get_class_counts(output_dir, classes)

        # Continuous capture mode logic
        if continuous_mode:
            save_folder = os.path.join(output_dir, class_name)
            os.makedirs(save_folder, exist_ok=True)
            roi_crop = frame[ry:ry+rh, rx:rx+rw]
            roi_resized = cv2.resize(roi_crop, (128, 128))
            timestamp_str = int(time.time() * 1000)
            fname = f"{subject_id}_{active_hand}_{class_name}_{timestamp_str}_{continuous_counter:04d}.png"
            cv2.imwrite(os.path.join(save_folder, fname), roi_resized)
            continuous_counter += 1

        # Draw ROI box
        roi_color = (0, 0, 255) if continuous_mode else (0, 255, 120)
        cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), roi_color, 2)
        mode_text = "[GRABANDO CONTINUO]" if continuous_mode else f"Rafaga: {burst_size} fotos"
        cv2.putText(display, f"ROI: {class_name} | {hand_text} ({mode_text})", (rx, ry - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, roi_color, 2)

        # Draw Top Info Bar
        cv2.rectangle(display, (0, 0), (w, 42), (20, 24, 30), -1)
        top_str = f"PARTICIPANTE: {subject_id} | MANO: {hand_text} | CLASE: {class_name} | [M] Cambio Mano | [P] Cambio Sujeto"
        cv2.putText(display, top_str,
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)

        # Draw Bottom Statistics Bar
        cv2.rectangle(display, (0, h - 38), (w, h), (20, 24, 30), -1)
        stats_str = " | ".join([f"{c}: {counts[c]}" for c in classes])
        total_all = sum(counts.values())
        cv2.putText(display, f"TOTAL: {total_all} imgs ({stats_str})",
                    (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1)

        cv2.imshow("Captura de Dataset Real - Lab 3 UMNG", display)

        key = cv2.waitKey(20 if continuous_mode else 1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            break
        elif key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4')]:
            current_class_idx = int(chr(key))
            print(f"[Clase Cambiada] -> {classes[current_class_idx]} (Total fotos clase: {counts[classes[current_class_idx]]})")
        elif key == ord('m') or key == ord('M'):
            active_hand_idx = (active_hand_idx + 1) % len(hands)
            print(f"\n[CAMBIO DE MANO] -> Mano activa: {hand_labels[hands[active_hand_idx]]}")
        elif key == ord('p') or key == ord('P'):
            subject_counter += 1
            subject_id = f"subj_{subject_counter:02d}"
            print(f"\n[CAMBIO DE PARTICIPANTE] -> Nuevo Sujeto ID: {subject_id}")
        elif key == ord('w') or key == ord('W') or key == ord('+'):
            burst_size = min(500, burst_size + 25)
            print(f"[Ráfaga ajustada] -> {burst_size} fotos")
        elif key == ord('s') or key == ord('S') or key == ord('-'):
            burst_size = max(10, burst_size - 25)
            print(f"[Ráfaga ajustada] -> {burst_size} fotos")
        elif key == ord('i') or key == ord('I') or key == ord('f') or key == ord('F'):
            flip_camera = not flip_camera
            print(f"[Cámara] Modo espejo: {'ACTIVADO' if flip_camera else 'DESACTIVADO'}")
        elif key == ord('c') or key == ord('C'):
            continuous_mode = not continuous_mode
            status = "INICIADA" if continuous_mode else "DETENIDA"
            print(f"[Grabación Continua] -> {status} para {class_name} ({subject_id} - {hand_text})")
        elif key == ord(' '):
            save_folder = os.path.join(output_dir, class_name)
            os.makedirs(save_folder, exist_ok=True)
            
            active_hand = hands[active_hand_idx]
            print(f"\nCapturando ráfaga de {burst_size} fotos | {subject_id} | {hand_labels[active_hand]} | {class_name}...")
            count = 0
            while count < burst_size:
                r, f = cap.read()
                if not r:
                    break
                if flip_camera:
                    f = cv2.flip(f, 1)
                roi_crop = f[ry:ry+rh, rx:rx+rw]
                roi_resized = cv2.resize(roi_crop, (128, 128))
                
                timestamp_str = int(time.time() * 1000)
                fname = f"{subject_id}_{active_hand}_{class_name}_{timestamp_str}_{count+1:03d}.png"
                cv2.imwrite(os.path.join(save_folder, fname), roi_resized)
                
                # Visual countdown flash
                f_flash = f.copy()
                cv2.rectangle(f_flash, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 3)
                cv2.putText(f_flash, f"Guardando ({subject_id} - {active_hand}): {count+1}/{burst_size}",
                            (rx + 10, ry + rh//2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                cv2.imshow("Captura de Dataset Real - Lab 3 UMNG", f_flash)
                cv2.waitKey(20)
                count += 1
                
            updated_total = len(glob.glob(os.path.join(save_folder, "*.png")))
            print(f"[Éxito] Ráfaga completada. Total fotos en {class_name}: {updated_total}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Webcam Dataset Collector for Lab 3 CNN")
    parser.add_argument("--output", type=str, default="dataset/raw", help="Output directory")
    parser.add_argument("--subject", type=str, default="subj_01", help="Subject/Person identifier (e.g. subj_01)")
    parser.add_argument("--burst", type=int, default=50, help="Number of images per spacebar burst")
    args = parser.parse_args()

    collect_webcam_dataset(
        output_dir=args.output,
        subject_id=args.subject,
        burst_size=args.burst
    )
