"""
Main Interactive Application (Webcam + CNN Perception + CoppeliaSim Control)
Laboratorio 3: CNN para Reconocimiento de Gestos (0 a 4 dedos)
Universidad Militar Nueva Granada - Inteligencia Artificial
"""

import sys
import os
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np

try:
    from src.cnn_inference import GestureInferenceEngine
except ImportError:
    from cnn_inference import GestureInferenceEngine


def run_app():
    print("=" * 70)
    print("LABORATORIO 3: CNN PARA RECONOCIMIENTO DE GESTOS Y COPPELIASIM")
    print("Universidad Militar Nueva Granada - Mecatrónica")
    print("=" * 70)
    print("Iniciando motor de inferencia y adaptador de CoppeliaSim...")
    
    engine = GestureInferenceEngine()
    engine.robot_adapter.connect_simulator()
    
    # Try opening webcam with high resolution
    cap = cv2.VideoCapture(0)
    using_camera = cap.isOpened()
    
    if using_camera:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("[Éxito] Cámara web HD inicializada a 1280x720 correctamente.")
    else:
        print("[Aviso] No se detectó cámara física. Iniciando en MODO SIMULADOR DE CÁMARA.")
        print("Usa los números [0, 1, 2, 3, 4] en la ventana para alternar gestos.")
        
    win_name = "Laboratorio 3: CNN Gesture Recognition"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)
    
    current_sim_gesture = 0
    synthetic_hand_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    flip_camera = True  # Invertir cámara horizontalmente por defecto (modo espejo natural)
    
    running = True
    frame_idx = 0
    
    print("[Cámara] Inversión horizontal activada por defecto (modo espejo). Presiona [I] para alternar.")
    
    while running:
        t_now = time.time()
        
        if using_camera:
            ret, frame = cap.read()
            if not ret or frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            elif flip_camera:
                frame = cv2.flip(frame, 1)  # Volteo horizontal (modo espejo)
        else:
            # Generate simulated camera frame with animated background and hand
            synthetic_hand_frame[:] = (35, 40, 45)
            # Add grid pattern
            for x in range(0, 640, 40):
                cv2.line(synthetic_hand_frame, (x, 0), (x, 480), (45, 50, 58), 1)
            for y in range(0, 480, 40):
                cv2.line(synthetic_hand_frame, (0, y), (640, y), (45, 50, 58), 1)
                
            # Draw interactive hand inside ROI
            rx, ry, rw, rh = engine.roi["x"], engine.roi["y"], engine.roi["w"], engine.roi["h"]
            cx, cy = rx + rw // 2, ry + rh // 2
            
            # Palm
            cv2.circle(synthetic_hand_frame, (cx, cy + 20), 45, (120, 165, 215), -1)
            
            # Draw fingers for current_sim_gesture
            angles = [-35, -15, 5, 25, 45]
            for f in range(current_sim_gesture):
                rad = np.radians(angles[f] - 90)
                tx = int(cx + 80 * np.cos(rad))
                ty = int(cy + 20 + 80 * np.sin(rad))
                cv2.line(synthetic_hand_frame, (cx, cy + 10), (tx, ty), (120, 165, 215), 14)
                cv2.circle(synthetic_hand_frame, (tx, ty), 7, (120, 165, 215), -1)
                
            frame = synthetic_hand_frame.copy()

        # Process frame through CNN and CoppeliaSim pipeline
        hud_frame, accepted_cmd = engine.process_frame(frame, timestamp=t_now)
        
        cv2.imshow("Laboratorio 3: CNN Gesture Recognition", hud_frame)
        
        # Key event handling
        key = cv2.waitKey(15) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            print("Cerrando aplicación...")
            running = False
        elif key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4')]:
            current_sim_gesture = int(chr(key))
            print(f"[Simulador de Gesto] Cambiado a Gesto {current_sim_gesture} ({chr(key)} dedos)")
        elif key == ord('i') or key == ord('I') or key == ord('f') or key == ord('F'):
            flip_camera = not flip_camera
            print(f"[Cámara] Inversión horizontal: {'ACTIVADA (Modo Espejo)' if flip_camera else 'DESACTIVADA (Original)'}")
        elif key == ord('r') or key == ord('R'):
            engine.command_filter.reset()
            print("[Filtro] Búfer temporal reiniciado.")
        elif key == ord(' '):
            print("[CoppeliaSim] Ejecutando ciclo completo de Pick & Place de prueba...")
            res = engine.robot_adapter.execute_pick_and_place_cycle(target_object_index=1)
            print("Ciclo completado:", res)

        frame_idx += 1

    if using_camera:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_app()
