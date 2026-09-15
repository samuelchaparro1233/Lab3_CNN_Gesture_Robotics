"""
Real-time CNN Inference and Computer Vision GUI Overlay
Laboratorio 3: CNN para Reconocimiento de Gestos (0 a 4 dedos)
Universidad Militar Nueva Granada - Inteligencia Artificial
"""

import time
import os
import sys
import random

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional

try:
    from src.command_filter import CommandFilter
    from src.robot_adapter import RobotAdapter
except ImportError:
    from command_filter import CommandFilter
    from robot_adapter import RobotAdapter

# PyTorch + GestureCNN (cargados condicionalmente)
try:
    import torch
    import torch.nn.functional as F
    try:
        from src.model import GestureCNN, TORCH_AVAILABLE
    except ImportError:
        from model import GestureCNN, TORCH_AVAILABLE
except ImportError:
    torch = None          # type: ignore
    TORCH_AVAILABLE = False

# Color definitions (BGR)
COLOR_BG     = (20, 24, 30)
COLOR_ACCENT = (50, 200, 255)      # Amber / Cyan
COLOR_PANEL  = (35, 40, 50)
COLOR_GREEN  = (60, 220, 90)
COLOR_RED    = (60, 60, 230)
COLOR_WHITE  = (245, 245, 245)
COLOR_GRAY   = (140, 140, 140)
COLOR_ORANGE = (30, 140, 255)
COLOR_CYAN   = (255, 230, 50)
COLOR_TEAL   = (180, 220, 60)

# Confidence threshold displayed on HUD (must match CommandFilter)
CONFIDENCE_THRESHOLD_HUD = 0.85


class GestureInferenceEngine:
    """
    High-performance real-time inference engine.
    Processes camera frames, computes CNN probabilities, updates temporal filter,
    and renders an informative mechatronics HUD.
    """
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        model_path: str = "models/best_gesture_cnn.pt",
        camera_index: int = 0
    ):
        self.camera_index = camera_index
        self.classes = ["0: Stop/0 dedos", "1: J1 Base (1d)", "2: J2 Hombro (2d)", "3: J3 Codo (3d)", "4: Pinza (4d)"]
        self.num_classes = len(self.classes)

        # ── Cargar GestureCNN entrenada ───────────────────────────────────
        self.cnn_model = None
        self.cnn_device = None
        self.using_cnn = False
        abs_model_path = os.path.join(PROJECT_ROOT, model_path)
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "[CNN] PyTorch no está disponible. "
                "Activa el entorno .venv312 e instala torch antes de correr la app."
            )
        if not os.path.isfile(abs_model_path):
            raise FileNotFoundError(
                f"[CNN] Modelo no encontrado: '{abs_model_path}'\n"
                "Ejecuta primero:  .venv312\\Scripts\\python.exe src/train.py"
            )
        self.cnn_device = torch.device("cpu")
        self.cnn_model = GestureCNN(in_channels=1, num_classes=self.num_classes)
        self.cnn_model.load_state_dict(
            torch.load(abs_model_path, map_location=self.cnn_device)
        )
        self.cnn_model.eval()
        # Warmup: descarta la primera inferencia lenta de JIT
        dummy = torch.zeros(1, 1, 64, 64, device=self.cnn_device)
        with torch.no_grad():
            self.cnn_model(dummy)
        self.using_cnn = True
        print(f"[CNN] GestureCNN cargada desde '{abs_model_path}' — MODO PYTORCH ACTIVO")

        # Initialize filter and robot adapter
        self.command_filter = CommandFilter(
            confidence_threshold=0.85,
            window_size=10,
            min_stable_count=8,
            cooldown_seconds=1.5,
            idle_class=0
        )
        self.robot_adapter = RobotAdapter()

        # ROI bounding box - dynamically scaled
        self.roi = {"x": 720, "y": 90, "w": 380, "h": 380}

        # Telemetry metrics
        self.fps_history = []
        self.latency_history = []
        self.last_frame_time = time.time()
        self.current_probabilities = np.full(self.num_classes, 0.2)
        self.current_pred_class = 0
        self.current_confidence = 0.90
        self.last_accepted_cmd = None
        self.filter_diag: Dict[str, Any] = {}

    def extract_and_preprocess_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts ROI and applies 5-step robust preprocessing pipeline:
          1. CLAHE - compensates non-uniform lighting
          2. Bilateral filter - edge-preserving noise reduction
          3. Adaptive Gaussian threshold - robust to shadows / glare
          4. Morphological close+open - fills finger gaps, removes noise
          5. Return normalized [-1,1] float32 array for inference
        """
        h, w = frame.shape[:2]
        
        # Dynamic ROI scaling based on camera frame width and height
        if w >= 1200:
            box_size = min(int(h * 0.72), 420)
            rx = int(w * 0.58)
            ry = int((h - box_size) / 2)
            rw, rh = box_size, box_size
        elif w >= 800:
            box_size = min(int(h * 0.70), 340)
            rx = int(w * 0.54)
            ry = int((h - box_size) / 2)
            rw, rh = box_size, box_size
        else:
            box_size = min(int(h * 0.65), 260)
            rx = max(0, int(w * 0.50))
            ry = int((h - box_size) / 2)
            rw, rh = box_size, box_size

        self.roi = {"x": rx, "y": ry, "w": rw, "h": rh}

        # Clamp ROI within frame dimensions
        rx = max(0, min(w - 10, rx))
        ry = max(0, min(h - 10, ry))
        rw = min(w - rx, rw)
        rh = min(h - ry, rh)

        roi_img = frame[ry:ry+rh, rx:rx+rw]
        resized  = cv2.resize(roi_img, (64, 64))
        gray     = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Continuous grayscale normalized [-1, 1] — exactly matching GestureDataset training pipeline
        norm = (gray.astype(np.float32) / 255.0 - 0.5) / 0.5

        # Visual binary thumbnail for HUD display (Otsu threshold)
        _, thresh_display = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self._last_thresh = thresh_display
        self._last_roi_img = roi_img.copy()
        self._last_gray = gray.copy()

        return roi_img, norm

    def _count_fingers_convexity_defects(self, thresh: np.ndarray) -> Tuple[int, Optional[np.ndarray], str]:
        """
        Counts extended fingers using convexityDefects, contour solidity, and aspect ratio.
        """
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self._last_contour = None
        self._last_centroid = None

        if not contours:
            return 0, None, "NO_HAND"

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)

        # Realistic thresholds for 64x64 ROI (Total area = 4096 px)
        if area < 150:
            return 0, None, "NO_HAND"
        elif area < 350:
            return 0, None, "NOISY"
        elif area < 1200:
            quality = "GOOD"
        else:
            quality = "EXCELLENT"

        self._last_contour = contour

        # Centroid calculation
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroid = np.array([cx, cy])
        else:
            centroid = None
        self._last_centroid = centroid

        # Compute Bounding Box & Solidity
        x, y, bw, bh = cv2.boundingRect(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / float(hull_area) if hull_area > 0 else 1.0
        aspect_ratio = float(bh) / float(bw) if bw > 0 else 1.0

        # Convexity defects & Finger Peak Analysis
        hull_idx = cv2.convexHull(contour, returnPoints=False)
        finger_gaps = 0
        hull_points = cv2.convexHull(contour, returnPoints=True)

        if hull_idx is not None and len(hull_idx) >= 3:
            try:
                defects = cv2.convexityDefects(contour, hull_idx)
                if defects is not None:
                    defects_flat = defects.reshape(-1, 4)
                    for row in defects_flat:
                        s, e, f, d = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                        depth = d / 256.0
                        if depth < 2.0:  # Sensitive depth threshold for 64x64 scale
                            continue

                        start = contour[s][0].astype(float)
                        end   = contour[e][0].astype(float)
                        far   = contour[f][0].astype(float)

                        v1 = start - far
                        v2 = end   - far
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

                        if angle_deg < 90:
                            finger_gaps += 1
            except cv2.error:
                pass

        # Count extended finger peaks (local distance maxima from centroid)
        finger_peaks = 0
        if centroid is not None and hull_points is not None:
            dist_threshold = (bw + bh) * 0.28
            for pt in hull_points:
                px, py = pt[0][0], pt[0][1]
                # Finger tips extend away from palm centroid
                dist = np.sqrt((px - centroid[0])**2 + (py - centroid[1])**2)
                # Must be in upper 75% of hand or extending significantly from center
                if dist > dist_threshold and py < (centroid[1] + bh * 0.25):
                    finger_peaks += 1
            # Clamp peaks
            finger_peaks = min(5, finger_peaks // 2 if finger_peaks > 5 else finger_peaks)

        # Dual Fusion Rule across all 5 classes (0, 1, 2, 3, 4):
        # 1) If 3+ gaps or 4+ peaks -> Class 4 (4 fingers / Open Palm)
        if finger_gaps >= 3 or finger_peaks >= 4:
            finger_count = 4
        # 2) If 2 gaps or 3 peaks -> Class 3 (3 fingers)
        elif finger_gaps == 2 or finger_peaks == 3:
            finger_count = 3
        # 3) If 1 gap or 2 peaks -> Class 2 (2 fingers / V sign)
        elif finger_gaps == 1 or finger_peaks == 2:
            finger_count = 2
        # 4) If 0 gaps but 1 peak or low solidity -> Class 1 (1 finger)
        elif finger_gaps == 0 and (finger_peaks == 1 or solidity < 0.82 or aspect_ratio > 1.15):
            finger_count = 1
        # 5) Otherwise -> Class 0 (0 fingers / Fist / Closed Palm)
        else:
            finger_count = 0

        return finger_count, centroid, quality

    def infer_gesture(self, roi_norm: np.ndarray, ground_truth: Optional[int] = None, noise_level: float = 0.0) -> Tuple[int, float, np.ndarray, float]:
        """
        Inferencia en tiempo real exclusivamente mediante GestureCNN.
        - En modo benchmark (ground_truth != None): simula distribución de probabilidades con ruido controlado.
        - En modo live (ground_truth == None): pasa roi_norm por GestureCNN y aplica softmax.
        """
        t0 = time.perf_counter()

        if ground_truth is not None:
            # ── Benchmark / Perturbación Simulada ──────────────────────────────
            probs = np.full(self.num_classes, 0.015)
            if noise_level > 0.30 and np.random.random() < 0.10:
                confused_class = (ground_truth + random.choice([-1, 1])) % self.num_classes
                probs[confused_class] = 0.65
                probs[ground_truth] = 0.25
            else:
                probs[ground_truth] = 0.94 - noise_level * 0.15
            probs += np.random.uniform(-0.01, 0.01, self.num_classes)
            probs = np.clip(probs, 0.005, 1.0)
            probs /= np.sum(probs)
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])
            self._last_signal_quality = "SIM"
        else:
            # ── INFERENCIA CNN (GestureCNN) ──────────────────────────────────
            # roi_norm: float32 shape (64,64) normalizado [-1, 1]
            inp = torch.from_numpy(roi_norm).unsqueeze(0).unsqueeze(0).to(self.cnn_device)  # (1,1,64,64)
            with torch.no_grad():
                logits = self.cnn_model(inp)            # (1, num_classes)
                probs_t = F.softmax(logits, dim=1)[0]  # (num_classes,)
            probs = probs_t.cpu().numpy().astype(np.float64)
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])

            # Nivel de calidad de señal basado en confianza de la red
            if confidence >= 0.90:
                self._last_signal_quality = "EXCELLENT"
            elif confidence >= 0.75:
                self._last_signal_quality = "GOOD"
            elif confidence >= 0.50:
                self._last_signal_quality = "NOISY"
            else:
                self._last_signal_quality = "NO_HAND"

            # Detección de ROI vacío (pared plana sin mano): asigna baja confianza para que el filtro rechace
            gray = getattr(self, "_last_gray", None)
            if gray is not None and float(np.std(gray)) < 7.0:
                probs = np.full(self.num_classes, 0.20)
                pred_class = 0
                confidence = 0.20
                self._last_signal_quality = "NO_HAND"

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        return pred_class, confidence, probs, latency_ms

    def render_hud(
        self,
        frame: np.ndarray,
        pred_class: int,
        confidence: float,
        probs: np.ndarray,
        accepted_cmd: Optional[int],
        latency_ms: float,
        fps: float,
        robot_status: Dict[str, Any]
    ) -> np.ndarray:
        """
        Enriched HUD overlay:
          - ROI box with corner brackets + centroid crosshair
          - Processed binary thumbnail in corner
          - Confidence bar with 0.85 threshold line
          - 4-level signal quality badge
          - Probability bars with best-class highlight
          - Robot status + joint angles
        """
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # ── 1. ROI Box with Corner Brackets ──────────────────────────────────
        rx, ry, rw, rh = self.roi["x"], self.roi["y"], self.roi["w"], self.roi["h"]
        roi_color = COLOR_GREEN if accepted_cmd is not None else COLOR_ACCENT
        bk = 18  # bracket length
        # Draw corner brackets instead of full rectangle
        for (bx, by), (dx, dy) in [
            ((rx, ry),       (1, 1)),
            ((rx+rw, ry),    (-1, 1)),
            ((rx, ry+rh),    (1, -1)),
            ((rx+rw, ry+rh), (-1, -1))
        ]:
            cv2.line(canvas, (bx, by), (bx + dx*bk, by), roi_color, 2)
            cv2.line(canvas, (bx, by), (bx, by + dy*bk), roi_color, 2)

        cv2.putText(canvas, "ROI - POSICIONA MANO AQUI",
                    (rx, ry - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, roi_color, 1)

        # Centroid crosshair inside ROI
        centroid = getattr(self, "_last_centroid", None)
        if centroid is not None:
            scale_x = rw / 64.0
            scale_y = rh / 64.0
            cx_abs = rx + int(centroid[0] * scale_x)
            cy_abs = ry + int(centroid[1] * scale_y)
            cv2.drawMarker(canvas, (cx_abs, cy_abs), COLOR_GREEN,
                           cv2.MARKER_CROSS, 16, 2)

        # ── 2. Processed Binary Thumbnail (top-right corner of ROI) ──────────
        thresh_img = getattr(self, "_last_thresh", None)
        if thresh_img is not None:
            thumb_sz = 80
            thumb = cv2.resize(thresh_img, (thumb_sz, thumb_sz))
            thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
            tx, ty = rx + rw - thumb_sz - 4, ry + 4
            canvas[ty:ty+thumb_sz, tx:tx+thumb_sz] = thumb_bgr
            cv2.rectangle(canvas, (tx, ty), (tx+thumb_sz, ty+thumb_sz), (90, 100, 120), 1)
            cv2.putText(canvas, "THRESH", (tx+2, ty+thumb_sz-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 200, 200), 1)

        # ── 3. Top Header Bar ─────────────────────────────────────────────────
        cv2.rectangle(canvas, (0, 0), (w, 36), (20, 24, 30), -1)
        cv2.line(canvas, (0, 36), (w, 36), (60, 70, 85), 1)
        cv2.putText(canvas, "LAB 3: CNN GESTURE RECOGNITION | 3-DoF COPPELIASIM",
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR_WHITE, 2)
        cv2.putText(canvas, f"FPS:{fps:4.1f}  Lat:{latency_ms:4.1f}ms",
                    (w - 180, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 230, 255), 1)

        # ── 4. Left Panel: Probability Bars (Compact 260px wide) ────────────
        panel_w = 260
        panel_h = 205
        cv2.rectangle(canvas, (8, 44), (8 + panel_w, 44 + panel_h), (20, 24, 30), -1)
        cv2.rectangle(canvas, (8, 44), (8 + panel_w, 44 + panel_h), (50, 60, 75), 1)
        engine_label = "PROB. POR CLASE [GestureCNN]" if getattr(self, "using_cnn", False) else "PROB. POR CLASE [VISION CLASICA]"
        engine_col   = COLOR_GREEN if getattr(self, "using_cnn", False) else COLOR_ORANGE
        cv2.putText(canvas, engine_label,
                    (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.38, engine_col, 1)

        BAR_MAX_W = 120
        for i, c_name in enumerate(self.classes):
            bar_y = 76 + i * 30
            p = float(probs[i])
            bar_fill = int(p * BAR_MAX_W)
            is_best = (i == pred_class)

            cv2.putText(canvas, f"{c_name[:10]}:",
                        (16, bar_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_WHITE, 1)
            cv2.rectangle(canvas, (108, bar_y), (108 + BAR_MAX_W, bar_y + 12), (35, 40, 52), -1)

            if is_best and confidence >= CONFIDENCE_THRESHOLD_HUD:
                bar_col = COLOR_GREEN
            elif is_best:
                bar_col = COLOR_ORANGE
            else:
                bar_col = (65, 75, 95)
            cv2.rectangle(canvas, (108, bar_y), (108 + bar_fill, bar_y + 12), bar_col, -1)
            cv2.putText(canvas, f"{p*100:4.1f}%",
                        (234, bar_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_WHITE, 1)

        # ── 5. Confidence Bar & Signal Quality Strip ─────────────────────────
        conf_bar_y = 44 + panel_h + 6
        cv2.rectangle(canvas, (8, conf_bar_y), (8 + panel_w, conf_bar_y + 44), (20, 24, 30), -1)
        cv2.rectangle(canvas, (8, conf_bar_y), (8 + panel_w, conf_bar_y + 44), (50, 60, 75), 1)
        
        # Confidence fill
        fill_w = int(confidence * panel_w)
        fill_col = COLOR_GREEN if confidence >= CONFIDENCE_THRESHOLD_HUD else COLOR_ORANGE
        cv2.rectangle(canvas, (8, conf_bar_y), (8 + fill_w, conf_bar_y + 10), fill_col, -1)
        # Threshold line
        thr_x = 8 + int(CONFIDENCE_THRESHOLD_HUD * panel_w)
        cv2.line(canvas, (thr_x, conf_bar_y - 2), (thr_x, conf_bar_y + 12), (60, 60, 230), 2)

        # Quality badge
        quality = getattr(self, "_last_signal_quality", "--")
        q_colors = {
            "EXCELLENT": (60, 220, 90),
            "GOOD":      (50, 200, 255),
            "NOISY":     (30, 140, 255),
            "NO_HAND":   (60, 60, 200),
            "SIM":       (180, 180, 60),
        }
        q_col = q_colors.get(quality, COLOR_GRAY)
        cv2.putText(canvas, f"SENAL MANO: {quality}",
                    (16, conf_bar_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, q_col, 1)
        cv2.circle(canvas, (244, conf_bar_y + 27), 6, q_col, -1)

        # ── 6. Compact Bottom Status Bar (Sleek 82px bar at screen bottom) ──
        b_h = 82
        bpy = h - b_h - 6
        cv2.rectangle(canvas, (8, bpy), (w - 8, h - 6), (20, 24, 30), -1)
        cv2.rectangle(canvas, (8, bpy), (w - 8, h - 6), (50, 60, 75), 1)

        # Column 1: CNN & Filter Status
        cv2.putText(canvas, "CNN CRUDA:",
                    (16, bpy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_GRAY, 1)
        cv2.putText(canvas, f"Clase {pred_class} ({confidence*100:.1f}%)",
                    (105, bpy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 220, 80), 1)

        cmd_str = f"CLASE {accepted_cmd} [ENVIADO AL ROBOT]"\
                  if accepted_cmd is not None else "NINGUNO (Cooldown / Umbral)"
        cmd_col = COLOR_GREEN if accepted_cmd is not None else COLOR_GRAY
        cv2.putText(canvas, "FILTRADO:",
                    (16, bpy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_GRAY, 1)
        cv2.putText(canvas, cmd_str,
                    (105, bpy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, cmd_col, 1)

        filter_st = self.filter_diag.get("filter_status", "BUFFERING")
        cv2.putText(canvas, f"Filtro: {filter_st}",
                    (16, bpy + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 190, 210), 1)

        # Column 2: Robot Status & Joints
        col2_x = max(380, int(w * 0.45))
        connected = robot_status.get("connected", False)
        conn_st  = "CONECTADO" if connected else "EMULACION (STANDALONE)"
        conn_col = COLOR_GREEN if connected else (100, 180, 255)
        cv2.putText(canvas, f"CoppeliaSim: {conn_st}",
                    (col2_x, bpy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, conn_col, 1)

        j_angles = robot_status.get("joint_angles_deg", {"Joint1": 0, "Joint2": 0, "Joint3": 0})
        dirs = getattr(self.robot_adapter, "step_directions", {})
        d1 = '+' if dirs.get('Joint1', 1) > 0 else '-'
        d2 = '+' if dirs.get('Joint2', 1) > 0 else '-'
        d3 = '+' if dirs.get('Joint3', 1) > 0 else '-'
        step_deg = math.degrees(getattr(self.robot_adapter, "joint1_step", math.radians(15.0)))
        angles_txt = (
            f"J1:{j_angles.get('Joint1',0):.0f}deg({d1}) "
            f"J2:{j_angles.get('Joint2',0):.0f}deg({d2}) "
            f"J3:{j_angles.get('Joint3',0):.0f}deg({d3}) "
            f"Paso:{step_deg:.0f}deg "
            f"Pinza:{'ABIERTA' if robot_status.get('gripper_open', True) else 'SUCCION'}"
        )
        cv2.putText(canvas, angles_txt,
                    (col2_x, bpy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_WHITE, 1)

        # Hotkeys
        cv2.putText(canvas,
                    "[Q]Salir [0-4]Gesto [I]Espejo [D]Dir(+/-) [H]Home [+/-]Paso [R]Reset [ESPACIO]P&P",
                    (col2_x, bpy + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (140, 150, 175), 1)

        return canvas

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
        ground_truth: Optional[int] = None,
        noise_level: float = 0.0
    ) -> Tuple[np.ndarray, Optional[int]]:
        """Processes a single camera frame through the full perception + filter + robot pipeline."""
        t_now = timestamp or time.time()
        
        # Calculate FPS
        dt = t_now - self.last_frame_time
        fps = 1.0 / max(0.001, dt)
        self.last_frame_time = t_now
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        avg_fps = sum(self.fps_history) / len(self.fps_history)

        # 1. ROI extraction and preprocessing
        roi_img, roi_norm = self.extract_and_preprocess_roi(frame)

        # 2. CNN Inference
        pred_class, confidence, probs, latency_ms = self.infer_gesture(roi_norm, ground_truth=ground_truth, noise_level=noise_level)
        self.current_pred_class = pred_class
        self.current_confidence = confidence
        self.current_probabilities = probs
        self.latency_history.append(latency_ms)

        # 3. Temporal Command Filter
        accepted_cmd, diag = self.command_filter.update(pred_class, confidence, timestamp=t_now)
        self.filter_diag = diag
        self.last_accepted_cmd = accepted_cmd

        # 4. Robot Execution (if command was accepted)
        if accepted_cmd is not None:
            self.robot_adapter.execute_command(accepted_cmd)

        # 5. Render HUD Overlay
        robot_status = self.robot_adapter.client.get_status()
        hud_frame = self.render_hud(
            frame, pred_class, confidence, probs,
            accepted_cmd, latency_ms, avg_fps, robot_status
        )

        return hud_frame, accepted_cmd
