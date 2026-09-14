"""
Hand Gesture Dataset Generator with Controlled Multi-Subject Variations
Laboratorio 3: CNN para Reconocimiento de Gestos (0 a 4 dedos)
Universidad Militar Nueva Granada - Inteligencia Artificial
"""

import os
import random
import math
import cv2
import numpy as np
from typing import Tuple, List, Dict


def draw_procedural_hand(
    canvas_size: Tuple[int, int],
    num_fingers: int,
    hand_color: Tuple[int, int, int],
    background_type: str = "plain",
    rotation_deg: float = 0.0,
    scale: float = 1.0,
    offset: Tuple[int, int] = (0, 0),
    noise_level: float = 0.0,
    blur_kernel: int = 0
) -> np.ndarray:
    """
    Renders a realistic hand gesture mask with palm, knuckles, and specified number of fingers.
    """
    h, w = canvas_size
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # 1. Background generation
    if background_type == "plain":
        bg_val = random.randint(30, 220)
        img[:] = (bg_val, bg_val, bg_val)
    elif background_type == "gradient":
        for y in range(h):
            val = int(40 + (180 - 40) * (y / h))
            img[y, :] = (val, val + random.randint(-10, 10), val)
    elif background_type == "cluttered":
        img[:] = random.randint(40, 180)
        for _ in range(random.randint(5, 12)):
            pt1 = (random.randint(0, w), random.randint(0, h))
            pt2 = (random.randint(0, w), random.randint(0, h))
            col = (random.randint(20, 230), random.randint(20, 230), random.randint(20, 230))
            cv2.rectangle(img, pt1, pt2, col, -1)
            cv2.line(img, pt1, pt2, (col[0]//2, col[1]//2, col[2]//2), random.randint(1, 4))

    # Base Hand coordinates (centered in 128x128 space, then scaled/rotated)
    base_center = (w // 2 + offset[0], h // 2 + 10 + offset[1])
    palm_radius = int(22 * scale)
    wrist_w = int(18 * scale)
    wrist_h = int(35 * scale)

    # Hand canvas overlay
    hand_layer = np.zeros((h, w, 4), dtype=np.uint8)

    # Draw Wrist
    wrist_pt1 = (base_center[0] - wrist_w, base_center[1])
    wrist_pt2 = (base_center[0] + wrist_w, base_center[1] + wrist_h)
    cv2.rectangle(hand_layer, wrist_pt1, wrist_pt2, (*hand_color, 255), -1)

    # Draw Palm (Ellipse / Circle)
    cv2.circle(hand_layer, base_center, palm_radius, (*hand_color, 255), -1)
    cv2.ellipse(hand_layer, (base_center[0], base_center[1] - 5), (palm_radius, int(palm_radius * 1.15)), 0, 0, 360, (*hand_color, 255), -1)

    # Finger specifications: [(base_angle_deg, length, width)]
    # Angles relative to top (0 = straight up, negative = left, positive = right)
    all_finger_defs = [
        {"name": "index",  "angle": -15.0, "len": 38 * scale, "w": 6.5 * scale, "knuckle_dx": -10 * scale, "knuckle_dy": -18 * scale},
        {"name": "middle", "angle":  -2.0, "len": 42 * scale, "w": 6.5 * scale, "knuckle_dx":  -1 * scale, "knuckle_dy": -22 * scale},
        {"name": "ring",   "angle":  12.0, "len": 37 * scale, "w": 6.0 * scale, "knuckle_dx":   8 * scale, "knuckle_dy": -20 * scale},
        {"name": "pinky",  "angle":  26.0, "len": 28 * scale, "w": 5.5 * scale, "knuckle_dx":  16 * scale, "knuckle_dy": -14 * scale},
        {"name": "thumb",  "angle": -48.0, "len": 26 * scale, "w": 7.5 * scale, "knuckle_dx": -18 * scale, "knuckle_dy":  -2 * scale},
    ]

    # Select which fingers are extended based on num_fingers (0 to 4)
    extended_indices = []
    if num_fingers == 0:
        extended_indices = [] # Fist (all folded)
    elif num_fingers == 1:
        # Index finger
        extended_indices = [0]
    elif num_fingers == 2:
        # Index + Middle ("V" sign)
        extended_indices = [0, 1]
    elif num_fingers == 3:
        # Index + Middle + Ring (or Thumb + Index + Middle)
        extended_indices = [0, 1, 2]
    elif num_fingers == 4:
        # Index + Middle + Ring + Pinky (4 fingers)
        extended_indices = [0, 1, 2, 3]
    elif num_fingers >= 5:
        # All 5 fingers (OOD / 5 fingers)
        extended_indices = [0, 1, 2, 3, 4]

    # Draw folded fingers / knuckles (for realistic fist/contours)
    for idx, f_def in enumerate(all_finger_defs):
        kx = int(base_center[0] + f_def["knuckle_dx"])
        ky = int(base_center[1] + f_def["knuckle_dy"])
        if idx not in extended_indices:
            # Folded finger: draw compact rounded knuckle
            cv2.circle(hand_layer, (kx, ky), int(f_def["w"] * 1.1), (*hand_color, 255), -1)
        else:
            # Extended finger: draw articulated capsule
            f_len = f_def["len"]
            rad = math.radians(f_def["angle"] - 90) # polar coordinate
            tip_x = int(kx + f_len * math.cos(rad))
            tip_y = int(ky + f_len * math.sin(rad))
            cv2.line(hand_layer, (kx, ky), (tip_x, tip_y), (*hand_color, 255), int(f_def["w"] * 2))
            cv2.circle(hand_layer, (tip_x, tip_y), int(f_def["w"]), (*hand_color, 255), -1)
            cv2.circle(hand_layer, (kx, ky), int(f_def["w"]), (*hand_color, 255), -1)

    # 2. Rotation & Affine Transformation
    if abs(rotation_deg) > 0.1:
        M = cv2.getRotationMatrix2D(base_center, rotation_deg, 1.0)
        hand_layer = cv2.warpAffine(hand_layer, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # 3. Composite Hand Layer onto Background
    alpha = hand_layer[:, :, 3] / 255.0
    for c in range(3):
        img[:, :, c] = (1.0 - alpha) * img[:, :, c] + alpha * hand_layer[:, :, c]

    # 4. Add Noise & Blur if requested
    if noise_level > 0.0:
        noise = np.random.normal(0, noise_level * 25, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if blur_kernel > 1:
        k = blur_kernel if blur_kernel % 2 != 0 else blur_kernel + 1
        img = cv2.GaussianBlur(img, (k, k), 0)

    return img


def generate_full_dataset(
    output_base_dir: str = "dataset",
    samples_per_class: int = 350,
    random_seed: int = 42
):
    """
    Generates a balanced, multi-subject gesture recognition dataset partitioned
    strictly across Train, Val, and Test subsets.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    classes = ["0_dedos", "1_dedo", "2_dedos", "3_dedos", "4_dedos"]
    subjects = [f"subj{i+1:02d}" for i in range(6)]  # subj01 to subj06

    # Subject assignments:
    # Train: subj01, subj02, subj03, subj04 (4 subjects, ~67%)
    # Val:   subj05 (1 subject, ~16.5%)
    # Test:  subj06 (1 subject, ~16.5% - completely unseen during training)
    split_map = {
        "subj01": "train",
        "subj02": "train",
        "subj03": "train",
        "subj04": "train",
        "subj05": "val",
        "subj06": "test"
    }

    # Subject skin tone profiles (BGR)
    skin_tones = {
        "subj01": (160, 190, 235), # Fair tone
        "subj02": (120, 160, 215), # Medium-light
        "subj03": (90, 135, 195),  # Medium-tan
        "subj04": (60, 100, 160),  # Deep-tan
        "subj05": (140, 175, 225), # Val Subject
        "subj06": (80, 120, 180),  # Test Subject (unseen)
    }

    # Ensure output folders exist
    for split in ["train", "val", "test", "ood"]:
        for c in classes:
            os.makedirs(os.path.join(output_base_dir, split, c), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, "ood", "ambiguous"), exist_ok=True)

    print(f"Generating balanced dataset ({samples_per_class} samples/class across 6 subjects)...")

    for class_idx, class_name in enumerate(classes):
        for subj_id in subjects:
            split = split_map[subj_id]
            subj_samples = samples_per_class // len(subjects)
            base_skin = skin_tones[subj_id]

            for i in range(subj_samples):
                # Varied parameters per sample
                bg_type = random.choice(["plain", "gradient", "cluttered"])
                rot = random.uniform(-25.0, 25.0)
                scale = random.uniform(0.80, 1.25)
                offset = (random.randint(-12, 12), random.randint(-12, 12))
                
                # Skin color jitter
                skin_jitter = (
                    int(np.clip(base_skin[0] + random.randint(-15, 15), 0, 255)),
                    int(np.clip(base_skin[1] + random.randint(-15, 15), 0, 255)),
                    int(np.clip(base_skin[2] + random.randint(-15, 15), 0, 255))
                )
                
                noise = random.uniform(0.0, 0.4) if random.random() < 0.3 else 0.0
                blur = random.choice([0, 3, 5]) if random.random() < 0.25 else 0

                # Generate 128x128 hand image
                img = draw_procedural_hand(
                    canvas_size=(128, 128),
                    num_fingers=class_idx,
                    hand_color=skin_jitter,
                    background_type=bg_type,
                    rotation_deg=rot,
                    scale=scale,
                    offset=offset,
                    noise_level=noise,
                    blur_kernel=blur
                )

                # Save sample
                fname = f"{subj_id}_{class_name}_sample{i+1:04d}.png"
                out_path = os.path.join(output_base_dir, split, class_name, fname)
                cv2.imwrite(out_path, img)

    # Generate Out-Of-Distribution (OOD) / Ambiguous Samples for testing
    print("Generating OOD & Ambiguous validation set...")
    for i in range(50):
        # 5 fingers (OOD)
        img_5 = draw_procedural_hand((128, 128), 5, (130, 170, 220), "cluttered", scale=1.1)
        cv2.imwrite(os.path.join(output_base_dir, "ood", "ambiguous", f"ood_5fingers_{i+1:03d}.png"), img_5)
        # Empty background (no hand)
        img_empty = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(output_base_dir, "ood", "ambiguous", f"ood_nohand_{i+1:03d}.png"), img_empty)

    print("Dataset generation complete!")


if __name__ == "__main__":
    generate_full_dataset("dataset", samples_per_class=360)
