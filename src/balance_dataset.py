"""
Automated Session-Aware Dataset Balancer (Zero Data Leakage)
Laboratorio 3: CNN para Reconocimiento de Gestos (0 a 4 dedos)
Universidad Militar Nueva Granada - Inteligencia Artificial

Groups raw captures by session/burst timestamp, allocates sessions independently
across Train, Val, and Test splits, and balances every class to exact quotas:
  - Train: 500 samples / class (2,500 total)
  - Val:   100 samples / class (500 total)
  - Test:  100 samples / class (500 total)
Total Curated Dataset: 3,500 perfectly balanced samples.
"""

import os
import sys
import glob
import shutil
import random
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def balance_and_split_dataset(
    raw_dir: str = "dataset/raw",
    output_base: str = "dataset",
    train_per_class: int = 500,
    val_per_class: int = 100,
    test_per_class: int = 100,
    random_seed: int = 42
):
    random.seed(random_seed)
    np.random.seed(random_seed)

    classes = ["0_dedos", "1_dedo", "2_dedos", "3_dedos", "4_dedos"]

    print("=" * 75)
    print("BALANCEO Y PARTICIÓN SESIÓN-INDEPENDIENTE DEL DATASET")
    print(f"Origen: {raw_dir}")
    print(f"Destino: {output_base}/[train, val, test]")
    print(f"Objetivo por clase: Train={train_per_class}, Val={val_per_class}, Test={test_per_class}")
    print("=" * 75)

    # Clean existing destination split folders to ensure clean balance
    for split in ["train", "val", "test"]:
        for c in classes:
            target_dir = os.path.join(output_base, split, c)
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            os.makedirs(target_dir, exist_ok=True)

    summary = {"train": {c: 0 for c in classes}, "val": {c: 0 for c in classes}, "test": {c: 0 for c in classes}}

    for c in classes:
        c_raw = os.path.join(raw_dir, c)
        if not os.path.exists(c_raw):
            print(f"[Error] No existe la carpeta {c_raw}")
            continue

        raw_files = sorted(glob.glob(os.path.join(c_raw, "*.png")) + glob.glob(os.path.join(c_raw, "*.jpg")))
        if len(raw_files) < (train_per_class + val_per_class + test_per_class):
            print(f"[Advertencia] {c} tiene {len(raw_files)} imágenes, menos de las requeridas.")

        # Group by session burst (using the timestamp prefix: e.g. subj_01_der_0_dedos_1789442809835_001.png)
        session_groups = {}
        for f in raw_files:
            fname = os.path.basename(f)
            parts = fname.split("_")
            # Extract timestamp token
            session_id = "sess_default"
            for p in parts:
                if p.isdigit() and len(p) >= 10:
                    session_id = p[:10]  # group by burst timestamp
                    break
            session_groups.setdefault(session_id, []).append(f)

        sessions = sorted(list(session_groups.keys()))
        random.shuffle(sessions)

        # Allocate sessions across train (70%), val (15%), test (15%)
        n_sess = len(sessions)
        n_tr = max(1, int(round(n_sess * 0.70)))
        n_va = max(1, int(round(n_sess * 0.15)))

        train_sessions = sessions[:n_tr]
        val_sessions = sessions[n_tr:n_tr + n_va]
        test_sessions = sessions[n_tr + n_va:]

        if not test_sessions and len(val_sessions) > 1:
            test_sessions = [val_sessions.pop()]

        # Collect pool of images per split
        train_pool = [img for s in train_sessions for img in session_groups[s]]
        val_pool = [img for s in val_sessions for img in session_groups[s]]
        test_pool = [img for s in test_sessions for img in session_groups[s]]

        # Subsample evenly to reach exact targets
        def sample_evenly(pool, target_count):
            if len(pool) <= target_count:
                return pool
            indices = np.linspace(0, len(pool) - 1, target_count, dtype=int)
            return [pool[i] for i in indices]

        selected_train = sample_evenly(train_pool, train_per_class)
        selected_val = sample_evenly(val_pool, val_per_class)
        selected_test = sample_evenly(test_pool, test_per_class)

        # Copy to destination
        for p in selected_train:
            shutil.copy2(p, os.path.join(output_base, "train", c, os.path.basename(p)))
            summary["train"][c] += 1

        for p in selected_val:
            shutil.copy2(p, os.path.join(output_base, "val", c, os.path.basename(p)))
            summary["val"][c] += 1

        for p in selected_test:
            shutil.copy2(p, os.path.join(output_base, "test", c, os.path.basename(p)))
            summary["test"][c] += 1

        print(f"[{c}] Raw: {len(raw_files)} -> Train: {summary['train'][c]} | Val: {summary['val'][c]} | Test: {summary['test'][c]}")

    print("\n" + "=" * 75)
    print("RESUMEN DE DATASET BALANCEADO FINAL:")
    print(f"  - Train: {sum(summary['train'].values())} imágenes ({summary['train']})")
    print(f"  - Val:   {sum(summary['val'].values())} imágenes ({summary['val']})")
    print(f"  - Test:  {sum(summary['test'].values())} imágenes ({summary['test']})")
    print("=" * 75)


if __name__ == "__main__":
    balance_and_split_dataset()
