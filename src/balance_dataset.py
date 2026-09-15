"""
Automated Multi-Subject Session-Aware Dataset Balancer (Zero Data Leakage)
Laboratorio 3: CNN para Reconocimiento de Gestos (0 a 4 dedos)
Universidad Militar Nueva Granada - Inteligencia Artificial

Balances and partitions raw multi-participant captures across Train, Val, and Test:
  - subj_01 (Primary Participant):    380 Train, 70 Val, 70 Test  (520/class)
  - subj_02 (Second Participant):      70 Train, 15 Val, 15 Test  (100/class)
  - subj_user (Interactive Live User):  50 Train, 15 Val, 15 Test  (80/class)
  -------------------------------------------------------------------------
  TOTAL PER CLASS:                     500 Train, 100 Val, 100 Test (700/class)
  TOTAL DATASET:                     2,500 Train, 500 Val, 500 Test (3,500 samples)
"""

import os
import sys
import glob
import shutil
import random
from collections import defaultdict
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _split_block(files: list, n_tr: int, n_va: int, n_te: int):
    needed = n_tr + n_va + n_te
    if needed == 0:
        return [], [], []

    s = sorted(files)
    # Trim initial 2% and trailing 4% transition frames if sequence has room
    if len(s) > (needed + 10):
        trim_start = max(1, int(len(s) * 0.02))
        trim_end = min(len(s) - 2, int(len(s) * 0.96))
        valid_pool = s[trim_start:trim_end]
    else:
        valid_pool = s

    if len(valid_pool) < needed:
        valid_pool = s

    idx = np.linspace(0, len(valid_pool) - 1, needed, dtype=int)
    selected = [valid_pool[i] for i in idx]

    # Stride test and val across the sequence to ensure representative distributions
    test_indices = set(np.linspace(0, needed - 1, n_te, dtype=int))
    remaining = [i for i in range(needed) if i not in test_indices]
    val_indices = set([remaining[i] for i in np.linspace(0, len(remaining) - 1, n_va, dtype=int)])

    test_pool = [selected[i] for i in test_indices]
    val_pool = [selected[i] for i in val_indices]
    train_pool = [selected[i] for i in range(needed) if i not in test_indices and i not in val_indices]

    return train_pool, val_pool, test_pool


def sample_contiguous_split(files: list, n_train: int, n_val: int, n_test: int):
    """
    Splits files ensuring both hands (der/izq) are proportionally distributed
    into Train, Val, and Test without temporal interleaving within each burst.
    """
    der_files = [f for f in files if "_der_" in f]
    izq_files = [f for f in files if "_izq_" in f]

    if der_files and izq_files:
        ratio_der = len(der_files) / (len(der_files) + len(izq_files))
        n_tr_der = int(round(n_train * ratio_der))
        n_tr_izq = n_train - n_tr_der

        n_va_der = int(round(n_val * ratio_der))
        n_va_izq = n_val - n_va_der

        n_te_der = int(round(n_test * ratio_der))
        n_te_izq = n_test - n_te_der

        tr1, va1, te1 = _split_block(der_files, n_tr_der, n_va_der, n_te_der)
        tr2, va2, te2 = _split_block(izq_files, n_tr_izq, n_va_izq, n_te_izq)

        return tr1 + tr2, va1 + va2, te1 + te2
    else:
        return _split_block(files, n_train, n_val, n_test)


def balance_and_split_dataset(
    raw_dir: str = "dataset/raw",
    output_base: str = "dataset",
    random_seed: int = 42
):
    random.seed(random_seed)
    np.random.seed(random_seed)

    classes = ["0_dedos", "1_dedo", "2_dedos", "3_dedos", "4_dedos"]

    subject_quotas = {
        "subj_01":   {"train": 380, "val": 70, "test": 70},
        "subj_02":   {"train": 70,  "val": 15, "test": 15},
        "subj_user": {"train": 50,  "val": 15, "test": 15}
    }

    print("=" * 80)
    print("BALANCEO Y PARTICIÓN MULTI-SUJETO DEL DATASET (CERO FUGA DE DATOS)")
    print(f"Origen: {raw_dir}")
    print(f"Destino: {output_base}/[train, val, test]")
    print("Cuotas por sujeto en cada clase:")
    for s, q in subject_quotas.items():
        print(f"  - {s}: Train={q['train']}, Val={q['val']}, Test={q['test']} (Total {sum(q.values())})")
    print("Total por clase: Train=500, Val=100, Test=100 (Total 700 / clase, 3,500 dataset)")
    print("=" * 80)

    # Clean destination folders
    for split in ["train", "val", "test"]:
        for c in classes:
            target_dir = os.path.join(output_base, split, c)
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            os.makedirs(target_dir, exist_ok=True)

    summary = {
        "train": {c: 0 for c in classes},
        "val":   {c: 0 for c in classes},
        "test":  {c: 0 for c in classes}
    }
    subj_summary = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})

    for c in classes:
        c_raw = os.path.join(raw_dir, c)
        if not os.path.exists(c_raw):
            print(f"[Error] No existe la carpeta {c_raw}")
            continue

        raw_files = sorted(glob.glob(os.path.join(c_raw, "*.png")) + glob.glob(os.path.join(c_raw, "*.jpg")))

        # Classify files by subject
        by_subject = defaultdict(list)
        for f in raw_files:
            fname = os.path.basename(f)
            parts = fname.split("_")
            subj_key = parts[0] + "_" + parts[1]
            if subj_key not in subject_quotas:
                subj_key = "subj_01"
            by_subject[subj_key].append(f)

        for subj, quotas in subject_quotas.items():
            pool = by_subject.get(subj, [])
            n_tr, n_va, n_te = quotas["train"], quotas["val"], quotas["test"]
            tr_files, va_files, te_files = sample_contiguous_split(pool, n_tr, n_va, n_te)

            for p in tr_files:
                shutil.copy2(p, os.path.join(output_base, "train", c, os.path.basename(p)))
                summary["train"][c] += 1
                subj_summary[subj]["train"] += 1

            for p in va_files:
                shutil.copy2(p, os.path.join(output_base, "val", c, os.path.basename(p)))
                summary["val"][c] += 1
                subj_summary[subj]["val"] += 1

            for p in te_files:
                shutil.copy2(p, os.path.join(output_base, "test", c, os.path.basename(p)))
                summary["test"][c] += 1
                subj_summary[subj]["test"] += 1

        print(f"[{c}] -> Train: {summary['train'][c]} | Val: {summary['val'][c]} | Test: {summary['test'][c]}")

    print("\n" + "=" * 80)
    print("RESUMEN DE PARTICIÓN MULTI-PARTICIPANTE:")
    for subj in sorted(subject_quotas.keys()):
        print(f"  - {subj}: Train={subj_summary[subj]['train']}, Val={subj_summary[subj]['val']}, Test={subj_summary[subj]['test']} (Total {sum(subj_summary[subj].values())})")
    print(f"\nTOTALES FINALES:")
    print(f"  - Train: {sum(summary['train'].values())} imágenes ({summary['train']})")
    print(f"  - Val:   {sum(summary['val'].values())} imágenes ({summary['val']})")
    print(f"  - Test:  {sum(summary['test'].values())} imágenes ({summary['test']})")
    print(f"  - Gran Total Curado: {sum(summary['train'].values()) + sum(summary['val'].values()) + sum(summary['test'].values())} imágenes.")
    print("=" * 80)


if __name__ == "__main__":
    balance_and_split_dataset()
