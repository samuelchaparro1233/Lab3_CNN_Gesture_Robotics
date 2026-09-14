"""
Automated Real Dataset Splitter (Strict Subject/Session Partitioning)
Laboratorio 3: CNN para Reconocimiento de Gestos (0 a 4 dedos)
Universidad Militar Nueva Granada - Inteligencia Artificial
"""

import os
import sys
import glob
import shutil
import random

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import argparse

def split_real_captured_dataset(
    raw_dir: str = "dataset/raw",
    output_base: str = "dataset",
    clean_destination: bool = False,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
):
    """
    Scans dataset/raw/ for user-captured images, groups them by subject/session,
    and cleanly copies them into dataset/train/, dataset/val/, and dataset/test/
    with zero data leakage.
    """
    random.seed(random_seed)
    classes = ["0_dedos", "1_dedo", "2_dedos", "3_dedos", "4_dedos"]

    if not os.path.exists(raw_dir):
        print(f"[Aviso] No existe la carpeta {raw_dir}. Primero captura fotos con: python src/collect_data.py")
        return

    print("=" * 75)
    print("ORGANIZANDO Y PARTICIONANDO FOTOS REALES POR SUJETO/SESIÓN")
    print(f"Origen: {raw_dir} -> Destino: {output_base}/[train, val, test]")
    if clean_destination:
        print("[Modo: Limpieza] Se limpiarán las carpetas previas antes de copiar.")
    else:
        print("[Modo: Acumulación] Las nuevas fotos se sumarán a las ya existentes.")
    print("=" * 75)

    # Prepare or clean destination class folders
    for split in ["train", "val", "test"]:
        for c in classes:
            target_f = os.path.join(output_base, split, c)
            if clean_destination and os.path.exists(target_f):
                shutil.rmtree(target_f)
            os.makedirs(target_f, exist_ok=True)

    total_copied = {"train": 0, "val": 0, "test": 0}

    for c in classes:
        c_raw = os.path.join(raw_dir, c)
        if not os.path.exists(c_raw):
            continue

        images = glob.glob(os.path.join(c_raw, "*.png")) + glob.glob(os.path.join(c_raw, "*.jpg"))
        if not images:
            continue

        # Group by subject / session prefix (e.g. 'subj01_0_dedos_...' -> 'subj01')
        subject_groups = {}
        for img_path in images:
            fname = os.path.basename(img_path)
            parts = fname.split("_")
            subj_id = parts[0] if len(parts) > 1 else "subj_default"
            subject_groups.setdefault(subj_id, []).append(img_path)

        subjs = list(subject_groups.keys())
        random.shuffle(subjs)

        # Partition by subjects if multiple, else partition files in temporal blocks
        if len(subjs) >= 3:
            n_tr = max(1, int(round(len(subjs) * train_ratio)))
            n_va = max(1, int(round(len(subjs) * val_ratio)))
            tr_subjs = subjs[:n_tr]
            va_subjs = subjs[n_tr:n_tr + n_va]
            te_subjs = subjs[n_tr + n_va:]
            if not te_subjs:
                te_subjs = [va_subjs.pop()] if len(va_subjs) > 1 else [tr_subjs.pop()]

            for s in tr_subjs:
                for p in subject_groups[s]:
                    shutil.copy2(p, os.path.join(output_base, "train", c, os.path.basename(p)))
                    total_copied["train"] += 1
            for s in va_subjs:
                for p in subject_groups[s]:
                    shutil.copy2(p, os.path.join(output_base, "val", c, os.path.basename(p)))
                    total_copied["val"] += 1
            for s in te_subjs:
                for p in subject_groups[s]:
                    shutil.copy2(p, os.path.join(output_base, "test", c, os.path.basename(p)))
                    total_copied["test"] += 1
        else:
            # Single or two subjects: chronological block split (no interleaving frames)
            images.sort()
            n_total = len(images)
            n_tr = int(n_total * train_ratio)
            n_va = int(n_total * val_ratio)

            for p in images[:n_tr]:
                shutil.copy2(p, os.path.join(output_base, "train", c, os.path.basename(p)))
                total_copied["train"] += 1
            for p in images[n_tr:n_tr + n_va]:
                shutil.copy2(p, os.path.join(output_base, "val", c, os.path.basename(p)))
                total_copied["val"] += 1
            for p in images[n_tr + n_va:]:
                shutil.copy2(p, os.path.join(output_base, "test", c, os.path.basename(p)))
                total_copied["test"] += 1

    print(f"\n[Éxito] Fotos organizadas:")
    print(f" - Train: {total_copied['train']} imágenes copiadas")
    print(f" - Val:   {total_copied['val']} imágenes copiadas")
    print(f" - Test:  {total_copied['test']} imágenes copiadas")
    print("\nAhora puedes re-entrenar la CNN con: python src/train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split real captured dataset into Train/Val/Test")
    parser.add_argument("--clean", action="store_true", help="Clean destination folders before copying")
    parser.add_argument("--raw_dir", type=str, default="dataset/raw", help="Raw images directory")
    args = parser.parse_args()

    split_real_captured_dataset(
        raw_dir=args.raw_dir,
        clean_destination=args.clean
    )
