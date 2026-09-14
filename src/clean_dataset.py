"""
Script para Limpiar / Borrar Imágenes del Dataset (Raw, Train, Val, Test, OOD)
Laboratorio 3: CNN para Reconocimiento de Gestos (0 a 4 dedos)
Universidad Militar Nueva Granada - Inteligencia Artificial
"""

import os
import sys
import glob
import shutil
import argparse
from typing import List, Dict, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATASET_CLASSES = ["0_dedos", "1_dedo", "2_dedos", "3_dedos", "4_dedos"]

def scan_images(target_dirs: List[str], subject_id: str = None) -> Tuple[List[str], int]:
    """Scans for image files in specified target directories."""
    files_to_delete = []
    total_bytes = 0

    for base_dir in target_dirs:
        if not os.path.exists(base_dir):
            continue
        
        # Search all png and jpg files recursively
        pattern_png = os.path.join(base_dir, "**", "*.png")
        pattern_jpg = os.path.join(base_dir, "**", "*.jpg")
        found_files = glob.glob(pattern_png, recursive=True) + glob.glob(pattern_jpg, recursive=True)

        for filepath in found_files:
            filename = os.path.basename(filepath)
            # Filter by subject_id if specified
            if subject_id and subject_id not in filename:
                continue
            
            files_to_delete.append(filepath)
            try:
                total_bytes += os.path.getsize(filepath)
            except OSError:
                pass

    return files_to_delete, total_bytes

def clean_dataset(
    targets: List[str],
    subject_id: str = None,
    confirm: bool = True,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Deletes saved image files based on specified target folders.
    """
    target_paths = []
    base_dataset = os.path.join(PROJECT_ROOT, "dataset")

    if "all" in targets:
        target_paths = [os.path.join(base_dataset, d) for d in ["raw", "train", "val", "test", "ood"]]
    else:
        if "raw" in targets:
            target_paths.append(os.path.join(base_dataset, "raw"))
        if "splits" in targets:
            target_paths.extend([os.path.join(base_dataset, d) for d in ["train", "val", "test"]])
        if "ood" in targets:
            target_paths.append(os.path.join(base_dataset, "ood"))

    files_to_delete, total_bytes = scan_images(target_paths, subject_id=subject_id)
    size_mb = total_bytes / (1024 * 1024)

    print("=" * 75)
    print("      LIMPIADOR DE IMÁGENES DEL DATASET - LAB 3 CNN")
    print("=" * 75)
    print(f"Directorio base: {base_dataset}")
    print(f"Objetivos:       {', '.join(targets)}")
    if subject_id:
        print(f"Filtro Sujeto:   {subject_id}")
    print(f"Archivos:        {len(files_to_delete)} imágenes encontradas")
    print(f"Espacio total:   {size_mb:.2f} MB")
    print("-" * 75)

    if not files_to_delete:
        print("[Info] No se encontraron imágenes para borrar en las carpetas seleccionadas.")
        return {"deleted_count": 0, "freed_mb": 0.0}

    # Display breakdown by folder
    folder_counts = {}
    for f in files_to_delete:
        rel = os.path.relpath(f, base_dataset)
        parent = rel.split(os.sep)[0]
        folder_counts[parent] = folder_counts.get(parent, 0) + 1

    for folder, count in folder_counts.items():
        print(f"  - dataset/{folder}: {count} imágenes")

    if dry_run:
        print("\n[DRY RUN] Modo simulación activado. No se eliminó ningún archivo.")
        return {"deleted_count": len(files_to_delete), "freed_mb": size_mb, "dry_run": True}

    if confirm:
        print("\n" + "!" * 75)
        ans = input("¿Está seguro de que desea BORRAR PERMANENTEMENTE estas imágenes? (s/N): ").strip().lower()
        if ans not in ['s', 'si', 'y', 'yes']:
            print("[Cancelado] Operación cancelada por el usuario. No se borró nada.")
            return {"deleted_count": 0, "freed_mb": 0.0, "cancelled": True}

    print("\nEliminando archivos...")
    deleted_count = 0
    for f in files_to_delete:
        try:
            os.remove(f)
            deleted_count += 1
        except Exception as e:
            print(f"[Error] No se pudo borrar {f}: {e}")

    print("=" * 75)
    print(f"SUCCESS: Se eliminaron {deleted_count} imágenes exitosamente.")
    print(f"Espacio liberado: {size_mb:.2f} MB.")
    print("=" * 75)

    return {"deleted_count": deleted_count, "freed_mb": size_mb}


def main():
    parser = argparse.ArgumentParser(description="Herramienta para borrar imágenes guardadas del dataset.")
    parser.add_argument(
        "--target",
        type=str,
        default="all",
        choices=["all", "raw", "splits", "ood"],
        help="Target folders to clean: 'raw' (fotos webcam), 'splits' (train/val/test), 'ood', or 'all' (defecto: all)."
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Borrar solo imágenes en dataset/raw."
    )
    parser.add_argument(
        "--splits-only",
        action="store_true",
        help="Borrar solo las particiones en dataset/train, dataset/val, dataset/test."
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Borrar únicamente las fotos pertenecientes a un sujeto específico (ej. 'subj_01')."
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Omitir la confirmación interactiva y borrar directamente."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostrar qué archivos se borrarían sin eliminar nada realmente."
    )

    args = parser.parse_args()

    targets = [args.target]
    if args.raw_only:
        targets = ["raw"]
    elif args.splits_only:
        targets = ["splits"]

    clean_dataset(
        targets=targets,
        subject_id=args.subject,
        confirm=not args.yes,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()
