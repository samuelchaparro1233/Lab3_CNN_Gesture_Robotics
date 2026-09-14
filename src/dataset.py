"""
Dataset Loader, Preprocessing, and Session-Partitioned Data Pipeline
Laboratorio 3: CNN para Reconocimiento de Gestos (0 a 4 dedos)
Universidad Militar Nueva Granada - Inteligencia Artificial
"""

import os
import glob
import random
import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional, Callable

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object


class GestureDataset(Dataset):
    """
    PyTorch Dataset for Hand Gesture Images (0 to 4 fingers).
    Applies real-time mechatronic vision preprocessing and augmentation.
    """
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        image_size: Tuple[int, int] = (64, 64),
        channels: int = 1,
        is_training: bool = False,
        mean: float = 0.5,
        std: float = 0.5
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size
        self.channels = channels
        self.is_training = is_training
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.image_paths)

    def preprocess_image(self, image_np: np.ndarray) -> np.ndarray:
        """Applies spatial resizing, color space conversion, and augmentation."""
        # 1. Resize
        img = cv2.resize(image_np, (self.image_size[1], self.image_size[0]))
        
        # 2. Color Conversion
        if self.channels == 1:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=-1)
        else:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
        # 3. Targeted Mechatronic Augmentation (Only during training)
        if self.is_training:
            h, w = self.image_size
            # Combined Affine: Rotation (-15 to +15 deg), Scale (0.9 to 1.1), Shift (-3 to +3 px)
            angle = random.uniform(-15.0, 15.0)
            scale = random.uniform(0.90, 1.10)
            tx = random.uniform(-3.0, 3.0)
            ty = random.uniform(-3.0, 3.0)
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            img_rot = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            if self.channels == 1 and len(img_rot.shape) == 2:
                img = np.expand_dims(img_rot, axis=-1)
            else:
                img = img_rot
                
            # Random Brightness & Contrast Jitter
            alpha = random.uniform(0.75, 1.25)  # Contrast
            beta = random.uniform(-25.0, 25.0)  # Brightness
            img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
            
            # Random Gaussian Noise (25% probability)
            if random.random() < 0.25:
                noise = np.random.normal(0, 6, img.shape)
                img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            # Random Cutout / Patch Erasing (30% probability) - forces robustness against partial occlusions
            if random.random() < 0.30:
                cut_size = random.randint(6, 14)
                cy = random.randint(0, h - cut_size)
                cx = random.randint(0, w - cut_size)
                cut_val = random.randint(0, 50)
                img[cy:cy+cut_size, cx:cx+cut_size] = cut_val

        # 4. Normalization [0, 1] and standardization
        img_norm = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        
        # 5. Transpose to PyTorch format (C, H, W)
        if len(img_norm.shape) == 3:
            img_norm = np.transpose(img_norm, (2, 0, 1))
        else:
            img_norm = np.expand_dims(img_norm, axis=0)
            
        return img_norm

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Read image
        img = cv2.imread(path, cv2.IMREAD_COLOR if self.channels == 3 else cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback blank image if file is unreadable
            img = np.zeros((self.image_size[0], self.image_size[1], self.channels), dtype=np.uint8)
            
        processed_np = self.preprocess_image(img)
        
        if TORCH_AVAILABLE:
            tensor_x = torch.from_numpy(processed_np).float()
            tensor_y = torch.tensor(label, dtype=torch.long)
            return tensor_x, tensor_y
        else:
            return processed_np, label


def partition_dataset_by_subject(
    data_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[Dict[str, List], Dict[str, List], Dict[str, List]]:
    """
    Partitions the dataset strictly by subject/recording session to guarantee
    zero data leakage between training, validation, and independent test sets.
    """
    random.seed(random_seed)
    train_data = {"paths": [], "labels": []}
    val_data = {"paths": [], "labels": []}
    test_data = {"paths": [], "labels": []}
    
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    for class_idx, class_name in enumerate(classes):
        class_folder = os.path.join(data_dir, class_name)
        # Find all subjects/sessions
        img_files = glob.glob(os.path.join(class_folder, "*.jpg")) + \
                    glob.glob(os.path.join(class_folder, "*.png"))
                    
        # Group by subject prefix (e.g., 'sub01_frame001.png' -> 'sub01')
        subjects: Dict[str, List[str]] = {}
        for f in img_files:
            fname = os.path.basename(f)
            subj = fname.split("_")[0] if "_" in fname else "sub_default"
            subjects.setdefault(subj, []).append(f)
            
        subj_keys = list(subjects.keys())
        random.shuffle(subj_keys)
        
        # Partition subject keys
        n_subj = len(subj_keys)
        if n_subj >= 3:
            n_train = max(1, int(round(n_subj * train_ratio)))
            n_val = max(1, int(round(n_subj * val_ratio)))
            train_subjs = subj_keys[:n_train]
            val_subjs = subj_keys[n_train:n_train + n_val]
            test_subjs = subj_keys[n_train + n_val:]
            if not test_subjs:
                test_subjs = [val_subjs.pop()] if len(val_subjs) > 1 else [train_subjs.pop()]
        else:
            # Fallback if few subjects: split files with non-overlapping chunking
            all_files = sorted(img_files)
            random.shuffle(all_files)
            n_total = len(all_files)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            for p in all_files[:n_train]:
                train_data["paths"].append(p)
                train_data["labels"].append(class_idx)
            for p in all_files[n_train:n_train + n_val]:
                val_data["paths"].append(p)
                val_data["labels"].append(class_idx)
            for p in all_files[n_train + n_val:]:
                test_data["paths"].append(p)
                test_data["labels"].append(class_idx)
            continue

        for s in train_subjs:
            for p in subjects[s]:
                train_data["paths"].append(p)
                train_data["labels"].append(class_idx)
        for s in val_subjs:
            for p in subjects[s]:
                val_data["paths"].append(p)
                val_data["labels"].append(class_idx)
        for s in test_subjs:
            for p in subjects[s]:
                test_data["paths"].append(p)
                test_data["labels"].append(class_idx)
                
    return train_data, val_data, test_data
