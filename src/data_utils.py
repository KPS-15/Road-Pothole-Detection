"""
Dataset preparation, inspection, deduplication, splitting, and preprocessing utilities.
Ensures reproducible stratified splits, data integrity, and strict consistency between
training, evaluation, and inference pipelines.
"""

from pathlib import Path
import hashlib
import json
import shutil
import re
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for headless/background runs
import matplotlib.pyplot as plt
# Using native numpy stratified splitting to avoid Application Control DLL blocks

from src.config import (
    CLASS_NAMES,
    CLASS_LABELS,
    VALID_IMAGE_EXTENSIONS,
    TRAIN_SPLIT_RATIO,
    VAL_SPLIT_RATIO,
    TEST_SPLIT_RATIO,
    RANDOM_SEED,
    IMG_HEIGHT,
    IMG_WIDTH,
    set_seed,
)


def compute_file_hash(filepath: Path, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file to detect exact duplicate images."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def extract_scene_key(filename: str) -> str:
    """
    Extract a scene or source-video identifier from a filename to group related
    dashcam frames and mitigate data leakage across splits.
    
    Examples:
        'dashcam_01_frame_042.jpg' -> 'dashcam_01'
        'videoA_001.png'           -> 'videoA'
        'pothole_scene3_12.jpg'    -> 'pothole_scene3'
        '142.jpg'                  -> 'scene_unknown_142'
    """
    name = Path(filename).stem
    # Match patterns like prefix_number or word-word
    match = re.match(r"^([a-zA-Z0-9]+(?:[-_][a-zA-Z]+)*)", name)
    if match and len(match.group(1)) > 3 and not match.group(1).isdigit():
        return match.group(1)
    # If filename is purely numeric or no delimiter, assign individual scene
    return f"scene_{name}"


def normalize_class_name(name: str) -> Optional[str]:
    """Map arbitrary folder/class names to canonical 'Normal' or 'Pothole'."""
    lower = name.lower().strip()
    if any(k in lower for k in ["pothole", "defect", "damage", "crack"]):
        return "Pothole"
    if any(k in lower for k in ["normal", "plain", "good", "undamaged", "clean", "smooth"]):
        return "Normal"
    return None


def inspect_image_file(path: Path) -> Tuple[bool, Optional[Tuple[int, int, int]], Optional[str]]:
    """
    Verify image integrity, format, dimensions, and channels.
    Returns: (is_valid, (width, height, channels), error_message)
    """
    try:
        with Image.open(path) as img:
            img.verify()
        # Re-open after verify() because verify() invalidates the image handle
        with Image.open(path) as img:
            img_format = img.format
            width, height = img.size
            mode = img.mode
            channels = len(mode) if mode in ("RGB", "RGBA", "CMYK") else 1
            if channels not in (1, 3, 4):
                return False, None, f"Unsupported channel configuration: {mode}"
            return True, (width, height, channels), None
    except Exception as e:
        return False, None, str(e)


def scan_dataset_directory(source_dir: Path) -> Dict[str, Any]:
    """
    Recursively scans the provided directory, cataloging images, validating integrity,
    tracking exact duplicates via SHA-256, and summarizing class distributions.
    """
    source_dir = Path(source_dir)
    results = {
        "valid_images": [],      # List of dicts with metadata
        "corrupted_images": [],  # List of dicts with failure reason
        "duplicates": [],        # List of duplicate file records
        "class_counts": {"Normal": 0, "Pothole": 0},
        "extensions": {},
        "resolutions": [],
        "hash_map": {},          # sha256 -> original_path
        "total_scanned": 0,
    }

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    # Recursive scan for all supported extensions
    for ext in VALID_IMAGE_EXTENSIONS:
        for file_path in source_dir.rglob(f"*{ext}"):
            results["total_scanned"] += 1
            ext_lower = file_path.suffix.lower()
            results["extensions"][ext_lower] = results["extensions"].get(ext_lower, 0) + 1

            # Determine class from parent folders
            canonical_class = None
            for parent in file_path.parents:
                c = normalize_class_name(parent.name)
                if c:
                    canonical_class = c
                    break

            if not canonical_class:
                # Check filename itself as fallback
                canonical_class = normalize_class_name(file_path.stem)

            if not canonical_class:
                # Cannot determine label
                results["corrupted_images"].append({
                    "path": str(file_path),
                    "reason": "Could not infer class (Normal/Pothole) from folder or filename",
                })
                continue

            # Check integrity
            is_valid, dims, err = inspect_image_file(file_path)
            if not is_valid:
                results["corrupted_images"].append({
                    "path": str(file_path),
                    "reason": f"Corrupted image: {err}",
                })
                continue

            # Check exact duplicates
            file_hash = compute_file_hash(file_path)
            if file_hash in results["hash_map"]:
                results["duplicates"].append({
                    "duplicate_path": str(file_path),
                    "original_path": results["hash_map"][file_hash],
                    "hash": file_hash,
                    "class": canonical_class,
                })
                continue

            # First unique occurrence
            results["hash_map"][file_hash] = str(file_path)
            scene_id = extract_scene_key(file_path.name)
            record = {
                "path": str(file_path),
                "filename": file_path.name,
                "class": canonical_class,
                "label": CLASS_LABELS[canonical_class],
                "hash": file_hash,
                "width": dims[0],
                "height": dims[1],
                "channels": dims[2],
                "scene_id": scene_id,
            }
            results["valid_images"].append(record)
            results["class_counts"][canonical_class] += 1
            results["resolutions"].append((dims[0], dims[1]))

    return results


def check_existing_official_splits(source_dir: Path) -> Optional[Dict[str, List[Path]]]:
    """
    Check if the dataset already contains standard train/val/test subdirectories.
    Returns a dict mapping 'train', 'validation', 'test' to list of paths if present.
    """
    source_dir = Path(source_dir)
    possible_train = [source_dir / "train", source_dir / "Train", source_dir / "training"]
    possible_val = [source_dir / "validation", source_dir / "Validation", source_dir / "val", source_dir / "Val"]
    possible_test = [source_dir / "test", source_dir / "Test", source_dir / "testing"]

    train_path = next((p for p in possible_train if p.exists() and p.is_dir()), None)
    val_path = next((p for p in possible_val if p.exists() and p.is_dir()), None)
    test_path = next((p for p in possible_test if p.exists() and p.is_dir()), None)

    if train_path and (val_path or test_path):
        return {
            "train": [train_path],
            "validation": [val_path] if val_path else [],
            "test": [test_path] if test_path else [],
        }
    return None


def partition_dataset_records(
    records: List[Dict[str, Any]],
    train_ratio: float = TRAIN_SPLIT_RATIO,
    val_ratio: float = VAL_SPLIT_RATIO,
    test_ratio: float = TEST_SPLIT_RATIO,
    seed: int = RANDOM_SEED,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Splits unique records into stratified train/validation/test partitions
    using native numpy random permutation for reproducible, robust stratification.
    """
    if not records:
        raise ValueError("Cannot partition empty list of records.")

    rng = np.random.RandomState(seed)

    # Group records by class label
    class_groups: Dict[int, List[Dict[str, Any]]] = {}
    for r in records:
        lbl = r["label"]
        class_groups.setdefault(lbl, []).append(r)

    train_records = []
    val_records = []
    test_records = []

    for lbl, group in class_groups.items():
        n = len(group)
        indices = rng.permutation(n)
        
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # Ensure remaining go to test
        n_test = n - n_train - n_val

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        train_records.extend([group[i] for i in train_indices])
        val_records.extend([group[i] for i in val_indices])
        test_records.extend([group[i] for i in test_indices])

    # Shuffle the combined partitions
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)

    return {
        "train": train_records,
        "validation": val_records,
        "test": test_records,
    }


def deploy_split_files(
    splits: Dict[str, List[Dict[str, Any]]],
    target_dir: Path,
) -> Dict[str, Dict[str, int]]:
    """
    Copies partition records into structured directories:
    target_dir/
      train/Normal, train/Pothole
      validation/Normal, validation/Pothole
      test/Normal, test/Pothole
    """
    target_dir = Path(target_dir)
    split_summary: Dict[str, Dict[str, int]] = {}

    for split_name, records in splits.items():
        split_summary[split_name] = {"Normal": 0, "Pothole": 0}
        for rec in records:
            c = rec["class"]
            dest_folder = target_dir / split_name / c
            dest_folder.mkdir(parents=True, exist_ok=True)
            src_file = Path(rec["path"])
            dest_file = dest_folder / src_file.name
            
            # If name collision occurs, append hash prefix
            if dest_file.exists() and dest_file != src_file:
                dest_file = dest_folder / f"{rec['hash'][:8]}_{src_file.name}"

            shutil.copy2(src_file, dest_file)
            split_summary[split_name][c] += 1

    return split_summary


def plot_class_distribution(
    split_summary: Dict[str, Dict[str, int]],
    output_path: Path,
) -> None:
    """Generates an publication-quality class distribution bar chart."""
    splits = list(split_summary.keys())
    normal_counts = [split_summary[s]["Normal"] for s in splits]
    pothole_counts = [split_summary[s]["Pothole"] for s in splits]

    x = np.arange(len(splits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    rects1 = ax.bar(x - width / 2, normal_counts, width, label="Normal Road", color="#2E7D32")
    rects2 = ax.bar(x + width / 2, pothole_counts, width, label="Pothole", color="#D32F2F")

    ax.set_ylabel("Number of Images", fontsize=11, fontweight="bold")
    ax.set_title("Dataset Class Distribution Across Splits", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in splits], fontsize=10, fontweight="bold")
    ax.legend(frameon=True, facecolor="#F5F5F5")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Attach count labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
