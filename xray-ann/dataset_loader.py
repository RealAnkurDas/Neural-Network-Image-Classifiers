from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import random
import re

import cv2
import numpy as np


CLASS_NAMES = ("NORMAL", "PNEUMONIA")
SOURCE_SPLITS = ("train", "val", "test")
OUTPUT_SPLITS = ("train", "val", "test")
SUPPORTED_IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}
_AUGMENTATION_SUFFIX = re.compile(r"_aug_\d+$", re.IGNORECASE)
_PNEUMONIA_PERSON_PREFIX = re.compile(r"^(person\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class ImageBatch:
    features: np.ndarray
    labels: np.ndarray
    group_ids: tuple[str, ...]


def source_group_key(class_name: str, image_path: Path) -> str:
    if class_name == "NORMAL":
        return _AUGMENTATION_SUFFIX.sub("", image_path.stem).casefold()
    if class_name == "PNEUMONIA":
        match = _PNEUMONIA_PERSON_PREFIX.match(image_path.stem)
        if match:
            return match.group(1).casefold()
        return image_path.stem.casefold()
    raise ValueError(f"Unsupported class name: {class_name}")


def _representative_image(image_paths: list[Path]) -> Path:
    non_augmented = [
        path for path in image_paths if not _AUGMENTATION_SUFFIX.search(path.stem)
    ]
    return min(non_augmented or image_paths, key=lambda path: path.name.casefold())


def _split_group_ids(group_ids: list[str], rng: random.Random) -> dict[str, set[str]]:
    if len(group_ids) < 3:
        raise ValueError(
            "At least three source groups per class are needed to split data"
        )

    shuffled = sorted(group_ids)
    rng.shuffle(shuffled)

    train_count = max(1, round(len(shuffled) * 0.8))
    validation_count = max(1, round(len(shuffled) * 0.1))
    if train_count + validation_count >= len(shuffled):
        train_count = len(shuffled) - 2
        validation_count = 1

    return {
        "train": set(shuffled[:train_count]),
        "val": set(shuffled[train_count : train_count + validation_count]),
        "test": set(shuffled[train_count + validation_count :]),
    }


def load_group_splits(
    dataset_root: Path,
    image_size: int,
    seed: int = 42,
) -> dict[str, ImageBatch]:
    if image_size < 1:
        raise ValueError("image_size must be a positive integer")

    grouped_paths: dict[str, dict[str, dict[str, Path]]] = {
        class_name: defaultdict(dict) for class_name in CLASS_NAMES
    }

    for source_split in SOURCE_SPLITS:
        split_root = dataset_root / source_split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Dataset split directory not found: {split_root}")

        for class_name in CLASS_NAMES:
            class_root = split_root / class_name
            if not class_root.is_dir():
                raise FileNotFoundError(f"Class directory not found: {class_root}")

            for image_path in sorted(
                class_root.iterdir(), key=lambda path: path.name.casefold()
            ):
                if (
                    not image_path.is_file()
                    or image_path.suffix.casefold() not in SUPPORTED_IMAGE_SUFFIXES
                ):
                    continue
                group_key = source_group_key(class_name, image_path)
                grouped_paths[class_name][group_key].setdefault(
                    image_path.name.casefold(), image_path
                )

    for class_name, groups in grouped_paths.items():
        if not groups:
            raise ValueError(f"No supported images found for class {class_name}")

    rng = random.Random(seed)
    output_paths: dict[str, list[tuple[str, int, Path]]] = {
        split_name: [] for split_name in OUTPUT_SPLITS
    }

    for label, class_name in enumerate(CLASS_NAMES, start=1):
        groups = grouped_paths[class_name]
        assignments = _split_group_ids(list(groups), rng)

        for split_name, group_ids in assignments.items():
            for group_id in sorted(group_ids):
                group_paths = sorted(
                    groups[group_id].values(), key=lambda path: path.name.casefold()
                )
                selected_paths = (
                    group_paths
                    if split_name == "train"
                    else [_representative_image(group_paths)]
                )
                public_group_id = f"{class_name}:{group_id}"
                output_paths[split_name].extend(
                    (public_group_id, label, path) for path in selected_paths
                )

    batches: dict[str, ImageBatch] = {}
    for split_name, records in output_paths.items():
        records.sort(key=lambda record: (record[0], record[2].name.casefold()))
        images = []
        labels = []
        group_ids = []

        for group_id, label, image_path in records:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            resized = cv2.resize(image, (image_size, image_size))
            images.append(resized.astype(np.float64).ravel() / 255.0)
            labels.append(label)
            group_ids.append(group_id)

        batches[split_name] = ImageBatch(
            features=np.asarray(images, dtype=np.float64),
            labels=np.asarray(labels, dtype=np.int64),
            group_ids=tuple(group_ids),
        )

    return batches
