from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = ("bart", "homer")
SUPPORTED_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}


def load_split(
    dataset_root: Path,
    split_name: str,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if image_size < 1:
        raise ValueError("image_size must be a positive integer")

    split_root = dataset_root / split_name
    if not split_root.is_dir():
        raise FileNotFoundError(f"Dataset split directory not found: {split_root}")

    images = []
    labels = []

    for label, class_name in enumerate(CLASS_NAMES, start=1):
        class_root = split_root / class_name
        if not class_root.is_dir():
            raise FileNotFoundError(f"Class directory not found: {class_root}")

        image_paths = sorted(
            (
                path
                for path in class_root.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            ),
            key=lambda path: path.name.lower(),
        )
        if not image_paths:
            raise ValueError(f"No supported images found in: {class_root}")

        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(grayscale, (image_size, image_size))
            images.append(resized.astype(np.float64).ravel() / 255.0)
            labels.append(label)

    return np.asarray(images, dtype=np.float64), np.asarray(labels, dtype=np.int64)
