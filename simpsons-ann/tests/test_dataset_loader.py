import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from dataset_loader import CLASS_NAMES, load_split


class LoadSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_image(self, split_name, class_name, filename):
        class_dir = self.dataset_root / split_name / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        image = np.full((12, 20, 3), 127, dtype=np.uint8)
        path = class_dir / filename
        self.assertTrue(cv2.imwrite(str(path), image))
        return path

    def test_loads_resized_images_with_stable_one_based_labels(self):
        self.write_image("training_set", "homer", "b.bmp")
        self.write_image("training_set", "homer", "a.bmp")
        self.write_image("training_set", "bart", "bart.bmp")

        images, labels = load_split(self.dataset_root, "training_set", 35)

        self.assertEqual(images.shape, (3, 35 * 35))
        self.assertEqual(labels.tolist(), [1, 2, 2])
        self.assertEqual(CLASS_NAMES, ("bart", "homer"))
        self.assertTrue(np.all((images >= 0.0) & (images <= 1.0)))

    def test_missing_split_raises_a_clear_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "Dataset split directory"):
            load_split(self.dataset_root, "training_set", 35)

    def test_missing_class_raises_a_clear_error(self):
        (self.dataset_root / "training_set" / "homer").mkdir(parents=True)

        with self.assertRaisesRegex(FileNotFoundError, "Class directory"):
            load_split(self.dataset_root, "training_set", 35)

    def test_unreadable_image_raises_with_its_path(self):
        self.write_image("training_set", "bart", "bart.bmp")
        homer_dir = self.dataset_root / "training_set" / "homer"
        homer_dir.mkdir(parents=True)
        bad_image = homer_dir / "broken.bmp"
        bad_image.write_text("not an image", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "broken.bmp"):
            load_split(self.dataset_root, "training_set", 35)


if __name__ == "__main__":
    unittest.main()
