import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from dataset_loader import CLASS_NAMES, OUTPUT_SPLITS, load_group_splits


class LoadGroupSplitsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_root = Path(self.temp_dir.name)
        for split_name in ("train", "val", "test"):
            for class_name in CLASS_NAMES:
                (self.dataset_root / split_name / class_name).mkdir(
                    parents=True, exist_ok=True
                )

        for index in range(12):
            source_split = ("train", "val", "test")[index % 3]
            normal_stem = f"IM-{index:04d}-0001"
            self.write_image(
                source_split, "NORMAL", f"{normal_stem}.jpeg", 50 + index
            )
            pneumonia_person = f"person{index + 1}"
            self.write_image(
                source_split,
                "PNEUMONIA",
                f"{pneumonia_person}_bacteria_1.jpeg",
                100 + index,
            )

            if index < 4:
                other_split = ("test", "train", "val")[index % 3]
                self.write_image(
                    other_split,
                    "NORMAL",
                    f"{normal_stem}_aug_{index}.jpg",
                    60 + index,
                )
                self.write_image(
                    other_split,
                    "PNEUMONIA",
                    f"{pneumonia_person}_bacteria_2.jpeg",
                    110 + index,
                )

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_image(self, split_name, class_name, filename, pixel_value):
        path = self.dataset_root / split_name / class_name / filename
        image = np.full((16, 20), pixel_value, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(path), image))
        return path

    def test_splits_source_groups_without_leakage(self):
        splits = load_group_splits(self.dataset_root, image_size=10, seed=7)

        self.assertEqual(set(splits), set(OUTPUT_SPLITS))
        group_sets = {
            split_name: set(batch.group_ids)
            for split_name, batch in splits.items()
        }
        for first, second in (
            ("train", "val"),
            ("train", "test"),
            ("val", "test"),
        ):
            self.assertFalse(group_sets[first] & group_sets[second])

        self.assertEqual(splits["train"].features.shape[1], 100)
        self.assertTrue(
            np.all(
                (splits["train"].features >= 0.0)
                & (splits["train"].features <= 1.0)
            )
        )
        self.assertEqual(set(splits["train"].labels.tolist()), {1, 2})

    def test_split_is_deterministic_for_a_seed(self):
        first = load_group_splits(self.dataset_root, image_size=10, seed=11)
        second = load_group_splits(self.dataset_root, image_size=10, seed=11)

        for split_name in OUTPUT_SPLITS:
            self.assertEqual(first[split_name].group_ids, second[split_name].group_ids)
            np.testing.assert_array_equal(
                first[split_name].labels, second[split_name].labels
            )

    def test_missing_split_raises_a_clear_error(self):
        (self.dataset_root / "val").rename(self.dataset_root / "validation")

        with self.assertRaisesRegex(FileNotFoundError, "Dataset split directory"):
            load_group_splits(self.dataset_root, image_size=10)

    def test_unreadable_image_raises_with_its_path(self):
        broken_image = (
            self.dataset_root / "train" / "NORMAL" / "unreadable.jpeg"
        )
        broken_image.write_text("not an image", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "unreadable.jpeg"):
            load_group_splits(self.dataset_root, image_size=10)


if __name__ == "__main__":
    unittest.main()
