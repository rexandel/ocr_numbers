import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Callable
from tqdm import tqdm

from augmentation_preview import (
    load_image,
    apply_random_rotation,
    apply_random_affine,
    apply_perspective_transform,
    apply_random_blur,
    apply_random_noise,
    apply_random_brightness_contrast
)


AVAILABLE_AUGMENTATIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    'rotation': apply_random_rotation,
    'affine': apply_random_affine,
    'perspective': apply_perspective_transform,
    'blur': apply_random_blur,
    'noise': apply_random_noise,
    'brightness': apply_random_brightness_contrast,
}


class DatasetAugmenter:

    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        augmentations: List[str]
    ):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.augmentations = augmentations

        for aug_name in augmentations:
            if aug_name not in AVAILABLE_AUGMENTATIONS:
                raise ValueError(
                    f"Неизвестная аугментация: '{aug_name}'. "
                    f"Доступные: {list(AVAILABLE_AUGMENTATIONS.keys())}"
                )

        self.stats = {
            'processed_images': 0,
            'created_augmentations': 0,
            'skipped_unlabeled': 0,
            'errors': 0
        }

    def read_labels(self, txt_path: Path) -> Dict[str, str]:
        labels = {}
        if not txt_path.exists():
            return labels

        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) == 2:
                    filename, label = parts
                    labels[filename] = label
        return labels

    def write_labels(self, txt_path: Path, labels: Dict[str, str]) -> None:
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(txt_path, 'w', encoding='utf-8') as f:
            for filename, label in sorted(labels.items()):
                f.write(f"{filename} {label}\n")

    def augment_image(self, image: np.ndarray, aug_name: str) -> np.ndarray:
        aug_func = AVAILABLE_AUGMENTATIONS[aug_name]
        return aug_func(image)

    def process_category(
        self,
        source_subdir: Path,
        output_subdir: Path,
        category: str
    ) -> None:
        txt_path = source_subdir / f"{category}.txt"
        images_dir = source_subdir / category

        output_txt_path = output_subdir / f"{category}.txt"
        output_images_dir = output_subdir / category

        if not txt_path.exists() or not images_dir.exists():
            return

        labels = self.read_labels(txt_path)
        if not labels:
            return

        output_images_dir.mkdir(parents=True, exist_ok=True)
        new_labels = {}

        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

        for img_path in tqdm(image_files, leave=False):
            stem = img_path.stem

            if stem not in labels:
                self.stats['skipped_unlabeled'] += 1
                continue

            label = labels[stem]

            try:
                image = load_image(str(img_path))

                output_original = output_images_dir / img_path.name
                cv2.imwrite(str(output_original), image)
                new_labels[stem] = label
                self.stats['processed_images'] += 1

                for aug_name in self.augmentations:
                    aug_image = self.augment_image(image, aug_name)
                    aug_filename = f"{stem}_{aug_name}{img_path.suffix}"
                    aug_stem = f"{stem}_{aug_name}"
                    output_aug_path = output_images_dir / aug_filename
                    cv2.imwrite(str(output_aug_path), aug_image)
                    new_labels[aug_stem] = label
                    self.stats['created_augmentations'] += 1

            except Exception:
                self.stats['errors'] += 1

        self.write_labels(output_txt_path, new_labels)

    def process_split(self, split: str) -> None:
        source_split_dir = self.source_dir / split
        output_split_dir = self.output_dir / split

        if not source_split_dir.exists():
            return

        for source_name_dir in sorted(source_split_dir.iterdir()):
            if not source_name_dir.is_dir():
                continue

            output_name_dir = output_split_dir / source_name_dir.name

            categories = []
            for item in source_name_dir.iterdir():
                if item.is_dir() and (source_name_dir / f"{item.name}.txt").exists():
                    categories.append(item.name)

            if not categories:
                for txt_file in source_name_dir.glob("*.txt"):
                    cat_name = txt_file.stem
                    if (source_name_dir / cat_name).is_dir():
                        categories.append(cat_name)

            for category in categories:
                self.process_category(source_name_dir, output_name_dir, category)

    def run(self, splits: Optional[List[str]] = None) -> None:
        if splits is None:
            splits = ['train', 'test']

        for split in splits:
            self.process_split(split)

        total = self.stats['processed_images'] + self.stats['created_augmentations']
        print(f"Обработано изображений: {self.stats['processed_images']}")
        print(f"Создано аугментаций: {self.stats['created_augmentations']}")
        print(f"Пропущено неразмеченных: {self.stats['skipped_unlabeled']}")
        print(f"Ошибок: {self.stats['errors']}")
        print(f"Всего изображений в новом датасете: {total}")


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    source_dir = project_root / "dataset"
    output_dir = project_root / "dataset_augmentation"

    splits = ['train']

    augmentations = [
        'rotation',
        'affine',
        'perspective',
        'blur',
        'noise',
        'brightness',
    ]

    augmenter = DatasetAugmenter(
        source_dir=str(source_dir),
        output_dir=str(output_dir),
        augmentations=augmentations
    )

    augmenter.run(splits=splits)


if __name__ == "__main__":
    main()
