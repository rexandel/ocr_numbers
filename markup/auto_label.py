from pathlib import Path
from typing import List


def label_folder(folder_path: str, label: str) -> int:
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Папка не найдена: {folder}")
        return 0
    
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files: List[Path] = []
    for ext in extensions:
        image_files.extend(folder.glob(ext))
    
    if not image_files:
        print(f"Изображения не найдены в: {folder}")
        return 0
    
    image_files = sorted(image_files)
    
    txt_path = folder.parent / f"{folder.name}.txt"
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        for img_path in image_files:
            stem = img_path.stem
            f.write(f"{stem} {label}\n")
    
    print(f"Создан файл: {txt_path}")
    print(f"Размечено изображений: {len(image_files)}")
    
    return len(image_files)


def main():
    folder = r"D:\GitHub\ocr_numbers\dataset\train\begickaya\text"
    label = "55"
    label_folder(folder, label)


if __name__ == "__main__":
    main()
