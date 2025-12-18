import os
from pathlib import Path
import sys

# Добавляем корневую директорию проекта в sys.path для импорта config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def windows_sort_key(filename):
    import re
    
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    
    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]
    
    return alphanum_key(filename)

def create_image_list_file():
    dataset_root = config.get_dataset_path()
    images_dir = dataset_root / "train" / "ruzhimmash" / "number"
    
    output_file = dataset_root / "train" / "ruzhimmash" / "number.txt"
    
    if not images_dir.exists():
        print(f"Ошибка: Папка {images_dir} не найдена!")
        return
    try:
        image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        image_files = [f for f in image_files if os.path.splitext(f)[1].lower() in image_extensions]
        
        image_names = [os.path.splitext(f)[0] for f in image_files]
        
        image_names.sort(key=windows_sort_key)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for filename in image_names:
                f.write(f"{filename} \n")
        
        print(f"Файл успешно создан: {output_file}")
        print(f"Количество записанных файлов: {len(image_names)}")
        if image_names:
            print("\nПервые 10 записей (как в проводнике Windows):")
            for i, filename in enumerate(image_names[:10]):
                print(f"{i+1:3d}. {filename}")
            
            if len(image_names) > 10:
                print(f"\n... и еще {len(image_names) - 10} файлов")
        
    except Exception as e:
        print(f"Ошибка при обработке файлов: {e}")

if __name__ == "__main__":
    create_image_list_file()