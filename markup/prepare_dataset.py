import pandas as pd
import shutil
from pathlib import Path
import random
from collections import defaultdict
import sys

# Добавляем корневую директорию проекта в sys.path для импорта config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def prepare_balanced_dataset(
    source_dir: str = None,
    output_dir: str = None,
    seed: int = 42
):

    random.seed(seed)
    
    if source_dir is None:
        # По умолчанию используем begickaya из dataset/train
        source_dir = config.get_dataset_path() / "train" / "begickaya"
    
    if output_dir is None:
        # По умолчанию сохраняем в dataset/train_balanced
        output_dir = config.get_training_dataset_path()
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if output_path.exists():
        print(f"Удаление старого датасета из {output_path}...")
        shutil.rmtree(output_path)
    
    train_path = output_path / "train"
    valid_path = output_path / "valid"
    train_path.mkdir(parents=True, exist_ok=True)
    valid_path.mkdir(parents=True, exist_ok=True)
    
    dataset_config = {
        'number': {
            'sample_ratio': 1.0,
            'valid_ratio': 0.25
        },
        'prod': {
            'sample_ratio': 0.10,
            'valid_ratio': 0.10
        },
        'year': {
            'sample_ratio': 0.10,
            'valid_ratio': 0.10
        },
        'text': {
            'sample_ratio': 0.10,
            'valid_ratio': 0.10
        }
    }
    
    all_datasets = {}
    for dataset_type in ["number", "prod", "year", "text"]:
        txt_file = source_path / f"{dataset_type}.txt"
        img_dir = source_path / dataset_type
        
        if not txt_file.exists() or not img_dir.exists():
            print(f"Предупреждение: {dataset_type} не найден, пропускаем")
            continue
        
        df = pd.read_csv(txt_file, sep=' ', header=None, names=['filename', 'label'])
        
        data = []
        for _, row in df.iterrows():
            filename = row['filename']
            label = str(row['label'])
            
            img_path = img_dir / f"{filename}.jpg"
            if not img_path.exists():
                for ext in ['.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    alt_path = img_dir / f"{filename}{ext}"
                    if alt_path.exists():
                        img_path = alt_path
                        break
                else:
                    continue
            
            data.append({
                'filename': filename,
                'label': label,
                'source': dataset_type,
                'img_path': img_path
            })
        
        all_datasets[dataset_type] = data
        print(f"{dataset_type}: {len(data)} примеров доступно")
    
    number_available = len(all_datasets.get('number', []))
    prod_available = len(all_datasets.get('prod', []))
    year_available = len(all_datasets.get('year', []))
    text_available = len(all_datasets.get('text', []))
    
    target_number = number_available  # 100% от number
    target_prod = int(prod_available * 0.10)  # 10% от prod
    target_year = int(year_available * 0.10)  # 10% от year
    target_text = int(text_available * 0.10)  # 10% от text
    
    print(f"\n{'='*60}")
    print("РАСПРЕДЕЛЕНИЕ:")
    print(f"number: {target_number} примеров (100%)")
    print(f"prod: {target_prod} примеров (10%)")
    print(f"year: {target_year} примеров (10%)")
    print(f"text: {target_text} примеров (10%)")
    total_target = target_number + target_prod + target_year + target_text
    print(f"ИТОГО: {total_target} примеров")
    
    train_data = []
    valid_data = []
    
    for dataset_type, target_count in [('number', target_number), ('prod', target_prod), ('year', target_year), ('text', target_text)]:
        if dataset_type not in all_datasets:
            continue
        
        data = all_datasets[dataset_type].copy()
        random.shuffle(data)
        
        selected = data[:target_count]
        
        valid_ratio = dataset_config[dataset_type]['valid_ratio']
        valid_count = int(len(selected) * valid_ratio)
        
        type_valid = selected[:valid_count]
        type_train = selected[valid_count:]
        
        train_data.extend(type_train)
        valid_data.extend(type_valid)
        
        print(f"\n{dataset_type}:")
        print(f"  Отобрано: {len(selected)}")
        print(f"  Train: {len(type_train)} ({100*(1-valid_ratio):.0f}%)")
        print(f"  Valid: {len(type_valid)} ({100*valid_ratio:.0f}%)")
    
    random.shuffle(train_data)
    random.shuffle(valid_data)
    
    print(f"\n{'='*60}")
    print("ИТОГОВОЕ РАСПРЕДЕЛЕНИЕ:")
    print(f"Train: {len(train_data)} примеров")
    print(f"Valid: {len(valid_data)} примеров")
    
    for split_name, split_data in [('Train', train_data), ('Valid', valid_data)]:
        by_type = defaultdict(int)
        for item in split_data:
            by_type[item['source']] += 1
        print(f"\n{split_name} по типам:")
        for dtype, count in sorted(by_type.items()):
            pct = 100 * count / len(split_data) if split_data else 0
            print(f"  {dtype}: {count} ({pct:.1f}%)")
    
    def save_dataset(data_list, target_path):
        labels_data = []
        
        for item in data_list:
            src_img = item['img_path']
            original_ext = src_img.suffix
            dst_filename = f"{item['filename']}{original_ext}"
            dst_img = target_path / dst_filename
            
            if dst_img.exists():
                base_name = item['filename']
                counter = 1
                while dst_img.exists():
                    dst_filename = f"{base_name}_{item['source']}_{counter}{original_ext}"
                    dst_img = target_path / dst_filename
                    counter += 1
            
            try:
                shutil.copy2(src_img, dst_img)
            except Exception as e:
                print(f"Ошибка при копировании {src_img}: {e}")
                continue
            
            labels_data.append({
                'filename': dst_filename,
                'words': item['label']
            })
        
        labels_df = pd.DataFrame(labels_data)
        labels_df.to_csv(target_path / "labels.csv", index=False, sep=',')
        print(f"Сохранено {len(labels_data)} примеров в {target_path}")
    
    print(f"\n{'='*60}")
    print("СОХРАНЕНИЕ:")
    save_dataset(train_data, train_path)
    save_dataset(valid_data, valid_path)
    
    print(f"\n{'='*60}")
    print("ГОТОВО!")
    print(f"Train: {train_path}")
    print(f"Valid: {valid_path}")

if __name__ == "__main__":
    prepare_balanced_dataset()

