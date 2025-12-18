import torch
from pathlib import Path
from PIL import Image
import sys
import yaml
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.helpers.settings import settings, settings_reader
from core.network.recognizer import recognizer
from core.helpers.text_encoder import text_encoder
from core.helpers.image_dataset import image_collator


def batch_label_and_review():

    CONFIG_PATH = "E:/GitHub/ocr/ocr_number/ocr_deti_kubgu/config/config.yaml"
    MODEL_PATH = "E:/GitHub/ocr/ocr_number/ocr_deti_kubgu/saved_models/universal_model_from_scratch/best_model.pth"
    INPUT_DIR = "E:/GitHub/ocr/dataset/train/ruzhimmash/number"
    AUTO_FILE = "E:/GitHub/ocr/auto_predictions.json"
    OUTPUT_FILE = "E:/GitHub/ocr/dataset/train/ruzhimmash/number.txt"

    print("\nЗагрузка модели...")
    
    config = settings_reader.load(CONFIG_PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = text_encoder(config.decoder.kind, config.alphabet)
    config.alphabet = encoder.symbols
    
    model = recognizer(config).to(device)
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    collator = image_collator(config)
    
    print(f"Модель загружена")
    
    print("АВТОМАТИЧЕСКАЯ РАЗМЕТКА")
    
    input_path = Path(INPUT_DIR)
    image_files = list(input_path.glob("*.jpg"))
    
    if not image_files:
        image_files = list(input_path.glob("*.jpeg"))
    
    print(f"Найдено {len(image_files)} изображений")
    
    auto_predictions = {}
    
    for image_path in tqdm(image_files, desc="Авторазметка"):
        filename = image_path.stem
        
        try:
            img = Image.open(image_path).convert('L')
            processed_img = collator._process(img)
            processed_img = processed_img.unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(processed_img)
                _, indices = output.max(2)
                preds_size = torch.IntTensor([output.size(1)])
                pred_str = encoder.decode(indices.view(-1).data.cpu(), preds_size.data.cpu())
                prediction = pred_str[0] if pred_str else ""
            
            auto_predictions[filename] = prediction
            
        except Exception as e:
            print(f"Ошибка при обработке {filename}: {e}")
            auto_predictions[filename] = ""
    
    with open(AUTO_FILE, 'w', encoding='utf-8') as f:
        json.dump(auto_predictions, f, ensure_ascii=False, indent=2)
    
    print(f"\nАвтопредсказания сохранены в {AUTO_FILE}")
    
    print("\n" + "="*30)
    print("ПРОВЕРКА И ИСПРАВЛЕНИЕ")
    print("Управление:")
    print("  Enter - принять предсказание")
    print("  введите текст - исправить")
    print("  's' - пропустить")
    print("  'q' - завершить")
    print("-"*60)
    
    final_labels = {}
    corrected = 0
    skipped = []
    
    for i, (filename, auto_pred) in enumerate(auto_predictions.items()):
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            path = input_path / f"{filename}{ext}"
            if path.exists():
                image_path = path
                break
        
        if not image_path:
            print(f"Файл не найден: {filename}")
            continue
        
        try:
            img = Image.open(image_path)
            img.show()
        except:
            print(f"Не удалось открыть: {filename}")
            continue
        
        print(f"\n[{i+1}/{len(auto_predictions)}] {filename}")
        print(f"Автопредсказание: '{auto_pred}'")
        
        while True:
            user_input = input("Введите правильный текст (Enter-принять, s-пропустить, q-выход): ").strip()
            
            if user_input.lower() == 'q':
                print("\nДосрочное завершение...")
                break
            elif user_input.lower() == 's':
                print(f"Пропущено: {filename}")
                skipped.append(filename)
                break
            elif user_input == '':
                final_labels[filename] = auto_pred
                break
            else:
                final_labels[filename] = user_input
                corrected += 1
                break
        
        if user_input.lower() == 'q':
            break
    
    print(f"\nСохранение {len(final_labels)} меток в {OUTPUT_FILE}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for filename, label in final_labels.items():
            f.write(f"{filename} {label}\n")
    
    print(f"\n" + "="*60)
    print("ИТОГИ:")
    print(f"Всего изображений: {len(auto_predictions)}")
    print(f"Размечено: {len(final_labels)}")
    print(f"Пропущено: {len(skipped)}")
    print(f"Исправлено: {corrected} ({corrected/len(final_labels)*100:.1f}%)")
    
    if corrected > 0:
        print(f"\nПримеры исправлений (первые 5):")
        corrections = [(k, auto_predictions[k], final_labels[k]) 
                      for k in final_labels.keys() if auto_predictions[k] != final_labels[k]]
        for filename, auto, final in corrections[:5]:
            print(f"  {filename}: '{auto}' -> '{final}'")
    
    print(f"\nРазметка завершена!")

if __name__ == "__main__":
    batch_label_and_review()