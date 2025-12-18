import torch.backends.cudnn as cudnn
from pathlib import Path
import argparse
from .trainer import trainer
from .helpers import settings_reader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default=None,
                        help='Путь к датасету (переопределяет params.yaml)')
    parser.add_argument('--weights', '-w', type=str, default=None,
                        help='Путь к весам для загрузки')
    args = parser.parse_args()
    
    cfg_path = str(Path(__file__).parent / "config" / "params.yaml")
    cudnn.benchmark = True
    cudnn.deterministic = True
    cfg = settings_reader.load(cfg_path)
    if cfg is None:
        print("failed to load config")
        return
    
    # Переопределяем путь к датасету, если указан
    if args.dataset:
        cfg.data.root = args.dataset
        print(f"Используется датасет: {cfg.data.root}")
    
    t = trainer(cfg, weights_path=args.weights)
    t.run()


if __name__ == "__main__":
    main()

