import torch.backends.cudnn as cudnn
from .trainer import trainer
from .helpers import settings_reader


def main():
    cfg_path = "ocr_number/core/config/params.yaml"
    cudnn.benchmark = True
    cudnn.deterministic = True
    cfg = settings_reader.load(cfg_path)
    if cfg is None:
        print("failed to load config")
        return
    t = trainer(cfg, weights_path=None)
    t.run()


if __name__ == "__main__":
    main()

