import cv2
import torch.utils.data as data
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps, ImageEnhance


class image_dataset(data.Dataset):
    def __init__(self, cfg, is_training):
        split = 'train' if is_training else 'valid'
        self.folder = f"{cfg.data.root}/{split}"
        self.df = pd.read_csv(f"{self.folder}/labels.csv", sep=',', usecols=['filename', 'words'], dtype={'words': str})
        self.length = len(self.df)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = f"{self.folder}/{row['filename']}"
        label = str(row['words'])
        img = Image.open(path).convert('L')
        return img, label


class image_loader(data.DataLoader):
    def __init__(self, cfg, is_training):
        ds = image_dataset(cfg, is_training)
        super().__init__(
            dataset=ds,
            batch_size=cfg.batch,
            shuffle=is_training,
            num_workers=cfg.data.workers,
            collate_fn=image_collator(cfg)
        )
        self._iter = None

    def next_batch(self):
        if self._iter is None:
            self._iter = super().__iter__()
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = super().__iter__()
            return next(self._iter)


class image_collator:
    def __init__(self, cfg):
        self.h = cfg.rectifier.height
        self.w = cfg.rectifier.width
        self.brightness_thresh = 80
        self.contrast_mult = 1.5
        self.sharpen_mult = 1.3
        self.tile = 8

    def _apply_clahe(self, arr):
        hist = cv2.calcHist([arr], [0], None, [256], [0, 256])
        hist_sum = np.sum(hist[50:200])
        clip = max(1.0, min(4.0, 1.0 + hist_sum / (arr.size * 0.1)))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(self.tile, self.tile))
        return clahe.apply(arr)

    def _enhance(self, img):
        arr = np.array(img)
        if np.mean(arr) < self.brightness_thresh:
            if len(arr.shape) == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            arr = self._apply_clahe(arr)
        img = Image.fromarray(arr)
        img = ImageEnhance.Contrast(img).enhance(self.contrast_mult)
        img = ImageEnhance.Sharpness(img).enhance(self.sharpen_mult)
        img = ImageOps.autocontrast(img, cutoff=1, ignore=5)
        return img

    def _process(self, img):
        img = self._enhance(img)
        w, h = img.size
        ratio = w / float(h)
        new_w = min(int(self.h * ratio), self.w)
        img = img.resize((new_w, self.h), resample=Image.BICUBIC)
        tensor = transforms.ToTensor()(img)
        tensor = tensor.sub(0.5).div(0.5)
        c, h, w = tensor.size()
        padded = torch.zeros((c, h, self.w))
        padded[:, :, :w] = tensor
        if self.w != w:
            padded[:, :, w:] = tensor[:, :, w - 1].unsqueeze(2).expand(c, h, self.w - w)
        return padded

    def __call__(self, batch):
        imgs, labels = zip(*batch)
        imgs = [self._process(img) for img in imgs]
        return torch.stack(imgs), list(labels)

