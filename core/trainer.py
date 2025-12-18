from .helpers import settings, image_loader, text_encoder
from .network import recognizer
import torch
import time
from tqdm import tqdm
from pathlib import Path


class trainer:
    def __init__(self, cfg, weights_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        self.output_dir = Path(f"ocr_number/core/weights/{self.cfg.name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.encoder = text_encoder(self.cfg.decoder.kind, self.cfg.alphabet)
        self.cfg.alphabet = self.encoder.symbols
        self.net = recognizer(self.cfg).to(self.device)
        if weights_path:
            self._load_weights(weights_path)
        self.loss_fn = self._create_loss()
        self.opt = self._create_optimizer()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, factor=0.5, patience=5)
        self.train_loader = image_loader(cfg, is_training=True)
        self.valid_loader = image_loader(cfg, is_training=False)

    def run(self):
        best_acc = 0
        step = 0
        t0 = time.time()
        while step < self.cfg.max_steps:
            self.net.train()
            epoch_loss = 0
            bar = tqdm(range(self.cfg.eval_interval), desc=f'step {step}/{self.cfg.max_steps}')
            for _ in bar:
                imgs, labels = self.train_loader.next_batch()
                imgs = imgs.to(self.device)
                bs = imgs.size(0)
                txt, lengths = self.encoder.encode(labels)
                txt, lengths = txt.to(self.device), lengths.to(self.device)
                preds = self.net(imgs, labels).log_softmax(2)
                pred_lens = torch.IntTensor([preds.size(1)] * bs).to(self.device)
                preds = preds.permute(1, 0, 2)
                torch.backends.cudnn.enabled = False
                loss = self.loss_fn(preds, txt, pred_lens, lengths)
                torch.backends.cudnn.enabled = True
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.optimizer.clip_norm)
                self.opt.step()
                step += 1
                epoch_loss += loss.item()
                bar.set_postfix(loss=loss.item())
            avg_loss = epoch_loss / self.cfg.eval_interval
            val_loss, val_acc = self._evaluate()
            self.scheduler.step(val_loss)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.net.state_dict(), str(self.output_dir / "best.pth"))
                tqdm.write(f'saved best model: {best_acc:.2f}%')
            tqdm.write(f'step {step} | train_loss: {avg_loss:.4f} | val_loss: {val_loss:.4f} | acc: {val_acc:.2f}%')

    def _evaluate(self):
        self.net.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in self.valid_loader:
                imgs = imgs.to(self.device)
                bs = imgs.size(0)
                preds = self.net(imgs, labels)
                pred_lens = torch.IntTensor([preds.size(1)] * bs).to(self.device)
                txt, lengths = self.encoder.encode(labels)
                txt, lengths = txt.to(self.device), lengths.to(self.device)
                loss = self.loss_fn(preds.log_softmax(2).permute(1, 0, 2), txt, pred_lens, lengths)
                total_loss += loss.item()
                _, pred_idx = preds.max(2)
                decoded = self.encoder.decode(pred_idx.view(-1).cpu(), pred_lens.cpu())
                for pred, gt in zip(decoded, labels):
                    if pred == gt:
                        correct += 1
                total += bs
        return total_loss / len(self.valid_loader), 100 * correct / total

    def _create_loss(self):
        if self.cfg.decoder.kind == "CTC":
            return torch.nn.CTCLoss(zero_infinity=True).to(self.device)
        raise ValueError("unknown decoder")

    def _create_optimizer(self):
        params = [p for p in self.net.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=self.cfg.optimizer.learning_rate, betas=self.cfg.optimizer.betas, weight_decay=self.cfg.optimizer.decay)

    def _load_weights(self, path):
        try:
            ckpt = torch.load(path, map_location=self.device)
            model_dict = self.net.state_dict()
            filtered = {k: v for k, v in ckpt.items() if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(filtered)
            self.net.load_state_dict(model_dict, strict=False)
            print(f"loaded {len(filtered)}/{len(ckpt)} weights")
        except Exception as e:
            print(f"failed to load weights: {e}")

