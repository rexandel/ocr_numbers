from torch import nn
import torch
from .layers.spatial_transform import thin_plate_spline
from .layers.backbone import conv_backbone, block_type
from .layers.recurrent import bidirectional_rnn


class recognizer(nn.Module):
    def __init__(self, cfg):
        super(recognizer, self).__init__()
        self.cfg = cfg
        self.rectifier = self._build_rectifier(cfg)
        self.encoder = self._build_encoder(cfg)
        self.pool = nn.AdaptiveAvgPool2d((None, 1))
        self.sequencer = self._build_sequencer(cfg)
        self.decoder = self._build_decoder(cfg)
        self._init_params()

    def _init_params(self):
        for name, param in self.named_parameters():
            if 'fc2' in name:
                continue
            try:
                if 'bias' in name:
                    torch.nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    torch.nn.init.kaiming_normal_(param)
            except:
                if 'weight' in name:
                    param.data.fill_(1)

    def forward(self, x, text=None, training=True):
        if self.rectifier:
            x = self.rectifier(x)
        if self.encoder:
            x = self.encoder(x)
        x = self.pool(x.permute(0, 3, 1, 2)).squeeze(3)
        if self.sequencer:
            x = self.sequencer(x)
        if self.decoder:
            x = self.decoder(x.contiguous())
        return x

    @staticmethod
    def _build_rectifier(cfg):
        if cfg.rectifier.kind == "TPS":
            layer = thin_plate_spline(
                fiducial_points=cfg.rectifier.fiducials,
                input_channels=cfg.input_channels,
                height=cfg.rectifier.height,
                width=cfg.rectifier.width
            )
            return nn.Sequential(layer, nn.Dropout(cfg.rectifier.dropout))
        return None

    @staticmethod
    def _build_encoder(cfg):
        if cfg.encoder.kind == "ResNet18":
            layer = conv_backbone(
                in_channels=cfg.input_channels,
                out_channels=cfg.encoder.channels,
                block=block_type.basic,
                layer_sizes=[2, 2, 2, 2]
            )
            return nn.Sequential(layer, nn.Dropout(cfg.encoder.dropout))
        return None

    @staticmethod
    def _build_sequencer(cfg):
        if not hasattr(cfg, 'sequencer'):
            return None
        if cfg.sequencer.kind == "BiLSTM":
            rnn1 = bidirectional_rnn(cfg.encoder.channels, cfg.sequencer.units, cfg.sequencer.units)
            rnn2 = bidirectional_rnn(cfg.sequencer.units, cfg.sequencer.units, cfg.sequencer.units)
            return nn.Sequential(rnn1, nn.Dropout(cfg.sequencer.dropout), rnn2, nn.Dropout(cfg.sequencer.dropout))
        return None

    @staticmethod
    def _build_decoder(cfg):
        if cfg.decoder.kind == "CTC":
            input_size = cfg.sequencer.units if hasattr(cfg, 'sequencer') else cfg.encoder.channels
            return nn.Sequential(nn.Linear(input_size, len(cfg.alphabet)))
        return None

