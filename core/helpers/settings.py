from pydantic import BaseModel, ValidationError
from typing import Literal, Optional, Tuple
import yaml
from pprint import pformat


class rectifier_settings(BaseModel):
    kind: Literal["TPS"]
    height: int
    width: int
    fiducials: int
    dropout: float


class encoder_settings(BaseModel):
    kind: Literal["ResNet18"]
    channels: int
    dropout: float


class sequencer_settings(BaseModel):
    kind: Literal["BiLSTM"]
    units: int
    dropout: float


class decoder_settings(BaseModel):
    kind: Literal["CTC"]


class optimizer_settings(BaseModel):
    learning_rate: float
    betas: Tuple[float, float]
    decay: float
    clip_norm: int


class data_settings(BaseModel):
    root: str
    workers: int


class settings(BaseModel):
    name: str
    alphabet: str
    batch: int
    input_channels: int
    max_steps: int
    eval_interval: int
    rectifier: rectifier_settings
    encoder: encoder_settings
    sequencer: sequencer_settings
    decoder: decoder_settings
    optimizer: optimizer_settings
    data: data_settings


class settings_reader:
    @staticmethod
    def load(path):
        ext = path.split(".")[-1].lower()
        if ext in ["yaml", "yml"]:
            return settings_reader._load_yaml(path)
        return None

    @staticmethod
    def _load_yaml(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return settings(**data)
        except FileNotFoundError:
            print(f"file not found: {path}")
            return None
        except yaml.YAMLError as e:
            print(f"yaml error: {e}")
            return None
        except ValidationError as e:
            print(f"validation error: {pformat(e.errors())}")
            return None

