"""
Загрузка конфигурации проекта из config.ini
"""

import configparser
from pathlib import Path


def get_config() -> configparser.ConfigParser:
    """Загружает конфигурацию из config.ini"""
    config = configparser.ConfigParser()
    config_path = Path(__file__).parent / "config.ini"
    config.read(config_path, encoding='utf-8')
    return config


def get_project_root() -> Path:
    """Возвращает корневую папку проекта"""
    return Path(__file__).parent


def get_dataset_path() -> Path:
    """Возвращает путь к папке dataset"""
    config = get_config()
    dataset_rel = config.get('paths', 'dataset', fallback='dataset')
    return get_project_root() / dataset_rel


def get_dataset_augmentation_path() -> Path:
    """Возвращает путь к папке dataset_augmentation"""
    config = get_config()
    aug_rel = config.get('paths', 'dataset_augmentation', fallback='dataset_augmentation')
    return get_project_root() / aug_rel
