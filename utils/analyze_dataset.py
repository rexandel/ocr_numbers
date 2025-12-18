import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_dataset_path

COLORS = ['#c2e5f5', '#5296b8', '#135669', '#1a7a8c', '#8bc4d9']
DATASET_PATH = str(get_dataset_path())


def count_images_in_folder(folder_path: str) -> int:
    if not os.path.exists(folder_path):
        return 0

    count = 0
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            count += 1
    return count


def get_dataset_stats(base_path: str, split: str, manufacturer: str) -> dict:
    manufacturer_path = os.path.join(base_path, split, manufacturer)

    if not os.path.exists(manufacturer_path):
        return {}

    stats = {}
    for item in os.listdir(manufacturer_path):
        item_path = os.path.join(manufacturer_path, item)
        if os.path.isdir(item_path):
            stats[item] = count_images_in_folder(item_path)

    return stats


def plot_bar_chart(stats: dict, title: str, colors: list, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    categories = list(stats.keys())
    values = list(stats.values())

    bar_colors = [colors[i % len(colors)] for i in range(len(categories))]
    bars = ax.bar(categories, values, color=bar_colors, edgecolor='white', linewidth=1.5)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            str(value),
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Категория', fontsize=11)
    ax.set_ylabel('Количество изображений', fontsize=11)
    ax.set_ylim(0, max(values) * 1.15 if values else 1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    return ax


def plot_pie_chart(stats: dict, title: str, colors: list, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    categories = list(stats.keys())
    values = list(stats.values())
    total = sum(values)

    pie_colors = [colors[i % len(colors)] for i in range(len(categories))]

    def autopct_format(pct):
        absolute = int(round(pct / 100. * total))
        return f'{pct:.1f}%\n({absolute})'

    wedges, texts, autotexts = ax.pie(
        values,
        labels=categories,
        colors=pie_colors,
        autopct=autopct_format,
        startangle=90,
        explode=[0.02] * len(categories),
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )

    for text in texts:
        text.set_fontsize(11)
        text.set_fontweight('bold')

    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    return ax


def create_all_charts():
    print("Сбор статистики датасета...")

    train_begickaya = get_dataset_stats(DATASET_PATH, 'train', 'begickaya')
    train_ruzhimmash = get_dataset_stats(DATASET_PATH, 'train', 'ruzhimmash')
    test_begickaya = get_dataset_stats(DATASET_PATH, 'test', 'begickaya')
    test_ruzhimmash = get_dataset_stats(DATASET_PATH, 'test', 'ruzhimmash')

    print("\n" + "=" * 50)
    print("СТАТИСТИКА ДАТАСЕТА")
    print("=" * 50)

    print("\nОБУЧАЮЩАЯ ВЫБОРКА (train)")
    print("-" * 30)
    print(f"Begickaya: {train_begickaya}")
    print(f"  Всего: {sum(train_begickaya.values())} изображений")
    print(f"Ruzhimmash: {train_ruzhimmash}")
    print(f"  Всего: {sum(train_ruzhimmash.values())} изображений")

    print("\nТЕСТОВАЯ ВЫБОРКА (test)")
    print("-" * 30)
    print(f"Begickaya: {test_begickaya}")
    print(f"  Всего: {sum(test_begickaya.values())} изображений")
    print(f"Ruzhimmash: {test_ruzhimmash}")
    print(f"  Всего: {sum(test_ruzhimmash.values())} изображений")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if train_begickaya:
        plot_bar_chart(train_begickaya, 'Обучающая выборка: Begickaya', COLORS, axes[0, 0])
    else:
        axes[0, 0].text(0.5, 0.5, 'Нет данных', ha='center', va='center', fontsize=14)
        axes[0, 0].set_title('Обучающая выборка: Begickaya', fontsize=14, fontweight='bold')

    if train_ruzhimmash:
        plot_bar_chart(train_ruzhimmash, 'Обучающая выборка: Ruzhimmash', COLORS[1:], axes[0, 1])
    else:
        axes[0, 1].text(0.5, 0.5, 'Нет данных', ha='center', va='center', fontsize=14)
        axes[0, 1].set_title('Обучающая выборка: Ruzhimmash', fontsize=14, fontweight='bold')

    if test_begickaya:
        plot_bar_chart(test_begickaya, 'Тестовая выборка: Begickaya', COLORS, axes[1, 0])
    else:
        axes[1, 0].text(0.5, 0.5, 'Нет данных', ha='center', va='center', fontsize=14)
        axes[1, 0].set_title('Тестовая выборка: Begickaya', fontsize=14, fontweight='bold')

    if test_ruzhimmash:
        plot_bar_chart(test_ruzhimmash, 'Тестовая выборка: Ruzhimmash', COLORS[1:], axes[1, 1])
    else:
        axes[1, 1].text(0.5, 0.5, 'Нет данных', ha='center', va='center', fontsize=14)
        axes[1, 1].set_title('Тестовая выборка: Ruzhimmash', fontsize=14, fontweight='bold')

    plt.suptitle('Распределение изображений в датасете', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_dir = os.path.join(os.path.dirname(__file__), 'dataset_analysis')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'dataset_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nСтолбчатые графики сохранены: {output_path}")
    plt.show()

    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

    if train_begickaya:
        plot_pie_chart(train_begickaya, 'Обучающая выборка: Begickaya', COLORS, axes2[0, 0])
    else:
        axes2[0, 0].text(0.5, 0.5, 'Нет данных', ha='center', va='center', fontsize=14)
        axes2[0, 0].set_title('Обучающая выборка: Begickaya', fontsize=14, fontweight='bold')

    if train_ruzhimmash:
        plot_pie_chart(train_ruzhimmash, 'Обучающая выборка: Ruzhimmash', COLORS[1:], axes2[0, 1])
    else:
        axes2[0, 1].text(0.5, 0.5, 'Нет данных', ha='center', va='center', fontsize=14)
        axes2[0, 1].set_title('Обучающая выборка: Ruzhimmash', fontsize=14, fontweight='bold')

    if test_begickaya:
        plot_pie_chart(test_begickaya, 'Тестовая выборка: Begickaya', COLORS, axes2[1, 0])
    else:
        axes2[1, 0].text(0.5, 0.5, 'Нет данных', ha='center', va='center', fontsize=14)
        axes2[1, 0].set_title('Тестовая выборка: Begickaya', fontsize=14, fontweight='bold')

    if test_ruzhimmash:
        plot_pie_chart(test_ruzhimmash, 'Тестовая выборка: Ruzhimmash', COLORS[1:], axes2[1, 1])
    else:
        axes2[1, 1].text(0.5, 0.5, 'Нет данных', ha='center', va='center', fontsize=14)
        axes2[1, 1].set_title('Тестовая выборка: Ruzhimmash', fontsize=14, fontweight='bold')

    plt.suptitle('Процентное соотношение категорий в датасете', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path_pie = os.path.join(output_dir, 'dataset_analysis_pie.png')
    plt.savefig(output_path_pie, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Круговые диаграммы сохранены: {output_path_pie}")
    plt.show()

    return {
        'train_begickaya': train_begickaya,
        'train_ruzhimmash': train_ruzhimmash,
        'test_begickaya': test_begickaya,
        'test_ruzhimmash': test_ruzhimmash
    }


if __name__ == "__main__":
    stats = create_all_charts()
