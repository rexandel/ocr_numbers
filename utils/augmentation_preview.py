import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Tuple


def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    return image


def apply_affine_transform(
    image: np.ndarray,
    scale: float = 1.0,
    shear_x: float = 0.0,
    shear_y: float = 0.0,
    translate_x: int = 0,
    translate_y: int = 0
) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    M = np.array([
        [scale, shear_x, translate_x + (1 - scale) * center[0] - shear_x * center[1]],
        [shear_y, scale, translate_y + (1 - scale) * center[1] - shear_y * center[0]]
    ], dtype=np.float32)

    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def apply_random_affine(
    image: np.ndarray,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    shear_range: float = 0.15,
    translate_range: int = 10
) -> np.ndarray:
    scale = np.random.uniform(*scale_range)
    shear_x = np.random.uniform(-shear_range, shear_range)
    shear_y = np.random.uniform(-shear_range, shear_range)
    translate_x = np.random.randint(-translate_range, translate_range + 1)
    translate_y = np.random.randint(-translate_range, translate_range + 1)

    return apply_affine_transform(
        image, scale, shear_x, shear_y, translate_x, translate_y
    )


def apply_rotation(image: np.ndarray, angle: float, scale: float = 1.0) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(
        image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101
    )


def apply_random_rotation(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    angle = np.random.uniform(-max_angle, max_angle)
    return apply_rotation(image, angle)


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_motion_blur(image: np.ndarray, kernel_size: int = 15, angle: int = 0) -> np.ndarray:
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = np.ones(kernel_size)
    kernel = kernel / kernel_size

    center = (kernel_size / 2, kernel_size / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))

    return cv2.filter2D(image, -1, kernel)


def apply_random_blur(image: np.ndarray, max_kernel: int = 7) -> np.ndarray:
    blur_type = np.random.choice(['gaussian', 'motion', 'none'])

    if blur_type == 'gaussian':
        kernel_size = np.random.choice([3, 5, 7])
        kernel_size = min(kernel_size, max_kernel)
        return apply_gaussian_blur(image, kernel_size)
    elif blur_type == 'motion':
        kernel_size = np.random.randint(5, 12)
        angle = np.random.randint(0, 180)
        return apply_motion_blur(image, kernel_size, angle)
    return image


def apply_gaussian_noise(
    image: np.ndarray, mean: float = 0, sigma: float = 25
) -> np.ndarray:
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_salt_pepper_noise(image: np.ndarray, amount: float = 0.02) -> np.ndarray:
    noisy = image.copy()

    num_salt = int(amount * image.size * 0.5)
    coords = [np.random.randint(0, i, num_salt) for i in image.shape]
    noisy[coords[0], coords[1]] = 255

    num_pepper = int(amount * image.size * 0.5)
    coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
    noisy[coords[0], coords[1]] = 0

    return noisy


def apply_random_noise(image: np.ndarray) -> np.ndarray:
    noise_type = np.random.choice(['gaussian', 'salt_pepper', 'none'])

    if noise_type == 'gaussian':
        sigma = np.random.uniform(10, 30)
        return apply_gaussian_noise(image, sigma=sigma)
    elif noise_type == 'salt_pepper':
        amount = np.random.uniform(0.01, 0.03)
        return apply_salt_pepper_noise(image, amount)
    return image


def apply_brightness_contrast(
    image: np.ndarray, brightness: float = 0, contrast: float = 1.0
) -> np.ndarray:
    result = image.astype(np.float32)
    result = result * contrast + brightness
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_random_brightness_contrast(
    image: np.ndarray,
    brightness_range: Tuple[int, int] = (-30, 30),
    contrast_range: Tuple[float, float] = (0.8, 1.2)
) -> np.ndarray:
    brightness = np.random.uniform(*brightness_range)
    contrast = np.random.uniform(*contrast_range)
    return apply_brightness_contrast(image, brightness, contrast)


def apply_perspective_transform(image: np.ndarray, strength: float = 0.1) -> np.ndarray:
    h, w = image.shape[:2]

    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    offset = int(min(w, h) * strength)
    dst_pts = src_pts + np.random.uniform(-offset, offset, src_pts.shape).astype(np.float32)

    dst_pts[:, 0] = np.clip(dst_pts[:, 0], 0, w)
    dst_pts[:, 1] = np.clip(dst_pts[:, 1], 0, h)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def apply_random_augmentation(
    image: np.ndarray,
    p_rotation: float = 0.5,
    p_affine: float = 0.5,
    p_perspective: float = 0.3,
    p_blur: float = 0.3,
    p_noise: float = 0.3,
    p_brightness: float = 0.5
) -> np.ndarray:
    result = image.copy()

    if np.random.random() < p_rotation:
        result = apply_random_rotation(result)

    if np.random.random() < p_affine:
        result = apply_random_affine(result)

    if np.random.random() < p_perspective:
        result = apply_perspective_transform(result)

    if np.random.random() < p_blur:
        result = apply_random_blur(result)

    if np.random.random() < p_noise:
        result = apply_random_noise(result)

    if np.random.random() < p_brightness:
        result = apply_random_brightness_contrast(result)

    return result


def apply_all_augmentations(image: np.ndarray) -> dict:
    return {
        'original': image.copy(),
        'affine_scale_up': apply_affine_transform(image, scale=1.15),
        'affine_scale_down': apply_affine_transform(image, scale=0.85),
        'affine_shear_x': apply_affine_transform(image, shear_x=0.15),
        'affine_shear_y': apply_affine_transform(image, shear_y=0.15),
        'rotate_5': apply_rotation(image, 5),
        'rotate_10': apply_rotation(image, 10),
        'rotate_15': apply_rotation(image, 15),
        'rotate_-10': apply_rotation(image, -10),
        'blur_gaussian_3': apply_gaussian_blur(image, kernel_size=3),
        'blur_gaussian_5': apply_gaussian_blur(image, kernel_size=5),
        'blur_motion_h': apply_motion_blur(image, kernel_size=10, angle=0),
        'blur_motion_diag': apply_motion_blur(image, kernel_size=10, angle=45),
        'noise_gaussian': apply_gaussian_noise(image, sigma=20),
        'noise_salt_pepper': apply_salt_pepper_noise(image, amount=0.02),
        'bright_up': apply_brightness_contrast(image, brightness=30),
        'bright_down': apply_brightness_contrast(image, brightness=-30),
        'contrast_up': apply_brightness_contrast(image, contrast=1.3),
        'contrast_down': apply_brightness_contrast(image, contrast=0.7),
    }


def visualize_augmentations(augmented: dict, figsize: tuple = (16, 12)) -> None:
    n_images = len(augmented)
    n_cols = 5
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for idx, (name, img) in enumerate(augmented.items()):
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(name, fontsize=9)
        axes[idx].axis('off')

    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(__file__), "augmentation_output")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'augmentation_demo.png'), dpi=150, bbox_inches='tight')
    plt.show()


def save_augmented_images(augmented: dict, output_dir: str, base_name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for name, img in augmented.items():
        filename = f"{base_name}_{name}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, img)


def demo_augmentation(image_path: str) -> dict:
    image = load_image(image_path)
    augmented = apply_all_augmentations(image)
    visualize_augmentations(augmented)
    return augmented


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import get_dataset_path

    dataset_path = str(get_dataset_path())

    test_image = None
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.jpg'):
                test_image = os.path.join(root, file)
                break
        if test_image:
            break

    if test_image:
        augmented = demo_augmentation(test_image)
        output_dir = os.path.join(os.path.dirname(__file__), "augmentation_output")
        base_name = os.path.splitext(os.path.basename(test_image))[0]
        save_augmented_images(augmented, output_dir, base_name)

        for _ in range(3):
            _ = apply_random_augmentation(load_image(test_image))
