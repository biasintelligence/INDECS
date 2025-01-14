import os
import numpy as np
import cv2
from tqdm import tqdm
from src.indecs.config import Config as app_config

def calculate_image_statistics(image_dir):
    """Calculate mean and std of images in directory."""
    pixel_sum = 0
    pixel_square_sum = 0
    pixel_count = 0

    # Get all image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Processing {len(image_files)} images...")

    # First pass: calculate mean
    for img_file in tqdm(image_files, desc="Calculating mean"):
        img_path = os.path.join(image_dir, img_file)
        # Read as grayscale since these are thermal images
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            pixel_sum += np.sum(img)
            pixel_count += img.size

    mean = pixel_sum / pixel_count

    # Second pass: calculate std
    for img_file in tqdm(image_files, desc="Calculating std"):
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            pixel_square_sum += np.sum((img - mean) ** 2)

    std = np.sqrt(pixel_square_sum / pixel_count)

    return mean, std


def main():
    # Path to your training images
    train_image_dir = app_config.ANNOTATIONS_YOLO_PATH

    mean, std = calculate_image_statistics(train_image_dir)

    print("\nDataset Statistics:")
    print(f"Mean: {mean:.3f}")
    print(f"Std:  {std:.3f}")

    print("\nFor MMDetection config:")
    print("img_norm_cfg = dict(")
    print(f"    mean=[{mean:.3f}],")
    print(f"    std=[{std:.3f}],")
    print("    to_rgb=False)")


if __name__ == "__main__":
    main()