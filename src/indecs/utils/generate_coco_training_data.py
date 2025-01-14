import os
import re
import json
import shutil
import argparse
from pathlib import Path
import random
from collections import defaultdict
import logging
from typing import Dict, List, Tuple

from src.indecs.config import Config as app_config


class YOLOToCOCOConverter:
    def __init__(self, input_dir: str, output_dir: str, train_ratio: float = 0.8):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.image_id = 1
        self.annotation_id = 1

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    @staticmethod
    def parse_filename(filename: str) -> Dict:
        """Parse YOLO filename into components."""
        parts = filename.split('_')

        # Extract components
        lek_number = parts[0]

        # Extract height and path type
        height_path = parts[1]
        height = height_path[:3]
        path_type = height_path[3:]

        # Extract date
        date_pattern = r'\d{8}'
        date_match = re.search(date_pattern, filename)
        date = date_match.group(0) if date_match else None

        # Extract frame number (last component before extension)
        frame_number = Path(filename).stem.split('_')[-1]

        return {
            'lek_number': lek_number,
            'height': height,
            'path_type': path_type,
            'date': date,
            'frame_number': frame_number
        }

    def group_files(self) -> Dict[str, List[Tuple[Path, Path]]]:
        """Group corresponding image and annotation files."""
        image_files = set(f.stem for f in self.input_dir.glob('*.jpg'))
        annotation_files = set(f.stem for f in self.input_dir.glob('*.txt'))

        # Find common files
        common_stems = image_files.intersection(annotation_files)

        # Group by stratification keys
        grouped_files = defaultdict(list)
        for stem in common_stems:
            img_path = self.input_dir / f"{stem}.jpg"
            ann_path = self.input_dir / f"{stem}.txt"

            # Parse filename to get stratification key
            parsed = self.parse_filename(stem)
            strat_key = f"{parsed['lek_number']}_{parsed['height']}_{parsed['path_type']}"

            grouped_files[strat_key].append((img_path, ann_path))

        return grouped_files

    @staticmethod
    def convert_yolo_to_coco_annotation(yolo_bbox: List[float], image_width: int, image_height: int) -> Dict:
        """Convert YOLO format bbox to COCO format."""
        x_center, y_center, width, height = yolo_bbox

        # Convert normalized coordinates to absolute
        x_center = float(x_center) * image_width
        y_center = float(y_center) * image_height
        width = float(width) * image_width
        height = float(height) * image_height

        # Convert to COCO format [x,y,width,height] where x,y is top-left corner
        x = x_center - width / 2
        y = y_center - height / 2

        return {
            'bbox': [x, y, width, height],
            'area': width * height,
            'iscrowd': 0
        }

    def create_coco_dataset(self, files: List[Tuple[Path, Path]], split: str) -> Dict:
        """Create COCO format dataset from list of files."""
        dataset = {
            'images': [],
            'annotations': [],
            'categories': [{'id': 1, 'name': 'sage_grouse'}]  # Assuming single class
        }

        for img_path, ann_path in files:
            # Add image
            dataset['images'].append({
                'id': self.image_id,
                'file_name': img_path.name,
                'width': 640,  # You might want to read actual dimensions
                'height': 640  # You might want to read actual dimensions
            })

            # Read and convert annotations
            with open(ann_path, 'r') as f:
                annotations = f.readlines()

            for ann in annotations:
                parts = ann.strip().split()
                class_id = int(parts[0])
                bbox = [float(x) for x in parts[1:5]]

                coco_ann = self.convert_yolo_to_coco_annotation(bbox, 640, 640)
                coco_ann.update({
                    'id': self.annotation_id,
                    'image_id': self.image_id,
                    'category_id': class_id + 1  # YOLO uses 0-based indexing
                })

                dataset['annotations'].append(coco_ann)
                self.annotation_id += 1

            self.image_id += 1

        return dataset

    def split_and_convert(self):
        """Main method to split data and convert to COCO format."""
        logging.info("Starting stratified split process")

        # Create output directories
        train_dir = self.output_dir / 'train'
        val_dir = self.output_dir / 'val'
        for d in [train_dir, val_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Group files by stratification criteria
        grouped_files = self.group_files()
        logging.info(f"Found {len(grouped_files)} unique strata for splitting")

        train_files = []
        val_files = []

        # Split each group maintaining ratio
        logging.info("\nStratification details:")
        for group, files in grouped_files.items():
            lek, height, path_type = group.split('_')
            total_files = len(files)
            split_idx = int(total_files * self.train_ratio)

            random.shuffle(files)
            group_train = files[:split_idx]
            group_val = files[split_idx:]

            train_files.extend(group_train)
            val_files.extend(group_val)

            logging.info(f"\nStratum: {group}")
            logging.info(f"  - Lek: {lek}")
            logging.info(f"  - Height: {height}")
            logging.info(f"  - Path Type: {path_type}")
            logging.info(f"  - Total images: {total_files}")
            logging.info(f"  - Train split: {len(group_train)} images ({len(group_train) / total_files * 100:.1f}%)")
            logging.info(f"  - Val split: {len(group_val)} images ({len(group_val) / total_files * 100:.1f}%)")

        # Create datasets
        train_dataset = self.create_coco_dataset(train_files, 'train')
        val_dataset = self.create_coco_dataset(val_files, 'val')

        # Copy files and save annotations
        for split, (files, dataset) in [
            ('train', (train_files, train_dataset)),
            ('val', (val_files, val_dataset))
        ]:
            split_dir = self.output_dir / split

            # Copy image files
            for img_path, _ in files:
                shutil.copy2(img_path, split_dir / img_path.name)

            # Save COCO annotations
            with open(split_dir / 'annotations.json', 'w') as f:
                json.dump(dataset, f, indent=2)

            logging.info(f"\nFinished creating {split} set:")
            logging.info(f"  - Total images: {len(files)}")
            logging.info(f"  - Total annotations: {len(dataset['annotations'])}")
            logging.info(f"  - Average annotations per image: {len(dataset['annotations']) / len(files):.1f}")


def main():
    parser = argparse.ArgumentParser(description='Convert YOLO annotations to COCO format')
    parser.add_argument('--input_dir', type=str, default=app_config.ANNOTATIONS_YOLO_PATH,
                        help='Directory containing YOLO format dataset (default: from config)')
    parser.add_argument('--output_dir', type=str, default=app_config.ANNOTATIONS_COCO_PATH,
                        help='Output directory for COCO format dataset (default: from config)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training set size (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    random.seed(args.seed)

    converter = YOLOToCOCOConverter(
        args.input_dir,
        args.output_dir,
        args.train_ratio
    )
    converter.split_and_convert()


if __name__ == '__main__':
    main()