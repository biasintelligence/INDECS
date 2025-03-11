import argparse
import os
import cv2 as cv
from src.indecs.detector import Detector
from src.indecs.config import Config

def process_images(image_dir, output_dir, threshold, nms):
    """Processes images in the given directory and saves results."""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    detector = Detector()

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)
            image = cv.imread(image_path)

            if image is None:
                print(f"Error: Could not read image {filename}")
                continue

            detections, coords, diams = detector.detect(image, threshold, nms)

            # You can add logic here to visualize detections or save results
            print(f"Processed {filename}: {len(detections)} detections")

            #Example of saving a copy of the image with bounding boxes.
            for coord in coords:
                x1, y1, x2, y2 = map(int, coord)
                cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.imwrite(os.path.join(output_dir, "detected_" + filename), image)

def main():
    parser = argparse.ArgumentParser(description="Process images with YOLOv4.")
    parser.add_argument("image_dir", help="Directory containing input images.")
    parser.add_argument("output_dir", help="Directory to save output images.")
    parser.add_argument("--threshold", type=float, default=0.25, help="Detection threshold.")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS threshold.")
    parser.add_argument("--download", action="store_true", help="Download model from huggingface")

    args = parser.parse_args()

    if args.download:
        Config.download_model()
    process_images(args.image_dir, args.output_dir, args.threshold, args.nms)

if __name__ == "__main__":
    main()