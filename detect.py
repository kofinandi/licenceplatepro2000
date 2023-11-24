import sys
import cv2
import numpy as np
import os
from tqdm.auto import tqdm

from DANI_license_plate_transformer import run_license_plate_transformer
from yolo import YOLO
from convolutional_ocr import ConvolutionalOCR


def load_image(image_path):
    try:
        # Open an image file
        img = cv2.imread(image_path)
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


convolutional_ocr = ConvolutionalOCR()


def process_image(img_file):
    candidates = yolo.get_license_plate_candidates(img_file)
    for img in candidates:
        opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        image_no_cimer, image_binary, concatenated = run_license_plate_transformer(opencvImage)
        det = convolutional_ocr.detect(image_no_cimer, draw_boxes=True)

        cv2.imshow(det, concatenated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # cv2.waitKey(0)


if __name__ == "__main__":
    # Check if the user provided an image path
    if len(sys.argv) != 2:
        print("Usage: python load_image.py <image_path>")
    else:
        # Get the image path from the command line argument
        image_path = sys.argv[1]

        yolo = YOLO(yolo_threshold=0.3, bb_scale_factor=1.5)

        if os.path.isdir(image_path):
            scale_factor = 1.2
            files = os.listdir(image_path)

            for fileName in files:
                process_image(os.path.join(image_path, fileName))
        elif os.path.isfile(image_path):
            process_image(image_path)
