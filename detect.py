import sys
import time
import cv2
import numpy as np
import os
from tqdm.auto import tqdm

from license_plate_cropper import LicensePlateCropper
from side_detector import SideDetector
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


convolutional_ocr = ConvolutionalOCR(conv_threshold=0.6, padding=0.15, nms_threshold=0.1, character_size=(0.8, 1.0), character_size_step=0.05, angle_range=0, angle_step=1)
side_detector = SideDetector(max_angle_dev=25, max_side_angle_dev=3, min_line_length_percentage=80, min_side_size_percentage_x=25, min_side_size_percentage_y=5, side_sample_num=2)
license_plate_cropper = LicensePlateCropper(side_detector, text_precentage=0.12, hue_range=(94, 140), saturation_threshold=80, value_threshold=60)  # hue_value might be added for blue filtering


def process_image(img_file):
    candidates = yolo.get_license_plate_candidates(img_file)
    for img in candidates:
        opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        start_time = time.time()
        image_cropped, image_binary, aspect_ratio, concatenated = license_plate_cropper.run_license_plate_transformer(opencvImage)
        transform_time = time.time() - start_time
        start_time = time.time()
        det = convolutional_ocr.detect(image_cropped, draw_boxes=True)
        text = convolutional_ocr.parse(det)
        ocr_time = time.time() - start_time
        print(f"Transform time: {transform_time}")
        print(f"OCR time: {ocr_time}")
        cv2.imshow(text, concatenated)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key == ord('q'):
            assert False

        start_time = time.time()
        image_cropped, image_binary, aspect_ratio, concatenated = license_plate_cropper.run_cropped_license_plate_transformer(0.7)
        transform_time = time.time() - start_time
        start_time = time.time()
        det = convolutional_ocr.detect(image_cropped, draw_boxes=True)
        text = convolutional_ocr.parse(det)
        ocr_time = time.time() - start_time
        print(f"Transform time: {transform_time}")
        print(f"OCR time: {ocr_time}")
        cv2.imshow(text, concatenated)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key == ord('q'):
            assert False

        


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

            #shuffle the files
            np.random.shuffle(files)
            
            for fileName in files:
                process_image(os.path.join(image_path, fileName))
        elif os.path.isfile(image_path):
            process_image(image_path)
