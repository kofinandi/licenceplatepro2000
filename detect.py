import sys
import time
import time

import argparse
from contextlib import closing
from multiprocessing import Pool

import cv2
import numpy as np
import os
from tqdm.auto import tqdm

from license_plate_cropper import LicensePlateCropper
from side_detector import SideDetector
from yolo import YOLO
from convolutional_ocr import ConvolutionalOCR

# Check if the user provided an image path
parser = argparse.ArgumentParser(
    prog='LicensePlate Pro 2000',
    description='Detecting Hungarian licenseplate images')

parser.add_argument('filename')  # positional argument
parser.add_argument('-s', '--show_detection', action=argparse.BooleanOptionalAction)
parser.add_argument('--export_images', action=argparse.BooleanOptionalAction)
parser.add_argument('--save_file', default='./data/detections.npy')
parser.add_argument('--img_filter_res', default='./data/img_filter_res')
parser.add_argument('--plate_conv_res', default='./data/plate_conv_res')
parser.add_argument('--process_count', default='8')
args = parser.parse_args()

print(args)

os.makedirs(args.img_filter_res, exist_ok=True)
os.makedirs(args.plate_conv_res, exist_ok=True)

convolutional_ocr = ConvolutionalOCR(conv_threshold=0.6, padding=0.15, nms_threshold=0.1, character_size=(0.8, 1.0), character_size_step=0.05, angle_range=0, angle_step=1)
side_detector = SideDetector(max_angle_dev=25, max_side_angle_dev=3, min_line_length_percentage=80, min_side_size_percentage_x=25, min_side_size_percentage_y=5, side_sample_num=2)
license_plate_cropper = LicensePlateCropper(side_detector, text_precentage=0.12, hue_range=(94, 140), saturation_threshold=80, value_threshold=60)  # hue_value might be added for blue filtering

save_interval = 50
processes = int(args.process_count)
scale_factor = 1.2 # On the big image, we detect the licenseplate's location, This is that bounding_boxes scale factor

# Get the image path from the command line argument
image_path = args.filename
show_detection = args.show_detection
yolo = YOLO(yolo_threshold=0.3, bb_scale_factor=1.5)


def process_image(img_file, show_detection=False):
    candidates = yolo.get_license_plate_candidates(img_file)
    text_predictions = []
    for img in candidates:
        opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        start_time = time.time()
        cropped_image, image_binary, aspect_ratio, concatenated = license_plate_cropper.run_license_plate_transformer(opencvImage)
        transform_time = time.time() - start_time
        start_time = time.time()
        det, picked_boxes = convolutional_ocr.detect(cropped_image, draw_boxes=show_detection)
        text = convolutional_ocr.parse(det)
        ocr_time = time.time() - start_time
        if args.export_images:
            cv2.imwrite(os.path.join(args.img_filter_res, os.path.basename(img_file)), concatenated)
            cv2.imwrite(os.path.join(args.plate_conv_res, os.path.basename(img_file)), picked_boxes)
        if show_detection:
            print(f"Transform time: {transform_time}")
            print(f"OCR time: {ocr_time}")
            cv2.imshow(text, concatenated)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == ord('q'):
                assert False
        text_predictions.append(text)
    return text_predictions


def process(img_path):
    predicted = process_image(img_path, show_detection=show_detection)
    desired = os.path.basename(img_path)
    return predicted, desired


if __name__ == "__main__":

    detections = np.load(args.save_file, allow_pickle=True).item() if os.path.exists(args.save_file) else {}
    print(f'We already have: {str(len(detections.keys()))} files')

    print('Processing ' + image_path)
    if os.path.isdir(image_path):
        files = os.listdir(image_path)
        files = [x for x in files if x not in detections.keys()]
        files = [os.path.join(image_path ,file) for file in files]
        print('Files count ' + str(len(files)))

        np.random.shuffle(files)

        i = 0
        with closing(Pool(processes)) as pool:
            with tqdm(total=len(files)) as pbar:
                # Map tasks to worker_function and store results
                for result in pool.imap_unordered(process, files):
                    predicted, desired = result
                    detections[desired] = predicted

                    if i % save_interval == 0:
                        np.save(args.save_file, detections)
                    pbar.update(1)
                    i = i + 1

    elif os.path.isfile(image_path):
        process(image_path)
    else:
        print('Can\'t open! It\'s not a file nor directory.')
