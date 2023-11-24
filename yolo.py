import torch
from PIL import Image
from transformers import YolosForObjectDetection, AutoFeatureExtractor


class YOLO:
    def __init__(self, yolo_threshold=0.5, bb_scale_factor=1.2):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained('nickmuchi/yolos-small-rego-plates-detection')
        self.model = YolosForObjectDetection.from_pretrained('nickmuchi/yolos-small-rego-plates-detection')

        self.yolo_threshold = yolo_threshold
        self.bb_scale_factor = bb_scale_factor

    def get_license_plate_candidates(self, file_name):
        image = Image.open(file_name)
        processed_outputs = self._predict_bboxes(image)
        boxes = self._get_license_plate_bounding(processed_outputs, self.yolo_threshold)

        predictions = []

        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = self._scale_bbox(box, self.bb_scale_factor)
            cropped_img = image.crop((xmin, ymin, xmax, ymax))
            predictions.append(cropped_img)

        return predictions

    @staticmethod
    def _get_license_plate_bounding(output_dict, threshold=0.5):
        keep = output_dict["scores"] > threshold
        boxes = output_dict["boxes"][keep].tolist()
        scores = output_dict["scores"][keep].tolist()
        labels = output_dict["labels"][keep].tolist()
        output = []
        for score, (xmin, ymin, xmax, ymax), label in zip(scores, boxes, labels):
            if label == 1:
                output.append([xmin, ymin, xmax, ymax])

        return output

    @staticmethod
    def _scale_bbox(bbox, scale_factor):

        # Get the center of the bbox
        xmin, ymin, xmax, ymax = bbox
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2

        # Calculate the scaled width and height
        scaled_width = (xmax - xmin) * scale_factor
        scaled_height = (ymax - ymin) * scale_factor

        # Calculate the new coordinates for the scaled bounding box
        scaled_xmin = int(center_x - scaled_width / 2)
        scaled_ymin = int(center_y - scaled_height / 2)
        scaled_xmax = int(center_x + scaled_width / 2)
        scaled_ymax = int(center_y + scaled_height / 2)

        return scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax

    def _predict_bboxes(self, img):
        inputs = self.feature_extractor(img, return_tensors="pt")
        outputs = self.model(**inputs)
        img_size = torch.tensor([tuple(reversed(img.size))])
        processed_outputs = self.feature_extractor.post_process(outputs, img_size)
        return processed_outputs[0]
