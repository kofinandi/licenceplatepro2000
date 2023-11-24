import cv2
import numpy as np
import os

class ConvolutionalOCR:
    def __init__(self):
        # create a list of all the template images
        self.template_imgs = {}
        for img_file in os.listdir('font_img'):
            self.template_imgs[img_file[0]] = cv2.imread('font_img/' + img_file, 0)

    def _apply_nms(self, boxes, threshold=0.5):
        # Sort boxes by their confidence score in descending order
        boxes.sort(key=lambda x: x[0], reverse=True)

        selected_boxes = []

        while len(boxes) > 0:
            # Select the box with the highest confidence (the first box after sorting)
            selected_box = boxes[0]
            selected_boxes.append(selected_box)

            # Calculate the area of the selected box
            selected_x1 = selected_box[1]
            selected_y1 = selected_box[2]
            selected_x2 = selected_x1 + selected_box[4]
            selected_y2 = selected_y1 + selected_box[3]
            selected_area = selected_box[3] * selected_box[4]

            # Remove the selected box from the list
            boxes = boxes[1:]

            # Iterate over the remaining boxes and remove those with high overlap
            remaining_boxes = []
            for box in boxes:
                x1 = max(selected_x1, box[1])
                y1 = max(selected_y1, box[2])
                x2 = min(selected_x2, box[1] + box[4])
                y2 = min(selected_y2, box[2] + box[3])

                intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
                box_area = box[3] * box[4]
                overlap = intersection_area / float(selected_area + box_area - intersection_area)

                # If overlap is less than the threshold, keep the box
                if overlap <= threshold:
                    remaining_boxes.append(box)

            boxes = remaining_boxes

        return selected_boxes
    
    def _draw_boxes(self, image, boxes):
        image_with_boxes = image.copy()
        for box in boxes:
            confidence, x, y, height, width, _ = box
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + width), int(y + height)
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 0, 0), 2)
        return image_with_boxes
    
    def detect(self, plate_img, draw_boxes=False):
        # Convert images to grayscale
        plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # resize the image
        height, width = plate_gray.shape
        plate_gray = cv2.resize(plate_gray, (int(height / 40 * 205), height))

        r = 0.2

        # pad the plate image with extra white pixels to avoid errors
        plate_gray = cv2.copyMakeBorder(plate_gray, int(height * r), int(height * r), int(width * r), int(width * r), cv2.BORDER_CONSTANT, value=[255, 255, 255])

        detections = []
        for key, template in self.template_imgs.items():
            character_ratio = 0.7

            template_height, template_width = template.shape
            template = cv2.resize(template, (int(height * character_ratio * template_width / template_height), int(height * character_ratio)))

            match = cv2.matchTemplate(plate_gray, template, cv2.TM_CCOEFF_NORMED)

            # print(key + ': ' + str(np.max(match)))

            # Set a threshold to identify matches
            threshold = 0.7
            locations = np.where(match >= threshold)
            # locations is a tuple of arrays, so we need to convert it to a n x 3 array format adding the third dimension as the value of the match
            locations = list(zip(*locations[::-1]))
            locations = [list([match[elem[1], elem[0]]]) + list(elem) + list(template.shape) + list(key) for elem in locations]
            detections.extend(locations)

        # Apply NMS
        picked_boxes = self._apply_nms(detections, threshold=0.1)

        # Draw bounding boxes around detected positions after NMS
        if draw_boxes:
            plate_with_boxes = self._draw_boxes(plate_gray, picked_boxes)
            cv2.imshow('Detected Characters', plate_with_boxes)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

        picked_boxes.sort(key=lambda x: x[1])
        # print the detected text
        text = ''
        for box in picked_boxes:
            text += box[5]
        # capitalize the text
        return text.upper()
