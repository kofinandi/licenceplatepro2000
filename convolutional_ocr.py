import cv2
import numpy as np
import os

class ConvolutionalOCR:
    def __init__(self, conv_threshold=0.65, nms_threshold=0.1, padding=0.2, plate_size=(40, 205), character_size=(0.6, 0.9), character_size_step=0.05, angle_range=6, angle_step=3):
        # create a list of all the template images
        self.template_imgs = {}
        for img_file in os.listdir('font_img_regi'):
            self.template_imgs[img_file[0]] = cv2.imread('font_img_regi/' + img_file, 0)

        self.conv_threshold = conv_threshold
        self.nms_threshold = nms_threshold
        self.padding = padding
        self.plate_size = plate_size

        self.character_size = character_size
        self.character_size_step = character_size_step
        self.angle_range = angle_range
        self.angle_step = angle_step

    def _apply_nms(self, boxes):
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
                if overlap <= self.nms_threshold:
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
    
    def _rotate_image(self, image, angle):
        image = cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        height, width = image.shape[:2]
        image_center = (width / 2, height / 2)
        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
        rotated_image = cv2.warpAffine(image, rotation_mat, (width, height))
        rotated_image = rotated_image[50:-50, 50:-50]
        return rotated_image
    
    def detect(self, plate_img, draw_boxes=False):
        # Convert images to grayscale
        plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # resize the image
        height, width = plate_gray.shape
        plate_gray = cv2.resize(plate_gray, (int(height / self.plate_size[0] * self.plate_size[1]), height))

        p = int(height * self.padding)

        # pad the plate image with the border pixels
        plate_gray = cv2.copyMakeBorder(plate_gray, p, p, p, p, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        detections = []
        for key, template in self.template_imgs.items():
            for character_ratio in np.linspace(self.character_size[0], self.character_size[1], num=int((self.character_size[1]-self.character_size[0])/self.character_size_step)+1):
                for angle in range(-self.angle_range, self.angle_range+1, self.angle_step):
                    template = self._rotate_image(template, angle)
                    template_height, template_width = template.shape
                    template = cv2.resize(template, (int(height * character_ratio * template_width / template_height), int(height * character_ratio)))
                    match = cv2.matchTemplate(plate_gray, template, cv2.TM_CCOEFF_NORMED)

                    locations = np.where(match >= self.conv_threshold)
                    # locations is a tuple of arrays, so we need to convert it to a n x 3 array format adding the third dimension as the value of the match
                    locations = list(zip(*locations[::-1]))
                    locations = [list([match[elem[1], elem[0]]]) + list(elem) + list(template.shape) + list(key) for elem in locations]
                    detections.extend(locations)

        # Apply NMS
        picked_boxes = self._apply_nms(detections)

        picked_boxes.sort(key=lambda x: x[1])
        # print the detected text
        text = ''
        for box in picked_boxes:
            text += box[5]

        text = text.upper()

        # Draw bounding boxes around detected positions after NMS
        if draw_boxes:
            plate_with_boxes = self._draw_boxes(plate_gray, picked_boxes)
            cv2.imshow(text, plate_with_boxes)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        # capitalize the text
        return text
    
    def parse(self, text):
        if len(text) == 7:
            text = text[:4] + text[4:].replace('O', '0')
            text = text[:4] + text[4:].replace('D', '0')
            text = text[:4] + text[4:].replace('I', '1')
            text = text[:4] + text[4:].replace('Q', '0')
            text = text[:4] + text[4:].replace('Z', '2')
            text = text[:4] + text[4:].replace('S', '5')
            text = text[:4] + text[4:].replace('B', '8')
            text = text[:4] + text[4:].replace('G', '6')
            text = text[:4] + text[4:].replace('T', '7')
            text = text[:4] + text[4:].replace('A', '4')

            text = text[:4].replace('0', 'O') + text[4:]
            text = text[:4].replace('1', 'I') + text[4:]
            text = text[:4].replace('2', 'Z') + text[4:]
            text = text[:4].replace('5', 'S') + text[4:]
            text = text[:4].replace('8', 'B') + text[4:]
            text = text[:4].replace('6', 'G') + text[4:]
            text = text[:4].replace('7', 'T') + text[4:]
            text = text[:4].replace('4', 'A') + text[4:]

            # format the text from AAAA111 to AA AA-111
            text = text[:2] + ' ' + text[2:4] + '-' + text[4:]

        if len(text) == 6:
            text = text[:3] + text[3:].replace('O', '0')
            text = text[:3] + text[3:].replace('D', '0')
            text = text[:3] + text[3:].replace('I', '1')
            text = text[:3] + text[3:].replace('Q', '0')
            text = text[:3] + text[3:].replace('Z', '2')
            text = text[:3] + text[3:].replace('S', '5')
            text = text[:3] + text[3:].replace('B', '8')
            text = text[:3] + text[3:].replace('G', '6')
            text = text[:3] + text[3:].replace('T', '7')
            text = text[:3] + text[3:].replace('A', '4')

            text = text[:3].replace('0', 'O') + text[3:]
            text = text[:3].replace('1', 'I') + text[3:]
            text = text[:3].replace('2', 'Z') + text[3:]
            text = text[:3].replace('5', 'S') + text[3:]
            text = text[:3].replace('8', 'B') + text[3:]
            text = text[:3].replace('6', 'G') + text[3:]
            text = text[:3].replace('7', 'T') + text[3:]
            text = text[:3].replace('4', 'A') + text[3:]

            # format the text from AAA111 to AAA-111
            text = text[:3] + '-' + text[3:]

        return text