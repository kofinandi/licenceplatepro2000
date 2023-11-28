import cv2
import numpy as np
import os
from side_detector import *
import pytesseract


# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust the path as per your Tesseract installation
class LicensePlateCropper:
    def __init__(self, side_detector, text_precentage=0.1, hue_range=(110, 160), saturation_threshold=100, value_threshold=150):
        self.side_detector = side_detector
        self.text_precentage = text_precentage
        self.saturation_threshold = saturation_threshold
        self.value_threshold = value_threshold
        self.hue_range = hue_range
        self.start_point = None
        self.end_point = None
        self.start_v = None
        self.end_v = None

    def __line_intersection(self, line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return (x, y)


    def __transform_image(self, image, points):
        # Convert the input points to a NumPy array
        points = np.array(points, dtype=np.float32)

        # Calculate the bounding box for the source region
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)

        # Calculate the aspect ratio of the source region
        aspect_ratio = (max_x - min_x) / (max_y - min_y)

        # Define the target rectangle based on the aspect ratio
        target_height = 80  # Adjust this value based on your preference
        target_width = int(target_height * aspect_ratio)

        target_rect = np.array([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]],
                            dtype=np.float32)

        # Calculate the perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(points, target_rect)

        # Apply the perspective transformation
        result = cv2.warpPerspective(image, matrix, (target_width, target_height))

        return result
    
    def __remove_colors_from_image(self, image):
        # Convert the image to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create masks for pixels above the threshold in H, S and V channels
        h_mask = np.logical_and(hsv[:,:,0] > self.hue_range[0], hsv[:,:,0] < self.hue_range[1])
        s_mask = hsv[:,:,1] > self.saturation_threshold
        v_mask = hsv[:,:,2] > self.value_threshold

        # Combine the masks
        combined_mask = np.logical_and(s_mask, v_mask)
        combined_mask = np.logical_and(combined_mask, h_mask)

        # Set the corresponding pixels to black in the original image
        image = image.copy()
        image[combined_mask] = [0, 0, 0]

        return image

    def detect_text_OCR(self, binary_image):
        # Specify the OCR engine mode
        custom_config = r'--oem 3 --psm 6'

        # Perform OCR on the preprocessed image
        text = pytesseract.image_to_string(binary_image, config=custom_config)
        return text

    def __is_blurry(self, image, threshold=100):
        # Compute the Laplacian of the image
        laplacian = cv2.Laplacian(image, cv2.CV_64F).var()

        # Compare the variance with the threshold
        return laplacian < threshold

    def __create_binary_image(self, image, sample_image):
        average_brightness = cv2.mean(sample_image)[0]

        # Get the pixel number in the image
        pixel_num_min = sample_image.shape[0] * sample_image.shape[1] * self.text_precentage

        i = 0.3
        while i < 1:
            i += 0.1
            # Set a threshold value
            threshold = average_brightness * i

            # Apply the threshold to create a binary mask
            binary_mask = (sample_image < int(threshold)).astype(np.uint8)

            # Flatten the image and mask to 1D arrays
            flat_image = sample_image.flatten()
            flat_masked_image = flat_image[binary_mask.flatten() == 1]

            if len(flat_masked_image) > int(pixel_num_min):
                break

        # Find the most frequently occurring pixel value below the threshold
        most_frequent_value = np.bincount(flat_masked_image).argmax()

        # print(f"Image is blurry: {is_blurry(image, 20)}")

        if self.__is_blurry(image, 20):
            _, binary_image = cv2.threshold(image, most_frequent_value + 25, 255, cv2.THRESH_BINARY)
        else:
            _, binary_image = cv2.threshold(image, most_frequent_value + 14, 255, cv2.THRESH_BINARY)

        return binary_image
    
    def line_length(self, p0, p1):
        return ((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)**0.5

    def run_license_plate_transformer(self, image):
        self.prev_image = image.copy()
        no_colors_image = self.__remove_colors_from_image(image)

        h, v = self.side_detector.get_bounding_lines(cv2.cvtColor(no_colors_image, cv2.COLOR_BGR2GRAY))   # 25, 3, 70, 40, 15

        if h is None or v is None:
            print('failed to find bounding lines')
            return None, None, None

        # Show the lines on the image
        image1 = no_colors_image.copy()
        cv2.line(image1, (h[0], h[1]), (h[2], h[3]), (0, 255, 0), 2)
        cv2.line(image1, (h[4], h[5]), (h[6], h[7]), (0, 255, 0), 2)
        cv2.line(image1, (v[0], v[1]), (v[2], v[3]), (0, 255, 0), 2)
        cv2.line(image1, (v[4], v[5]), (v[6], v[7]), (0, 255, 0), 2)

        # print(image.shape)

        # print(f"end_y: {self.end_y}")
        # print(f"start_x: {self.start_x}")
        # print(f"end_x: {self.end_x}")

        # Get the intersection points
        p1 = self.__line_intersection(((h[0], h[1]), (h[2], h[3])), ((v[0], v[1]), (v[2], v[3])))
        p2 = self.__line_intersection(((h[0], h[1]), (h[2], h[3])), ((v[4], v[5]), (v[6], v[7])))
        p3 = self.__line_intersection(((h[4], h[5]), (h[6], h[7])), ((v[4], v[5]), (v[6], v[7])))
        p4 = self.__line_intersection(((h[4], h[5]), (h[6], h[7])), ((v[0], v[1]), (v[2], v[3])))

        self.start_point = p4
        self.end_point = p3
        self.start_v = p1
        self.end_v = p2

        image2 = no_colors_image.copy()
        # Show the points on the image
        cv2.circle(image2, (int(p1[0]), int(p1[1])), 3, (0, 255, 0), -1)
        cv2.circle(image2, (int(p2[0]), int(p2[1])), 3, (0, 255, 0), -1)
        cv2.circle(image2, (int(p3[0]), int(p3[1])), 3, (0, 255, 0), -1)
        cv2.circle(image2, (int(p4[0]), int(p4[1])), 3, (0, 255, 0), -1)

        # Get the rectangle image
        cropped_image = self.__transform_image(image, [p1, p2, p3, p4])

        # Make the image4 grayscale
        image_binary = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        # Create a binary image
        image_binary = self.__create_binary_image(image_binary, cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY))

        # scale all 4 images to a fixed width
        width = 250
        image1 = cv2.resize(image1, (width, int(image1.shape[0] * width / image1.shape[1])))
        image2 = cv2.resize(image2, (width, int(image2.shape[0] * width / image2.shape[1])))
        cropped_image_v = cv2.resize(cropped_image, (width, int(cropped_image.shape[0] * width / cropped_image.shape[1])))
        image_binary = cv2.cvtColor(image_binary, cv2.COLOR_GRAY2RGB)
        image_binary_v = cv2.resize(image_binary, (width, int(image_binary.shape[0] * width / image_binary.shape[1])))

        # Insert a red line at the bottom of the first 4 images
        image1 = cv2.line(image1, (0, image1.shape[0] - 1), (image1.shape[1] - 1, image1.shape[0] - 1), (0, 0, 255), 2)
        image2 = cv2.line(image2, (0, image2.shape[0] - 1), (image2.shape[1] - 1, image2.shape[0] - 1), (0, 0, 255), 2)
        cropped_image_v = cv2.line(cropped_image_v, (0, cropped_image_v.shape[0] - 1), (cropped_image_v.shape[1] - 1, cropped_image_v.shape[0] - 1), (0, 0, 255), 2)

        # Show all three images in one line as the output of the cell
        concatenated = np.concatenate((image1, image2, cropped_image_v, image_binary_v), axis=0)

        aspect_ratio = self.line_length(p1, p2) / self.line_length(p2, p3)

        return cropped_image, image_binary, aspect_ratio, concatenated
    
    def run_cropped_license_plate_transformer(self, height_ratio=0.4):
        if self.start_point is None or self.end_point is None or self.start_v is None or self.end_v is None:
            return None, None, None, None

        end_y = int(max(self.start_point[1], self.end_point[1]))
        start_y = int(max(end_y - int(self.prev_image.shape[0] * height_ratio), 0))

        start_top_point = self.__line_intersection((self.start_point,self.start_v), ((0, start_y), (self.prev_image.shape[1], start_y)))
        end_top_point = self.__line_intersection((self.end_point,self.end_v), ((0, start_y), (self.prev_image.shape[1], start_y)))

        start_x = int(min(self.start_point[0], start_top_point[0])) - 10
        end_x = int(max(self.end_point[0], end_top_point[0])) + 10

        end_y = end_y + 10

        cropped_image = self.prev_image[start_y:end_y, start_x:end_x]

        return self.run_license_plate_transformer(cropped_image)


