import cv2
import numpy as np
import math

class SideDetector:
    def __init__(self, max_angle_dev=10, max_side_angle_dev=10, min_line_length_percentage=40, min_side_size_percentage_x=70, min_side_size_percentage_y=45, side_sample_num=3):
        self.max_angle_dev = max_angle_dev
        self.max_side_angle_dev = max_side_angle_dev
        self.min_line_length_percentage = min_line_length_percentage
        self.min_side_size_percentage_x = min_side_size_percentage_x
        self.min_side_size_percentage_y = min_side_size_percentage_y
        self.side_sample_num = side_sample_num

    def __distance_point_to_line(self, x_middle, y_middle, x0, y0, x1, y1):
        numerator = abs((y1 - y0) * x_middle - (x1 - x0) * y_middle + x1 * y0 - y1 * x0)
        denominator = math.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)
        distance = numerator / denominator
        return distance


    def __get_line_pixels(self, image, x0, y0, x1, y1):
        # Implement the Bresenham's line algorithm
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        is_horizontal = dx > dy

        if is_horizontal:
            err = dx / 2.0
            while x != x1:
                if 0 <= x < len(image[0]) and 0 <= y < len(image):
                    if 0 <= y - self.side_sample_num and y + self.side_sample_num < len(image):
                        points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                if 0 <= x < len(image[0]) and 0 <= y < len(image):
                    if 0 <= x - self.side_sample_num and x + self.side_sample_num < len(image[0]):
                        points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        if 0 <= x < len(image[0]) and 0 <= y < len(image):
            if (not is_horizontal) and 0 <= x - self.side_sample_num and x + self.side_sample_num < len(image[0]):
                points.append((x, y))
            if is_horizontal and 0 <= y - self.side_sample_num and y + self.side_sample_num < len(image):
                points.append((x, y))

        pixels = np.empty((len(points), 2), dtype=np.int16)
        
        cnt = 0
        if is_horizontal:
            for x, y in points:
                neg_sum = 0
                pos_sum = 0
                for i in range(self.side_sample_num + 1):
                    neg_sum += int(image[y - i][x])
                    pos_sum += int(image[y + i][x])
                pixels[cnt] = (neg_sum, pos_sum)
                cnt += 1
        else:
            for x, y in points:
                neg_sum = 0
                pos_sum = 0
                for i in range(self.side_sample_num + 1):
                    neg_sum += int(image[y][x - i])
                    pos_sum += int(image[y][x + i])
                pixels[cnt] = (neg_sum, pos_sum)
                cnt += 1

        return pixels

    def __get_best_side_pair(self, image, is_horizontal, min_side_distance, min_line_length):
        im_size_x = len(image[0])
        im_size_y = len(image)
        if is_horizontal:
            im_size = im_size_y
            min_angle_d = -self.max_angle_dev
            max_angle_d = self.max_angle_dev
            mid_angle_d = 0
        else:
            im_size = im_size_x
            min_angle_d = 90 - self.max_angle_dev
            max_angle_d = 90 + self.max_angle_dev
            mid_angle_d = 90

        all_min_values = {}
        all_max_values = {}

        for angle_d in range(min_angle_d, max_angle_d):
            angle = np.radians(angle_d)
            if is_horizontal:
                line_length = int(im_size_x / np.cos(angle))
            else:
                line_length = int(im_size_y / np.sin(angle))

            all_contrasts = []
            for i in range(im_size):
                if is_horizontal:
                    if mid_angle_d > angle_d:
                        x0, y0 = 0, i
                        x1, y1 = int(x0 + line_length * np.cos(angle)), int(y0 + line_length * np.sin(angle))
                    else:
                        x1, y1 = im_size_x, i
                        x0, y0 = int(x1 - line_length * np.cos(angle)), int(y1 - line_length * np.sin(angle))
                else:
                    if mid_angle_d > angle_d:
                        x1, y1 = i, im_size_y
                        x0, y0 = int(x1 - line_length * np.cos(angle)), int(y1 - line_length * np.sin(angle))
                    else:
                        x0, y0 = i, 0
                        x1, y1 = int(x0 + line_length * np.cos(angle)), int(y0 + line_length * np.sin(angle))

                if self.__distance_point_to_line(im_size_x / 2, im_size_y / 2, x0, y0, x1, y1) < min_side_distance / 2:
                    break

                pixels = self.__get_line_pixels(image, x0, y0, x1, y1)
                pixels = np.array(pixels, dtype=np.int16)
                if pixels.shape[0] < min_line_length:
                    continue
                total_contrast = pixels[:, 0] - pixels[:, 1]
                all_contrasts.append((i, total_contrast.mean()))

            if len(all_contrasts) == 0:
                continue
            all_contrasts = np.array(all_contrasts)
            min_index = all_contrasts[all_contrasts[:, 1].argmin()][0]
            min_value = all_contrasts[:, 1].min()
            all_min_values[angle_d] = (min_index, min_value)

            all_contrasts = []
            for i in range(im_size - 1, -1, -1):
                if is_horizontal:
                    if mid_angle_d > angle_d:
                        x1, y1 = im_size_x, i
                        x0, y0 = int(x1 - line_length * np.cos(angle)), int(y1 - line_length * np.sin(angle))
                    else:
                        x0, y0 = 0, i
                        x1, y1 = int(x0 + line_length * np.cos(angle)), int(y0 + line_length * np.sin(angle))
                else:
                    if mid_angle_d > angle_d:
                        x0, y0 = i, 0
                        x1, y1 = int(x0 + line_length * np.cos(angle)), int(y0 + line_length * np.sin(angle))
                    else:
                        x1, y1 = i, im_size_y
                        x0, y0 = int(x1 - line_length * np.cos(angle)), int(y1 - line_length * np.sin(angle))

                if self.__distance_point_to_line(im_size_x / 2, im_size_y / 2, x0, y0, x1, y1) < min_side_distance / 2:
                    break

                pixels = self.__get_line_pixels(image, x0, y0, x1, y1)
                pixels = np.array(pixels, dtype=np.int16)
                if pixels.shape[0] < min_line_length:
                    continue
                total_contrast = pixels[:, 0] - pixels[:, 1]
                all_contrasts.append((i, total_contrast.mean()))

            if len(all_contrasts) == 0:
                continue
            all_contrasts = np.array(all_contrasts)
            max_index = all_contrasts[all_contrasts[:, 1].argmax()][0]
            max_value = all_contrasts[:, 1].max()
            all_max_values[angle_d] = (max_index, max_value)

        all_values = []
        for angle_d_min in range(min_angle_d, max_angle_d):
            for angle_d_max in range(angle_d_min - self.max_side_angle_dev, angle_d_min + self.max_side_angle_dev):
                if angle_d_min not in all_min_values or angle_d_max not in all_max_values:
                    continue
                min_index, min_value = all_min_values[angle_d_min]
                max_index, max_value = all_max_values[angle_d_max]
                angle_min = np.radians(angle_d_min)
                angle_max = np.radians(angle_d_max)

                if is_horizontal:
                    line_length_min = int(im_size_x / np.cos(angle_min))
                    line_length_max = int(im_size_x / np.cos(angle_max))
                else:
                    line_length_min = int(im_size_y / np.sin(angle_min))
                    line_length_max = int(im_size_y / np.sin(angle_max))

                if is_horizontal:
                    if mid_angle_d > angle_d_min:
                        min_x0, min_y0 = 0, min_index
                        min_x1, min_y1 = int(min_x0 + line_length_min * np.cos(angle_min)), int(
                            min_y0 + line_length_min * np.sin(angle_min))
                    else:
                        min_x1, min_y1 = im_size_x, min_index
                        min_x0, min_y0 = int(min_x1 - line_length_min * np.cos(angle_min)), int(
                            min_y1 - line_length_min * np.sin(angle_min))

                    if mid_angle_d > angle_d_max:
                        max_x1, max_y1 = im_size_x, max_index
                        max_x0, max_y0 = int(max_x1 - line_length_max * np.cos(angle_max)), int(
                            max_y1 - line_length_max * np.sin(angle_max))
                    else:
                        max_x0, max_y0 = 0, max_index
                        max_x1, max_y1 = int(max_x0 + line_length_max * np.cos(angle_max)), int(
                            max_y0 + line_length_max * np.sin(angle_max))
                else:
                    if mid_angle_d > angle_d_min:
                        min_x1, min_y1 = min_index, im_size_y
                        min_x0, min_y0 = int(min_x1 - line_length_min * np.cos(angle_min)), int(
                            min_y1 - line_length_min * np.sin(angle_min))
                    else:
                        min_x0, min_y0 = min_index, 0
                        min_x1, min_y1 = int(min_x0 + line_length_min * np.cos(angle_min)), int(
                            min_y0 + line_length_min * np.sin(angle_min))

                    if mid_angle_d > angle_d_max:
                        max_x0, max_y0 = max_index, 0
                        max_x1, max_y1 = int(max_x0 + line_length_max * np.cos(angle_max)), int(
                            max_y0 + line_length_max * np.sin(angle_max))
                    else:
                        max_x1, max_y1 = max_index, im_size_y
                        max_x0, max_y0 = int(max_x1 - line_length_max * np.cos(angle_max)), int(
                            max_y1 - line_length_max * np.sin(angle_max))

                all_values.append((min_x0, min_y0, min_x1, min_y1, max_x0, max_y0, max_x1, max_y1, max_value - min_value,
                                min_index, angle_d_min, angle_d_max))

        if len(all_values) == 0:
            return None
        all_values = np.array(all_values)
        best_index = all_values[:, 8].argmax()
        #print(f"best value angle_d_min: {all_values[best_index, 10]}")
        #print(f"best value angle_d_max: {all_values[best_index, 11]}")
        return int(all_values[best_index, 0]), int(all_values[best_index, 1]), int(all_values[best_index, 2]), int(
            all_values[best_index, 3]), int(all_values[best_index, 4]), int(all_values[best_index, 5]), int(
            all_values[best_index, 6]), int(all_values[best_index, 7])


    def get_bounding_lines(self, image):
        im_size_x = len(image[0])
        im_size_y = len(image)
        min_line_length_x = int(self.min_line_length_percentage * im_size_x / 100)
        min_line_length_y = int(self.min_line_length_percentage * im_size_y / 100)

        min_side_size_x = int(self.min_side_size_percentage_x * im_size_x / 100)
        min_side_size_y = int(self.min_side_size_percentage_y * im_size_y / 100)

        # Get the best horizontal pair
        best_horizontal_pair = self.__get_best_side_pair(image, True, min_side_size_y, min_line_length_x)
        # Get the best vertical pair
        best_vertical_pair = self.__get_best_side_pair(image, False, min_side_size_x, min_line_length_y)

        return best_horizontal_pair, best_vertical_pair
