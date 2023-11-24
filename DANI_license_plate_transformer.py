# %%
import cv2
import numpy as np
import os
from DANI_side_detector import *
import pytesseract


# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust the path as per your Tesseract installation

# %%
def line_intersection(line1, line2):
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


# %%
def transform_image(image, points):
    # Convert the input points to a NumPy array
    points = np.array(points, dtype=np.float32)

    # Calculate the bounding box for the source region
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)

    # Calculate the aspect ratio of the source region
    aspect_ratio = (max_x - min_x) / (max_y - min_y)

    # Define the target rectangle based on the aspect ratio
    target_height = 300  # Adjust this value based on your preference
    target_width = int(target_height * aspect_ratio)

    target_rect = np.array([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]],
                           dtype=np.float32)

    # Calculate the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(points, target_rect)

    # Apply the perspective transformation
    result = cv2.warpPerspective(image, matrix, (target_width, target_height))

    return result


# %%
def detect_text_OCR(binary_image):
    # Specify the OCR engine mode
    custom_config = r'--oem 3 --psm 6'

    # Perform OCR on the preprocessed image
    text = pytesseract.image_to_string(binary_image, config=custom_config)
    return text


# %%
def is_blurry(image, threshold=100):
    # Compute the Laplacian of the image
    laplacian = cv2.Laplacian(image, cv2.CV_64F).var()

    # Compare the variance with the threshold
    return laplacian < threshold


# %%
def create_binary_image(image, sample_image):
    average_brightness = cv2.mean(sample_image)[0]

    # Get the pixel number in the image
    pixel_num_min = sample_image.shape[0] * sample_image.shape[1] * 0.1

    i = 0.3
    while i < 1:
        i += 0.03
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

    if is_blurry(image, 20):
        _, binary_image = cv2.threshold(image, most_frequent_value + 25, 255, cv2.THRESH_BINARY)
    else:
        _, binary_image = cv2.threshold(image, most_frequent_value + 14, 255, cv2.THRESH_BINARY)

    return binary_image


# %%
def make_colors_in_image_white(image):
    # Convert the colored image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Get the average saturation
    average_saturation = cv2.mean(hsv_image)[1]

    saturation_threshold = average_saturation * 1.2  # adjust as needed

    v_channel = hsv_image[:, :, 2]
    # Define a threshold
    threshold = cv2.mean(hsv_image)[2] * 0.9  # adjust as needed

    # Create a mask for pixels below the threshold
    mask = v_channel < threshold

    # Get the values of V channel for pixels below the threshold
    values_below_threshold = v_channel[mask]

    # Get the most common value (mode) below the threshold
    most_common_value = np.argmax(np.bincount(values_below_threshold))

    # Create a mask based on the thresholds
    mask = (hsv_image[:, :, 1] > saturation_threshold) & (hsv_image[:, :, 2] > most_common_value * 1.2)

    # Set pixels to white where the mask is True
    hsv_image[mask] = [0, 0, 255]  # [Hue, Saturation, Value] for white color in OpenCV

    # Convert the modified HSV image back to BGR
    result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return result_image


# %%
# Get the bounding lines and show them on the image. Do this for all images in the folder
def run_license_plate_transformer(image):
    h, v = get_bounding_lines(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 25, 3, 70, 40, 15)

    if h is None or v is None:
        print('failed to find bounding lines')
        return None, None, None

    # Show the lines on the image
    image1 = image.copy()
    cv2.line(image1, (h[0], h[1]), (h[2], h[3]), (0, 255, 0), 2)
    cv2.line(image1, (h[4], h[5]), (h[6], h[7]), (0, 255, 0), 2)
    cv2.line(image1, (v[0], v[1]), (v[2], v[3]), (0, 255, 0), 2)
    cv2.line(image1, (v[4], v[5]), (v[6], v[7]), (0, 255, 0), 2)

    # Get the intersection points
    p1 = line_intersection(((h[0], h[1]), (h[2], h[3])), ((v[0], v[1]), (v[2], v[3])))
    p2 = line_intersection(((h[0], h[1]), (h[2], h[3])), ((v[4], v[5]), (v[6], v[7])))
    p3 = line_intersection(((h[4], h[5]), (h[6], h[7])), ((v[4], v[5]), (v[6], v[7])))
    p4 = line_intersection(((h[4], h[5]), (h[6], h[7])), ((v[0], v[1]), (v[2], v[3])))

    image2 = image.copy()
    # Show the points on the image
    cv2.circle(image2, (int(p1[0]), int(p1[1])), 3, (0, 255, 0), -1)
    cv2.circle(image2, (int(p2[0]), int(p2[1])), 3, (0, 255, 0), -1)
    cv2.circle(image2, (int(p3[0]), int(p3[1])), 3, (0, 255, 0), -1)
    cv2.circle(image2, (int(p4[0]), int(p4[1])), 3, (0, 255, 0), -1)

    # Get the rectangle image
    image3 = transform_image(image, [p1, p2, p3, p4])

    image_no_cimer = image3.copy()

    # Make the colors in the image white
    # image_no_cimer = make_colors_in_image_white(image_no_cimer)

    # Make the image4 grayscale
    image_binary = cv2.cvtColor(image_no_cimer, cv2.COLOR_BGR2GRAY)

    # Create a binary image
    image_binary = create_binary_image(image_binary, cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY))

    # scale all 4 images to a fixed width
    width = 250
    image1 = cv2.resize(image1, (width, int(image1.shape[0] * width / image1.shape[1])))
    image2 = cv2.resize(image2, (width, int(image2.shape[0] * width / image2.shape[1])))
    image3 = cv2.resize(image3, (width, int(image3.shape[0] * width / image3.shape[1])))
    image_no_cimer = cv2.resize(image_no_cimer, (width, int(image_no_cimer.shape[0] * width / image_no_cimer.shape[1])))
    image_binary = cv2.resize(image_binary, (width, int(image_binary.shape[0] * width / image_binary.shape[1])))

    # Insert a red line at the bottom of the first 4 images
    image1 = cv2.line(image1, (0, image1.shape[0] - 1), (image1.shape[1] - 1, image1.shape[0] - 1), (0, 0, 255), 2)
    image2 = cv2.line(image2, (0, image2.shape[0] - 1), (image2.shape[1] - 1, image2.shape[0] - 1), (0, 0, 255), 2)
    image3 = cv2.line(image3, (0, image3.shape[0] - 1), (image3.shape[1] - 1, image3.shape[0] - 1), (0, 0, 255), 2)
    image_no_cimer = cv2.line(image_no_cimer, (0, image_no_cimer.shape[0] - 1),
                              (image_no_cimer.shape[1] - 1, image_no_cimer.shape[0] - 1), (0, 0, 255), 2)

    # Perform OCR on the rectangle image
    print(detect_text_OCR(cv2.resize(image_binary, (205, 40))))

    image_binary = cv2.cvtColor(image_binary, cv2.COLOR_GRAY2RGB)

    # Show all three images in one line as the output of the cell
    concatenated = np.concatenate((image1, image2, image3, image_binary), axis=0)

    return image_no_cimer, image_binary, concatenated
