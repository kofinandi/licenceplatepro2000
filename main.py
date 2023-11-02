import cv2
import pytesseract

# Load the image and detect license plates (you need to implement this step)
image_path = 'data/images/MYC-860.jpg'
license_plate = cv2.imread(image_path)

license_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)

# Apply OCR to recognize text on the license plate
predicted_result = pytesseract.image_to_string(license_plate, lang='eng',
                                               config='--oem 3 --psm 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "")

# Print the recognized text (this may require further post-processing)
print(f"Recognized text on the license plate: [" + filter_predicted_result + "]")