import sys
import cv2


def load_image(image_path):
    try:
        # Open an image file
        img = cv2.imread(image_path)
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


if __name__ == "__main__":
    # Check if the user provided an image path
    if len(sys.argv) != 2:
        print("Usage: python load_image.py <image_path>")
    else:
        # Get the image path from the command line argument
        image_path = sys.argv[1]

        # Load the image
        image = load_image(image_path)

        cv2.imshow("Image", image)
        cv2.waitKey(0)
