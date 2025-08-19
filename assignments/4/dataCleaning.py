import cv2
import numpy as np
import os

# Path to the folder containing your images
input_folder = "../../data/external/double_mnist/train/0"
output_folder = "../../data/external/double_mnist2/train/0"

# Process each image in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        # Load the image in grayscale
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Find contours
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If two contours (zeros) are detected, remove one
        if len(contours) >= 2:
            # Fill the first detected zero with black
            cv2.drawContours(img, [contours[0]], -1, (0, 0, 0), thickness=cv2.FILLED)

        # Save the processed image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img)

print("Processing complete.")
