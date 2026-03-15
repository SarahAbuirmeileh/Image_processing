import cv2
import numpy as np
from skimage.filters import threshold_triangle

# Load and convert image to grayscale
image = cv2.imread('./assets/c1.jpg')
if image is None:
    print("Error: Could not load image.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Compute triangle threshold
tri_thresh = threshold_triangle(gray)

# Apply binary inverse thresholding
_, binary_mask = cv2.threshold(gray, tri_thresh, 255, cv2.THRESH_BINARY_INV)

# Clean noise
kernel = np.ones((3, 3), np.uint8)
cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)

# Extract foreground using the mask
result = cv2.bitwise_and(image, image, mask=cleaned_mask)

# Display and save
cv2.imshow("Grayscale", gray)
cv2.imshow("Triangle Mask", cleaned_mask)
cv2.imshow("Extracted Foreground", result)
cv2.imwrite('foreground_triangle.png', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
