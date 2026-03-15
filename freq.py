import cv2
import numpy as np
import os


def load_image(path):
    """Load an image from the given path."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {path}")
    return image


def detect_lines(gray_image, low_threshold=55, high_threshold=200,
                 hough_threshold=50, min_line_length=10, max_line_gap=10):
    """Detect lines using Canny edge detection followed by Hough Line Transform."""
    edges = cv2.Canny(gray_image, low_threshold, high_threshold, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=hough_threshold,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines


def create_line_mask(image_shape, lines, thickness=2):
    """Create a binary mask with detected lines drawn in white."""
    mask = np.zeros(image_shape, dtype=np.uint8)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
        cv2.imshow("Lines", mask)
    return mask


def remove_lines_from_image(original_image, mask, inpaint_radius=3):
    """Inpaint detected lines using the mask to remove them from the original image."""
    return cv2.inpaint(original_image, mask, inpaint_radius, cv2.INPAINT_TELEA)


def main(input_path, output_path):
    """Main function to process the image and remove lines."""
    image = load_image(input_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lines = detect_lines(gray)
    mask = create_line_mask(gray.shape, lines)

    inpainted_image = remove_lines_from_image(image, mask)
    cv2.imwrite(output_path, inpainted_image)
    print(f"Saved cleaned image to: {output_path}")


if __name__ == "__main__":
    INPUT_IMAGE_PATH = "./image.png"
    OUTPUT_IMAGE_PATH = "./image_without_lines.png"
    main(INPUT_IMAGE_PATH, OUTPUT_IMAGE_PATH)
    cv2.waitKey(0)
