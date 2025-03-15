import cv2 as cv
from enum import Enum
import logging
import numpy as np


class ImageType(Enum):
    GRAYSCALE = cv.IMREAD_GRAYSCALE
    COLOR = cv.IMREAD_COLOR


class ThresholdingImage:
    """
    Class to read an image and apply thresholding manually.
    """

    def __init__(self, image_path: str, image_type: ImageType):
        """
        Initializes the ThresholdingImage instance.

        :param image_path: Path to the image file.
        :param image_type: Image type (GRAYSCALE or COLOR).
        """
        self.image_path = image_path
        self.image_type = image_type
        self.image = self._read_image()

    def _read_image(self):
        """
        Reads an image from the given path.

        :return: Loaded image (grayscale or color) or None if loading fails.
        """
        image = cv.imread(self.image_path, self.image_type.value)

        if image is None:
            logging.error(f"Failed to load image from {self.image_path}. Please check the file path.")

        else:
            # Convert to grayscale if the requested type is COLOR (so processing is always done on grayscale)
            if self.image_type == ImageType.COLOR:
                image = self.convert_to_gray_scale(image)

        return image

    @staticmethod
    def convert_to_gray_scale(image):
        """
        Converts a colored image to grayscale.

        :param image: Colored image.
        :return: Grayscale image.
        """
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image is not None else None

    def thresholding(self, lower_threshold: int, upper_threshold: int):
        """
        Applies manual thresholding to the image.

        :param lower_threshold: Lower threshold value (0-255).
        :param upper_threshold: Upper threshold value (0-255).

        :return: Thresholded image (NumPy array) or None if no image is loaded.
        """

        if self.image is None:
            logging.warning("Thresholding skipped: No image loaded.")
            return None

        height, width = self.image.shape
        # Create new image (empty array) with same dimensions as the original one
        thresholded_image = np.zeros((height, width), dtype=np.uint8)

        # Loop across all pixels in the original image and apply  thresholding
        for i in range(height):
            for j in range(width):
                current_pixel_value = self.image[i, j]
                if current_pixel_value > upper_threshold or current_pixel_value < lower_threshold:
                    thresholded_image[i, j] = 0
                else:
                    thresholded_image[i, j] = current_pixel_value

        return thresholded_image

    @staticmethod
    def display_image(image, image_name: str):
        """
        Displays an image on the screen.

        :param image: Image to display.
        :param image_name: Window title.
        """
        if image is None:
            logging.warning("Display skipped: No image to show.")
            return

        cv.imshow(image_name, image)
        cv.waitKey(0)
        cv.destroyAllWindows()
