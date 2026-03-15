import cv2 as cv
import numpy as np
import logging
from matplotlib import pyplot as plt


class CoinSegmentation:
    def __init__(self, image_path, upper_threshold, lower_threshold):
        self.image_path = image_path

        self.image = None
        self.grayscale_image = None
        self.thresholded_image = None

        self.upper_threshold = upper_threshold 
        self.lower_threshold = lower_threshold


    def _read_image(self):
        """
        Read image
        """
        image = cv.imread(self.image_path, cv.IMREAD_COLOR)
        if image is None:
            logging.error(f"Failed to load image from {self.image_path}.")
        return image


    def convert_to_grayscale(self, image):
        """
        Convert RGB image into HSV and split the HSV into its' three channels
        """
        grayscale_image = None
        if image is not None:
            grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return grayscale_image
        

    def create_threshold_mask(self, image, lower_bound, upper_bound):
        """
        Creates a binary mask where values in the given image are 
        between the specified lower and upper bounds (inclusive).

        Returns:
            np.ndarray: Binary mask (0 or 1).
        """
        mask = np.zeros_like(image, dtype=np.uint8)
        mask[(image >= lower_bound) & (image <= upper_bound)] = 255
        return mask


    def dilation(self, image, structuring_element):
        kernel_height, kernel_width = structuring_element.shape
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2

        padded_image = np.pad(
            image,
            ((pad_height, pad_height), (pad_width, pad_width)),
            mode='constant',
            constant_values=255
        )

        dilated_image = np.zeros_like(image)
        image_height, image_width = image.shape

        for i in range(image_height):
            for j in range(image_width):
                region = padded_image[i:i + kernel_height, j:j + kernel_width]
                masked = region[structuring_element == 1]
                dilated_image[i, j] = np.max(masked)

        return dilated_image


    def erosion(self, image, structuring_element):
        kernel_height, kernel_width = structuring_element.shape
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2

        padded_image = np.pad(
            image,
            ((pad_height, pad_height), (pad_width, pad_width)),
            mode='constant',
            constant_values=255
        )

        eroded_image = np.zeros_like(image)
        image_height, image_width = image.shape

        for i in range(image_height):
            for j in range(image_width):
                region = padded_image[i:i + kernel_height, j:j + kernel_width]
                masked = region[structuring_element == 1]
                eroded_image[i, j] = np.min(masked)

        return eroded_image

    
    def open(self, image, structuring_element):
        eroded_image = self.erosion(image, structuring_element)
        dilated_image = self.dilation(eroded_image, structuring_element)
        return dilated_image
    
    
    def close(self, image, structuring_element):
        dilated_image = self.dilation(image, structuring_element)
        eroded_image = self.erosion(dilated_image, structuring_element)
        return eroded_image


    def start(self):
        self.image = self._read_image()
        self.grayscale_image = self.convert_to_grayscale(self.image)
        cv.imshow("Grayscale image",self.grayscale_image)

        self.thresholded_image = self.create_threshold_mask(self.grayscale_image, self.upper_threshold, self.lower_threshold)
        cv.imshow("Thresholded image", self.thresholded_image)

        structuring_element = np.ones((7, 7), dtype=np.uint8)
        opened_mask = self.open(self.thresholded_image, structuring_element)

        structuring_element = np.ones((5, 5), dtype=np.uint8)
        cleaned_mask = self.close(opened_mask, structuring_element)
        cv.imshow("Cleaned mask", cleaned_mask)

        restored_image = cv.bitwise_and(self.image, self.image, mask=cleaned_mask)
        cv.imshow("Coins", restored_image)


if __name__ == "__main__":

    image_path = "./assets/c1.jpg"
    upper_threshold = 0
    lower_threshold = 250

    coinSegmentation = CoinSegmentation(image_path, upper_threshold, lower_threshold)
    coinSegmentation.start()

    cv.waitKey(0)
