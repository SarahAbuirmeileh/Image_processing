import cv2 as cv
import numpy as np
import logging


class DetectionObjectThroughColor:
    def __init__(self, image_path, h_lower_bound, h_upper_bound, s_lower_bound, s_upper_bound):
        self.image_path = image_path
        self.h_lower_bound = h_lower_bound
        self.h_upper_bound = h_upper_bound
        self.s_lower_bound = s_lower_bound
        self.s_upper_bound = s_upper_bound

        self.image = None

        self.hsv_img = None
        self.h_image = None
        self.s_image = None
        self.v_image = None

    def _read_image(self):
        """
        Read image
        """
        image = cv.imread(self.image_path, cv.IMREAD_COLOR)
        if image is None:
            logging.error(f"Failed to load image from {self.image_path}.")
        return image

    def get_hsv_image(self):
        """
        Convert RGB image into HSV and split the HSV into its' three channels
        """
        self.hsv_img = cv.cvtColor(self.image, cv.COLOR_BGR2HSV)
        self.h_image, self.s_image, self.v_image = cv.split(self.hsv_img)

    def create_threshold_mask(self, image, lower_bound, upper_bound):
        """
        Creates a binary mask where values in the given image are 
        between the specified lower and upper bounds (inclusive).

        Returns:
            np.ndarray: Binary mask (0 or 1).
        """
        mask = np.zeros_like(image, dtype=np.uint8)
        mask[(image >= lower_bound) & (image <= upper_bound)] = 1
        return mask

    def intersect_images(self, image1, image2):
        """
        Intersect 2 mask images and return where they are 1 
        """
        intersection = np.zeros_like(image1, dtype=np.uint8)
        intersection[(image1 == 1) & (image2 == 1)] = 1

        return intersection

    def overlay_image_with_mask(self, image, mask, color=(0, 0, 255)):
        """
        Create new image that overlays te original image with  a scpeicifed color where the mask is not zero  
        """
        # Create a red mask
        color_mask = np.zeros_like(image)
        color_mask[:, :] = color

        # Blend mask color with the original image
        overlay_image = image.copy()
        mask_bool = mask.astype(bool)

        overlay_image[mask_bool] = cv.addWeighted(
            image[mask_bool], 0.5,
            color_mask[mask_bool], 0.5,
            0
        )

        # Trivial way (might be more readable)
        # for i in range(mask.shape[0]):       
        #     for j in range(mask.shape[1]):
        #         if mask[i, j] != 0:
        #             original_pixel = image[i, j].astype(np.int16)  
        #             overlay_pixel = np.array(color, dtype=np.int16)
        #             blended_pixel = ((original_pixel + overlay_pixel) // 2).astype(np.uint8)
        #             overlay_image[i, j] = blended_pixel

        return overlay_image

    def start(self):
        self.image = self._read_image()
        if self.image is None:
            return

        self.get_hsv_image()

        green_area = self.create_threshold_mask(self.h_image, self.h_lower_bound, h_upper_bound)
        saturated_image = self.create_threshold_mask(self.s_image, s_lower_bound, s_upper_bound)
        intersection_image = self.intersect_images(green_area, saturated_image)

        overlay_image = self.overlay_image_with_mask(self.image, intersection_image)
        cv.imshow('Original image', self.image)
        cv.imshow('Detected object with red mask', overlay_image)


if __name__ == "__main__":
    image1 = "./assets/playground1.png"
    image2 = "./assets/playground2.png"
    image3 = "./assets/playground3.jpeg"

    images = [image3, image1, image2]

    h_lower_bound = 30
    h_upper_bound = 55
    s_lower_bound = 100
    s_upper_bound = 255

    for image in images:
        football_playground_detection = DetectionObjectThroughColor(image, h_lower_bound, h_upper_bound, s_lower_bound,
                                                                    s_upper_bound)
        football_playground_detection.start()
        cv.waitKey(0)
