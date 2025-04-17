import cv2 as cv
import numpy as np
import logging


class EdgeDetection:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = self._read_image()
        self.sobel_kernel = np.array([[-1, 0, 1],
                                      [-2, 0, 2],
                                      [-1, 0, 1]], dtype=np.float32)

        self.gaussian_kernel = self.generate_gaussian_kernel(kernel_size=5, sigma=4.0)
        # self.gaussian_kernel = np.array([[1,4,7,4,1],
            #                              [4,16,26,16,4],
            #                              [7,26,41,26,7],
            #                              [4,16,26,16,4],
            #                              [1,4,7,4,1]], dtype=np.float32)
        # self.gaussian_kernel /= np.sum(self.gaussian_kernel)

    def _read_image(self):
        image = cv.imread(self.image_path, cv.IMREAD_GRAYSCALE)
        if image is None:
            logging.error(f"Failed to load image from {self.image_path}.")
        return image

    def threshold(self, image, threshold: int):
        if image is None:
            logging.warning("No image to threshold :(")
            return None

        return np.where(image >= threshold, 255, 0).astype(np.uint8)

    def display_image(self, image, image_name: str):
        if image is None:
            logging.warning("No image to show :(")
            return

        # Invert image to display 0 as white and 255 as black
        inverted_image = 255 - image

        cv.imshow(image_name, inverted_image)
        key = cv.waitKey(0)
        cv.destroyAllWindows()

    def generate_gaussian_kernel(self, kernel_size=5, sigma=1.0):
        k = kernel_size // 2
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

        for y in range(-k, k + 1):
            for x in range(-k, k + 1):
                exponent = -(x**2 + y**2) / (2 * sigma**2)
                kernel[y + k, x + k] = (1 / (2 * np.pi * sigma**2)) * np.exp(exponent)

        kernel /= np.sum(kernel)  # Normalize
        return kernel

    # Trivial way to do convolution
    def convolution_trivial_way(self, image, kernel):
        kernel_height, kernel_width = kernel.shape
        image_height, image_width = image.shape
        
        kh = kernel_height // 2
        kw = kernel_width // 2

        kernel = np.flipud(kernel)
        kernel = np.fliplr(kernel)

        output = np.zeros((image_height, image_width))

        # Start from kh, kw and subtract them to skip the borders
        for ix in range(kh, image_height-kh):
            for iy in range(kw, image_width-kw):
                sum = 0
                for kx in range(kernel_height):
                    for kw in range(kernel_width):
                        sum += image[ix + kx - kh, iy + kw - kw] * kernel[kx, kw]
                output[ix, iy] = sum

        return output

    def convolution(self, image, kernel, is_kernel_mode=False):
        if image is None or kernel is None:
            logging.warning("No input for convolution")
            return None

        image_height, image_width = image.shape
        kernel_height, kernel_width = kernel.shape

        kh = kernel_height // 2
        kw = kernel_width // 2

        # Flip the kernel for correct convolution 
        kernel = np.flipud(kernel)
        kernel = np.fliplr(kernel)

        if is_kernel_mode:
            # Full convolution (used for kernel * kernel)
            output_height = image_height + kernel_height - 1
            output_width = image_width + kernel_width - 1
            output = np.zeros((output_height, output_width), dtype=np.float32)

            # Pad iamge to cover all kernel positions 
            image = np.pad(image, ((kernel_height - 1, kernel_height - 1), (kernel_width - 1, kernel_width - 1)), mode='constant', constant_values=0)

            for ix in range(output_height):
                for iy in range(output_width):
                    # Extract the region of the image under the kernel 
                    # (extract the # of rows & columns centered at (ix, iy), extending by half the kernel size )
                    region = image[ix:ix + kernel_height, iy:iy + kernel_width]
                    output[ix, iy] = np.sum(region * kernel)
        else:
           output_height = image_height - kernel_height + 1
           output_width = image_width - kernel_width + 1
           output = np.zeros((output_height, output_width), dtype=np.float32)

           for ix in range(output_height):
            for iy in range(output_width):
                # Extract the region of the image fully under the kernel
                    region = image[ix:ix + kernel_height, iy:iy + kernel_width]
                    output[ix, iy] = np.sum(region * kernel)

        return output

    def sobel_edge_detection(self, threshold):
        sobel_image = self.convolution(self.image, self.sobel_kernel)
        thresholded = self.threshold(sobel_image, threshold)
        self.display_image(thresholded, "Sobel Edge Detection")

    def sobel_edge_detection_with_smoothing(self, threshold):
        # smoothed_image = self.convolution(self.image, self.gaussian_kernel)
        # sobel_edge_detection = self.convolution(smoothed_image, self.sobel_kernel)
        # thresholded_image = self.threshold(sobel_edge_detection, threshold)
        # self.display_image(thresholded_image, "Smoothed Sobel Edge Detection")
        
        # Combine the two kernels
        combined_kernels = self.convolution(self.gaussian_kernel, self.sobel_kernel, True)
        sobel_edge_detection = self.convolution(self.image, combined_kernels)
        thresholded_image = self.threshold(sobel_edge_detection, threshold)
        self.display_image(thresholded_image, "Smoothed Sobel Edge Detection")

    def canny_edge_detection(self,lower_threshold, upper_threshold ):
        canny_edge_detection = cv.Canny(self.image, lower_threshold, upper_threshold)
        edge_detection.display_image(canny_edge_detection, "Canny Edge Detection")


if __name__ == "__main__":
    # image = "./assets/flower.png"
    # threshold = 25

    image = "./assets/hand_ct.jpeg"
    threshold = 35
    
    lower_threshold = 20
    upper_threshold = 60

    edge_detection = EdgeDetection(image)
    edge_detection.sobel_edge_detection(threshold)
    edge_detection.sobel_edge_detection_with_smoothing(threshold)

    edge_detection.canny_edge_detection(lower_threshold, upper_threshold)
