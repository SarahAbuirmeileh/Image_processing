import cv2 as cv
import numpy as np

class VideoCartoonoization:
    def __init__(self, video_path):

        self.video_path = video_path
        self.images = []


    def read_video(self):

        captured = cv.VideoCapture(self.video_path)

        if (captured.isOpened() == False):
            print("Error opening the video {self.video_path}")
        
        while(captured.isOpened()):
            success, frame = captured.read()

            if success == True:
                self.images.append(frame)
            else:
                break

        captured.release()

    
    def display_video(self, images, video_name, frame_rate = 40):
        
        seconds = int(1000 / frame_rate)

        for image in images:
            cv.imshow(video_name, image)

            if cv.waitKey(seconds) == ord('0'):
                break

    
    def generate_edges(self, image):

        grey_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return cv.Canny(grey_img, 10, 200)


    def reduceColorsHSV(self, image, factor):

        hsv_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv_img)

        size = image.shape
        for i in range(size[0]):
            for j in range(size[1]):
                h[i, j] = self.reduceValHSV(h[i, j], factor)
                s[i, j] = int(s[i, j] * 0.75)
                v[i, j] = min(int(v[i, j] * 1.5), 255)

        hsv_img = cv.merge([h, s, v])
        rgb_img = cv.cvtColor(hsv_img, cv.COLOR_HSV2BGR)
        return rgb_img


    def reduceValHSV(self, val, factor):

        return factor * (val // factor)


    def smooth_image(self, image):

        return cv.GaussianBlur(image,(5,5),0)  


    def water_color_image(self, image, thickness, factor):

        smoothed_image = self.smooth_image(image)
        edge_image = self.generate_edges(smoothed_image)
        thick_image = self.adjust_edge_thickness(edge_image, thickness)
        reduced_color_image = self.reduceColorsHSV(smoothed_image, factor)
        size = image.shape

        for i in range(size[0]):
            for j in range(size[1]):
                if thick_image[i, j] != 0:
                    reduced_color_image[i, j][0] = 0
                    reduced_color_image[i, j][1] = 0
                    reduced_color_image[i, j][2] = 0

        return reduced_color_image


    def water_color_video(self, images, thickness, factor):

        water_colored_images = []
        for image in images:
            water_colored_images.append(self.water_color_image(image, thickness, factor))

        return water_colored_images

    
    def adjust_edge_thickness(self, image, thickness): 

        structuring_element = np.ones((thickness, thickness), np.uint8)
        dilated_image = cv.dilate(image, structuring_element, iterations=1)
        return dilated_image

    
    def start(self, edge_thickness = 1, colors_number = 10, frame_rate = 1):

        self.read_video()
        watered_video = self.water_color_video(self.images[:5], edge_thickness, colors_number)
        self.display_video(watered_video, "Cartoonized Video", frame_rate)



if __name__ == "__main__":

    cartooinze_video = VideoCartoonoization("./assets/flower.mp4")
    cartooinze_video.start()
    
    cv.destroyAllWindows()
