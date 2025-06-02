from thresholdingOpenCV import ImageType, ThresholdingImage

if __name__ == "__main__":
    
    # Test gray-scale image 
    image_path = "./assets/brain_tumor_mri.jpg"
    image_type = ImageType.GRAYSCALE
    
    # Test colored image
    # image_path = "./assets/bird.jpg"
    # image_type = ImageType.COLOR

    # Read the input from the user
    lower_threshold_input = int(input("Enter the lower threshold value (0-255): "))
    upper_threshold_input = int(input("Enter the upper threshold value (0-255): "))

    # Make sure that the values lies within the range
    lower_threshold = max(0, lower_threshold_input)
    upper_threshold = min(255, upper_threshold_input)

    image = ThresholdingImage(image_path, image_type=image_type)
    thresholded_image = image.thresholding(lower_threshold, upper_threshold)

    if thresholded_image is not None:
        image.display_image(thresholded_image, "Thresholded Image")
    else:
        print("Failed to apply thresholding. Please check the input image.")
