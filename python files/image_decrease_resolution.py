from PIL import Image
import cv2
import pickle

def decrease_resolution(image_path, factor):
    # Open the image
    image = Image.open(image_path)
    print('image',image)
    print("image width",image.width)
    print("image height",image.height)
    print(type(image))

    # Calculate the new width and height\
    new_width = image.width // factor
    new_height = image.height //3

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    # Save the resized image
    resized_image.save('resized_image.jpg')

# Example usage
image_path = '/home/usama/Downloads/PHOTO-2019-03-16-15-25-40.jpg'
decrease_resolution(image_path, 2)




