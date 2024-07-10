# To select the rectangle in the input image and generate its mask image.but the patch in the mask image is rgb patch

import cv2
import numpy as np

# Read the input image
img = cv2.imread('/home/usama/downsampled_image1.png')

# Create a black mask image with the same size as the input image
mask = np.zeros_like(img)

# Define the rectangular windows
rectangles = [
    ((100, 100), (200, 200)),
    #((300, 300), (400, 400))
   # ((500, 500), (800, 800)),
    #((700, 700), (1000, 1000)),
    #((900, 900), (1000, 1000)),
    #((1000, 1000), (1200, 1200))
]

# Draw the rectangles on the mask image
for rect in rectangles:
    cv2.rectangle(mask, rect[0], rect[1], (255, 255, 255), -1)

# Use the mask image to mask out the regions of the input image that correspond to the rectangular windows
masked_img = cv2.bitwise_and(img, mask)
cv2.imwrite('downsampled_image1_masked.png', masked_img)

# Display the masked image
#cv2.imshow('Masked Image', masked_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()