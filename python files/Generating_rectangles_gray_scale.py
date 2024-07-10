# Code for generating  number of gray scale rectangles on the mask image 

import cv2
import numpy as np

# Read the input image
img = cv2.imread('')

# Create a black mask image with the same size as the input image
mask = np.zeros_like(img)

# Define the rectangular windows
rectangles = [
   ((100, 100), (200, 200)),
    #((300, 300), (400, 400))
    #((500, 500), (800, 800))
    #((700, 700), (1000, 1000))
    #((900, 900), (1000, 1000)),
    #((1000, 1000), (1200, 1200))
]

# Draw the rectangles on the mask image
for rect in rectangles:
    cv2.rectangle(mask, rect[0], rect[1], (255, 255, 255), -1)
    

# Use the mask image to mask out the regions of the input image that correspond to the rectangular windows
# Convert the patch image to grayscale
gray_patch = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary mask
thresh = cv2.threshold(gray_patch, 0, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite('downsampled_image_masked11.png', thresh)

#masked_img = cv2.bitwise_and(img, mask)
#cv2.imwrite('masked1122.png', masked_img)

# Display the masked image
#cv2.imshow('Masked Image', masked1122_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()