import cv2
import numpy as np
image = cv2.imread('/home/usama/zoning_maps/1-page-001.jpg')
print("Original image shape",image.shape)
height, width = image.shape[:2]
new_width = ((width - 1) // 320 + 1) * 320
new_height = ((height - 1) // 320 + 1) * 320
pad_width = new_width - width
pad_height = new_height - height
top = pad_height // 2
bottom = pad_height - top
left = pad_width // 2
right = pad_width - left
padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
padded_width=padded_image[0]
print("Padded image shape",padded_image.shape)
