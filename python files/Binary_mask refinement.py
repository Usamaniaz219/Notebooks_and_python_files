import cv2
import numpy as np
from skimage.io import imread
from skimage import filters
import matplotlib.pyplot as plt
import pickle
import cv2
import numpy as np
from skimage.io import imread
from skimage import filters
import matplotlib.pyplot as plt
import pickle
from skimage.morphology import disk, binary_erosion, binary_dilation, binary_opening, binary_closing

from scipy.ndimage import binary_fill_holes





"""
def thresh(index: int, min_blob_size=5):
  with open('images2/2021 City of Duvall Zoning Map Final_202202151708363112-page-001/labelled_image', 'rb') as file:
        labelled_image = pickle.load(file)
  

  '''
    Get threshold to make mask using the otsus method, and apply a correction
    passed in conservative (-100;100) as a percentage of th.
  '''

  # blur and get level using otsus
  color_index_mat = np.array([index], np.uint8)
  mask1 = cv2.compare(labelled_image, color_index_mat, cv2.CMP_EQ)  
  
  # morph operators
  #kernel = cv2.getStructuringElement(cv2.MORPH_CLOSE, (5, 5))
  #kernel=np.ones((3,3))
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  mask = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
  mask=cv2.medianBlur(mask,5)

  # remove small connected blobs
  # find connected components
  #n_components, output, stats, centroids = cv2.connectedComponentsWithStats(
   #   mask1, connectivity=8)
  # remove background class
  #sizes = stats[1:, -1]
  #n_components = n_components - 1

  # remove blobs
  #mask_clean = np.zeros((output.shape))
  # for every component in the image, keep it only if it's above min_blob_size
  #for i in range(0, n_components):
   # if sizes[i] >=3:
    #  mask_clean[output == i + 1] = 255

  fig, axs = plt.subplots(1, 2, figsize=(15,15))
  axs[0].imshow(mask1, cmap='gray')
  axs[0].set_title('Binary image')

  axs[1].imshow(mask, cmap='gray')
  axs[1].set_title('clean image')
    
  plt.show()    
      

  return mask

thresh(5)
"""

def vectorize_zone(index: int):
    with open('images2/ca_san_carlos/labelled_image', 'rb') as file:
        labelled_image = pickle.load(file)
    
    color_index_mat = np.array([index], np.uint8)
    mask = cv2.compare(labelled_image, color_index_mat, cv2.CMP_EQ)
    #cv2.imwrite(f'maps_output_cca/2021 City of Duvall Zoning Map Final_202202151708363112-page-001/mask_result5_{index}.jpg',mask)
    #cv2.medianBlur(mask,3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    processed_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    processed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    processed_mask=cv2.medianBlur(mask,7)
   

    
    # Save the final processed image
    # Define the kernel for erosion
    #kernel = np.ones((3, 3), dtype=np.uint8)

    # Perform erosion using cv2.erode
    #processed_mask = cv2.erode(processed_mask, kernel, iterations=1)
    #cv2.medianBlur(processed_mask,5)
    cv2.imwrite(f'maps_output_cca/ca_san_carlos/processed_result6_{index}.jpg', processed_mask)

if __name__ == "__main__":
    vectorize_zone(22)


"""
def refine_mask(index: int):
    with open('images2/2021 City of Duvall Zoning Map Final_202202151708363112-page-001/labelled_image', 'rb') as file:
        labelled_image = pickle.load(file)
    color_index_mat = np.array([index], np.uint8)
    mask = cv2.compare(labelled_image, color_index_mat, cv2.CMP_EQ)
    
    # Binarize the image
    threshold = filters.threshold_otsu(mask)
    print("Threshold",threshold)
    binary_image = (mask <= threshold).astype(np.uint8) * 255
    print("Binary image",binary_image)
    
    # Create a figure to display the images
    #fig, axs = plt.subplots(1, 2, figsize=(15, 15))

    eroded1 = binary_erosion(binary_image, disk(1))
    eroded4 = binary_erosion(binary_image, disk(4))

    fig, axs = plt.subplots(1, 3, figsize=(15,15))
    axs[0].imshow(binary_image,cmap='gray')
    axs[0].set_title('Binary image')

    axs[1].imshow(eroded1, cmap='gray')
    axs[1].set_title('Eroded r=1')

    axs[2].imshow(eroded4, cmap='gray')
    axs[2].set_title('Eroded r=4')
    
    # Display original image and binary image side by side
    #axs[0].imshow(mask, cmap='gray')
    #axs[0].set_title('Original')
    
    #axs[1].imshow(binary_image, cmap='gray')
    #axs[1].set_title('Binary')


    dilated1 = binary_dilation(binary_image, disk(1))
    dilated4 = binary_dilation(binary_image, disk(4))

    fig, axs = plt.subplots(1, 3, figsize=(15,15))
    axs[0].imshow(binary_image,cmap='gray')
    axs[0].set_title('Binary image')

    axs[1].imshow(dilated1, cmap='gray')
    axs[1].set_title('Dilated r=1')

    axs[2].imshow(dilated4, cmap='gray')
    axs[2].set_title('Dilated r=4')

    filled = binary_fill_holes(binary_image)

    fig, axs = plt.subplots(1, 2, figsize=(15,15))
    axs[0].imshow(binary_image, cmap='gray')
    axs[0].set_title('Binary image')

    axs[1].imshow(filled, cmap='gray')
    axs[1].set_title('Holes filled')
    
    plt.show()

refine_mask(1)

"""