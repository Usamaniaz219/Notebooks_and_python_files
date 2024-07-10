import pickle
import numpy as np
import cv2




#def process_tile(tile):
    # Apply edge detection to the tile using Canny or any other edge detection method
    #edges = cv2.Canny(tile, threshold1=10, threshold2=50)  # Adjust thresholds as needed

    # Apply connected component algorithm on the edges
    #num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges)

    # Initialize an empty mask
    #mask = np.zeros_like(tile)

   # for label in range(1, num_labels):
        #area = stats[label, cv2.CC_STAT_AREA]
       # centroid = centroids[label]

      #  if area >= 20:  # Adjust the area threshold as needed
     #       mask[labels == label] = 255

    # Apply morphological operations to remove small gaps and noise
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    #return mask








#def calculate_eccentricity(stats, label):
 ##  minor_axis_length = stats[label, cv2.CC_STAT_HEIGHT]
   # eccentricity = np.sqrt(1 - (minor_axis_length / major_axis_length)**2)
    #return eccentricity

#def process_tile(tile):
 #   # Apply connected component algorithm on the tile
  #  mask = cv2.medianBlur(tile, 5)
  #  num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    
   # for label in range(1, num_labels):
    #    area = stats[label, cv2.CC_STAT_AREA]
     #   eccentricity = calculate_eccentricity(stats, label)
      #  
       # if area >= 3 and eccentricity < 0.98:
        #    mask[labels == label] = 255
        #else:
         #   mask[labels == label] = 0  # Set small or elongated regions to black
            
    # Apply morphological operation to fill tiny holes
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    #return mask

TILE_SIZE = 100
OVERLAP = 5

def process_tile(tile):
    # Apply connected component algorithm on the tile
    mask = cv2.medianBlur(tile, 5)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    #print("Centriods",len(centroids))
    
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        centroid = centroids[label]
       
        
        if area >= 10 and centroid[1] > 10:
            mask[labels == label] = 255
        else:
            mask[labels == label] = 0  # Set small regions or regions near the edges to black
            
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    kernel1=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel1)
    return mask




def vectorize_zone(index: int):
    with open('images2/Ocean Ridge_FL/labelled_image', 'rb') as file:
        labelled_image = pickle.load(file)
    
    color_index_mat = np.array([index], np.uint8)
    mask = cv2.compare(labelled_image, color_index_mat, cv2.CMP_EQ)
    cv2.imwrite(f'maps_output_cca/Ocean Ridge_FL/mask_result_{index}.jpg',mask)
    cv2.medianBlur(mask,5)

    height, width = mask.shape
    processed_mask = np.zeros_like(mask)

    for y in range(0, height - TILE_SIZE + 1, TILE_SIZE - OVERLAP):
        for x in range(0, width - TILE_SIZE + 1, TILE_SIZE - OVERLAP):
            tile = mask[y:y+TILE_SIZE, x:x+TILE_SIZE]
            processed_tile = process_tile(tile)
            processed_mask[y:y+TILE_SIZE, x:x+TILE_SIZE] = processed_tile


    cv2.medianBlur(processed_mask,5)



    cv2.imwrite(f'maps_output_cca/Ocean Ridge_FL/processed_result3_{index}.jpg', processed_mask)

if __name__ == "__main__":
    vectorize_zone(4)

    



















"""
def denoise_binary_mask(binary_mask, min_area_threshold):
    # Apply connected component labeling
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)

    # Create a copy of the input image to store the denoised result
    denoised_mask = np.zeros_like(binary_mask)

    # Remove small connected components (noise) by setting their pixels to 0
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= min_area_threshold:
            denoised_mask[labels == label] = 255

    return denoised_mask """



#from skimage.morphology import skeletonize, thin


"""
def process_tile(tile):
    # Apply skeletonization to the tile
    skeleton = skeletonize(tile)
    
    # Apply inverse skeletonization to get cleaned mask image
    cleaned_mask = thin(skeleton)
    
    return cleaned_mask
    """
""""


"""


""" 
def process_tile(tile):
    # Apply Otsu thresholding to the tile
    #_, thresholded_tile = cv2.threshold(tile, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #print("thresholded tile",thresholded_tile)
    
    # Apply connected component algorithm on the thresholded tile
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tile,connectivity=4)
    
    processed_mask = np.zeros_like(tile)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        
        if area >= 3:
            processed_mask[labels == label] = 255
    
    return processed_mask
"""



"""
def process_tile(tile):
    # Apply connected component algorithm on the tile
    mask = cv2.medianBlur(tile, 5)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >=3:
            mask[labels == label] = 255
    return mask
"""




""""
def process_tile(tile):
    # Apply connected component algorithm on the tile
    mask = cv2.medianBlur(tile, 5)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    
    processed_mask = np.zeros_like(mask)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        
        if area >= 5:
            mask[labels == label] = 255
            contours, _ = cv2.findContours((labels == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                contour = contours[0]
                perimeter = cv2.arcLength(contour, True)
                area = cv2.contourArea(contour)
                
                if perimeter > 0:
                    convexity = 4 * np.pi * area / (perimeter ** 2)
                    
                    if convexity >= 0.2:
                        processed_mask[labels == label] = 255
    
    return processed_mask
"""
"""
def process_tile(tile):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tile, connectivity=4)
    
    processed_mask = np.zeros_like(tile)
    
    size_thresholds = [10,20, 50,100, 150,200,250]  # Set your desired size thresholds
    
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        
        if area >= 3:
            for size_threshold in size_thresholds:
                if area >= size_threshold:
                    processed_mask[labels == label] = 255
                    break  # Exit loop after applying the first suitable threshold
    
    return processed_mask
"""







# Example usage
# Load an image or tile
# tile = cv2.imread('tile.png', cv2.IMREAD_GRAYSCALE)

# Process the tile
# processed_tile = process_tile(tile)

# Display or save the processed tile
# cv2.imshow('Processed Tile', processed_tile)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""
def process_tile(tile):
    # Apply connected component algorithm on the tile
    mask = cv2.medianBlur(tile, 5)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= 3:
            mask[labels == label] = 255
        else:
            mask[labels == label] = 0  # Set small regions to black
    
    # Create a copy of the mask to apply morphological operations
    filled_mask = mask.copy()
    
    # Apply morphological operation to fill tiny holes
    kernel = np.ones((9, 9), np.uint8)
    filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, kernel)
    
    # Use the filled_mask to only fill the holes without affecting the original regions
    mask[filled_mask == 0] = 0
    
    return mask

"""



"""
def process_tile(tile):
    # Apply connected component algorithm on the tile
    mask = cv2.medianBlur(tile, 5)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
   
        
        if area >= 0:
            mask[labels == label] = 255
            #mask = cv2.fastNlMeansDenoising(mask, None, h=20, templateWindowSize=15, searchWindowSize=21)
        #else:
            #mean_filter_kernel = np.ones((5,5),np.float32)/(5*5)
            #mask = cv2.filter2D(mask,-1,mean_filter_kernel)
         #   mask[labels == label] = 0  # Set small or irregular regions to black
    
    return mask 
"""
""" important 
def process_tile(tile):
    # Apply connected component algorithm on the tile
    #mask = cv2.medianBlur(tile, 5)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tile)
    
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        
        if area >= 10:
            tile[labels == label] = 255
            # You can apply further denoising here if needed
            # mask = cv2.fastNlMeansDenoising(mask, None, h=20, templateWindowSize=15, searchWindowSize=21)
        else:
            tile[labels == label] = 0  # Set small or irregular regions to black
    
    # Find contours in the mask after area-based filtering
    contours, _ = cv2.findContours(tile, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an empty mask to draw contours
    contour_mask = np.zeros_like(tile)
    
    # Draw the found contours on the contour mask
    cv2.drawContours(contour_mask, contours, -1, 255, thickness=cv2.FILLED)
    
    return contour_mask


def vectorize_zone(index: int):
    with open('images2/2021 City of Duvall Zoning Map Final_202202151708363112-page-001/labelled_image', 'rb') as file:
        labelled_image = pickle.load(file)
    
    color_index_mat = np.array([index], np.uint8)
    mask = cv2.compare(labelled_image, color_index_mat, cv2.CMP_EQ)
    #cv2.imwrite(f'maps_output_cca/2021 City of Duvall Zoning Map Final_202202151708363112-page-001/mask_result5_{index}.jpg',mask)
    #cv2.medianBlur(mask,5)

    height, width = mask.shape
    processed_mask = np.zeros_like(mask)

    for y in range(0, height - TILE_SIZE + 1, TILE_SIZE - OVERLAP):
        for x in range(0, width - TILE_SIZE + 1, TILE_SIZE - OVERLAP):
            tile = mask[y:y+TILE_SIZE, x:x+TILE_SIZE]
            processed_tile = process_tile(tile)
            processed_mask[y:y+TILE_SIZE, x:x+TILE_SIZE] = processed_tile
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))        
    processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel1)  
    #
    processed_mask=cv2.dilate(processed_mask,(5,5))
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))   
    #processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)        
    # Save the final processed im
    # Define the kernel for erosion
    #kernel = np.ones((3, 3), dtype=np.uint8)

    # Perform erosion using cv2.erode
    #processed_mask = cv2.erode(processed_mask, kernel, iterations=1)
    cv2.medianBlur(processed_mask,7)
    #processed_mask=cv2.erode(processed_mask,(5,5))
    cv2.imwrite(f'maps_output_cca/2021 City of Duvall Zoning Map Final_202202151708363112-page-001/processed_result5_{index}.jpg', processed_mask)

if __name__ == "__main__":
    vectorize_zone(9)



"""


























"""
def denoise_binary_mask(binary_mask, min_area_threshold, max_aspect_ratio_threshold):
    # Apply connected component labeling
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)

    # Create a copy of the input image to store the denoised result
    denoised_mask = np.zeros_like(binary_mask)

    # Remove small and non-regular connected components (noise) by setting their pixels to 0
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]

        aspect_ratio = width / height

        if area >= min_area_threshold and aspect_ratio <= max_aspect_ratio_threshold:
            denoised_mask[labels == label] = 255

    return denoised_mask

def vectorize_zone(index: int):
    with open('images3/0001/labelled_image', 'rb') as file:
        labelled_image = pickle.load(file)

    color_index_mat = np.array([index], np.uint8)
    mask = cv2.compare(labelled_image, color_index_mat, cv2.CMP_EQ)
    cv2.imwrite('maps_output_results_median/0001/mean__mask1.jpg', mask)

    mask = cv2.medianBlur(mask, 5)
    
    # Filter and denoise the connected components
    min_area_threshold = 10
    max_aspect_ratio_threshold = 4  # Adjust this threshold based on your specific requirement
    denoised_mask = denoise_binary_mask(mask, min_area_threshold, max_aspect_ratio_threshold)

    # Save the denoised connected component as the final result
    cv2.imwrite('maps_output_results_median/0001/extracted_cluster_large5.jpg', denoised_mask)

vectorize_zone(1)"""











"""
"""

























"""
def vectorize_zone(index: int):
    with open('images3/0001/labelled_image', 'rb') as file:
        labelled_image = pickle.load(file)

    color_index_mat = np.array([index], np.uint8)
    mask = cv2.compare(labelled_image, color_index_mat, cv2.CMP_EQ)
    cv2.imwrite('maps_output_results_median/0001/mean__mask14.jpg', mask)

    mask = cv2.medianBlur(mask, 5)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    # Calculate the mean area ratio of the valid connected components
    total_area_ratio = 0
    num_valid_components = 0

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > 10 and stats[label, cv2.CC_STAT_WIDTH] > 5:
            # Calculate the bounding rectangle area
            rect_area = stats[label, cv2.CC_STAT_WIDTH] * stats[label, cv2.CC_STAT_HEIGHT]

            area_ratio = area / rect_area
            total_area_ratio += area_ratio
            num_valid_components += 1

    if num_valid_components > 0:
        average_area_ratio = total_area_ratio / num_valid_components
    else:
        average_area_ratio = 0

    # Create an empty mask for the valid connected components
    contour_mask = np.zeros_like(mask)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > 1:
            # Calculate the bounding rectangle area
            rect_area = stats[label, cv2.CC_STAT_WIDTH] * stats[label, cv2.CC_STAT_HEIGHT]

            # Compute the contour area ratio
            area_ratio = area / rect_area

            # Filter connected components based on area ratio threshold
            if (average_area_ratio - 0.96) <= area_ratio <= (average_area_ratio + 0.96) and stats[label, cv2.CC_STAT_WIDTH] > 5:
                component_mask = np.zeros_like(mask, dtype=np.uint8)
                component_mask[labels == label] = 255
                contour_mask = cv2.bitwise_or(contour_mask, component_mask)

    cv2.medianBlur(contour_mask, 5)
    cv2.imwrite('maps_output_results_median/0001/extracted_cluster_large14.jpg', contour_mask)

vectorize_zone(14)
"""