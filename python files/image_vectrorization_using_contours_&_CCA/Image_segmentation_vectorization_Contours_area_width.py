import cv2
import pickle
import numpy as np

def vectorize_zone(index: int):
    with open('images3/0001/labelled_image', 'rb') as file:
        labelled_image = pickle.load(file)
    color_index_mat = np.array([index], np.uint8)
    mask = cv2.compare(labelled_image, color_index_mat, cv2.CMP_EQ)
    cv2.imwrite('maps_output_results_median/0001/mean__mask14.jpg', mask)
    mask=cv2.medianBlur(mask, 5)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Create an empty mask for all contours
    contour_mask = np.zeros_like(mask)
    total_area_ratio = 0
    num_valid_contours = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10 and cv2.boundingRect(contour)[2] > 5:
            # Calculate the bounding rectangle area
            rect_area = cv2.boundingRect(contour)[2] * cv2.boundingRect(contour)[3]

            area_ratio = area / rect_area
            total_area_ratio += area_ratio
            num_valid_contours += 1

    if num_valid_contours > 0:
        average_area_ratio = total_area_ratio / num_valid_contours
        print("Average Area Ratio", average_area_ratio)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1:
                # Calculate the bounding rectangle area
                rect_area = cv2.boundingRect(contour)[2] * cv2.boundingRect(contour)[3]
                # Compute the contour area ratio
                area_ratio = area / rect_area
                # Filter contours based on area ratio threshold
                #if (average_area_ratio - 0.96) < area_ratio < (average_area_ratio + 0.96) and cv2.boundingRect(contour)[2] > 5:
                cv2.drawContours(contour_mask, [contour], 0, 255, thickness=cv2.FILLED)

    cv2.medianBlur(contour_mask, 5)
    cv2.imwrite('maps_output_results_median/0001/extracted_cluster_large14.jpg', contour_mask)

vectorize_zone(14)




"""
def vectorize_zone(index: int):
    with open('images4/il_green_oaks/labelled_image', 'rb') as file:
        labelled_image = pickle.load(file)
    color_index_mat = np.array([index], np.uint8)
    mask = cv2.compare(labelled_image, color_index_mat, cv2.CMP_EQ)
    cv2.imwrite('maps_output_results_median/il_green_oaks/mean__mask15.jpg', mask)
    mask=cv2.medianBlur(mask, 5)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Create an empty mask for all contours
    contour_mask = np.zeros_like(mask)
    total_area_ratio = 0
    num_valid_contours = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10 and cv2.boundingRect(contour)[2] > 5:
            # Calculate the bounding rectangle area
            rect_area = cv2.boundingRect(contour)[2] * cv2.boundingRect(contour)[3]

            area_ratio = area / rect_area
            total_area_ratio += area_ratio
            num_valid_contours += 1

    if num_valid_contours > 0:
        average_area_ratio = total_area_ratio / num_valid_contours
        print("Average Area Ratio",average_area_ratio)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area >1 :
                # Calculate the bounding rectangle area
                rect_area = cv2.boundingRect(contour)[2] * cv2.boundingRect(contour)[3]
                # Compute the contour area ratio
                area_ratio = area / rect_area
                # Filter contours based on area ratio threshold
                if area_ratio >=(average_area_ratio-0.96) and cv2.boundingRect(contour)[2] >5:
                    cv2.drawContours(contour_mask, [contour], 0, 255, thickness=cv2.FILLED)
    cv2.medianBlur(contour_mask,5)

    cv2.imwrite('maps_output_results_median/il_green_oaks/extracted_cluster_large15.jpg', contour_mask)

vectorize_zone(15)

"""










#def vectorize_zone(index: int):
 #   with open('/home/usama/Downloads/labelled_image (1)', 'rb') as file:
  #      labelled_image = pickle.load(file)
   # color_index_mat = np.array([index], np.uint8)
    #mask = cv2.compare(labelled_image, color_index_mat, cv2.CMP_EQ)
    
    #cv2.imwrite('maps_output/5/mean__mask6.jpg', mask)
    #contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask for all contours
    #contour_mask = np.zeros_like(mask)

    #for contour in contours:
     #   area = cv2.contourArea(contour)
      #  if area > 10 and cv2.boundingRect(contour)[2] > 5:  
       #     cv2.drawContours(contour_mask, [contour], 0, 255, thickness=cv2.FILLED)
    
    #print("type of laballed image is :",type(labelled_image))
    #print("type of mask image is :",type(contour_mask))
    #print("contour mask is ",contour_mask)        

    #cv2.imwrite('maps_output/5/extracted_cluster_large6.jpg',contour_mask)

#vectorize_zone(6)
