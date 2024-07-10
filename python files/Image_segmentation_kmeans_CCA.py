import numpy as np
import cv2

def segment_zoning_map(image_path, num_clusters, target_cluster):
    image = cv2.imread(image_path)
    pixels = image.reshape(-1, 3).astype(np.float32)
    print("Pixels", pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    segmented_image = labels.reshape(image.shape[:2])
    segmented_image = segmented_image.astype(np.uint8)
    segmented_image = (segmented_image * (255 // (num_clusters-1))).astype(np.uint8)
    cv2.imwrite('segmented_image.jpg',segmented_image)
    
    # Connected component analysis
    #num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(segmented_image)

    # Extract all components with the target label
    #component_masks = []
    #for label in range(1, num_labels):
     #   component_mask = np.uint8(labels == label)
      #  component_masks.append(component_mask)
       # extracted_component = cv2.bitwise_and(image, image, mask=component_mask)
        #cv2.imwrite(f'extracted_component_{label}.jpg', extracted_component)

    # Extract the target cluster from the k-means segmented image
   # cluster_mask = np.uint8(segmented_image == target_cluster)
   # extracted_cluster = cv2.bitwise_and(image, image, mask=cluster_mask)
    #cv2.imshow("Extracted Cluster11", extracted_cluster)

    # Identify the labels of all connected components
    #component_labels = np.unique(labels[labels != 0])

    # Create a mask to combine all components
    #combined_mask = np.zeros_like(cluster_mask)

    # Combine all components with the target label
    #for component_label in component_labels:
     #   component_mask = np.uint8(labels == component_label)
      #  combined_mask |= component_mask

    # Apply the combined mask to the extracted cluster
    #extracted_cluster11 = cv2.bitwise_and(extracted_cluster, extracted_cluster, mask=combined_mask)

   # cv2.imwrite('extracted_cluster.jpg', extracted_cluster)
   # cv2.imshow("Original Image", image)
   # cv2.imshow("Segmented Image", segmented_image)
   # cv2.imshow("Extracted Cluster", extracted_cluster)
   # cv2.waitKey(0)
   # cv2.destroyAllWindows()

    unique_labels = np.unique(segmented_image)
    print("Unique Labels:", unique_labels)

num_clusters = 13  # Adjust this value based on your image and desired segmentation
target_cluster = 0   # Adjust this value to the desired zone you want to extract

image_path = "/home/usama/zoning_maps/sd_beresford_page-0001.jpg"
segment_zoning_map(image_path, num_clusters, target_cluster)
