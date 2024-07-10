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
    
    cluster_mask = np.uint8(segmented_image == target_cluster)
    extracted_cluster = cv2.bitwise_and(image, image, mask=cluster_mask)
    
    # Connected component analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cluster_mask)
    print("num_labels",num_labels)


    if num_labels > 1:
        largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

    # Find all labels (components) related to the largest connected component
        related_labels = []
        for i in range(1, num_labels):
            if i != largest_label:
                area = stats[i, cv2.CC_STAT_AREA]
                if area >0:  # Set your desired threshold for related components
                    related_labels.append(i)

        # Create a binary mask for the largest connected component and related components
        connected_mask = np.zeros_like(labels, dtype=np.uint8)
        connected_mask[labels == largest_label] = 255
        for label in related_labels:
            connected_mask[labels == label] = 255

        # Apply the binary mask to the image using bitwise AND operation
        extracted_cluster = cv2.bitwise_and(extracted_cluster, extracted_cluster, mask=connected_mask)

    
    cv2.imwrite('extracted_cluster_large.jpg', extracted_cluster)
    cv2.imshow("Original Image", image)
    cv2.imshow("Segmented Image", segmented_image)
    cv2.imshow("Extracted Cluster", extracted_cluster)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    unique_labels = np.unique(segmented_image)
    print("Unique Labels:", unique_labels)

num_clusters=20  # Adjust this value based on your image and desired segmentation
target_cluster=169  # Adjust this value to the desired zone you want to extract

image_path = "/home/usama/usama_dev_test/Stroke-Based-Scene-Text-Erasing/example/_inference_result/img_387.png"
segment_zoning_map(image_path, num_clusters, target_cluster)

#unique_labels = np.unique(segmented_image)
#print("Unique Labels:", unique_labels)










