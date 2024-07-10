import os
from scipy import stats
import  cv2
from sklearn.neighbors import NearestNeighbors
import numpy as np



# Read input image
input_image_path = 'image_vectorization/1/zoningmapdec14-page-001.jpg'
img = cv2.imread(input_image_path)
# dtype=img.dtype
# print(dtype)
z = img.reshape((-1, 3))


# Convert to np.float32
z = np.float32(z)
# z_type=z.dtype
# print(os.getcwd())

# help(cv2.kmeans)

# Define criteria, number of clusters (K), and apply k-means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Calculate the number of unique colors in the image
# num_unique_colors = len(np.unique(z, axis=0))
K=16
attempts = 3

ret, labels, center = cv2.kmeans(z, K, None, criteria, 10,cv2.KMEANS_PP_CENTERS, attempts)


# Perform post-processing to adjust cluster centers
for k in range(K):

#     # Get all points assigned to this cluster
    cluster_points = z[labels.flatten() == k]
#     # print("cluster points",cluster_points)
    
    # Calculate the mean RGB value for this cluster
    mean_color = np.mean(cluster_points, axis=0)

#     # median_color=np.median(cluster_points,axis=0)
#     mode_color = stats.mode(cluster_points, axis=0).mode[0]
    
#     # Assign the mean color as the new cluster center
#     # center[k] = mean_color
#     center[k] = mode_color

# Convert back into uint8, and make the original image
center = np.uint8(center)
quantized = center[labels.flatten()]

# Reshape the clustered image
quantized_img = quantized.reshape((img.shape))
img = cv2.resize(img,(960,960))
quantized_img = cv2.resize(quantized_img,(960,960))
cv2.imshow("original image",img)
cv2.waitKey(0)
cv2.imshow("quantized image",quantized_img)
cv2.waitKey(0)
cv2.imwrite(f"image_quantization_segmentation_outputs/Kmeans_quantized_{K}.jpg", quantized_img)
 
    

# #  Convert the image to RGB format
# map_image_rgb = cv2.cvtColor(quantized_img, cv2.COLOR_BGR2RGB)

# # # Reshape the image to a 2D array of RGB values
# rgb_values = map_image_rgb.reshape((-1, 3))

# # # Create a dictionary to count RGB occurrences
# rgb_counts = {}

# # # Count RGB occurrences
# for rgb in rgb_values:
#     rgb_tuple = tuple(rgb)
#     if rgb_tuple in rgb_counts:
#         rgb_counts[rgb_tuple] += 1
#     else:
#         rgb_counts[rgb_tuple] = 1

# # # Rank RGB values by occurrence count
# ranked_rgb = sorted(rgb_counts.items(), key=lambda x: x[1], reverse=True)

# top_50_rgb = ranked_rgb[:50]
# for rank, (rgb, count) in enumerate(top_50_rgb, start=1):
#     # Exclude RGB values (255, 255, 255) and (0, 0, 0)
#     if rgb == (255, 255, 255) or rgb == (0, 0, 0):
#         continue

#     print(f"Rank: {rank}, RGB: {rgb}, Count: {count}")

#     mask = np.all(map_image_rgb == rgb, axis=-1)
#     # mask = cv2.inRange(img, np.array([rgb]), np.array([rgb]))
#     # output_image = np.zeros_like(map_image_rgb)
#     # output_image[mask] = map_image_rgb[mask]

#      # Convert the output image back to BGR format if needed
#     # output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
   
#     # denoised_image = cv2.fastNlMeansDenoising(output_image_bgr, None, 40, templateWindowSize=7, searchWindowSize=21)
#     # kernel = np.ones((3, 3), np.uint8)
#     # kernel1 = np.ones((5, 5), np.uint8)

#     # output_image_bgr = cv2.morphologyEx(output_image_bgr, cv2.MORPH_OPEN, kernel)
#     # output_image_bgr = cv2.morphologyEx(output_image_bgr, cv2.MORPH_CLOSE, kernel1)
#     # output_image_bgr = cv2.medianBlur(output_image_bgr, 5) 



# # Save the extracted color range

#     cv2.imwrite(f"image_quantization_segmentation_outputs/output_mask{rank}.jpg", mask)
 
    
