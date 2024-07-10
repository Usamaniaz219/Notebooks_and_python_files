import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

def quantize_image_with_gmm(image, n_colors):
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape((-1, 3))

    # Fit a Gaussian Mixture Model with the desired number of components (colors)
    gmm = GaussianMixture(n_components=n_colors, random_state=0)
    gmm.fit(pixels)

    # Get the labels (color indices) for each pixel
    labels = gmm.predict(pixels)

    # Get the means (colors) for each label
    colors = gmm.means_.astype(int)

    # Replace each pixel with its quantized color
    quantized_image = colors[labels].reshape(image.shape)

    return quantized_image

if __name__ == "__main__":
    # Load an image
    image = cv2.imread("image_vectorization/1/zoningmapdec14-page-001.jpg")  # Replace with your image path

    # Define the number of colors (clusters) for quantization
    n_colors = 16

    # Perform image color quantization using GMM
    quantized_image = quantize_image_with_gmm(image, n_colors)

    # Display the original and quantized images
    # cv2.imshow("Original Image", image)
    # cv2.imshow("Quantized Image", quantized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Save the quantized image
    cv2.imwrite("image_quantization_segmentation_outputs/quantized_img_GMM_16.jpg", quantized_image)




























# import cv2
# import numpy as np
# from sklearn.mixture import GaussianMixture
# import time

# # Read input image
# input_image_path = 'image_vectorization/1/zoningmapdec14-page-001.jpg'
# img = cv2.imread(input_image_path)

# # Reshape the image into a feature vector of pixel values
# pixel_values = img.reshape(-1, 3)

# # Convert to np.float32
# pixel_values = np.float32(pixel_values)

# # Create a Gaussian Mixture Model (GMM) with the desired number of components (clusters)
# num_components = 31  # Adjust this based on your preference
# gmm = GaussianMixture(n_components=num_components, covariance_type='full')

# # Measure the start time
# start_time = time.time()

# # Fit the GMM to the pixel values
# gmm.fit(pixel_values)

# # Predict the labels for each pixel
# labels = gmm.predict(pixel_values)

# # Get the RGB values of the GMM cluster centers
# cluster_centers = np.uint8(gmm.means_)

# # Create an output image with the segmented colors
# segmented_img = cluster_centers[labels.flatten()].reshape(img.shape)

# # Measure the end time
# end_time = time.time()

# # Calculate the time taken for the process
# elapsed_time = end_time - start_time
# print(f"Time taken: {elapsed_time:.2f} seconds")

# # Save or display the segmented image
# cv2.imwrite("segmented_image.jpg", segmented_img)

# # Automatically generate masks for each cluster
# for cluster_id in range(num_components):
#     mask = (labels == cluster_id).reshape(img.shape[:2]).astype(np.uint8) * 255
#     cv2.imwrite(f"images1/mask{cluster_id}.jpg", mask)
