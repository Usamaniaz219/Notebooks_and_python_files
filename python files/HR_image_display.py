import cv2

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        original_x = int(x * scale_factor)
        original_y = int(y * scale_factor)

        cv2.circle(high_resolution_image, (original_x, original_y), 5, (0, 255, 0), -1)
        cv2.putText(high_resolution_image, f'({x},{y})', (original_x + 10, original_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("High Resolution Image", high_resolution_image)


image_path = '/home/usama/zoning_maps/il_dixmoor(map on page 21).jpg'
high_resolution_image = cv2.imread(image_path)

screen_width, screen_height = 1920, 1080
scale_factor = min(high_resolution_image.shape[1] / screen_width, high_resolution_image.shape[0] / screen_height)
print("Scale factor:", scale_factor)

low_resolution_width = int(high_resolution_image.shape[1] / scale_factor)
low_resolution_height = int(high_resolution_image.shape[0] / scale_factor)
low_resolution_image = cv2.resize(high_resolution_image, (low_resolution_width, low_resolution_height))

cv2.namedWindow("High Resolution Image", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("High Resolution Image", mouse_callback)

cv2.circle(low_resolution_image, (0, 0), 5, (0, 255, 0), 2)
cv2.putText(low_resolution_image, "", (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv2.imshow("High Resolution Image", low_resolution_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

