import cv2
import numpy as np

# Global variables
ref_points = []
background_points = []
foreground_points = []

def mouse_callback(event, x, y, flags, param):
    global image,ref_points, background_points, foreground_points

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image", image)

def select_points(image_path):
    global image, ref_points,background_points, foreground_points

    image = cv2.imread(image_path)
    cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Image", mouse_callback)

    while True:
        cv2.imshow("Image", image)
        key = cv2.waitKey(1) & 0xFF

        # Press 'r' to reset the points
        if key == ord("r"):
            image = cv2.imread(image_path)
            ref_points = []
            background_points = []
            foreground_points = []

        # Press 'b' to add a background point
        elif key == ord("b") and ref_points:
            background_points.append(ref_points[-1])
            cv2.circle(image, ref_points[-1], 3, (0, 0, 255), -1)
            cv2.imshow("Image", image)

        # Press 'f' to add a foreground point
        elif key == ord("f") and ref_points:
            foreground_points.append(ref_points[-1])
            cv2.circle(image, ref_points[-1], 3, (0, 255, 0), -1)
            cv2.imshow("Image", image)

        # Press 'q' to quit the program
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    return background_points, foreground_points

# Example usage
image_path = "/home/usama/usama_dev_test/Stroke-Based-Scene-Text-Erasing/example/_inference_result/img_393.png"
background_points, foreground_points = select_points(image_path)

print("Background Points:", background_points)
print("Foreground Points:", foreground_points)





