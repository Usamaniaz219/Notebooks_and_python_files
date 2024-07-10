import cv2

def draw_rectangles(image_path):
    image = cv2.imread(image_path)
    cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
    cv2.imshow('Image', image)
    rectangle_coordinates = []
    def draw_rectangle_callback(event, x, y, flags, param):
        nonlocal rectangle_coordinates
        if event == cv2.EVENT_LBUTTONDOWN:
            rectangle_coordinates.append([(x, y)])

        elif event == cv2.EVENT_LBUTTONUP:
            rectangle_coordinates[-1].append((x, y))
            cv2.rectangle(image, rectangle_coordinates[-1][0], rectangle_coordinates[-1][1], (0, 255, 0), 2)
            cv2.imshow('Image', image)
    cv2.setMouseCallback('Image', draw_rectangle_callback)

    while True:
        cv2.imshow('Image', image)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc key
            break
    cv2.destroyAllWindows()
    return rectangle_coordinates

image_path = "/home/usama/zoning_maps/il_dixmoor(map on page 21).jpg"
rectangle_coordinates = draw_rectangles(image_path)

for i, coordinates in enumerate(rectangle_coordinates):
    x1, y1 = coordinates[0][0], coordinates[0][1]
    x2, y2 = coordinates[1][0], coordinates[0][1]
    x3, y3 = coordinates[1][0], coordinates[1][1]
    x4, y4 = coordinates[0][0], coordinates[1][1]
    print(f'{x1}, {y1}, {x2}, {y2}, {x3}, {y3}, {x4}, {y4}')
    #print(f'Rectangle {i+1} Coordinates: {x1}, {y1}, {x2}, {y2}, {x3}, {y3}, {x4}, {y4}')

