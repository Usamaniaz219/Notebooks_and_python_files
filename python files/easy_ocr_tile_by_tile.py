import cv2
import numpy as np
import easyocr

def perform_ocr_on_tile(tile, language='en'):
    reader = easyocr.Reader([language])  # Load the OCR model into memory
    result = reader.detect(tile)  # Perform text detection on the tile
    return result

def detect_text_in_tiles(image, tile_size):
    # Get the dimensions of the image
    image_height, image_width = image.shape[:2]

    # Calculate the number of tiles in the x and y directions
    num_tiles_x = int(np.ceil(image_width / tile_size))
    num_tiles_y = int(np.ceil(image_height / tile_size))

    # Initialize a list to store the bounding box coordinates
    bounding_boxes = []
    output_image = np.copy(image)

    # Iterate over each tile
    for tile_y in range(num_tiles_y):
        for tile_x in range(num_tiles_x):
            # Calculate the starting and ending coordinates of the current tile
            start_x = tile_x * tile_size
            end_x = min(start_x + tile_size, image_width)
            start_y = tile_y * tile_size
            end_y = min(start_y + tile_size, image_height)

            # Extract the current tile from the image
            tile = image[start_y:end_y, start_x:end_x]

            # Perform text detection on the current tile using the detection model
            reader = easyocr.Reader(['en'],gpu=True) # this needs to run only once to load the model into memory
            result = reader.readtext(tile,text_threshold=0.7,rotation_info=[90,180,270,360],y_ths=0.98,mag_ratio=0.6,
                                     canvas_size =3000,adjust_contrast=0.8,contrast_ths=0.1,min_size=13)
            #result = reader.readtext(tile)
            #print("result", result)
            # Check if any bounding boxes were returned
            if len(result) > 0:
                # Extract the bounding box coordinates and text from the result
                bounding_boxes_tile = result

                # Map the bounding box coordinates back to the original image coordinates
                for bbox ,text, confidence in  bounding_boxes_tile:
                    try:
                        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = bbox
                    except ValueError:
                        continue
                    bounding_boxes.append(bbox)
                    x1 += start_x
                    y1 += start_y
                    x2 += start_x
                    y2 += start_y
                    x3 += start_x
                    y3 += start_y
                    x4 += start_x
                    y4 += start_y

                    mapped_bbox =  [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    #print("Mapped BBox", mapped_bbox)
                    bounding_boxes.append(mapped_bbox)
                    
                    # Draw bounding box on the tile image
                    # Draw bounding box on the output image
                    points = [(int(x1), int(y1)), (int(x2), int(y2)), (int(x3), int(y3)), (int(x4), int(y4))]
                    rectangle = cv2.minAreaRect(np.array(points))
                    box = cv2.boxPoints(rectangle)
                    box = np.int0(box)
                    cv2.drawContours(output_image, [box], 0, (0, 0, 255), 2)


   
            #for bbox in bounding_boxes:
             ##   x2, y2 = bbox[1]
               #  x3, y3 = bbox[2]
                # x4, y4 = bbox[3]
                 #print("BBOX:",bbox) 
                 #x1, y1 = bbox[0]
                 #x3, y3 = bbox[2]  
                 # Create a rectangle using the given points
                # points = [(int(x1), int(y1)), (int(x2), int(y2)), (int(x3), int(y3)), (int(x4), int(y4))]
                 #rectangle = cv2.minAreaRect(np.array(points))
                 #box = cv2.boxPoints(rectangle)
                 #box = np.int0(box) 
                 #cv2.drawContours(tile, [box], 0, (0, 0, 255), 2)  
                 #cv2.rectangle(tile, (bbox[0], (bbox[1])), ((bbox[2], bbox[3]), (0, 0, 255), 2))
                 #cv2.putText(tile, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Show the tile with bounding box
                 #cv2.imshow('Tile with Bounding Box', tile)
                 #cv2.waitKey(0)
    #cv2.destroyAllWindows()
      # Show the image with bounding boxes
    cv2.imshow('Image with Bounding Boxes', output_image)
    cv2.imwrite('OUTput_tile.jpg',output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return bounding_boxes
image = cv2.imread('/home/usama/zoning_maps/090122-Zoning_11x17-1-scaled.jpg')
tile_size = 800
bounding_boxes = detect_text_in_tiles(image, tile_size)
print("Bounding Boxes", bounding_boxes)
