
import cv2
import numpy as np
import time
from imutils.object_detection import non_max_suppression
import os


def scale_up_rectangle(x1, y1, x2, y2, scale_factor):
    # Calculate the width and height of the original rectangle
    width = x2 - x1
    height = y2 - y1

    # Calculate the new width and height 
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Calculate the new coordinates of the  rectangle
    new_x1 = x1 - int((new_width - width) / 2)
    new_y1 = y1 - int((new_height - height) / 2)
    new_x2 = new_x1 + new_width
    new_y2 = new_y1 + new_height

    return new_x1, new_y1, new_x2, new_y2

def transform_image(input_path, output_path, tile_width, tile_height, model):
    # Load the original image
    original_image = cv2.imread(input_path)


    input_height, input_width =     original_image.shape[:2]
    new_width = ((input_width - 1) // 320 + 1) * 320
    new_height =((input_height - 1) // 320 + 1) * 320
    # Calculate the amount of padding required
    pad_width = new_width - input_width
    pad_height = new_height - input_height


    # Calculate the padding amounts for each side
    top = pad_height // 2
    
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left

    print("top",top)
    print("bottom",bottom)
    print("left",left)
    print("right",right)

    original_image = cv2.copyMakeBorder(original_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Calculate the number of rows and columns
    num_rows = original_image.shape[0] // tile_height
    num_cols = original_image.shape[1] // tile_width

    # Create an empty list to store the tiles
    tiles = []

    # Iterate through the image and extract each tile
    for row in range(num_rows):
        for col in range(num_cols):
            x = col * tile_width
            y = row * tile_height
            tile = original_image[y:y+tile_height, x:x+tile_width]
            tiles.append(tile)

    # Perform some processing on each tile
    processed_tiles = []
    for tile in tiles:
        # Pass the tile through your model for processing
        processed_tile = model(tile)
        processed_tiles.append(processed_tile)

    # Create an empty canvas with the same size as the original image
    transformed_img = np.zeros_like(original_image)

    # Iterate through the processed tiles and place them back onto the transform_img
    for row in range(num_rows):
        for col in range(num_cols):
            x = col * tile_width
            y = row * tile_height
            processed_tile = processed_tiles[row * num_cols + col]
            transformed_img[y:y+tile_height, x:x+tile_width] = processed_tile

    padded_height, padded_width = transformed_img.shape[:2]

    # Set the original image dimensions
    original_width = 2750
    original_height = 2125

    # Calculate the amount of padding
    padding_width = padded_width - original_width
    padding_height = padded_height - original_height

    # Check if padding exists
    if padding_width < 0 or padding_height < 0:
        print("Padded image dimensions are smaller than the original image. No padding to remove.")
    else:
        # Calculate the cropping boundaries
        left_padding = padding_width // 2
        right_padding = padded_width - left_padding
        top_padding = padding_height // 2
        bottom_padding = padded_height - top_padding

        # Crop the padded image to remove the padding
        transformed_img = transformed_img[top_padding:bottom_padding, left_padding:right_padding]
       
    # Save the transformed image
    cv2.imwrite(output_path, transformed_img)

def detect_text(image):
    orig = image.copy()
    mask = np.zeros_like(image)
    layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]
    net = cv2.dnn.readNet('frozen_east_text_detection.pb')
   
    blob = cv2.dnn.blobFromImage(image, 1.0, (320, 320),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()
  
    print("[INFO] text detection took {:.6f} seconds".format(end - start))
  
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        
        xData0 = geometry[0, 0, y]
        #print("xdata",xData0)
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
 
            if scoresData[x] < 0.5:
                continue
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    boxes = non_max_suppression(np.array(rects), probs=confidences)
 
    for (startX, startY, endX, endY) in boxes:

        startX, startY, endX, endY = scale_up_rectangle(startX, startY, endX, endY, 1.65)

        # Ensure startX is within the allowed range of 0 to 320
        startX = int(max(0, startX))
        startX = int(min(startX, 320))

        # Ensure startY is within the allowed range of 0 to 320
        startY = int(max(0, startY))
        startY = int(min(startY, 320))

        # Ensure endX is within the allowed range of 0 to 320
        endX = int(max(0, endX))
        endX = int(min(endX, 320))

        # Ensure endY is within the allowed range of 0 to 320
        endY = int(max(0, endY))
        endY = int(min(endY, 320))

        print("new_startX",startX)
        print("new_startY",startY)
        print("new_endX",endX)
        print("new_endY",endY)
        cv2.rectangle(mask, (startX, startY), (endX, endY), (255, 255, 255), -1)
        gray_patch = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray_patch, 0, 255, cv2.THRESH_BINARY)[1]
        cv2.imwrite('downsampled_image_masked05.png', thresh)
       
        #cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)
        #cv2.imwrite('tiled_out_images/tiled_out_images.jpg',orig)

    return mask



transform_image("/home/usama/usama-dev/image_super_resolution_models/Usama_SWINIR/SwinIR/input/mo_ash_grove.jpg", 'transformed_image11.jpg', 320, 320, detect_text)

