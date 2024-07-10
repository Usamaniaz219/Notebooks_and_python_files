import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle

def vectorize_zone(index: int ):
        #with open('/home/usama/Downloads/labelled_image', 'rb') as file:
         #   labelled_image = pickle.load(file)
        labelled_image=cv2.imread('segmented_image.jpg')
        color_index_mat = np.array([index], np.uint8)
        #i5=plt.figure(5)
        #plt.imshow(labelled_image)
        #i5.show()
        mask = cv2.compare(labelled_image, color_index_mat, cv2.CMP_EQ)
        cv2.imwrite('mean__mask.jpg', mask)
        #i6=plt.figure(6)
        #plt.imshow(mask)
        #i6.show()
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        #i7=plt.figure(7)
        #plt.imshow(closing)
        #i7.show()
        cv2.imwrite('closing.jpg', closing)

        #blurred = cv2.blur(closing, (10, 10))

        #_, threshed = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)

        ## for test purpose
        #i1=plt.figure(1)
        #plt.imshow(threshed)
        #i1.show()
        
        threshed = cv2.dilate(closing, np.ones((5, 5)), iterations=4)
        threshed = cv2.erode(threshed, np.ones((3, 3)), iterations=3)
        cv2.imshow('extracted image',threshed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('threshed.jpg', threshed)
        
        #threshed = cv2.dilate(threshed, np.ones((5, 5)), iterations=4)
        #i2=plt.figure(2)
        #plt.imshow(threshed)
        #i2.show()
        #threshed = cv2.erode(threshed, np.ones((7, 7)), iterations=3)
        #i3=plt.figure(3)
        #plt.imshow(threshed)
        #i3.show()
        #threshed = cv2.erode(threshed, np.ones((3, 3)), iterations=1)
        #i4=plt.figure(4)
        #plt.imshow(threshed)
       
        #i4.show()
        
        #return threshed
# 0,21,42,63,84,105,126,147,168,189,210,231,252


vectorize_zone(126)