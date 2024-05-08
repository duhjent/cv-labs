import cv2 as cv
import numpy as np

img = cv.imread('./standard_test_images/lena_color_256.tif', cv.IMREAD_COLOR)

zeros = np.zeros((img.shape[0], img.shape[1]))

red = np.stack([img[:,:,0], zeros.copy(), zeros.copy()], axis=2)
green = np.stack([zeros.copy(), img[:,:,1], zeros.copy()], axis=2)
blue = np.stack([zeros.copy(), zeros.copy(), img[:,:,2]], axis=2)

cv.imwrite('./percolor/red.png', red)
cv.imwrite('./percolor/green.png', green)
cv.imwrite('./percolor/blue.png', blue)
cv.imwrite('./percolor/zorig.png', img)
