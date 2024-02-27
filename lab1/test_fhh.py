import cv2 as cv
import math
import matplotlib.pyplot as plt
import os

img = cv.imread('./standard_test_images/cameraman.tif')
beta = .4
l, a, b = cv.split(cv.cvtColor(img, cv.COLOR_BGR2LAB))

g = cv.normalize(l, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
g = cv.pow(g, beta, None)
g = cv.exp(-g, None)
c = (255.)/(math.exp(-1.) - 1.)
g = (g - 1) * c
g = g.astype('uint8')
# print(g.min(), g.max())
# plt.imshow(g, cmap='gray')
# plt.show()

img_out = cv.merge((g, a, b))
img_out = cv.cvtColor(img_out, cv.COLOR_LAB2BGR)

plt.imshow(img_out)
plt.show()

