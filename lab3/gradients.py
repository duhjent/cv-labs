import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def roberts_cross(img):
    kernel1 = np.array([[1, 0], [0, -1]])
    kernel2 = np.array([[0, 1], [-1, 0]])

    img1 = cv.filter2D(img, -1, kernel1)
    img2 = cv.filter2D(img, -1, kernel2)

    z = np.sqrt(img1**2 + img2**2)

    return z

def kirsh(img):
    g = []
    g.append(np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]))
    g.append(np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]))
    g.append(np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]))
    g.append(np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]))
    g.append(np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]))
    g.append(np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]))
    g.append(np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]))
    g.append(np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]))

    outs = [cv.filter2D(img, -1, g_i) for g_i in g]

    outs = np.stack(outs, axis=2)

    out = outs.max(axis=2)

    out = cv.normalize(out, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)

    return out

def sobel_filter(img):
    kernel1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernel2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    img1 = cv.filter2D(img, -1, kernel1)
    img2 = cv.filter2D(img, -1, kernel2)

    out = np.sqrt(img1**2 + img2**2)

    out = cv.normalize(out, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)

    return out

def prewitt_filter(img):
    kernel1 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernel2 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    img1 = cv.filter2D(img, -1, kernel1)
    img2 = cv.filter2D(img, -1, kernel2)

    out = np.sqrt(img1**2 + img2**2)

    out = cv.normalize(out, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)

    return out

def laplace_filter(img):
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    out = cv.filter2D(img, -1, kernel)
    out[out < 0] = 0

    out = cv.normalize(out, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)

    return out

def canny(img, treshold1, treshold2):
    return cv.Canny(np.uint8(img * 255.0), treshold1, treshold2) / 255.0


test_img = cv.imread('../lab1/standard_test_images/cameraman.tif', cv.IMREAD_GRAYSCALE)
x = test_img / 255.0

out = laplace_filter(x)
out = cv.normalize(out, None, 0, 255, cv.NORM_MINMAX, cv.CV_32F)
# plt.imshow(out, cmap='gray')
# plt.show()
cv.imwrite('./out_images/test_laplace.png', out)
