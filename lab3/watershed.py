import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def watershed(img: np.ndarray) -> np.ndarray:
    ...

test_img = cv.imread('../lab1/standard_test_images/cameraman.tif', cv.IMREAD_GRAYSCALE)

out = watershed(test_img)
out = cv.normalize(out, None, 0, 255, cv.NORM_MINMAX, cv.CV_32F)
plt.imshow(out, cmap='gray')
plt.show()
cv.imwrite(f'./out_images/test_watershed.png', out)

