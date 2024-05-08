import cv2 as cv
import numpy as np
from noise import motion_blur, gaussian_blur
from filters import inverse_filter, wiener_filter, get_gaussian_kernel
from iter_methods import lucy_richardson_filter, van_cittert_algo
from metrics import PSNR, SSIM
import pandas as pd

img = cv.imread('../lab1/standard_test_images/cameraman.tif', cv.IMREAD_GRAYSCALE)
cv.imwrite('./out_images/orig.png', img)
cv.imshow('orig', img)

img_blurred1 = img.copy()
img_blurred1 = motion_blur(img_blurred1, 45, (2, 0))
img_blurred1 = gaussian_blur(img_blurred1, 3)
cv.imwrite('./out_images/blurred1.png', img)

img_blurred2 = img.copy()
img_blurred2 = motion_blur(img_blurred2, 45, (6, 0))
img_blurred2 = gaussian_blur(img_blurred2, 5)
cv.imwrite('./out_images/blurred2.png', img)

exit()


wiener = wiener_filter(img_blurred, get_gaussian_kernel(3), 1e-5)
inverse = inverse_filter(img_blurred, get_gaussian_kernel(3), 1e-2)
van_cittert = van_cittert_algo(img_blurred, n_iter=3, kernel_size=3)
lucy_richardson = lucy_richardson_filter(img_blurred, np.ones((5, 5)) / 25, 10)

# cv.imshow('wiener', wiener)
# cv.imshow('inverse', inverse)
cv.imshow('van cittert', van_cittert)
cv.imshow('lucy richardson', lucy_richardson)

metadata = [
    {'name': 'wiener', 'img': wiener},
    {'name': 'inverse', 'img': inverse},
    {'name': 'van cittert', 'img': van_cittert},
    {'name': 'lucy-richardson', 'img': lucy_richardson},
]

data = [{'name': md['name'], 'PSNR': PSNR(img, md['img']), 'SSIM': SSIM(img, md['img'])} for md in metadata]

print(pd.DataFrame(data))

cv.waitKey(0)
