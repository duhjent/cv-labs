import cv2 as cv
import numpy as np
from noise import motion_blur, gaussian_blur
from filters import inverse_filter, wiener_filter, get_gaussian_kernel
from iter_methods import lucy_richardson_filter, van_cittert_algo
from metrics import PSNR, SSIM
import pandas as pd

img = cv.imread('../lab1/standard_test_images/cameraman.tif', cv.IMREAD_GRAYSCALE)
cv.imwrite('./out_images/orig.png', img)
# cv.imshow('orig', img)

img_blurred1 = img.copy()
img_blurred1 = motion_blur(img_blurred1, 45, (2, 0))
img_blurred1 = gaussian_blur(img_blurred1, 3)
cv.imwrite('./out_images/blurred1.png', img_blurred1)

img_blurred2 = img.copy()
img_blurred2 = motion_blur(img_blurred2, 45, (6, 0))
img_blurred2 = gaussian_blur(img_blurred2, 5)
cv.imwrite('./out_images/blurred2.png', img_blurred2)

wiener1 = wiener_filter(img_blurred1, get_gaussian_kernel(3), 1e-5)
inverse1 = inverse_filter(img_blurred1, get_gaussian_kernel(3), 1e-2)
van_cittert1 = van_cittert_algo(img_blurred1, n_iter=3, kernel_size=3)
lucy_richardson1 = lucy_richardson_filter(img_blurred1, np.ones((5, 5)) / 25, 10)

wiener2 = wiener_filter(img_blurred2, get_gaussian_kernel(3), 1e-5)
inverse2 = inverse_filter(img_blurred2, get_gaussian_kernel(3), 1e-2)
van_cittert2 = van_cittert_algo(img_blurred2, n_iter=3, kernel_size=3)
lucy_richardson2 = lucy_richardson_filter(img_blurred2, np.ones((5, 5)) / 25, 10)

cv.imwrite('./out_images/wiener1.png', wiener1)
cv.imwrite('./out_images/inverse1.png', inverse1)
cv.imwrite('./out_images/van_cittert1.png', van_cittert1)
cv.imwrite('./out_images/lucy_richardson1.png', lucy_richardson1)

cv.imwrite('./out_images/wiener2.png', wiener2)
cv.imwrite('./out_images/inverse2.png', inverse2)
cv.imwrite('./out_images/van_cittert2.png', van_cittert2)
cv.imwrite('./out_images/lucy_richardson2.png', lucy_richardson2)


# cv.imshow('wiener', wiener)
# cv.imshow('inverse', inverse)
# cv.imshow('van cittert', van_cittert)
# cv.imshow('lucy richardson', lucy_richardson)

metadata = [
    {'name': 'blur', 'img1': img_blurred1, 'img2': img_blurred2},
    {'name': 'wiener', 'img1': wiener1, 'img2': wiener2},
    {'name': 'inverse', 'img1': inverse1, 'img2': inverse2},
    {'name': 'van cittert', 'img1': van_cittert1, 'img2': van_cittert2},
    {'name': 'lucy-richardson', 'img1': lucy_richardson1, 'img2': lucy_richardson2},
]

data = [{'name': md['name'],
         'PSNR1': PSNR(img, md['img1']),
         'SSIM1': SSIM(img, md['img1']),
         'PSNR2': PSNR(img, md['img2']),
         'SSIM2': SSIM(img, md['img2'])}
           for md in metadata]

df = pd.DataFrame(data)
df.to_csv('./metrics.csv', index=False, float_format='%.3f')

# cv.waitKey(0)
