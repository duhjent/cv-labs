from helpers import histogram_equalization, histogram_equalization_cv, clahe_cv, fhh, gamma_correction
import cv2 as cv
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity

def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal.
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def SSIM(original, processed):
    original_grayscale = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
    processed_grayscale = cv.cvtColor(processed, cv.COLOR_BGR2GRAY)

    return structural_similarity(original_grayscale, processed_grayscale, data_range=processed_grayscale.max() - processed_grayscale.min())

img = cv.imread('./standard_test_images/cameraman.tif')


img_he = histogram_equalization(img)
img_he_cv = histogram_equalization_cv(img)
img_clahe1 = clahe_cv(img, 40, (8, 8))
img_clahe2 = clahe_cv(img, 60, (8, 8))
img_fhh1 = fhh(img, .4)
img_fhh2 = fhh(img, .8)
img_gc1 = gamma_correction(img, .8)
img_gc2 = gamma_correction(img, 1.2)

data = [
    {'name': 'HE (custom)', 'PSNR': PSNR(img, img_he), 'SSIM': SSIM(img, img_he)},
    {'name': 'HE (OpenCV)', 'PSNR': PSNR(img, img_he_cv), 'SSIM': SSIM(img, img_he_cv)},
    {'name': 'CLAHE (40, 8x8)', 'PSNR': PSNR(img, img_clahe1), 'SSIM': SSIM(img, img_clahe1)},
    {'name': 'CLAHE (60, 8x8)', 'PSNR': PSNR(img, img_clahe2), 'SSIM': SSIM(img, img_clahe2)},
    {'name': 'FHH (beta=.4)', 'PSNR': PSNR(img, img_fhh1), 'SSIM': SSIM(img, img_fhh1)},
    {'name': 'FHH (beta=.8)', 'PSNR': PSNR(img, img_fhh2), 'SSIM': SSIM(img, img_fhh2)},
    {'name': 'Gamma Correction (.8)', 'PSNR': PSNR(img, img_gc1), 'SSIM': SSIM(img, img_gc1)},
    {'name': 'Gamma Correction (1.2)', 'PSNR': PSNR(img, img_gc2), 'SSIM': SSIM(img, img_gc2)},
]

df = pd.DataFrame(data)
df.to_csv('./metrics.csv')

# print(pd.DataFrame(data).to_string(index=False))
