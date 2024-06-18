import numpy as np
import cv2 as cv
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
    if len(original.shape) != 2:
        original = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
        processed = cv.cvtColor(processed, cv.COLOR_BGR2GRAY)

    return structural_similarity(original, processed, data_range=processed.max() - processed.min())
