import cv2 as cv
import numpy as np

def van_cittert_algo(img: np.ndarray, n_iter: int=5, kernel_size: int=3, sigma: float=0.1) -> np.ndarray:
    result = np.zeros(img.shape, dtype=np.float32)
    for _ in range(n_iter):
        laplacian = cv.Laplacian(result, cv.CV_32F, ksize=kernel_size)
        result = result + sigma*(img - laplacian)
    
    return (255*(result - np.min(result))/np.ptp(result)).astype(np.uint8)

def lucy_richardson_filter(img: np.ndarray, kernel: np.ndarray, n_iter: int) -> np.ndarray:
    result: np.ndarray = np.ones(img.shape)
    kernel_transpose = np.flip(kernel)
    for _ in range(n_iter):
        est_img = cv.filter2D(result, -1, kernel)
        ratio = img / est_img
        correction_factor = cv.filter2D(ratio, -1, kernel_transpose)
        result = result * correction_factor

    return (255*(result - np.min(result))/np.ptp(result)).astype(np.uint8)
