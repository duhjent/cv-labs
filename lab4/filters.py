import cv2 as cv
import numpy as np

def get_gaussian_kernel(size: int) -> np.ndarray:
    empty = np.zeros((size, size))
    empty[size//2, size//2] = 1
    return cv.GaussianBlur(empty, (size, size), 0)

def wiener_filter(img: np.ndarray, kernel: np.ndarray, K: float) -> np.ndarray:
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    noise_var = np.var(img)    
    img_fft = np.fft.fft2(img)
    kernel_fft = np.fft.fft2(kernel, s=img.shape)
    kernel_fft_conj = np.conj(kernel_fft)
    kernel_norm = np.abs(kernel_fft) ** 2
    img_fft_filtered = kernel_fft_conj * img_fft / (kernel_norm + K*noise_var/kernel_norm)
    img_filtered = np.real(np.fft.ifft2(img_fft_filtered))
    img_filtered = np.clip(img_filtered, 0, 255)
    img_filtered = img_filtered.astype(np.uint8)
    return img_filtered


def inverse_filter(img: np.ndarray, kernel: np.ndarray, eps: float) -> np.ndarray:
    fft_image = np.fft.fft2(img)
    fft_kernel = np.fft.fft2(kernel, s=img.shape)
    fft_kernel = np.conj(fft_kernel) / (np.abs(fft_kernel)**2 + eps)

    result = np.fft.ifft2(fft_image * fft_kernel)
    result = np.real(result)

    return np.clip(result, 0, 255).astype(np.uint8)

