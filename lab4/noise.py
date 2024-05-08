import cv2 as cv
import numpy as np

def motion_blur(img: np.ndarray, angle: int, axes: tuple[int, int]) -> np.ndarray:
    psf = np.zeros((50, 50, 3))
    psf = cv.ellipse(psf, 
                    (25, 25), # center
                    axes, # axes -- 22 for blur length, 0 for thin PSF 
                    angle, # angle of motion in degrees
                    0, 360, # ful ellipse, not an arc
                    (1, 1, 1), # white color
                    thickness=-1) # filled

    psf /= psf[:,:,0].sum() # normalize by sum of one channel 
                            # since channels are processed independently

    return cv.filter2D(img, -1, psf)

def gaussian_blur(img: np.ndarray, radius: int) -> np.ndarray:
    return cv.GaussianBlur(img, (radius, radius), 0)
