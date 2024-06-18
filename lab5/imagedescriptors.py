import cv2 as cv
import numpy as np

hog = cv.HOGDescriptor(_winSize=(28, 28),
                       _blockSize=(14, 14),
                       _blockStride=(7, 7),
                       _cellSize=(7, 7),
                       _nbins=9)


def extract_hog_features_opencv(images: np.ndarray) -> np.ndarray:
    hog_features = []
    for image in images:
        image = image.astype(np.uint8)
        features = hog.compute(image)
        hog_features.append(features.flatten())
    return np.array(hog_features)
