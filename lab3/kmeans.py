import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.cluster import KMeans

def kmeans(img: np.ndarray, k: int, sample_size=1000) -> np.ndarray:
    image_array = img.reshape(-1, 1)
    image_array_sample = shuffle(image_array, n_samples=sample_size)
    kmeans = KMeans(n_clusters=k).fit(image_array_sample)
    labels = kmeans.predict(image_array)
    out = kmeans.cluster_centers_[labels].reshape(img.shape)

    return out

test_img = cv.imread('../lab1/standard_test_images/cameraman.tif', cv.IMREAD_GRAYSCALE)

for k in [2, 4, 8, 10, 15]:
    out = kmeans(test_img, k)
    out = cv.normalize(out, None, 0, 255, cv.NORM_MINMAX, cv.CV_32F)
    plt.imshow(out, cmap='gray')
    plt.show()
    cv.imwrite(f'./out_images/test_kmeans_{k}.png', out)

