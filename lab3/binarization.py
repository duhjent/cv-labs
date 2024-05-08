import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def otsu_intraclass_variance(image, threshold):
    return np.nansum(
        [
            np.mean(cls) * np.var(image, where=cls)
            for cls in [image >= threshold, image < threshold]
        ]
    )

def otsu(img: np.ndarray) -> np.ndarray:
    otsu_threshold = min(
        range(np.min(img) + 1, np.max(img)),
        key=lambda th: otsu_intraclass_variance(img, th),
    )

    return cv.threshold(img, otsu_threshold, 255, cv.THRESH_BINARY)[1]

def compute_sharp_peak(img: np.ndarray) -> int:
    hist = cv.calcHist([img],[0],None,[256],[0,256])
    freq = (hist / hist.sum()).ravel()
    ph = []
    for k in range(256):
        if k == 0:
            p1, p2, n1, n2 = 255, 254, 1, 2
        elif k == 1:
            p1, p2, n1, n2 = 0, 255, 2, 3
        elif k == 254:
            p1, p2, n1, n2 = 253, 252, 255, 0
        elif k == 255:
            p1, p2, n1, n2 = 254, 253, 0, 1
        else:
            p1, p2, n1, n2 = k-1, k-2, k+1, k+2
        
        if freq[k] > max(freq[p1], freq[p2], freq[n1], freq[n2]):
            ph.append(freq[k])

    if len(ph) == 0:
        return 0

    ph = np.array(ph)

    result = len(ph[ph > ph.mean()])
    return result

def partition_img(img: np.ndarray, K) -> np.ndarray:
    w, h = img.shape
    parts = np.s_[:w//2,:h//2], np.s_[:w//2,h//2+1:], np.s_[w//2+1:,:h//2], np.s_[w//2+1:,h//2+1:]
    for p in parts:
        img_part = img[p]
        n = compute_sharp_peak(img_part)
        if n > 2:
            avg = img_part.mean()
            pr = len(img_part[img_part > avg]) / len(img_part[img_part < avg])
            pp = K * pr
            if w > pp and h > pp:
                img[p] = partition_img(img_part, K)
                continue

        img[p] = cv.threshold(img_part,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

    return img

# https://sci-hub.se/https://link.springer.com/article/10.1007/s00138-011-0402-4
def partitioning(img: np.ndarray, K) -> np.ndarray:
    n = compute_sharp_peak(img)
    if n <= 2:
        return cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    out = img.copy()

    partition_img(out, K)

    return out

test_img = cv.imread('../lab1/standard_test_images/cameraman.tif', cv.IMREAD_GRAYSCALE)

out = otsu(test_img)
out = cv.normalize(out, None, 0, 255, cv.NORM_MINMAX, cv.CV_32F)
# plt.imshow(out, cmap='gray')
# plt.axis('off')
# plt.show()
cv.imwrite(f'./out_images/test_otsu.png', out)

