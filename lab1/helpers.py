import requests
import numpy as np
import cv2 as cv
import matplotlib
from typing import List
import math

def open_from_web(url):
    resp = requests.get(url)
    arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
    img = cv.imdecode(arr, 1)

    return img

def visualize_image_with_hist(img, axs):
    axs[0].axis('off')
    axs[0].imshow(img)

    axs[1].hist(img.flatten(), 256, [0, 256])

def histogram_equalization(img):
    l, a, b = cv.split(cv.cvtColor(img, cv.COLOR_BGR2LAB))
    hist, _ = np.histogram(l.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_m = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_m = cdf_m.astype('uint8')
    l_new = cdf_m[l]
    return cv.cvtColor(cv.merge((l_new, a, b)), cv.COLOR_LAB2BGR)

def histogram_equalization_cv(img):
    l, a, b = cv.split(cv.cvtColor(img, cv.COLOR_BGR2LAB))
    l_new = cv.equalizeHist(l)
    return cv.cvtColor(cv.merge((l_new, a, b)), cv.COLOR_LAB2BGR)

def clahe(img, clip_limit, tile_grid_size):
    img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(img_lab)

    x_tile_size = l.shape[0] / tile_grid_size[0]
    y_tile_size = l.shape[1] / tile_grid_size[1]

    x_tiles = [0] + [(i + 1) * math.floor(x_tile_size) - 1 for i in range(tile_grid_size[0] - 1)] + [l.shape[0] - 1]
    y_tiles = [0] + [(i + 1) * math.floor(y_tile_size) - 1 for i in range(tile_grid_size[1] - 1)] + [l.shape[1] - 1]

    grid = [[0]*tile_grid_size[0]] * tile_grid_size[1]

    for i in range(1, len(x_tiles)):
        for j in range(1, len(y_tiles)):
            grid[i-1][j-1] = l[x_tiles[i-1]:x_tiles[i], y_tiles[j-1]:y_tiles[j]]

    l_new = l
    return cv.cvtColor(cv.merge((l_new, a, b)), cv.COLOR_LAB2BGR)

def clahe_cv(img, clip_limit, tile_grid_size):
    l, a, b = cv.split(cv.cvtColor(img, cv.COLOR_BGR2LAB))
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_new = clahe.apply(l)

    return cv.cvtColor(cv.merge((l_new, a, b)), cv.COLOR_LAB2BGR)

# https://www.inase.org/library/2015/vienna/bypaper/BICHE/BICHE-09.pdf
def fhh(img, beta):
    c = (255.)/(math.exp(-1.) - 1.)
    l, a, b = cv.split(cv.cvtColor(img, cv.COLOR_BGR2LAB))

    g = cv.normalize(l, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
    g = cv.pow(g, beta, None)
    g = cv.exp(-g, None)
    g = (g - 1) * c
    g = g.astype('uint8')

    img_out = cv.merge((g, a, b))
    img_out = cv.cvtColor(img_out, cv.COLOR_LAB2BGR)

    return img_out

def gamma_correction(img, gamma):
    g = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, )
    g = g ** (1/gamma)
    g *= 255
    g = g.astype('uint8')

    return g

