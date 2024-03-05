from helpers import gamma_correction, open_from_web, visualize_image_with_hist, clahe, clahe_cv, fhh, histogram_equalization, histogram_equalization_cv
import matplotlib.pyplot as plt
import os
import cv2 as cv

# test_img_url = 'https://plantcv.readthedocs.io/en/stable/img/documentation_images/nonuniform_illumination/working_video.png'

# test_img = open_from_web(test_img_url)

# fig, axs = plt.subplots(1, 2)
# fig.suptitle('Test Original')
# visualize_image_with_hist(test_img, axs)
# plt.savefig('./out_images/test-orig.png')
# plt.close()

# test_img_he = histogram_equalization(test_img)
# test_img_he_cv = histogram_equalization_cv(test_img)

# test_img_clahe = clahe(test_img, 40, (8,8))
# test_img_clahe_cv = clahe_cv(test_img, 40, (8,8))

# test_img_fhh = fhh(test_img, .8)

# fig, axs = plt.subplots(1, 2)
# fig.suptitle('Test HE')
# visualize_image_with_hist(test_img_he, axs)
# plt.savefig('./out_images/test-he.png')
# plt.close()

# fig, axs = plt.subplots(1, 2)
# fig.suptitle('Test HE OpenCV')
# visualize_image_with_hist(test_img_he_cv, axs)
# plt.savefig('./out_images/test-he_cv.png')
# plt.close()

# fig, axs = plt.subplots(1, 2)
# fig.suptitle('Test CLAHE')
# visualize_image_with_hist(test_img_clahe, axs)
# plt.savefig('./out_images/test-clahe.png')
# plt.close()

# fig, axs = plt.subplots(1, 2)
# fig.suptitle('Test CLAHE OpenCV')
# visualize_image_with_hist(test_img_clahe_cv, axs)
# plt.savefig('./out_images/test-clahe_cv.png')
# plt.close()

# fig, axs = plt.subplots(1, 2)
# fig.suptitle('Test FHH')
# visualize_image_with_hist(test_img_fhh, axs)
# plt.savefig('./out_images/test-fhh.png')
# plt.close()

for file in os.listdir('./standard_test_images'):
    file_path = './standard_test_images/' + file
    title = file.split('.')[0]

    img = cv.imread(file_path)

    img_he = histogram_equalization(img)
    img_he_cv = histogram_equalization_cv(img)

    clahe_clip_limit = 40
    clahe_tile_grid_size = (8,8)
    img_clahe = clahe(img, clahe_clip_limit, clahe_tile_grid_size)
    img_clahe_cv = clahe_cv(img, clahe_clip_limit, clahe_tile_grid_size)

    fhh_beta = .8
    img_fhh = fhh(img, fhh_beta)

    fig, axs = plt.subplots(8, 2, figsize=(10, 50))

    visualize_image_with_hist(img, axs[0])
    visualize_image_with_hist(img_he, axs[1])
    visualize_image_with_hist(img_he_cv, axs[2])
    visualize_image_with_hist(img_clahe, axs[3])
    visualize_image_with_hist(img_clahe_cv, axs[4])
    visualize_image_with_hist(img_fhh, axs[5])
    visualize_image_with_hist(gamma_correction(img, .8), axs[6])
    visualize_image_with_hist(gamma_correction(img, 1.2), axs[7])

    plt.savefig(f'./out_images/singles/{title}.png')
    plt.close()
