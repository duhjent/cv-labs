from helpers import open_from_web, visualize_image_with_hist, clahe, clahe_cv, histogram_equalization, histogram_equalization_cv
import matplotlib.pyplot as plt

test_img_url = 'https://plantcv.readthedocs.io/en/stable/img/documentation_images/nonuniform_illumination/working_video.png'

test_img = open_from_web(test_img_url)

fig, axs = plt.subplots(1, 2)
fig.suptitle('Test Original')
visualize_image_with_hist(test_img, axs)
plt.savefig('./out_images/test-orig.png')

test_img_he = histogram_equalization(test_img)
test_img_he_cv = histogram_equalization_cv(test_img)

test_img_clahe = clahe(test_img, 40, (8,8))
test_img_clahe_cv = clahe_cv(test_img, 40, (8,8))

fig, axs = plt.subplots(1, 2)
fig.suptitle('Test HE')
visualize_image_with_hist(test_img_he, axs)
plt.savefig('./out_images/test-he.png')

fig, axs = plt.subplots(1, 2)
fig.suptitle('Test HE OpenCV')
visualize_image_with_hist(test_img_he_cv, axs)
plt.savefig('./out_images/test-he_cv.png')

fig, axs = plt.subplots(1, 2)
fig.suptitle('Test CLAHE')
visualize_image_with_hist(test_img_clahe, axs)
plt.savefig('./out_images/test-clahe.png')

fig, axs = plt.subplots(1, 2)
fig.suptitle('Test CLAHE OpenCV')
visualize_image_with_hist(test_img_clahe_cv, axs)
plt.savefig('./out_images/test-clahe_cv.png')
