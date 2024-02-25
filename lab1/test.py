from helpers import open_from_web, visualize_image_with_hist
import matplotlib.pyplot as plt

img = open_from_web('https://plantcv.readthedocs.io/en/stable/img/documentation_images/nonuniform_illumination/corrected_img.jpg')

f, axs = plt.subplots(1, 2)
f.tight_layout()

visualize_image_with_hist(img, axs)

plt.savefig('./test.png')
