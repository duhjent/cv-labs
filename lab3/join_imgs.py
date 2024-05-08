import cv2 as cv
import matplotlib.pyplot as plt

imgs = []

for k in [10, 20, 40, 50, 100]:
    imgs.append(cv.imread(f'./out_images/test_part_{k}.png', cv.IMREAD_GRAYSCALE))

fig, axs = plt.subplots(2, 3)
fig.tight_layout()

for i, img in enumerate(imgs):
    axs[i%2][i//2].imshow(img, cmap='gray')
    axs[i%2][i//2].axis('off')

axs[-1][-1].axis('off')

plt.savefig('./out_images/test_part_join.png')
plt.show()
