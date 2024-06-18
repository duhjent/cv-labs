from read import read_mnist
from skimage.util.shape import view_as_blocks
import matplotlib.pyplot as plt

images, labels = read_mnist('./mnist-csv/mnist_train.csv')

fig, axs = plt.subplots(4, 2)
fig.subplots_adjust(wspace=0, hspace=0)
# fig.tight_layout()
i = 0

for row in axs:
    for ax in row:
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')
        i += 1

# plt.show()
plt.savefig('./examples.png')
