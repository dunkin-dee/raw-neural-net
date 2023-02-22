import cv2
import numpy as np
np.set_printoptions(linewidth=200)
image_data = cv2.imread('fashion_mnist_images/train/1/0050.png',
  cv2.IMREAD_UNCHANGED)
import matplotlib.pyplot as plt
plt.imshow(image_data)
plt.show()

