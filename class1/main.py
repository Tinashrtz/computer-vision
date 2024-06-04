import numpy as np
from matplotlib import pyplot as plt

img = plt.imread("face.jpg")

height, width, channel = img.shape

img2 = img[200:, 300:, :]

print(img[200, 300, :])

# Gray Scale (r+g+b) / 3
grayscale (0.299 * img[:,:,0]) + (0.587 ∙ img[:,:,1]) + (0.114 ∙ img[:,:,2])

plt.hist(img[:,:,0])
plt.imshow(img[:,:,2],cmap="gray")

img = img[:,:,0]

img[img>127] = 255
img[img<=127] = 0


plt.imshow(img[:,:,0],cmap="summer")
plt.show()
