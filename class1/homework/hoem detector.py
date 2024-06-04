import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("Home.jpg",0)

print(img.shape)

kernel_chap = np.array([[1, -1],
                        [1,-1]])
kernel_rast = np.array([[-1, 1],
                    [-1,1]])
kernel_paein = np.array([[-1, -1],
                    [1,1]])
kernel_bala = np.array([[1, 1],
                    [-1,-1]])
result1 = cv.filter2D(img, -1, kernel_chap)
result2 = cv.filter2D(img, -1, kernel_rast)
result3 = cv.filter2D(img, -1, kernel_paein)
result4 = cv.filter2D(img, -1, kernel_bala)

result = (result1 + result2 + result3 + result4) / 4  # Average the results

plt.imshow(result)
plt.show()