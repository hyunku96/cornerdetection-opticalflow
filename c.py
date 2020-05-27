import cv2
import numpy as np

f1 = cv2.imread('f1.PNG', cv2.IMREAD_ANYCOLOR)
f2 = cv2.imread('f2.png', cv2.IMREAD_ANYCOLOR)

h1, w1 = f1.shape[0], f1.shape[1]
h2, w2 = f2.shape[0], f2.shape[1]
h = min(h1, h2)
w = min(w1, w2)
print(h, w)
frame1 = np.zeros((h, w, 3))
frame2 = np.zeros((h, w, 3))
for i in range(h):
    for j in range(w):
        frame1[i][j] = f1[i][j]
        frame2[i][j] = f2[i][j]

cv2.imwrite('f1.PNG', frame1)
cv2.imwrite('f2.PNG', frame2)