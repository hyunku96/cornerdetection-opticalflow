import numpy as np
import cv2

class point:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)


def conv(img, filter):
    boundary = filter.shape[0] // 2
    output = np.zeros((img.shape[0] - boundary * 2, img.shape[1] - boundary * 2))
    for i in range(boundary, img.shape[0] - boundary):
        for j in range(boundary, img.shape[1] - boundary):
            area = img[i - boundary:i + boundary + 1, j - boundary:j + boundary + 1]
            sum = abs(np.sum(area * filter))
            if sum > sobel_thres:
                output[i - boundary, j - boundary] = sum

    return output

# read image
frame1 = cv2.imread('f1.png', cv2.IMREAD_GRAYSCALE)
frame2 = cv2.imread('f2.png', cv2.IMREAD_GRAYSCALE)
height, width = frame1.shape

# parameters of corner detection
sobel_thres = 0

# gradient(Sobel filter)
gx = np.array([[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]])
gy = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])

ix = conv(frame1, gx)
iy = conv(frame1, gy)
ixx = ix * ix
iyy = iy * iy
ixy = ix * iy

corner_thres = 0.001
k = 0.04
window_size = 3

keypoints = []
boundary = window_size//2
R = np.zeros((height-boundary*2 - 2, width-boundary*2 - 2))
for i in range(boundary, ixx.shape[0] - boundary):
    for j in range(boundary, ixx.shape[1] - boundary):
        ixx_sum = np.sum(ixx[i-boundary:i+boundary+1, j-boundary:j+boundary+1])
        ixy_sum = np.sum(ixy[i - boundary:i + boundary + 1, j - boundary:j + boundary + 1])
        iyy_sum = np.sum(iyy[i - boundary:i + boundary + 1, j - boundary:j + boundary + 1])
        det = ixx_sum * iyy_sum - ixy_sum**2
        trace = ixx_sum + iyy_sum
        R[i-boundary, j-boundary] = det - k * trace**2  # 왜 그런가?

output = cv2.imread('f1.PNG', cv2.IMREAD_ANYCOLOR)
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        if R[i,j] > corner_thres*np.max(R):
            cv2.line(output, (j+boundary*2+2, i+boundary*2+2), (j+boundary*2+2, i+boundary*2+2), (255, 0, 0), 3)
            keypoints.append(point(i+boundary*2, j+boundary*2))

print(len(keypoints))
cv2.imshow('output', output)
cv2.waitKey()
