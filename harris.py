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
            output[i - boundary, j - boundary] = abs(np.sum(area * filter))

    return output

# read image
frame1 = cv2.imread('f1.png', cv2.IMREAD_GRAYSCALE)
frame2 = cv2.imread('f2.png', cv2.IMREAD_GRAYSCALE)
height, width = frame1.shape

# parameters of corner detection
corner_thres = 0.001
k = 0.04
window_size = 3

# gradient(Sobel filter)
gx = np.array([[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]])
gy = np.array([[1, 0, -1],
               [2, 0, -2],
               [1, 0, -1]])


ix = conv(frame1, gx)
iy = conv(frame1, gy)
ixx = ix * ix
iyy = iy * iy
ixy = ix * iy

# Harris corner detection
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
        R[i-boundary, j-boundary] = det - k * trace**2


for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        if R[i,j] > corner_thres*np.max(R):
            keypoints.append(point(i+boundary*2, j+boundary*2))

#print(len(keypoints))

# parameters of L-K optical flow
flow_size = 51
edge = flow_size//2

# L-K optical flow
it = frame2 - frame1
mv = []
for n in range(len(keypoints)):
    A = []
    x, y = keypoints[n].x, keypoints[n].y

    Ix = np.ravel(ix[x-edge:x+edge+1, y - edge:y+edge+1])
    Iy = np.ravel(iy[x-edge:x+edge+1, y - edge:y+edge+1])
    It = np.ravel(it[x-edge:x+edge+1, y - edge:y+edge+1])
    A.append(Ix)
    A.append(Iy)  # 2, 9
    A = np.array(A).T
    AtA = np.dot(A.T, A)
    if AtA[0, 0] * AtA[1, 1] - AtA[0,1] * AtA[1,0]:
        inv_AtA = np.linalg.inv(AtA)
        if A.shape[0] == len(It):
            mv.append(np.dot(inv_AtA, np.dot(A.T, -It)))


output = cv2.imread('f1.png', cv2.IMREAD_ANYCOLOR)
for i in range(len(mv)):
    x, y = keypoints[i].x, keypoints[i].y
    cv2.line(output, (y, x), (y, x), (255, 0, 0), 3)
    cv2.line(output, (y, x), (int(y+mv[i][1]), int(x+mv[i][0])), (240, 200, 0), 1)


cv2.imshow('output', output)
o2 = cv2.imread('f2.png', cv2.IMREAD_ANYCOLOR)
cv2.imshow('dest', o2)
cv2.waitKey()