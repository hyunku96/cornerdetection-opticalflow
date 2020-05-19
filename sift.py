import numpy as np
import cv2
import math

# load picture
class point:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)

def resize(img, size):
    if size > 1:  # bilinear
        size = int(size)
        output = np.zeros((height * size, width * size))
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                a, b, c, d = img[i//size][j//size], \
                             img[i//size][min(j//size+1, width-1)],\
                             img[min(i//size+1, height-1)][j//size], \
                             img[min(i//size+1, height-1)][min(j//size+1, width-1)]
                col_w1 = (i - i // size)/size
                col_w2 = 1 - col_w1
                row_w1 = (j - j // size)/size
                row_w2 = 1 - row_w1
                output[i, j] = int(row_w2 * (a * col_w2 + c * col_w1) + row_w1 * (b * col_w2 + d * col_w1))

        return output
    elif size == 1:
        return img
    else:
        h, w = int(height*size), int(width*size)
        output = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                output[i, j] = img[int(i / size), int(j / size)]

        return output

def gaussianfilter(size, sigma):
    filter = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            filter[i, j] = math.exp(-((i-size//2)**2 + (j-size//2)**2)/(2*sigma**2))/(2*math.pi*sigma**2)
    filter = filter * (1/np.sum(filter))  # normalize
    return filter

def conv(img, filter):
    boundary = filter_size//2
    output = np.zeros((img.shape[0]-boundary*2, img.shape[1]-boundary*2))
    for i in range(boundary, img.shape[0]-boundary):
        for j in range(boundary, img.shape[1]-boundary):
            area = img[i-boundary:i+boundary+1, j-boundary:j+boundary+1]
            output[i-boundary, j-boundary] = np.sum(np.ravel(area) * np.ravel(filter))

    return output


frame1 = cv2.imread('frame1.png', cv2.IMREAD_GRAYSCALE)
frame2 = cv2.imread('frame2.png', cv2.IMREAD_GRAYSCALE)
height, width = frame1.shape

## sift
k = 2**(1/3)
h_thres = 1
octave_num = 4
picture_num = 5
sigma = 1.6
filter_size = 3
keypoint_threshold = 0
# make octave(scaling)
octaves = []
for o in range(octave_num):
    octave = []
    img = resize(frame1, 2 / (2**o))  # base img
    for p in range(picture_num):
        octave.append(conv(img, gaussianfilter(filter_size, sigma*(k**p))))
    octaves.append(octave)
octaves = np.array(octaves)
# make DoG and extract keypoints
DoGs = []
keypoints = []
count = 0
for o in range(octave_num):
    DoG = []
    for p in range(picture_num-1):
        DoG.append(abs(octaves[o][p] - octaves[o][p+1]))   # 4, 238, 350
    DoG = np.reshape(DoG, (-1, int(((2/(2**o))*height - filter_size//2*2)), int(((2/(2**o))*width - filter_size//2*2))))
    print(np.max(DoG))
    for d in range(len(DoG)-2):  # sigma axis
        for i in range(DoG[d].shape[0]-2):
            for j in range(DoG[d].shape[1]-2):
                max = np.max(DoG[d:d+3, i:i+3, j:j+3])
                if max == DoG[d+1][i+1][j+1]:
                    if max > keypoint_threshold:
                        count += 1
                        keypoints.append(point((i+1)/(2/(2**o))+filter_size//2, (j+1)/(2/(2**o))+filter_size//2))  #

    DoGs.append(DoG)

# print(DoGs.shape)  4, 4
print(count)
# keypoint localize. all points are treated as same
mask = np.zeros((height, width))
for i in range(len(keypoints)):
    mask[keypoints[i].x][keypoints[i].y] = 1

keypoints.clear()
for i in range(height):
    for j in range(width):
        if mask[i][j] == 1:
            keypoints.append(point(i, j))
print(len(keypoints))

# Hessian(calc h and select keypoint)
#for i in range(len(keypoints)):

# save keypoints

## optical flow
# calc mv by least square

## visualize