# implementation of the ISEDA algorithm in python

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import ndimage

args = sys.argv[1:]

if len(args) < 1:
    print("Usage: iseda.py [file]")
    exit(1)

# import image from the command line
TEST_IMAGE = Image.open(args[0])
TEST_IMAGE_WIDTH = TEST_IMAGE.width
TEST_IMAGE_HEIGHT = TEST_IMAGE.height

MEDIAN_FILTER_SIZE = 10

# turn that into an np array to process
# since this is a RGB image, the array has the shape of NDArray[height, width, 3]
image_mat = np.asarray(TEST_IMAGE)



# STEP 1: PROCESS IMAGE
plt.figure('Image processing steps')

# show before processing
plt.subplot(2, 2, 1)
plt.imshow(image_mat)
plt.title('Unprocessed image')

# add up the channels
image_mat = image_mat.sum(2)

# divide by 3 to average
image_mat = np.divide(image_mat, 3)

# show grayscale image
plt.subplot(2, 2, 2)
plt.imshow(image_mat, cmap='gray')
plt.title('Grayscale image')

# apply a median filter on the image
image_mat = ndimage.median_filter(image_mat, size=MEDIAN_FILTER_SIZE)

pixel = np.argmax(image_mat)
print('Brightest pixel: {}'.format(pixel))

# show filtered image
plt.subplot(2, 2, 3)
plt.imshow(image_mat, cmap='gray')
plt.title('Median filtered')

# calculate threshold (Eq. 10)
gray_threshold = image_mat.min() + (image_mat.max() - image_mat.min()) / 4

# threshold image into binary image
image_mat = np.where(image_mat > gray_threshold, 1, 0)

# show thresholded image
plt.subplot(2, 2, 4)
plt.imshow(image_mat, cmap='gray')
plt.title('Thresholded binary image')

plt.show()



# STEP 2: CALCULATING INERTIAL PROPERTIES
plt.figure('Calculating inertial properties')

# calculate centroid
mass = image_mat.sum()
print('mass of lumped sum: {}'.format(mass))

# this gets all the points that are equal to 1 and then adds those coordinates
# up and then divides that by the "mass" of the body
centroid = np.argwhere(image_mat == 1).sum(0) / mass

plt.subplot(1, 3, 1)
plt.plot(*np.flip(centroid), marker='o', color='green')
plt.imshow(image_mat, cmap='gray')
plt.title('Calculated centroid')

# calculate moment of inertia
Iy, Ix = ((np.argwhere(image_mat == 1) - centroid) ** 2).sum(0)

# calculate product of inertia
Ixy = ((np.argwhere(image_mat == 1) - centroid).prod(1)).sum()

print('I: [{}, {}], Ixy: {}'.format(Ix, Iy, Ixy))

tr = Ix + Iy # trace of inertia tensor
det = Ix * Iy - Ixy * Ixy # determinant of tensor

# calculate the eigenvalues (trivial for 2x2)
Lmax = 0.5 * (tr + np.sqrt(tr * tr - 4 * det)) # minor axis
Lmin = 0.5 * (tr - np.sqrt(tr * tr - 4 * det)) # major axis

# calculate the eigenvector associated with Lmax (the minor axis)
wmax = np.array([1, (Lmax - Ix) / -Ixy])
wmax = wmax / np.linalg.norm(wmax) # normalize the vector
print(wmax)

wmin = np.array([1, (Lmin - Ix) / -Ixy])
wmin = wmin / np.linalg.norm(wmin)

# calculate a rough (very very rough) estimate of the radius of the moon as well
# as the (semi-)minor axis
estimated_moon_radius = 2 * np.sqrt(Lmax/mass)
estimated_minor_axis = 2 * np.sqrt(Lmin/mass)

plt.subplot(1, 3, 2)

# draw the rough minor axis
plt.arrow(
    *np.flip(centroid - wmax * estimated_minor_axis),
    *np.flip(2 * wmax * estimated_minor_axis),
    color='red'
)

# draw the rough major axis
plt.arrow(
    *np.flip(centroid - wmin * estimated_moon_radius),
    *np.flip(2 * wmin * estimated_moon_radius),
    color='blue'
)

# also put the centroid point
plt.plot(*np.flip(centroid), marker='o', color='green')

plt.imshow(image_mat, cmap='gray')
plt.title('Calculated axes of symmetry')

print('Lmax: {}\nLmin: {}'.format(Lmax, Lmin))
print('Estimated moon radius: {} pixels'.format(estimated_moon_radius))

# offset between each line to query
offset = wmin * estimated_moon_radius / 11

plt.subplot(1, 3, 3)

# display all the lines that will be used when placing the transition points by
# correlation
for i in range(-11, 12):
    plt.arrow(*np.flip(centroid - (wmax * estimated_minor_axis) + (i * offset)), *np.flip(2 * wmax * estimated_minor_axis), color='gray')

# draw the offset between points
plt.arrow(*np.flip(centroid), *np.flip(offset), color='blue')

# draw the centroid point
# plt.plot(*np.flip(centroid), marker='o', color='green')

plt.imshow(image_mat, cmap='gray')
plt.title('Lines to query')

plt.show()



# Step 3: PLACING TRANSITION POINTS 
plt.figure('Placing transition points')

# using a start point and a direction of a ray, find a point of intersection on
# the edge of the image canvas
def find_edge_point(start, dir):
    # ensure that we only get valid indices in the image
    W = TEST_IMAGE_WIDTH - 1 
    H = TEST_IMAGE_HEIGHT - 1

    # edge case: no direction
    if dir[0] == 0 and dir[1] == 0:
        return start

    # edge case: dir is facing in only one direction
    if dir[0] == 0 or dir[1] == 0:
        return (np.array([start[0],0]) if dir[1] < 0 else np.array([start[0],W])) if dir[0] == 0 else (np.array([0,start[1]]) if dir[0] < 0 else np.array([H,start[1]]))

    slope = dir[0] / dir[1] # y/x

    x_1 = start[1] - start[0]/slope # intersection with y = 0
    if x_1 >= 0 and x_1 <= W and dir[0] < 0:
        return np.array([0,x_1])

    x_2 = start[1] - (start[0] - H)/slope # intersection with y = H
    if x_2 >= 0 and x_2 <= W and dir[0] > 0:
        return np.array([H,x_2])

    y_1 = start[0] - start[1] * slope # intersection with x = 0
    if y_1 >= 0 and y_1 <= H and dir[1] < 0:
        return np.array([y_1,0])

    y_2 = start[0] - (start[1] - W) * slope # intersection with x = W
    return np.array([y_2,W])


lines = []
for i in range(-11, 12):
    start = find_edge_point(centroid + (i * offset), wmax)
    end = find_edge_point(centroid + (i * offset), -wmax)
    lines.append(np.array([start, end]))

plt.subplot(2, 2, 1)

# show the lines
for line in lines:
    plt.arrow(*np.flip(line[0]), *np.flip(line[1] - line[0]), color='gray')

plt.imshow(image_mat, cmap='gray')
plt.title('Extending the lines')

plt.show()

mask = np.array([0,0,0,0,1,1,1,1])


