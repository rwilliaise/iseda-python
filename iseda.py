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

estimated_moon_radius = 2 * np.sqrt(Lmax/mass)
estimated_minor_axis = 2 * np.sqrt(Lmin/mass)

plt.subplot(1, 3, 2)
plt.arrow(*np.flip(centroid - wmax * estimated_minor_axis), *np.flip(2 * wmax * estimated_minor_axis), color='red')
plt.arrow(*np.flip(centroid - wmin * estimated_moon_radius), *np.flip(2 * wmin * estimated_moon_radius), color='blue')
plt.plot(*np.flip(centroid), marker='o', color='green')
plt.imshow(image_mat, cmap='gray')
plt.title('Calculated axes of symmetry')

print('Lmax: {}\nLmin: {}'.format(Lmax, Lmin))
print('Estimated moon radius: {} pixels'.format(estimated_moon_radius))

# offset between each line to query
offset = wmin * estimated_moon_radius / 11

plt.subplot(1, 3, 3)

for i in range(-11, 12):
    plt.arrow(*np.flip(centroid - (wmax * estimated_minor_axis) + (i * offset)), *np.flip(2 * wmax * estimated_minor_axis), color='gray')
plt.arrow(*np.flip(centroid), *np.flip(offset), color='blue')
plt.plot(*np.flip(centroid), marker='o', color='green')
plt.imshow(image_mat, cmap='gray')
plt.title('Lines to query')

plt.show()



# Step 3: CROSS-CORRELATION
wmax_slope = wmax[0] / wmax[1] # rise over run (y/x)
y_intercept = centroid[0] - centroid[1] * wmax_slope
print('y_intercept: {}'.format(y_intercept))


start_point = np.array([y_intercept, 0])
if y_intercept < 0 or y_intercept > TEST_IMAGE_HEIGHT:
    x_intercept = centroid[1] - centroid[0] / wmax_slope
    

mask = np.array([0,0,0,0,1,1,1,1])


