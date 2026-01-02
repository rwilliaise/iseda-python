# implementation of the ISEDA algorithm in python

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys

from scipy import ndimage
from scipy.linalg import svd

args = sys.argv[1:]

if len(args) < 1:
    print("Usage: iseda.py [file]")
    exit(1)

# import image from the command line
TEST_IMAGE = Image.open(args[0])
TEST_IMAGE_WIDTH = TEST_IMAGE.width
TEST_IMAGE_HEIGHT = TEST_IMAGE.height

MEDIAN_FILTER_SIZE = 5

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

pixel = np.unravel_index(np.argmax(image_mat), image_mat.shape)
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
    if np.allclose(start, end):
        continue
    lines.append(np.array([start, end]))

# extending the lines plot 
plt.subplot(2, 2, 1)

# show the lines
for i, line in enumerate(lines):
    color = 'yellow' if i == int(len(lines) / 2) else 'gray'
    plt.arrow(*np.flip(line[0]), *np.flip(line[1] - line[0]), color=color)

plt.imshow(image_mat, cmap='gray')
plt.title('Extending the lines')

# get a bilinearly interpolated value for a point on the image
def sample_bilinear(point):
    floored_point = np.floor(point)
    fy, fx = floored_point
    iy = int(fy)
    ix = int(fx)

    if ix + 1 >= TEST_IMAGE_WIDTH or iy + 1 >= TEST_IMAGE_HEIGHT:
        return 0

    # coordinate relative to the pixel on the image
    py, px = point - floored_point
    
    # top-left value
    v0 = image_mat[iy][ix]
    # top-right value
    v1 = image_mat[iy][ix + 1]
    # bottom-left value
    v2 = image_mat[iy + 1][ix]
    # bottom-right value
    v3 = image_mat[iy + 1][ix + 1]
    
    # interpolate the value according to the pixel-relative coordinate
    return (1 - py) * (v0 * (1 - px) + v1 * px) + py * (v2 * (1 - px) + v3 * px)
    
mask = np.array([0,0,0,0,1,1,1,1])
correlations = []

# sample the points along each line
for line in lines:
    # cut each line up based on its line
    length = np.linalg.norm(line[1] - line[0])
    count = int(length)

    # generate a bunch of points on the line using the new count
    points = np.linspace(line[0], line[1], int(count))

    # now sample (using bilinear interpolation) using the points on the line
    samples = np.array([sample_bilinear(point) for point in points])

    # and then cross-correlate using the mask
    correlation = np.correlate(mask, samples)

    correlations.append(correlation)

# sampling the image plot
plt.subplot(2, 2, 2)

line_index = int(len(correlations) / 2)
correlation = correlations[line_index]

plt.plot(np.flip(correlation), 'r.-')
plt.xlabel('Position along middle line (left-most point to right-most point)')
plt.ylabel('Value of cross-correlation along middle line')
plt.title('Mask cross-correlation along middle line')

transition_points = []

for i, correlation in enumerate(correlations):
    # first find the line associated with this correlation
    line = lines[i]

    # then get the direction vector underlying it 
    length = np.linalg.norm(line[1] - line[0])
    dir = (line[1] - line[0]) / length

    # get the maximum correlation
    max_correlation = correlation.max()

    if max_correlation == 0:
        transition_points.append(np.array([]))
        continue
    
    # get the first value which passes the halfway point
    i0 = np.argmax(correlation > max_correlation / 2)
    # then, get the value after that which is less than the halfway point
    i1 = len(correlation) - np.argmax(np.flip(correlation) > max_correlation / 2)
    
    # using the point indices, we can get the point in space
    # TODO: these should be taking into account the size of each sample line
    p0 = line[1] - dir * i0
    p1 = line[1] - dir * i1
    
    # now add the points to the array
    points = np.array([p0, p1])
    transition_points.append(points)   

# points associated with each line plot
plt.subplot(2, 2, 3)

for i, line in enumerate(lines):
    color = 'yellow' if i == int(len(lines) / 2) else 'gray'
    points = transition_points[i]

    plt.arrow(*np.flip(line[0]), *np.flip(line[1] - line[0]), color=color)

    # pass any dead lines
    if len(points) == 0:
        continue

    plt.plot(*np.flip(points[0]), marker='o', color=color)
    plt.plot(*np.flip(points[1]), marker='o', color=color)

plt.imshow(image_mat, cmap='gray')
plt.title('Calculating the transition points')

# Lay the found points over the original image
plt.subplot(2, 2, 4)

transition_points = np.array(
    [point for point in transition_points if point.size != 0]
)

plt.plot(transition_points[:, 0, 1], transition_points[:, 0, 0], marker='x', color='red')
plt.plot(transition_points[:, 1, 1], transition_points[:, 1, 0], marker='x', color='blue')

plt.imshow(TEST_IMAGE)
plt.title('Compared to the original image')

plt.show()


# Step 4: CIRCLE FITTING THE POINTS
plt.figure('Circle fitting the points')

def taubin_fit(coords):
    x = coords[:, 0]
    y = coords[:, 1]
    centroid = np.array([x.mean(), y.mean()])
    X = x - centroid[0]
    Y = y - centroid[1]
    Z = X * X + Y * Y
    
    Zmean = np.mean(Z)
    Z0 = (Z - Zmean) / (2 * np.sqrt(Zmean))
    ZXY = np.array([Z0, X, Y])
    
    _, _, V = svd(ZXY.transpose(), full_matrices=False)
    V = V.transpose()
    A = V[:, 2]
    A[0] /= 2 * np.sqrt(Zmean)
    A = np.hstack((A, -Zmean * A[0]))

    if abs(A[0]) < 1e-8:
        print('warn: curve too straight, failed to fit onto a circle')
        return 0, 0, np.inf, np.inf
    
    c = -(A[1:3]).transpose() / A[0] / 2 + centroid
    xc = c[0]
    yc = c[1]
    r = np.sqrt(A[1] * A[1] + A[2] * A[2] - 4 * A[0] * A[3]) / abs(A[0]) / 2

    dx = x - xc
    dy = y - yc
    rmsd = np.sqrt(np.mean((np.sqrt(dx ** 2 + dy ** 2) - r) ** 2))
    return xc, yc, r, rmsd

red_points = transition_points[:, 0, :]
blue_points = transition_points[:, 1, :]

ax = plt.subplot(1, 3, 1)

xc0, yc0, r0, rmsd0 = taubin_fit(red_points)
print(('red points guessed radius: {}').format(r0))

plt.imshow(TEST_IMAGE)
img_xbounds = ax.get_xlim()
img_ybounds = ax.get_ylim()
plt.plot(red_points[:, 1], red_points[:, 0], marker='x', color='red')
plt.plot(yc0, xc0, marker='x', color='yellow')
ax.add_patch(patches.Rectangle((yc0-r0,xc0-r0), r0 * 2, r0 * 2, edgecolor='yellow', facecolor='none', lw=2))
ax.add_patch(patches.Circle((yc0, xc0), r0, edgecolor='green', facecolor='none', lw=2))
plt.title('Fitting the red points')
ax.set_xlim(img_xbounds)
ax.set_ylim(img_ybounds)

ax = plt.subplot(1, 3, 2)

xc1, yc1, r1, rmsd1 = taubin_fit(blue_points)
print(('blue points guessed radius: {}').format(r1))

plt.imshow(TEST_IMAGE)
plt.plot(blue_points[:, 1], blue_points[:, 0], marker='x', color='blue')
plt.plot(yc1, xc1, marker='x', color='yellow')
ax.add_patch(patches.Rectangle((yc1-r1,xc1-r1), r1 * 2, r1 * 2, edgecolor='yellow', facecolor='none', lw=2))
ax.add_patch(patches.Circle((yc1, xc1), r1, edgecolor='green', facecolor='none', lw=2))
plt.title('Fitting the blue points')
ax.set_xlim(img_xbounds)
ax.set_ylim(img_ybounds)

plt.subplot(1, 3, 3)

chosen_points = red_points if rmsd0 < rmsd1 else blue_points

plt.imshow(TEST_IMAGE)
plt.plot(chosen_points[:, 1], chosen_points[:, 0], marker='x', color='yellow')
plt.title('Choosing the best set of points')

print('final rmsd of points: {}'.format(min(rmsd0, rmsd1)))

plt.show()

