
from PIL import Image
from scipy.linalg import svd

import hashlib
import os
import math
import numpy as np
import pygame
import sys
import json

# a really rough rendition of ISEDA (no median filter) to quickly get points
def extract_points(image):
    image_width = image.width
    image_height = image.height

    image_mat = np.asarray(image)

    # STEP 1: image processing
    # add up the channels
    image_mat = image_mat.sum(2)

    # divide by 3 to average
    image_mat = np.divide(image_mat, 3)

    # calculate threshold (Eq. 10)
    gray_threshold = image_mat.min() + (image_mat.max() - image_mat.min()) / 4

    # threshold image into binary image
    image_mat = np.where(image_mat > gray_threshold, 1, 0)

    # STEP 2: intertial properties
    # calculate centroid
    mass = image_mat.sum()
        
    # this gets all the points that are equal to 1 and then adds those coordinates
    # up and then divides that by the "mass" of the body
    centroid = np.argwhere(image_mat == 1).sum(0) / mass

    # calculate moment of inertia
    Iy, Ix = ((np.argwhere(image_mat == 1) - centroid) ** 2).sum(0)

    # calculate product of inertia
    Ixy = ((np.argwhere(image_mat == 1) - centroid).prod(1)).sum()

    tr = Ix + Iy # trace of inertia tensor
    det = Ix * Iy - Ixy * Ixy # determinant of tensor

    # calculate the eigenvalues (trivial for 2x2)
    Lmax = 0.5 * (tr + np.sqrt(tr * tr - 4 * det)) # minor axis
    Lmin = 0.5 * (tr - np.sqrt(tr * tr - 4 * det)) # major axis

    # calculate the eigenvector associated with Lmax (the minor axis)
    wmax = np.array([1, (Lmax - Ix) / -Ixy])
    wmax = wmax / np.linalg.norm(wmax) # normalize the vector

    wmin = np.array([1, (Lmin - Ix) / -Ixy])
    wmin = wmin / np.linalg.norm(wmin)

    # calculate a rough (very very rough) estimate of the radius of the moon as well
    # as the (semi-)minor axis
    estimated_moon_radius = 2 * np.sqrt(Lmax/mass)
    estimated_minor_axis = 2 * np.sqrt(Lmin/mass)

    # offset between each line to query
    offset = wmin * estimated_moon_radius / 11

    # STEP 3: place points
    
    # using a start point and a direction of a ray, find a point of intersection on
    # the edge of the image canvas
    def find_edge_point(start, dir):
        # ensure that we only get valid indices in the image
        W = image_width - 1 
        H = image_height - 1

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


    # get a bilinearly interpolated value for a point on the image
    def sample_bilinear(point):
        floored_point = np.floor(point)
        fy, fx = floored_point
        iy = int(fy)
        ix = int(fx)

        if ix + 1 >= image_width or iy + 1 >= image_height:
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

    transition_points = np.array(
        [point for point in transition_points if point.size != 0]
    )

    # Step 4: circle fit the points
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
        return rmsd

    red_points = transition_points[:, 0, :]
    blue_points = transition_points[:, 1, :]
    rmsd0 = taubin_fit(red_points)
    rmsd1 = taubin_fit(blue_points)
    return red_points if rmsd0 > rmsd1[0] else blue_points

if __name__ == '__main__':
    args = sys.argv[1:]

    if len(args) < 1:
        print('Usage: label.py [file] <file.json>')
        exit(1)

    image_test = Image.open(args[0])
    points = extract_points(image_test)

    print(points)

    pygame.init()
    
    RADIUS = 5
    WIDTH = 1280
    HEIGHT = 960

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    running = True

    pygame.font.init()
    default_font = pygame.font.SysFont('Comic Sans MS', 12)

    image_sur = pygame.image.frombuffer(image_test.tobytes(), image_test.size,
                                        image_test.mode) # pyright: ignore
    image_sur = pygame.transform.scale(image_sur, 
            (image_test.width * HEIGHT / image_test.height, HEIGHT))
    sur_rect = image_sur.get_rect()

    dragging_point = None

    scale = HEIGHT / image_test.height
    point_texts = [default_font.render(str(idx),
                                       False, 'white') for
                   idx, _ in enumerate(points)]
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                for point in points:
                    dx = event.pos[0] - point[1] * scale
                    dy = event.pos[1] - point[0] * scale
                    if math.sqrt(dx * dx + dy * dy) < RADIUS: 
                        dragging_point = point
                        break

            if event.type == pygame.MOUSEBUTTONUP:
                dragging_point = None

            if event.type == pygame.MOUSEMOTION:
                if dragging_point is not None:
                    dragging_point[0] = event.pos[1] / scale
                    dragging_point[1] = event.pos[0] / scale

        screen.fill('black')
        screen.blit(image_sur, sur_rect)

        for idx, point in enumerate(points):
            pygame.draw.circle(screen, 'red',
                    pygame.Vector2(point[1] * scale, point[0] * scale), RADIUS)
            screen.blit(point_texts[idx], (point[1] * scale, point[0] * scale))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

    path = os.path.splitext(args[0])[0] + '.json'
    if len(args) >= 2:
        path = args[1]

    if os.path.exists(path):
        print('Output file already exists.', path)
        exit(1)

    with open(args[0], 'rb') as image_fp:
        with open(path, 'w') as fp:
            out = [np.flip(point).tolist() for point in points]
            digest = hashlib.file_digest(image_fp, 'sha256')
            fp.write(json.dumps({
                'image': {
                    'name': os.path.basename(args[0]),
                    'width': image_test.width,
                    'height': image_test.height,
                    'sha256': digest.hexdigest(),
                },
                'target_points': out
            }, indent='\t'))

