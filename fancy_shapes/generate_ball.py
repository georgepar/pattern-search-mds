#!/usr/bin/env python
"""
Practically generates radom points in the cube [-1,1]x[-1,1]x[-1,1]
and keeps the ppints that are inside the radius
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import Isomap
from scipy.spatial import distance_matrix

npoints = 3000
radius = 0.9

use_noise = False
noise_std = 1e-2
show_plots = False
epsilon = 1e-9


def check_if_symmetric(a, tol=1e-9):
    return np.allclose(a, a.T, atol=tol)


def get_points(npoints, radius, use_noise=False, noise_std=1e-2):
    phi = np.random.uniform(0, 2.0 * np.pi, npoints)
    costheta = np.random.uniform(-1.0, 1.0, npoints)
    u = np.random.uniform(0.0, 1.0, npoints)
    theta = np.arccos(costheta)
    r = radius * np.cbrt(u)
    sintheta = np.sin(theta)
    x = r * sintheta * np.cos(phi)
    y = r * sintheta * np.sin(phi)
    z = r * costheta
    p = np.stack((x, y, z), axis=1)
    if use_noise:
        noise_x = np.random.normal(0, noise_std, npoints)
        noise_y = np.random.normal(0, noise_std, npoints)
        noise_z = np.random.normal(0, noise_std, npoints)
        noise = np.stack((noise_x, noise_y, noise_z), axis=1)
        p += noise
    p = np.around(p, decimals=6)
    return p


def main():
    # set the random seed so as to generate the same sequence every time the
    # program is called
    np.random.seed(42)

    points = get_points(npoints, radius, use_noise=use_noise,
                        noise_std=noise_std)

    # EUCLIDEAN DISTANCE CALCULATION FOR BALL
    eucl_distances = distance_matrix(points, points)
    filename = "Euclidean_ball_geopar.dat"
    np.savetxt(filename, eucl_distances, delimiter=',')

    # GEODESIC DISTANCE CALCULATION FOR BALL
    geodesic_distances = (Isomap(n_neighbors=8,
                                 n_components=3,
                                 n_jobs=4)
                          .fit(points)
                          .dist_matrix_)
    if not check_if_symmetric(geodesic_distances):
        print("Geodesic distance matrix not symmetric")
    filename = "Geodesic_ball_geopar.dat"
    np.savetxt(filename, geodesic_distances, delimiter=',')

    max_geod = np.max(geodesic_distances)
    print("Max geodesic distance between two points:", max_geod)

    filename = "ball_coords_geopar.dat"
    np.savetxt(filename, points, delimiter=',')

    print("Number of vectors:", points.shape[0])

    # print(points)

    if show_plots:
        # PLOT

        xx = points[:, 0]
        yy = points[:, 1]
        zz = points[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xx, yy, zz)

        plt.show()

if __name__ == '__main__':
    main()
