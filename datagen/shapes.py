import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import Isomap
from scipy.spatial import distance_matrix


class Shape(object):
    def __init__(self, seed=42, n_neighbors=8, dim=3, n_jobs=4):
        np.random.seed(seed)
        self.n_neighbors = n_neighbors
        self.dim = dim
        self.n_jobs = n_jobs
        self.points = None
        self.euclidean_d = None
        self.geodesic_d = None

    def generate(self, npoints, use_cache=True):
        if (use_cache and
           self.points is not None and
           npoints == self.points.shape[0]):
            return self.points
        self.points = np.random.rand(npoints, self.dim)
        return self.points

    def euclidean_distances(self, points=None, use_cache=True):
        if use_cache and self.euclidean_d is not None:
            return self.euclidean_d
        if points is None:
            points = self.points
        self.euclidean_d = distance_matrix(points, points)
        return self.euclidean_d

    def geodesic_distances(self, points=None, use_cache=True):
        if use_cache and self.geodesic_d is not None:
            return self.geodesic_d
        if points is None:
            points = self.points
        self.geodesic_d = (Isomap(n_neighbors=self.n_neighbors,
                                  n_components=self.dim,
                                  n_jobs=self.n_jobs)
                           .fit(points)
                           .dist_matrix_)
        return self.geodesic_d


class Ball(Shape):
    def __init__(
            self, radius, use_noise=True, noise_std=1e-2,
            seed=42, n_neighbors=8, dim=3, n_jobs=4):
        super(Ball, self).__init__(
            seed=seed, n_neighbors=n_neighbors, dim=dim, n_jobs=n_jobs)
        self.radius = radius
        self.use_noise = use_noise
        self.noise_std = noise_std
        self.points = None

    def generate(self, npoints, use_cache=True):
        if (use_cache and
           self.points is not None and
           npoints == self.points.shape[0]):
            return self.points
        phi = np.random.uniform(0, 2.0 * np.pi, npoints)
        costheta = np.random.uniform(-1.0, 1.0, npoints)
        u = np.random.uniform(0.0, 1.0, npoints)
        theta = np.arccos(costheta)
        r = self.radius * np.cbrt(u)
        sintheta = np.sin(theta)
        x = r * sintheta * np.cos(phi)
        y = r * sintheta * np.sin(phi)
        z = r * costheta
        p = np.stack((x, y, z), axis=1)
        if self.use_noise:
            noise_x = np.random.normal(0, self.noise_std, npoints)
            noise_y = np.random.normal(0, self.noise_std, npoints)
            noise_z = np.random.normal(0, self.noise_std, npoints)
            noise = np.stack((noise_x, noise_y, noise_z), axis=1)
            p += noise
        p = np.around(p, decimals=6)
        self.points = p
        return p
