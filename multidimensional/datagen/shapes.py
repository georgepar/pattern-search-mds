import os

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from sklearn.utils.graph import  graph_shortest_path
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform

import multidimensional.config as config


class Shape(object):
    def __init__(self,
                 X=None,
                 name='random',
                 seed=42,
                 n_neighbors=8,
                 dim=3,
                 use_noise=True,
                 noise_std=1e-2,
                 n_jobs=4):
        np.random.seed(seed)
        self.name = name
        self.n_neighbors = n_neighbors
        self.dim = dim
        self.n_jobs = n_jobs
        self.points = X
        self.euclidean_d = None
        self.sqeuclidean_d = None
        self.geodesic_d = None
        self.use_noise = use_noise
        self.noise_std = noise_std

    def generate(self, npoints, use_cache=True):
        if (use_cache and
           self.points is not None and
           npoints == self.points.shape[0]):
            return self.points
        self.points = np.random.rand(npoints, self.dim)
        return self.points

    def noise_round_points(self, p):
        if self.use_noise:
            noise_x = np.random.normal(0, self.noise_std, p.shape[0])
            noise_y = np.random.normal(0, self.noise_std, p.shape[0])
            noise_z = np.random.normal(0, self.noise_std, p.shape[0])
            noise = np.stack((noise_x, noise_y, noise_z), axis=1)
            p += noise
        p = np.around(p, decimals=6)
        return p

    def euclidean_distances(self, points=None, use_cache=True):
        if use_cache and self.euclidean_d is not None:
            return self.euclidean_d
        if points is None:
            points = self.points
        self.euclidean_d = distance_matrix(points, points)
        return self.euclidean_d

    def sqeuclidean_distances(self, points=None, use_cache=True):
        if use_cache and self.sqeuclidean_d is not None:
            return self.euclidean_d
        if points is None:
            points = self.points
        self.sqeuclidean_d = squareform(pdist(points, metric='sqeuclidean'))
        return self.sqeuclidean_d

    def geodesic_distances(self, points=None, use_cache=True):
        if use_cache and self.geodesic_d is not None:
            return self.geodesic_d
        if points is None:
            points = self.points
        dist = self.euclidean_distances()
        nbrs_inc = np.argsort(dist, axis=1)
        max_dist = -1
        for i in range(dist.shape[0]):
            achieved_neighbors = 0
            while achieved_neighbors < min(self.n_neighbors, dist.shape[0]):
                j = achieved_neighbors
                if max_dist < dist[i][nbrs_inc[i][j]]:
                    max_dist = dist[i][nbrs_inc[i][j]]
                achieved_neighbors += 1
        nbrs = (NearestNeighbors(algorithm='auto',
                                 n_neighbors=self.n_neighbors,
                                 radius=max_dist,
                                 n_jobs=self.n_jobs)
                .fit(points))
        kng = radius_neighbors_graph(
            nbrs, max_dist, mode='distance', n_jobs=self.n_jobs)
        self.geodesic_d = graph_shortest_path(kng, method='D', directed=False)
        return self.geodesic_d

    def _save_data(self, x, base_name):
        if x is not None:
            save_file = os.path.join(
                config.DATA_DIR, base_name.format(self.name))
            np.savetxt(save_file, x, delimiter=',')

    def save(self):
        self._save_data(self.points, '{}_coords.dat')
        self._save_data(self.euclidean_d, 'Euclidean_{}.dat')
        self._save_data(self.geodesic_d, 'Geodesic_{}.dat')

    def instance(self, npoints, distance='euclidean'):
        if self.points is None:
            points = self.generate(npoints)
        else:
            points = self.points
        if distance == 'euclidean':
            dist = self.euclidean_distances()
        elif distance == 'sqeuclidean':
            dist = self.sqeuclidean_distances()
        else:
            dist = self.geodesic_distances()
        return points, dist

    def plot3d(self):
        if self.points is None:
            return
        xx = self.points[:, 0]
        yy = self.points[:, 1]
        zz = self.points[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xx, yy, zz)
        filename = '{}_{}_{}_{}'.format(
            self.name,
            self.points.shape[0],
            self.dim,
            'noise' if self.use_noise else 'no_noise')
        plt.savefig(os.path.join(config.REPORT_DIR, filename))


class Ball(Shape):
    def __init__(self,
                 radius=0.9,
                 name='ball',
                 use_noise=True,
                 noise_std=1e-2,
                 seed=42,
                 n_neighbors=8,
                 dim=3,
                 n_jobs=4):
        super(Ball, self).__init__(
            name=name,
            seed=seed,
            n_neighbors=n_neighbors,
            dim=dim,
            use_noise=use_noise,
            noise_std=noise_std,
            n_jobs=n_jobs)
        self.radius = radius

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
        self.points = self.noise_round_points(p)
        return self.points


class Sphere(Shape):
    def __init__(self,
                 radius=0.9,
                 name='sphere',
                 use_noise=True,
                 noise_std=1e-2,
                 seed=42,
                 n_neighbors=8,
                 dim=3,
                 n_jobs=4):
        super(Sphere, self).__init__(
            name=name,
            seed=seed,
            n_neighbors=n_neighbors,
            dim=dim,
            use_noise=use_noise,
            noise_std=noise_std,
            n_jobs=n_jobs)
        self.radius = radius

    @staticmethod
    def _get_coords(theta, phi):
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        return x, y, z

    def generate(self, npoints, use_cache=True):
        phi = np.random.uniform(0, 2.0 * np.pi, npoints)
        costheta = np.random.uniform(-1.0, 1.0, npoints)
        theta = np.arccos(costheta)
        x, y, z = self._get_coords(theta, phi)
        p = np.stack((x, y, z), axis=1)
        self.points = self.noise_round_points(p)
        return self.points


class CutSphere(Shape):
    def __init__(self,
                 radius=0.9,
                 cut_theta=0.5 * np.pi,
                 name='cut-sphere',
                 use_noise=True,
                 noise_std=1e-2,
                 seed=42,
                 n_neighbors=8,
                 dim=3,
                 n_jobs=4):
        super(CutSphere, self).__init__(
            name=name,
            seed=seed,
            n_neighbors=n_neighbors,
            dim=dim,
            use_noise=use_noise,
            noise_std=noise_std,
            n_jobs=n_jobs)
        self.radius = radius
        self.cut_theta = cut_theta

    @staticmethod
    def _get_coords(theta, phi):
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        return x, y, z

    def generate(self, npoints, use_cache=True):
        phi = np.random.uniform(0, 2.0 * np.pi, npoints)
        costheta = np.random.uniform(np.cos(self.cut_theta), 1.0, npoints)
        theta = np.arccos(costheta)
        # cut_theta = theta[theta < self.cut_theta]
        x, y, z = self._get_coords(theta, phi)
        p = np.stack((x, y, z), axis=1)
        self.points = self.noise_round_points(p)
        return self.points


class Spiral(Shape):
    def __init__(self,
                 name='spiral',
                 angle_start=np.pi,
                 angle_stop=4*np.pi,
                 r_stop=0.9,
                 r_start=0.1,
                 depth=10,
                 use_noise=True,
                 noise_std=1e-2,
                 seed=42,
                 n_neighbors=8,
                 dim=3,
                 n_jobs=4):
        super(Spiral, self).__init__(
            name=name,
            seed=seed,
            n_neighbors=n_neighbors,
            dim=dim,
            use_noise=use_noise,
            noise_std=noise_std,
            n_jobs=n_jobs)
        self.angle_start = angle_start
        self.angle_stop = angle_stop
        self.r_start = r_start
        self.r_stop = r_stop
        self.depth = depth

    def generate(self, npoints, use_cache=True):
        rows = np.round(npoints / self.depth) - 1
        angle_step = float(self.angle_stop - self.angle_start) / rows
        distance_step = float(self.r_stop - self.r_start) / rows
        angle = self.angle_start
        distance = self.r_start
        points = []
        while angle <= self.angle_stop:
            for i in range(self.depth):
                x = -0.9 + (1.8 * i) / (self.depth - 1)
                y = distance * np.cos(angle)
                z = distance * np.sin(angle)
                points.append([x, y, z])

            distance += distance_step
            angle += angle_step
        p = np.array(points)
        self.points = self.noise_round_points(p)
        return self.points


class SpiralHole(Shape):
    def __init__(self,
                 name='spiral-with-hole',
                 angle_start=np.pi,
                 angle_stop=4*np.pi,
                 r_stop=0.9,
                 r_start=0.1,
                 depth=10,
                 use_noise=True,
                 noise_std=1e-2,
                 seed=42,
                 n_neighbors=8,
                 dim=3,
                 n_jobs=4):
        super(SpiralHole, self).__init__(
            name=name,
            seed=seed,
            n_neighbors=n_neighbors,
            dim=dim,
            use_noise=use_noise,
            noise_std=noise_std,
            n_jobs=n_jobs)
        self.angle_start = angle_start
        self.angle_stop = angle_stop
        self.r_start = r_start
        self.r_stop = r_stop
        self.depth = depth
        self.angle_hole_start = float(360 + 45) * np.pi / 180
        self.angle_hole_stop = float(360 + 135) * np.pi / 180

    def generate(self, npoints, use_cache=True):
        rows = np.round(npoints / self.depth) - 1
        angle_step = float(self.angle_stop - self.angle_start) / rows
        distance_step = float(self.r_stop - self.r_start) / rows
        angle = self.angle_start
        distance = self.r_start
        points = []
        while angle <= self.angle_stop:
            for i in range(self.depth):
                x = -0.9 + (1.8 * i) / (self.depth - 1)
                y = distance * np.cos(angle)
                z = distance * np.sin(angle)

                min_hole = np.round(int(2 * self.depth / 3))
                max_hole = np.round(int(self.depth / 3))
                if (self.angle_hole_stop >= angle >= self.angle_hole_start and
                   min_hole > i >= max_hole):
                    pass
                else:
                    points.append([x, y, z])
            distance += distance_step
            angle += angle_step
        p = np.array(points)
        self.points = self.noise_round_points(p)
        return self.points
