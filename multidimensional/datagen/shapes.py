import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import (
    NearestNeighbors, radius_neighbors_graph, kneighbors_graph)
from sklearn.utils.graph import  graph_shortest_path
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn import datasets

import multidimensional.config as config
import multidimensional.common as common


class Shape(object):
    def __init__(self,
                 X=None,
                 name='random',
                 seed=42,
                 n_neighbors=12,
                 dim=3,
                 use_noise=False,
                 noise_std=1e-2,
                 n_jobs=4):
        np.random.seed(seed)
        self.seed = seed
        self.name = name
        self.n_neighbors = n_neighbors
        self.dim = dim
        self.n_jobs = n_jobs
        self.euclidean_d = None
        self.sqeuclidean_d = None
        self.geodesic_d = None
        self.use_noise = use_noise
        self.noise_std = noise_std
        self.points = self.add_noise(X)
        self.color = None

    def generate(self, npoints, use_cache=True):
        if (use_cache and
           self.points is not None and
           npoints == self.points.shape[0]):
            return self.points
        self.points = np.random.rand(npoints, self.dim)
        return self.points

    def add_noise(self, x):
        if self.use_noise:
            n = np.random.normal(0, self.noise_std, x.shape)
            x = x + n
        return x

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
        self.euclidean_d = common.DISTANCE_MATRIX(points)
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
        # kng = kneighbors_graph(nbrs,
        #                        self.n_neighbors,
        #                        mode='distance',
        #                        n_jobs=self.n_jobs)
        self.geodesic_d = graph_shortest_path(kng, method='D', directed=False)
        return self.geodesic_d

    def geodesic_radius(self, points=None, use_cache=True):
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

    def geodesic_neighbors(self, points=None, use_cache=True):
        if use_cache and self.geodesic_d is not None:
            return self.geodesic_d
        if points is None:
            points = self.points
        nbrs = (NearestNeighbors(algorithm='auto',
                                 n_neighbors=self.n_neighbors,
                                 n_jobs=self.n_jobs)
                .fit(points))
        kng = kneighbors_graph(nbrs,
                               self.n_neighbors,
                               mode='distance',
                               n_jobs=self.n_jobs)
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

    def instance(self, npoints=0, distance='euclidean', geomethod='neigh'):
        if self.points is None:
            points = self.generate(npoints)
        else:
            points = self.points
        if distance == 'euclidean':
            dist = self.euclidean_distances()
        elif distance == 'sqeuclidean':
            dist = self.sqeuclidean_distances()
        else:
            if geomethod == 'radius':
                dist = self.geodesic_radius()
            else:
                dist = self.geodesic_neighbors()
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
                 X=None,
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
                 X=None,
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
                 X=None,
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
                 X=None,
                 name='spiral',
                 angle_start=np.pi,
                 angle_stop=4*np.pi,
                 r_stop=0.9,
                 r_start=0.1,
                 depth=12,
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
        self.color = self.points[:, 0]
        return self.points


class SpiralHole(Shape):
    def __init__(self,
                 X=None,
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


class SwissRoll(Shape):
    def generate(self, npoints, use_cache=True):
        noise_std = 0 if not self.use_noise else self.noise_std
        self.points, self.color = datasets.samples_generator.make_swiss_roll(
            n_samples=npoints, noise=noise_std, random_state=self.seed)
        return self.points


class SCurve(Shape):
    def generate(self, npoints, use_cache=True):
        noise_std = 0 if not self.use_noise else self.noise_std
        self.points, self.color = datasets.samples_generator.make_s_curve(
            n_samples=npoints, noise=noise_std, random_state=self.seed)
        return self.points


class ToroidalHelix(Shape):
    def generate(self, npoints, use_cache=True):
        param = -1
        t = np.arange(1, npoints) / float(npoints)
        e_t = t ** (param * 2.0 * np.pi)
        self.color = e_t
        p = np.array([
            (2 + np.cos(8 * e_t)) * np.cos(e_t),
            (2 + np.cos(8 * e_t)) * np.sin(e_t),
            np.sin(8 * e_t)]).T
        self.points = self.noise_round_points(p)
        return self.points


class SwissHole(Shape):
    def generate(self, npoints, use_cache=True):
        param = 1
        tt = (3 * np.pi / 2.0) * (1 + 2.0 * np.random.rand(2 * npoints))
        h = 21 * np.random.rand(2 * npoints)
        kl = np.zeros(2 * npoints)

        for ii in range(2 * npoints):
            if 9 < tt[ii] < 12:
                if 9 < h[ii] < 14:
                    kl[ii] = 1
        tt = tt[kl == 0]
        h = h[kl == 0]
        p = np.array([tt * np.cos(tt), h, param * tt * np.sin(tt)]).T
        self.points = self.noise_round_points(p)
        self.color = tt
        return self.points


class PuncturedSphere(Shape):
    def generate(self, npoints, use_cache=True):
        param = .5
        inc = 9.0 / np.sqrt(npoints)
        yy, xx = map(lambda z: z.flatten(),
                     np.mgrid[-5:5:inc, -5:5:inc])
        rr2 = xx ** 2 + yy ** 2
        ii = np.argsort(rr2)
        y = np.array([xx[ii[:npoints]].T, yy[ii[:npoints]].T])
        a = 4.0 / (4 + np.sum(y ** 2, axis=0))
        p = np.array([a * y[0, :], a * y[1, :], param * 2 * (1 - a)]).T
        self.points = self.noise_round_points(p)
        self.color = self.points[:, 2]
        return self.points


class CornerPlane(Shape):
    def generate(self, npoints, use_cache=True):
        k = 0
        x_max = int(np.floor(np.sqrt(npoints)))
        y_max = int(np.ceil(npoints / float(x_max)))
        corner_point = int(np.floor(y_max / 2.0))
        p = np.zeros((x_max * y_max, 3))
        color = np.zeros(x_max * y_max)
        param = 330
        for xx in range(0, x_max):
            for yy in range(0, y_max):
                if yy <= corner_point:
                    p[k, :] = np.array([xx, yy, 0])
                    color[k] = yy
                else:
                    p[k, :] = np.array([
                        xx,
                        corner_point + (yy - corner_point) * np.cos(param * np.pi / 180),
                        (yy - corner_point) * np.sin(param * np.pi / 180)])
                    color[k] = yy
                k += 1
        self.points = self.noise_round_points(p)
        self.color = color
        return self.points


class TwinPeaks(Shape):
    def generate(self, npoints, use_cache=True):
        param = 1
        xy = 1 - 2 * np.random.rand(2, npoints)
        p = np.array([
            xy[1, :],
            xy[0, :],
            param * np.sin(np.pi * xy[0, :]) * np.tanh(3 * xy[1, :])]
        ).T
        self.points = self.noise_round_points(p)
        self.color = self.points[:, 2]
        return self.points


class Gaussian(Shape):
    def generate(self, npoints, use_cache=True):
        param = 1
        std = param
        p = std * np.random.randn(npoints, 3)
        p[:, 2] = (1.0 / ((std ** 2) * 2 * np.pi) * np.exp(
            (-p[:, 0] ** 2 - p[:, 1] ** 2) / (2 * std ** 2)))
        self.points = self.noise_round_points(p)
        self.color = self.points[:, 2]
        return self.points


class Clusters3D(Shape):
    def generate(self, npoints, use_cache=True):
        param = 10
        num_clusters = max(1, param)
        colors = cm.rainbow(np.linspace(0, 1, num_clusters + 1))
        centers = 10 * np.random.rand(num_clusters, 3)
        d = common.DISTANCE_MATRIX(centers)
        min_d = np.min(d[d > 0])
        n2 = npoints - (num_clusters - 1) * 9
        p = np.zeros((npoints, 3))
        color = np.zeros((npoints, 4))
        k = 0
        for i in range(num_clusters):
            for j in range(int(np.ceil(n2 / num_clusters))):
                p[k, :] = (
                    centers[i, :] +
                    (np.random.rand(1, 3) - 0.5) * min_d / np.sqrt(12))
                color[k] = colors[i + 1]
                k += 1
            if i < num_clusters - 1:
                for t in np.arange(0.1, 0.9, 0.1):
                    p[k, :] = (
                        centers[i, :] +
                        (centers[i + 1, :] - centers[i, :]) * t)
                    color[k] = colors[0]
                    k = k + 1
        self.points = self.noise_round_points(p)
        self.color = color
        return self.points


class DataBuilder(object):
    def __init__(self):
        self.shape_map = {
            'sphere': Sphere,
            'cut-sphere': CutSphere,
            'ball': Ball,
            'random': Shape,
            'real': Shape,
            'spiral': Spiral,
            'spiral-hole': SpiralHole,
            'swissroll': SwissRoll,
            'swisshole': SwissHole,
            'toroid-helix': ToroidalHelix,
            's-curve': SCurve,
            'punctured-sphere': PuncturedSphere,
            'gaussian': Gaussian,
            'clusters-3d': Clusters3D,
            'twin-peaks': TwinPeaks,
            'corner-plane': CornerPlane
        }
        self.type = None
        self.dim = None
        self.distance = 'euclidean'
        self.npoints = None
        self.use_noise = False
        self.points = None
        self.n_neighbors = 12
        self.noise_std = 1e-2

    def with_type(self, t):
        self.type = t
        return self

    def with_dim(self, dim):
        self.dim = dim
        return self

    def with_distance(self, distance):
        self.distance = distance
        return self

    def with_npoints(self, npoints):
        self.npoints = npoints
        return self

    def with_noise(self, noise_std):
        if noise_std == 0:
            return self
        self.use_noise = True
        self.noise_std = noise_std
        return self

    def with_points(self, points):
        self.type = 'real'
        self.points = points
        return self

    def with_neighbors(self, n_neighbors):
        self.n_neighbors = n_neighbors
        return self

    def build(self):
        shape = self.shape_map[self.type]
        s = shape(X=self.points,
                  dim=self.dim,
                  n_neighbors=self.n_neighbors,
                  use_noise=self.use_noise,
                  noise_std=self.noise_std,
                  n_jobs=4)
        x, d = s.instance(npoints=self.npoints, distance=self.distance)
        return x, d, s.color
