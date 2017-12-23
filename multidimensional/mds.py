#!/usr/bin/python3

import logging
import os

from datetime import datetime
import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import euclidean_distances
import config
import common
import datagen.shapes

import point_filters as pf
import radius_updates as ru
import approximate_mds
from sklearn.manifold import MDS as skMDS

logging.basicConfig()
LOG = logging.getLogger("MDS")
LOG.setLevel(logging.DEBUG)


class MDS(object):
    def __init__(self,
                 target_dimensions,
                 point_filter,
                 radius_update,
                 starting_radius=1.0,
                 max_turns=1000,
                 patience=np.Inf,
                 error_barrier=1e-2,
                 radius_barrier=1e-3,
                 explore_dim_percent=.5,
                 uniform_init=True,
                 keep_history=True):
        self.target_dimensions = target_dimensions
        self.point_filter = point_filter
        self.radius_update = radius_update
        self.starting_radius = starting_radius
        self.max_turns = max_turns
        self.patience = patience
        self.error_barrier = error_barrier
        self.radius_barrier = radius_barrier
        self.uniform_init = uniform_init
        self.keep_history = keep_history
        self.explore_dim_percent = explore_dim_percent
        self.history = {}
        if keep_history:
            self.history = {
                'radius': [],
                'error': [],
                'xs_files': []
            }
            self.history_path = os.path.join(
                config.HISTORY_DIR,
                str(datetime.now().strftime("%Y%m%d%H%M%S")))
            common.mkdir_p(self.history_path)

    @staticmethod
    def _log_iteration(turn, radius, prev_error, error):
        LOG.info("Turn {0}: Radius {1}: (prev, error decrease, error): "
                 "({2}, {3}, {4})"
                 .format(turn, radius, prev_error, prev_error - error, error))

    @staticmethod
    def _init_pertubations(dim):
        return np.concatenate((-np.eye(dim), np.eye(dim)))

    def _stop_conditions(self, turn, patience_cnt, error, radius):
        return (turn >= self.max_turns or
                patience_cnt >= self.patience or
                error <= self.error_barrier or
                radius <= self.radius_barrier)

    def _history(self, turn, radius, error, xs):

        xs_file = os.path.join(self.history_path, 'epoch_{}'.format(turn))
        np.savetxt(xs_file, xs, delimiter=',')
        self.history['radius'].append(radius)
        self.history['error'].append(error)
        self.history['xs_files'].append(xs_file)

    def _history_done(self, turn):
        cfg_file = os.path.join(self.history_path, 'config')
        with open(cfg_file, 'w') as fd:
            fd.writelines([
                'DIM={}\n'.format(self.target_dimensions),
                'EPOCHS={}'.format(turn)
            ])

    @common.timemethod
    def fit(self, x, d_goal=None, init_x=False):
        if init_x:
            xs = x
        else:
            xs = common.random_table(x.shape[0],
                                     self.target_dimensions,
                                     uniform=self.uniform_init)
        d_goal = d_goal if d_goal is not None else distance_matrix(x, x)
        d_current = distance_matrix(xs, xs)
        points = np.arange(xs.shape[0])

        radius = self.starting_radius
        turn = 0
        radius_burnout = 0
        patience_cnt = 0
        error = common.MSE2(d_goal, d_current)
        prev_error = np.Inf
        LOG.info("Starting Error: {}".format(error))

        if self.keep_history:
            self._history(turn, radius, error, xs)

        while not self._stop_conditions(
                turn, patience_cnt, error, radius):
            turn += 1
            radius_burnout += 1

            if error >= prev_error:
                patience_cnt += 1
            radius, radius_burnout = self.radius_update.update(
                radius, turn, error, prev_error, burnout=radius_burnout)
            prev_error = error
            filtered_points = self.point_filter.filter(
                points, turn=turn, d_goal=d_goal, d_current=d_current)
            test_error = error
            for point in filtered_points:
                error_i = common.MSE(d_goal[point], d_current[point])
                optimum_error, optimum_k, optimum_step = (
                    common.BEST_PERTUBATION(
                        xs, radius, d_current, d_goal, point,
                        percent=self.explore_dim_percent))
                test_error -= (error_i - optimum_error)
                d_current = common.UPDATE_DISTANCE_MATRIX(
                    xs, d_current, point, optimum_step, optimum_k)
                xs[point, optimum_k] += optimum_step
                error = test_error
            self._log_iteration(turn, radius, prev_error, error)
            if self.keep_history:
                self._history(turn, radius, error, xs)
                self._history_done(turn)
        LOG.info("Ending Error: {}".format(error))
        return xs


@common.timefunc
def smacof(m, D_goal, x_init=None):
    x_mds =m.fit_transform(D_goal, init=x_init)
    return x_mds

if __name__ == '__main__':
    X = None
    with open('/home/geopar/projects/hidden-set-mds/real_data/glove.men.300d.txt') as fd:
        d = [map(float, l.strip().split()[1:]) for l in fd.readlines()]
        X = np.array(d)
        print(X.shape)
    np.random.seed(42)
    shape = datagen.shapes.Shape(X=X, use_noise=False, dim=300)
    X_real, D_goal = shape.instance(751, distance='euclidean')
    print(np.all(X == X_real))
    point_f = pf.FixedStochasticFilter(keep_percent=1)
    rad_up = ru.AdaRadiusHalving(tolerance=.5 * 1e-3)
    x = X_real
    m = MDS(
        100, point_f, rad_up,
        starting_radius=4,
        explore_dim_percent=.3,
        max_turns=10000,
        keep_history=False)

    x_mds = m.fit(x, d_goal=D_goal, init_x=False)
    #m = skMDS(verbose=2, max_iter=100000, n_components=100, dissimilarity='precomputed', n_init=4, n_jobs=4)

    #smacof(m, D_goal, x_init=None)

    # np.savetxt('glove.men.mds.50d.txt', x_mds, delimiter=' ')

