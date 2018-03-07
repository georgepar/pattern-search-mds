#!/usr/bin/python3

import logging
import os
import time

from datetime import datetime
import numpy as np
from scipy.spatial import distance_matrix

# import config
import common
import datagen.shapes
import point_filters as pf
import radius_updates as ru

from history import  HistoryObserver
# import approximate_mds
# from sklearn.manifold import MDS as skMDS


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
                 dissimilarities='precomputed',
                 uniform_init=True,
                 history_color=None,
                 keep_history=True,
                 history_path=str(datetime.now().strftime("%Y%m%d%H%M%S"))):
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
        self.num_epochs = 0
        self.dissimilarities = dissimilarities
        if keep_history:
            self.history_color = history_color
            self.history_observer = HistoryObserver(path=history_path)

    @staticmethod
    def _log_iteration(turn, radius, prev_error, error):
        LOG.info("Turn {0}: Radius {1}: (prev, error decrease, error): "
                 "({2}, {3}, {4})"
                 .format(turn, radius, prev_error, prev_error - error, error))

    def _stop_conditions(self, turn, patience_cnt, error, radius):
        return (turn >= self.max_turns or
                patience_cnt >= self.patience or
                error <= self.error_barrier or
                radius <= self.radius_barrier)

    def fit(self, x, x_init=None):
        if self.dissimilarities == 'precomputed':
            d_goal = x
            xs = common.random_table(x.shape[0],
                                     self.target_dimensions,
                                     uniform=self.uniform_init)
        else:
            if x_init is not None:
                xs = x_init
            else:
                xs = common.random_table(x.shape[0],
                                         self.target_dimensions,
                                         uniform=self.uniform_init)
            x = x.astype(np.float64)
            d_goal = common.DISTANCE_MATRIX(x)
        d_current = common.DISTANCE_MATRIX(xs)
        points = np.arange(xs.shape[0])

        radius = self.starting_radius
        turn = 0
        radius_burnout = 0
        patience_cnt = 0
        error = common.MSE2(d_goal, d_current)
        prev_error = np.Inf
        #errors = np.zeros(xs.shape[0])
        LOG.info("Starting Error: {}".format(error))

        avg_epoch_time = 0
        if self.keep_history:
            self.history_observer.epoch(turn, radius, error, xs, avg_epoch_time)

        while not self._stop_conditions(
                turn, patience_cnt, error, radius):
            t0 = time.time()
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
                # error = common.HJ(xs, radius, d_current, d_goal, point, error, error_i,
                #         percent=self.explore_dim_percent)a
            t1 = time.time()
            avg_epoch_time += (t1 - t0)
            LOG.info("Epoch took: {}".format(t1 - t0))
            self._log_iteration(turn, radius, prev_error, error)
            if self.keep_history:
                self.history_observer.epoch(turn, radius, error, xs, epoch_time=avg_epoch_time)
        self.num_epochs = turn
        LOG.info("Avg epoch time: {}".format(avg_epoch_time / float(self.num_epochs)))
        LOG.info("Ending Error: {}".format(error))
        return xs

    @common.timemethod
    def fit_transform(self, x, x_init=None):
        return self.fit(x, x_init=x_init)

    def plot_history(self):
        if self.keep_history:
            if self.target_dimensions == 2:
                self.history_observer.plot2d(self.num_epochs,
                                             color=self.history_color)
            if self.target_dimensions == 3:
                self.history_observer.plot3d(self.num_epochs,
                                             color=self.history_color)


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
    X_real, D_goal = shape.instance(npoints=751, distance='euclidean')
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

