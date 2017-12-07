#!/usr/bin/python3

import logging
import os
import sys

from datetime import datetime
import numpy as np
from scipy.spatial import distance_matrix

import config
import common
import datagen.shapes

import point_filters as pf
import radius_updates as ru

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
        pertubations_unscaled = self._init_pertubations(self.target_dimensions)
        points = np.arange(xs.shape[0])

        radius = self.starting_radius
        turn = 0
        radius_burnout = 0
        patience_cnt = 0
        error = common.compute_mds_error(d_goal, d_current)
        prev_error = np.Inf
        LOG.info("Starting Error: {}".format(error))

        if self.keep_history:
            self._history(turn, radius, error, xs)

        prev_pertubations = np.zeros((xs.shape[0], xs.shape[1]))

        while not self._stop_conditions(
                turn, patience_cnt, error, radius):
            xorg = xs.copy()
            turn += 1
            radius_burnout += 1

            if error >= prev_error:
                patience_cnt += 1
            radius, radius_burnout = self.radius_update.update(
                radius, turn, error, prev_error, burnout=radius_burnout)
            prev_error = error
            pertubations = radius * pertubations_unscaled

            filtered_points = self.point_filter.filter(
                points, turn=turn, d_goal=d_goal, d_current=d_current)
            for point in filtered_points:
                error_i = common.MSE(d_goal[point], d_current[point])
                optimum_error, optimum_k = common.BEST_PERTUBATION(
                    xs, xorg, pertubations, d_current, d_goal, point)
                xs[point] = xorg[point] + pertubations[optimum_k]
                prev_pertubations[point, :] = pertubations[optimum_k]
                d_current = common.UPDATE_DISTANCE_MATRIX(xs, d_current, point)
                error -= (error_i - optimum_error)
            self._log_iteration(turn, radius, prev_error, error)
            if self.keep_history:
                self._history(turn, radius, error, xs)
        self._history_done(turn)
        LOG.info("Ending Error: {}".format(error))
        return xs


if __name__ == '__main__':
    np.random.seed(42)
    shape = datagen.shapes.Sphere(use_noise=False)
    #shape = datagen.shapes.Ball(use_noise=False)
    X_real, D_goal = shape.instance(1404, distance='geodesic')
    point_f = pf.StochasticFilter()
    rad_up = ru.AdaRadiusHalving()
    m = MDS(2, point_f, rad_up, starting_radius=1, keep_history=True)
    x_mds = m.fit(X_real, d_goal=D_goal, init_x=False)
