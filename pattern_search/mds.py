#!/usr/bin/python3

import logging
import numpy as np
from sklearn.base import BaseEstimator

from . import common

logging.basicConfig()
LOG = logging.getLogger("MDS")
LOG.setLevel(logging.INFO)


def _radius_update(radius, error, prev_error, tolerance=1e-4):
    if error >= prev_error or prev_error - error <= error * tolerance:
        return radius * 0.5, 0
    return radius


def _point_sampling(points, keep_percent=1.0, turn=-1, recalculate_each=-1):
    if keep_percent > 1.0 or 1.0 - keep_percent < 1e-5:
        return points
    if turn > 0 and recalculate_each > 0 and turn % recalculate_each == 0:
        return points
    keep = int(points.shape[0] * keep_percent)
    return np.random.choice(points, size=keep, replace=False)


class MDS(BaseEstimator):
    def __init__(self,
                 target_dimensions,
                 starting_radius=1.0,
                 max_turns=1000,
                 patience=np.Inf,
                 error_barrier=1e-2,
                 radius_barrier=1e-3,
                 explore_dim_percent=1.0,
                 sample_points_percent=1.0,
                 radius_update_tolerance=1e-4,
                 dissimilarity='precomputed'):
        self.radius_update_tolerance = radius_update_tolerance
        self.sample_points = sample_points_percent
        self.target_dimensions = target_dimensions
        self.starting_radius = starting_radius
        self.max_turns = max_turns
        self.patience = patience
        self.error_barrier = error_barrier
        self.radius_barrier = radius_barrier
        self.explore_dim_percent = explore_dim_percent
        self.num_epochs = 0
        self.dissimilarity = dissimilarity

    @staticmethod
    def _log_iteration(turn, radius, prev_error, error):
        LOG.debug("Turn {0}: Radius {1}: (prev, error decrease, error): "
                  "({2}, {3}, {4})"
                  .format(turn, radius, prev_error, prev_error - error, error))

    def _stop_conditions(self, turn, patience_cnt, error, radius):
        return (turn >= self.max_turns or
                patience_cnt >= self.patience or
                error <= self.error_barrier or
                radius <= self.radius_barrier)

    def fit(self, X, init=None):
        if self.dissimilarity == 'precomputed':
            d_goal = X
            xs = np.random.rand(X.shape[0], self.target_dimensions)
        else:
            if init is not None:
                xs = init
            else:
                xs = np.random.rand(X.shape[0], self.target_dimensions)
            X = X.astype(np.float64)
            d_goal = common.DISTANCE_MATRIX(X)
        d_current = common.DISTANCE_MATRIX(xs)
        points = np.arange(xs.shape[0])

        radius = self.starting_radius
        turn = 0
        patience_cnt = 0
        error = common.MSE2(d_goal, d_current)
        prev_error = np.Inf
        LOG.debug("Starting Error: {}".format(error))

        while not self._stop_conditions(turn, patience_cnt, error, radius):
            turn += 1

            if error >= prev_error:
                patience_cnt += 1
            radius = _radius_update(radius, error, prev_error, tolerance=self.radius_update_tolerance)
            prev_error = error
            filtered_points = _point_sampling(points, keep_percent=self.sample_points)
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
        self.num_epochs = turn
        LOG.debug("Ending Error: {}".format(error))
        return xs

    @common.timemethod
    def fit_transform(self, X, init=None):
        return self.fit(X, init=init)
