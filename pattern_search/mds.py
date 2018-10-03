import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_random_state

from mds_fast import (
    distance_matrix,
    update_distance_matrix,
    c_pertub_error as best_pertubation,
    mse as mse1d,
    mse2 as mse2d,
)


def _log_iteration(turn, radius, prev_error, error):
    print("Turn {0}: Radius {1}: (prev, error decrease, error): "
          "({2}, {3}, {4})"
          .format(turn, radius, prev_error, prev_error - error, error))


def _radius_update(radius, error, prev_error, tolerance=1e-4):
    if error >= prev_error or prev_error - error <= error * tolerance:
        return radius * 0.5
    return radius


def _point_sampling(points, keep_percent=1.0, turn=-1, recalculate_each=-1):
    if keep_percent > 1.0 or 1.0 - keep_percent < 1e-5:
        return points
    if turn > 0 and recalculate_each > 0 and turn % recalculate_each == 0:
        return points
    keep = int(points.shape[0] * keep_percent)
    return np.random.choice(points, size=keep, replace=False)


def pattern_search_mds(d_goal, init=None, n_components=2, starting_radius=1.0,
                       radius_update_tolerance=1e-4, sample_points=1.0,
                       explore_dim_percent=1.0, max_iter=1000,
                       radius_barrier=1e-3, n_jobs=1, verbose=0,
                       random_state=None):
    n_samples = d_goal.shape[0]
    random_state = check_random_state(random_state)
    xs = (init if init is not None
          else random_state.rand(n_samples, n_components))
    d_current = distance_matrix(xs)
    points = np.arange(xs.shape[0])

    radius = starting_radius
    turn = 0
    error = mse2d(d_goal, d_current)
    prev_error = np.Inf
    if verbose:
        print("Starting Error: {}".format(error))

    while turn <= max_iter and radius > radius_barrier:
        turn += 1
        radius = _radius_update(
            radius, error, prev_error, tolerance=radius_update_tolerance)
        prev_error = error
        filtered_points = _point_sampling(points, keep_percent=sample_points)
        for point in filtered_points:
            point_error = mse1d(d_goal[point], d_current[point])
            optimum_error, optimum_k, optimum_step = best_pertubation(
                xs, radius, d_current, d_goal, point,
                percent=explore_dim_percent, n_jobs=n_jobs)
            error -= (point_error - optimum_error)
            d_current = update_distance_matrix(
                xs, d_current, point, optimum_step, optimum_k)
            xs[point, optimum_k] += optimum_step
        if verbose >= 2:
            _log_iteration(turn, radius, prev_error, error)
    if verbose:
        print("Ending Error: {}".format(error))
    return xs, error, turn


class MDS(BaseEstimator):
    def __init__(self,
                 n_components=2,
                 starting_radius=1.0,
                 max_iter=1000,
                 radius_barrier=1e-3,
                 explore_dim_percent=1.0,
                 sample_points=1.0,
                 radius_update_tolerance=1e-4,
                 verbose=0,
                 random_state=None,
                 n_jobs=1,
                 dissimilarity='precomputed'):
        self.radius_update_tolerance = radius_update_tolerance
        self.sample_points = sample_points
        self.n_components = n_components
        self.starting_radius = starting_radius
        self.max_iter = max_iter
        self.radius_barrier = radius_barrier
        self.explore_dim_percent = explore_dim_percent
        self.num_epochs = 0
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = 1
        self.dissimilarity = dissimilarity

    def fit_transform(self, X, init=None):
        X = X.astype(np.float64)
        X = check_array(X)
        d_goal = (X if self.dissimilarity == 'precomputed'
                  else distance_matrix(X))
        self.embedding_, self.error_, self.n_iter_ = pattern_search_mds(
            d_goal, init=init, n_components=self.n_components,
            starting_radius=self.starting_radius, max_iter=self.max_iter,
            sample_points=self.sample_points,
            explore_dim_percent=self.explore_dim_percent,
            radius_update_tolerance=self.radius_update_tolerance,
            radius_barrier=self.radius_barrier,
            n_jobs=self.n_jobs, verbose=self.verbose,
            random_state=self.random_state
        )
        return self.embedding_

    def fit(self, X, init=None):
        self.fit_transform(X, init=init)
        return self
