#!/usr/bin/python3
import errno
import functools
import os
import time

import numpy as np
from scipy.spatial import distance_matrix
import mds_utils


UPDATE_DISTANCE_MATRIX = mds_utils.update_distance_matrix
UPDATE_DISTANCE_MATRIX2 = mds_utils.update_distance_matrix2
BEST_PERTUBATION = mds_utils.pertub_error
BEST_PERTUBATION2 = mds_utils.pertub_error1
NORMR = mds_utils.dist_from_point
MSE = mds_utils.mse
MSE2 = mds_utils.mse2
SUM = np.sum
SQRT = np.sqrt


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def timefunc(func):
    """
    Decorator that measure the time it takes for a function to complete
    Usage:
      @timethis
      def time_consuming_function(...):
    """
    @functools.wraps(func)
    def timed(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        elapsed = '{0}'.format(te - ts)
        print('{f} took: {t} sec'.format(
            f=func.__name__, t=elapsed))
        return result
    return timed


def timemethod(func):
    """
    Decorator that measure the time it takes for a function to complete
    Usage:
      @timethis
      def time_consuming_function(...):
    """
    @functools.wraps(func)
    def timed(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        elapsed = '{0}'.format(te - ts)
        print('{c}.{f} took: {t} sec'.format(
            c=args[0].__class__.__name__, f=func.__name__, t=elapsed))
        return result
    return timed


def random_table(n, m, uniform=True):
    if uniform:
        return np.random.rand(n, m)
    else:
        return np.random.randn(n, m)


def compute_mds_error(d_goal, d_current, axis=None):
    return SUM((d_goal - d_current) ** 2, axis=axis) / 2.0


def update_distance_matrix(xs, d_current, idx, i):
    norm = NORMR(xs[idx], xs[i])
    d_current[idx, i] = norm
    d_current[i, idx] = norm
    return d_current


def order_of_magnitude(x):
    return int(np.floor(np.log10(x)))


def instance(num_vectors, dimensions):
    print("Dimensions=", dimensions)
    x_real = random_table(num_vectors, dimensions)
    return x_real, distance_matrix(x_real, x_real)
