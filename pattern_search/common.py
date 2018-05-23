#!/usr/bin/python3
import errno
import functools
import os
import time

import numpy as np
from scipy.spatial import distance_matrix
import mds_utils

DISTANCE_MATRIX = mds_utils.distance_matrix
UPDATE_DISTANCE_MATRIX = mds_utils.update_distance_matrix
BEST_PERTUBATION = mds_utils.c_pertub_error
NORMR = mds_utils.dist_from_point
MSE = mds_utils.mse
MSE2 = mds_utils.mse2
SUM = np.sum
SQRT = np.sqrt
HJ = mds_utils.hooke_j


def load_embeddings(data_file):
    with open(data_file, 'r') as fd:
        lines = fd.readlines()
        words = [l.strip().split()[0] for l in lines]
        d = [map(float, l.strip().split()[1:]) for l in lines]
        xs = np.array(d)
    return words, xs


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
      @timefunc
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
      @timemethod
      def time_consuming_method(...):
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
