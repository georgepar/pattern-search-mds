import os
import sys
sys.path.append('../')

import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver

import multidimensional
import multidimensional.common
import multidimensional.mds
import multidimensional.point_filters
import multidimensional.radius_updates

import config

ex = Experiment("1404 vectors - 3 -> 2 dimensions - Sphere Data - New version")
ex.observers.append(MongoObserver.create(
    url=config.SACRED_MONGO_URL,
    db_name=config.SACRED_DB
))


@ex.config
def cfg():
    coords_file = os.path.join(config.DATA_DIR, 'sphere_coords.dat')
    distance_matrix_file = os.path.join(config.DATA_DIR,
                                        'Geodesic_sphere.dat')
    target_dim = 2
    point_filter = multidimensional.point_filters.StochasticFilter(
        min_points_per_turn=0.1, max_points_per_turn=0.8)
    radius_update = multidimensional.radius_updates.AdaRadiusHalving()


@ex.automain
def experiment(coords_file,
               distance_matrix_file,
               target_dim,
               point_filter,
               radius_update,
               _run):
    xs = np.loadtxt(coords_file, delimiter=',')
    d_goal = np.loadtxt(distance_matrix_file, delimiter=',')
    m = multidimensional.mds.MDS(
        target_dim, point_filter, radius_update, keep_history=True)
    m.fit(xs, d_goal=d_goal)
    for i, error in enumerate(m.history['error']):
        _run.log_scalar('mds.mse.error', error, i + 1)
    for i, radius in enumerate(m.history['radius']):
        _run.log_scalar('mds.step', radius, i + 1)

    return m.history['error'][-1]
        # for i, xs in enumerate(m.history['xs_files']):
    #     _run.add_artifact(xs, name='xs_{}'.format(i + 1))
