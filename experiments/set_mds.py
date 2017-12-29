from collections import namedtuple

import os
import sys
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from sacred import Experiment
from sacred.observers import MongoObserver

sys.path.append('../')


import multidimensional
import multidimensional.common
import multidimensional.mds
import multidimensional.point_filters
import multidimensional.radius_updates
import multidimensional.datagen.shapes as datagen

import config

EXPERIMENT_NAME = 'Swissroll_big_set-MDS'

KEEP_HISTORY = True

ex = Experiment(EXPERIMENT_NAME)
ex.observers.append(MongoObserver.create(
    url=config.SACRED_MONGO_URL,
    db_name=config.SACRED_DB
    #db_name='test'
))


@ex.config
def cfg():
    data_file = None # os.path.join(config.DATA_DIR, 'glove.men.300d.txt')
    data_type = 'gaussian' # 'toroid-helix'
    dim = 100
    distance = 'geodesic'
    npoints = 5000
    n_neighbors = 12
    noise_std = 0
    target_dim = 2
    point_filter = (multidimensional
                    .point_filters
                    .FixedStochasticFilter(keep_percent=1))
    radius_update = (multidimensional
                     .radius_updates
                     .AdaRadiusHalving(tolerance=1e-3))
    radius_barrier = 1e-3
    explore_dim_percent = 1
    starting_radius = 8
    max_turns = 10000


@ex.automain
def experiment(
        data_file, data_type, dim, distance, npoints, n_neighbors,
        noise_std, target_dim, point_filter, radius_update, radius_barrier,
        explore_dim_percent, starting_radius, max_turns, _run):
    xs = None
    if data_file is not None:
        xs = multidimensional.common.load_embeddings(data_file)

    xs, d_goal, color = (datagen.DataBuilder()
                         .with_dim(dim)
                         .with_distance(distance)
                         .with_noise(noise_std)
                         .with_npoints(npoints)
                         .with_neighbors(n_neighbors)
                         .with_points(xs)
                         .with_type(data_type)
                         .build())

    m = multidimensional.mds.MDS(
        target_dim,
        point_filter,
        radius_update,
        starting_radius=starting_radius,
        radius_barrier=radius_barrier,
        max_turns=max_turns,
        explore_dim_percent=explore_dim_percent,
        keep_history=KEEP_HISTORY,
        history_color=color,
        history_path=EXPERIMENT_NAME,
        dissimilarities='precomputed')
    x = m.fit(d_goal)

    m.plot_history()

    history = m.history_observer.history
    for i, error in enumerate(history['error']):
        _run.log_scalar('mds.mse.error', error, i + 1)
    for i, radius in enumerate(history['radius']):
        _run.log_scalar('mds.step', radius, i + 1)
    start_points = history['xs_files'][0]
    _run.add_artifact(start_points, name='points_start')
    end_points = history['xs_files'][-1]
    _run.add_artifact(end_points, name='points_end')
    if len(history['xs_images']) > 0:
        start_image = history['xs_images'][0]
        _run.add_artifact(start_image, name='points_image_start')
        end_image = history['xs_images'][-1]
        _run.add_artifact(end_image, name='points_image_end')
    if history['animation'] is not None:
        _run.add_artifact(history['animation'], name='animation')
    return m.history_observer.history['error'][-1]

