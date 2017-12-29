import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver

sys.path.append('../')


import multidimensional
import multidimensional.common
import multidimensional.mds
import multidimensional.point_filters
import multidimensional.radius_updates
import multidimensional.datagen.shapes as datagen
from multidimensional import smacof

import config

EXPERIMENT_NAME = 'Gaussian_smacof'

ex = Experiment(EXPERIMENT_NAME)
ex.observers.append(MongoObserver.create(
    url=config.SACRED_MONGO_URL,
    db_name=config.SACRED_DB
))


@ex.config
def cfg():
    data_file = None # os.path.join(config.DATA_DIR, 'glove.men.300d.txt')
    data_type = 'gaussian'
    dim = 3
    distance = 'geodesic'
    npoints = 1000
    n_neighbors = 12
    noise_std = 0
    target_dim = 2
    max_turns = 10000


@multidimensional.common.timefunc
def smacof_fit(m, d_goal):
    return m.fit_transform(d_goal)


@ex.automain
def experiment(
        data_file, data_type, dim, distance, npoints, n_neighbors,
        noise_std, target_dim, max_turns, _run):
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
    m = smacof.MDS(n_components=target_dim,
                   n_init=1,
                   max_iter=max_turns,
                   verbose=2,
                   dissimilarity='precomputed',
                   history_path=EXPERIMENT_NAME)
    x = smacof_fit(m, d_goal)
    m.history_observer.plot(m.n_iter_, target_dim)

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
