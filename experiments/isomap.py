import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn import datasets, manifold

sys.path.append('../')


import multidimensional
import multidimensional.common
import multidimensional.mds
import multidimensional.point_filters
import multidimensional.radius_updates
import multidimensional.datagen.shapes as datagen

import config

EXPERIMENT_NAME = 'Swissroll'

KEEP_HISTORY = True

ex = Experiment(EXPERIMENT_NAME)
ex.observers.append(MongoObserver.create(
    url=config.SACRED_MONGO_URL,
    db_name=config.SACRED_DB
))


@ex.config
def cfg():
    data_file = None # os.path.join(config.DATA_DIR, 'glove.men.300d.txt')
    data_type = 'swissroll'
    dim = 3
    distance = 'geodesic'
    npoints = 1500
    n_neighbors = 12
    noise_std = 0
    target_dim = 2
    max_turns = 10000


@multidimensional.common.timefunc
def isomap(m, xs):
    return m.fit_transform(xs)


@ex.automain
def experiment(
        data_file, data_type, dim, distance, npoints, n_neighbors,
        noise_std, target_dim, max_turns):
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
    m = manifold.Isomap(n_components=target_dim,
                        n_neighbors=n_neighbors,
                        max_iter=max_turns)
    x = isomap(m, xs)

    fig = plt.figure()

    ax = plt.axes(projection='3d')
    ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2], c=color, cmap=plt.cm.Spectral)
    ax.set_title("Original data")
    #plt.show()

    ax = plt.axes(projection='3d')
    ax.scatter(x[:, 0], x[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title('Projected data')
    plt.show()

    # history = m.history_observer.history
    # for i, error in enumerate(history['error']):
    #     _run.log_scalar('mds.mse.error', error, i + 1)
    # for i, radius in enumerate(history['radius']):
    #     _run.log_scalar('mds.step', radius, i + 1)
    # start_points = history['xs_files'][0]
    # _run.add_artifact(start_points, name='points_start')
    # end_points = history['xs_files'][-1]
    # _run.add_artifact(end_points, name='points_end')
    # if len(history['xs_images']) > 0:
    #     start_image = history['xs_images'][0]
    #     _run.add_artifact(start_image, name='points_image_start')
    #     end_image = history['xs_images'][-1]
    #     _run.add_artifact(end_image, name='points_image_end')
    # if history['animation'] is not None:
    #     _run.add_artifact(history['animation'], name='animation')
    # return m.history_observer.history['error'][-1]
    #
