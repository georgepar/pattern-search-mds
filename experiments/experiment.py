from collections import namedtuple

import os
import sys
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn import manifold

sys.path.append('../')


import multidimensional
import multidimensional.common
import multidimensional.mds
import multidimensional.point_filters
import multidimensional.radius_updates
import multidimensional.datagen.shapes as datagen

import config

EXPERIMENT_NAME = 'Swissroll'

KEEP_HISTORY = False

ex = Experiment(EXPERIMENT_NAME)
ex.observers.append(MongoObserver.create(
    url=config.SACRED_MONGO_URL,
    db_name=config.SACRED_DB
))


RESULT_IMAGE = 'punctured_sphere_noise.png'

@ex.config
def cfg():
    data_file = None # os.path.join(config.DATA_DIR, 'glove.men.300d.txt')
    data_type = 'punctured-sphere' # 'toroid-helix'
    dim = 100
    distance = 'geodesic'
    npoints = 1000
    n_neighbors = 12
    noise_std = 1e-2
    target_dim = 2
    point_filter = (multidimensional
                    .point_filters
                    .FixedStochasticFilter(keep_percent=1))
    radius_update = (multidimensional
                     .radius_updates
                     .AdaRadiusHalving(tolerance=1e-3))
    radius_barrier = 1e-3
    explore_dim_percent = 1
    starting_radius = 1
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
    dim_reduction = namedtuple('dim_reduction', 'name method data')
    MDS_proposed = dim_reduction(
        'MDS proposed',
        multidimensional.mds.MDS(
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
            dissimilarities='precomputed'),
        d_goal)
    LLE = dim_reduction(
        'LLE',
        manifold.LocallyLinearEmbedding(n_neighbors,
                                        target_dim,
                                        eigen_solver='auto',
                                        method='standard'),
        xs)
    LTSA = dim_reduction(
        'LTSA',
        manifold.LocallyLinearEmbedding(n_neighbors,
                                        target_dim,
                                        eigen_solver='auto',
                                        method='ltsa'),
        xs)
    HessianLLE = dim_reduction(
        'HessianLLE',
        manifold.LocallyLinearEmbedding(n_neighbors,
                                        target_dim,
                                        eigen_solver='auto',
                                        method='hessian'),
        xs)
    ModifiedLLE = dim_reduction(
        'ModifiedLLE',
        manifold.LocallyLinearEmbedding(n_neighbors,
                                        target_dim,
                                        eigen_solver='auto',
                                        method='modified'),
        xs)
    Isomap = dim_reduction(
        'Isomap',
        manifold.Isomap(n_neighbors, target_dim),
        xs)
    mds = dim_reduction(
        'MDS SMACOF',
        manifold.MDS(n_components=target_dim,
                     n_init=1,
                     max_iter=max_turns,
                     verbose=2,
                     dissimilarity='precomputed'),
        d_goal)
    SpectralEmbedding = dim_reduction(
        'SpectralEmbedding',
        manifold.SpectralEmbedding(n_components=target_dim,
                                   n_neighbors=n_neighbors),
        xs)
    tSNE = dim_reduction(
        'tSNE',
        manifold.TSNE(n_components=target_dim, init='pca', random_state=0),
        xs)

    methods = [MDS_proposed, mds, Isomap, LLE, HessianLLE, ModifiedLLE, LTSA]

    fig = plt.figure(figsize=(20, 10))
    plt.suptitle("Manifold Learning with %i points, %i neighbors, %.3f noise"
                 % (npoints, n_neighbors, noise_std), fontsize=14)
    ax = fig.add_subplot(251, projection='3d')
    ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2], c=color, cmap=plt.cm.Spectral)
    plt.title("Original Manifold")
    for i, method in enumerate(methods):
        print("Running {}".format(methods[i].name))
        try:
            t0 = time.time()
            x = methods[i].method.fit_transform(methods[i].data)
            t1 = time.time()
            ax = fig.add_subplot(2, 5, i + 2)

            # Plot the 2 dimensions.
            plt.scatter(x[:, 0], x[:, 1], c=color, cmap=plt.cm.Spectral)
            plt.title(methods[i].name + "(%.2g sec)" % (t1-t0))
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            plt.axis('tight')
            # With high noise level, some of the models fail.
        except Exception as e:
            print(e)
            ax = fig.add_subplot(2, 5, i + 2)
            plt.title(methods[i].name + " did not run")
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            plt.axis('tight')
    plt.savefig(RESULT_IMAGE)
    plt.show()

    # m.plot_history()

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

