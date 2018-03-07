from collections import namedtuple

import os
import sys
import time

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import NullFormatter

import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn import manifold, decomposition

sys.path.append('../')


import multidimensional
import multidimensional.common
import multidimensional.mds
import multidimensional.point_filters
import multidimensional.radius_updates
import multidimensional.datagen.shapes as datagen
import multidimensional.smacof

import config


EXPERIMENT_NAME = 'clusters_3d_overshoot'

KEEP_HISTORY = True

ex = Experiment(EXPERIMENT_NAME)
ex.observers.append(MongoObserver.create(
    url=config.SACRED_MONGO_URL,
    db_name=config.SACRED_DB
    # db_name='test'
))


RESULT_IMAGE = EXPERIMENT_NAME + '.png'

@ex.config
def cfg():
    data_type = 'swissroll'
    # {
    #     'sphere': Sphere,
    #     'cut-sphere': CutSphere,
    #     'ball': Ball,
    #     'random': Shape,
    #     'real': Shape,
    #     'spiral': Spiral,
    #     'spiral-hole': SpiralHole,
    #     'swissroll': SwissRoll,
    #     'swisshole': SwissHole,
    #     'toroid-helix': ToroidalHelix,
    #     's-curve': SCurve,
    #     'punctured-sphere': PuncturedSphere,
    #     'gaussian': Gaussian,
    #     'clusters-3d': Clusters3D,
    #     'twin-peaks': TwinPeaks,
    #     'corner-plane': CornerPlane
    # }
    dim = 3
    distance = 'geodesic'
    npoints = 3000
    n_neighbors = 30
    noise_std = 0
    target_dim = 2
    point_filter = (multidimensional
                    .point_filters
                    .FixedStochasticFilter(keep_percent=1))
    radius_update = (multidimensional
                     .radius_updates
                     .AdaRadiusHalving(tolerance=1e-4))
    radius_barrier = 1e-3
    explore_dim_percent = 1
    starting_radius = 32
    max_turns = 10000


@ex.automain
def experiment(
        data_type, dim, distance, npoints, n_neighbors,
        noise_std, target_dim, point_filter, radius_update, radius_barrier,
        explore_dim_percent, starting_radius, max_turns, _run):

    xs = None
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
    mds = dim_reduction(
        'r0=16',
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
            history_path=EXPERIMENT_NAME+'_mds_proposed',
            dissimilarities='precomputed'),
        d_goal)

    mds_pess = dim_reduction(
        'r0=1',
        multidimensional.mds.MDS(
            target_dim,
            point_filter,
            radius_update,
            starting_radius=1,
            radius_barrier=radius_barrier,
            max_turns=max_turns,
            explore_dim_percent=explore_dim_percent,
            keep_history=KEEP_HISTORY,
            history_color=color,
            history_path=EXPERIMENT_NAME+'_mds_proposed',
            dissimilarities='precomputed'),
        d_goal)

    mds_opt = dim_reduction(
        'r0=65536',
        multidimensional.mds.MDS(
            target_dim,
            point_filter,
            radius_update,
            starting_radius=65536,
            radius_barrier=radius_barrier,
            max_turns=max_turns,
            explore_dim_percent=explore_dim_percent,
            keep_history=KEEP_HISTORY,
            history_color=color,
            history_path=EXPERIMENT_NAME+'_mds_proposed',
            dissimilarities='precomputed'),
        d_goal)


    methods = [mds, mds_pess, mds_opt]
    #methods = [MDS_proposed, mds]
    fig = plt.figure(figsize=(20, 10))

    #plt.suptitle("Learning %s with %i points, %.3f noise"
    #             % (data_type, npoints, noise_std), fontsize=14)
    ax = fig.add_subplot(141, projection='3d', aspect=1)
    ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2], c=color, cmap=plt.cm.Spectral)
    plt.title("Original Manifold", fontsize=32)
    for i, method in enumerate(methods):
        print("Running {}".format(methods[i].name))
        try:
            t0 = time.time()
            x = methods[i].method.fit_transform(methods[i].data)
            t1 = time.time()
            ax = fig.add_subplot("14{}".format(i + 2), aspect=1)
            # Plot the 2 dimensions.
            ax.scatter(x[:, 0], x[:, 1], c=color, cmap=plt.cm.Spectral)
            plt.title(methods[i].name)
            #ax.xaxis.set_major_formatter(NullFormatter())
            #ax.yaxis.set_major_formatter(NullFormatter())
            plt.axis('tight')

            #plt.show()
            # With high noise level, some of the models fail.
        except Exception as e:
            print(e)
            ax = fig.add_subplot("33{}".format(i + 2), aspect=1)
            plt.title(methods[i].name + " did not run", fontsize=32)
            # ax.xaxis.set_major_formatter(NullFormatter())
            # ax.yaxis.set_major_formatter(NullFormatter())
            plt.axis('tight')
    plt.tight_layout()
    plt.savefig(RESULT_IMAGE)
    plt.show()

    # m.plot_history()

    history = mds.method.history_observer.history
    for i, error in enumerate(history['error']):
        _run.log_scalar('mds.mse.error', error, i + 1)
    for i, radius in enumerate(history['radius']):
        _run.log_scalar('mds.step', radius, i + 1)
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

    history = mds_pess.method.history_observer.history
    for i, error in enumerate(history['error']):
        _run.log_scalar('pessimistic.mse.error', error, i + 1)
    for i, radius in enumerate(history['radius']):
        _run.log_scalar('pessimistic.step', radius, i + 1)

    history = mds_opt.method.history_observer.history
    for i, error in enumerate(history['error']):
        _run.log_scalar('optimistic.mse.error', error, i + 1)
    for i, radius in enumerate(history['radius']):
        _run.log_scalar('optimistic.step', radius, i + 1)


    return mds.method.history_observer.history['error'][-1]

