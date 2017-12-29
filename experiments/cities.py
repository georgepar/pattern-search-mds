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


RESULT_IMAGE = 'cities_noise.png'

@ex.config
def cfg():
    data_file = 'cities.csv'  # os.path.join(config.DATA_DIR, 'glove.men.300d.txt')
    target_dim = 2
    noise = 0#100000
    point_filter = (multidimensional
                    .point_filters
                    .FixedStochasticFilter(keep_percent=1))
    radius_update = (multidimensional
                     .radius_updates
                     .AdaRadiusHalving(tolerance=1e-20))
    radius_barrier = 1e-10
    explore_dim_percent = 1
    starting_radius = 256
    max_turns = 10000


@ex.automain
def experiment(
        data_file, target_dim, noise, point_filter, radius_update, radius_barrier,
        explore_dim_percent, starting_radius, max_turns, _run):
    with open(data_file) as fd:
        lines = [l for l in fd.readlines()]
        symbols = lines[0].strip().split(',')
        d_goal = np.array([map(float, l.strip().split(','))
                           for l in lines[1:]])

    if noise > 0:
        d_goal[4, 6] += noise
        d_goal[6, 4] += noise

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
            history_color=None,
            history_path=EXPERIMENT_NAME,
            dissimilarities='precomputed'),
        d_goal)
    mds = dim_reduction(
        'MDS SMACOF',
        manifold.MDS(n_components=target_dim,
                     n_init=4,
                     max_iter=max_turns,
                     verbose=2,
                     dissimilarity='precomputed'),
        d_goal)

    methods = [MDS_proposed, mds]

    fig = plt.figure(figsize=(20, 10))
    plt.suptitle("US cities distances", fontsize=14)

    x = methods[0].method.fit_transform(methods[0].data)
    #x[:, 1] = -x[:, 1]
    #x = -x
    x[:, 0] = -x[:, 0]
    ax = fig.add_subplot(211)
    ax.scatter(x[:, 0], x[:, 1])
    for i, txt in enumerate(symbols):
        ax.annotate(symbols[i], (x[i, 0], x[i, 1]))
    plt.title("MDS proposed")

    x = methods[1].method.fit_transform(methods[1].data)
    x[:, 0] = -x[:, 0]
    #x = -x
    ax = fig.add_subplot(212)
    ax.scatter(x[:, 0], x[:, 1])
    for i, txt in enumerate(symbols):
        ax.annotate(symbols[i], (x[i, 0], x[i, 1]))
    plt.title("SMACOF MDS")

    #plt.savefig(RESULT_IMAGE)
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

