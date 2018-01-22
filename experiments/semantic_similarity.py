from collections import namedtuple

import os
import sys
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn import manifold, decomposition

try:
    import ujson as json
except ImportError:
    import json

try:
    import cPickle as pickle
except ImportError:
    import pickle

sys.path.append('../')
sys.path.append('../mmfeat')

import multidimensional.approximate_mds as amds
import multidimensional
import multidimensional.common
import multidimensional.mds
import multidimensional.smacof
import multidimensional.point_filters
import multidimensional.radius_updates
import multidimensional.datagen.shapes as datagen

import config
import multidimensional.config as project_config

import mmfeat
import mmfeat.space

EXPERIMENT_NAME = 'Semantic_similarity_MEN'

KEEP_HISTORY = False

ex = Experiment(EXPERIMENT_NAME)
ex.observers.append(MongoObserver.create(
    url=config.SACRED_MONGO_URL,
    #db_name=config.SACRED_DB
    db_name='test'
))


RESULT_IMAGE = 'semantic_similarity_men.png'

@ex.config
def cfg():
    ground_truth_men = os.path.join(
        project_config.BASE_PATH,
        'mmfeat/datasets/men.json')
    ground_truth_simlex = os.path.join(
        project_config.BASE_PATH,
        'mmfeat/datasets/simlex.json')
    data_file = os.path.join(
       project_config.BASE_PATH,
       'real_data/glove.semantic.300d.txt')
    data_file_small = os.path.join(
        project_config.BASE_PATH,
        'real_data/glove.semantic.50d.txt')
    # data_file = os.path.join(
    #     project_config.BASE_PATH,
    #     'real_data/simrel-mikolov.pkl'
    # )
    data_type = 'real' # 'toroid-helix'
    dim = 300
    distance = 'euclidean'
    npoints = 1577
    n_neighbors = 66
    noise_std = 0
    target_dim = 10
    point_filter = (multidimensional
                    .point_filters
                    .FixedStochasticFilter(keep_percent=1, recalculate_each=100000))
    radius_update = (multidimensional
                     .radius_updates
                     .AdaRadiusHalving(tolerance=.5*1e-3, burnout_tolerance=100000))
    radius_barrier = 1e-3
    explore_dim_percent = .9
    starting_radius = 8
    max_turns = 1000000


def load_ground_truth(gt_file):
    with open(gt_file) as fd:
        gt = json.load(fd)
    return gt


def load_pkl(data_file):
    with open(data_file) as fd:
        d = pickle.load(fd).items()
    words = [x[0] for x in d]
    vectors = [map(float, x[1]) for x in d]
    return words, np.array(vectors)


def semantic_similarity(model_name, ground_truth, words, vectors):
    d = dict(zip(words, vectors.tolist()))
    space = mmfeat.space.Space(d, modelDescription=model_name)
    corr, sig, test_pairs, _ = space.pearson(ground_truth)
    return {
        'correlation': corr,
        'significance': sig,
        'test_pairs': test_pairs,
    }


class Identity(object):
    def __init__(self):
        pass

    def fit_transform(self, X):
        return X


@ex.automain
def experiment(
        ground_truth_men, ground_truth_simlex, data_file, data_file_small, data_type, dim, distance, npoints,
        n_neighbors, noise_std, target_dim, point_filter, radius_update,
        radius_barrier, explore_dim_percent, starting_radius, max_turns,
        _run):
    xs = None
    xs_small = None
    words = None
    if data_file is not None:
        if data_file.endswith('pkl'):
            words, xs = load_pkl(data_file)
        else:
            words, xs = multidimensional.common.load_embeddings(data_file)

    if data_file_small is not None:
        words, xs_small = multidimensional.common.load_embeddings(data_file_small)
    men = load_ground_truth(ground_truth_men)
    simlex = load_ground_truth(ground_truth_simlex)

    xs, d_goal, color = (datagen.DataBuilder()
                         .with_dim(dim)
                         .with_distance(distance)
                         .with_noise(noise_std)
                         .with_npoints(npoints)
                         .with_neighbors(n_neighbors)
                         .with_points(xs)
                         .with_type(data_type)
                         .build())

    xs_small, d_goal_small, color = (datagen.DataBuilder()
                               .with_dim(dim)
                               .with_distance(distance)
                               .with_noise(noise_std)
                               .with_npoints(npoints)
                               .with_neighbors(n_neighbors)
                               .with_points(xs_small)
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
        multidimensional.smacof.MDS(n_components=target_dim,
                                    n_init=1,
                                    max_iter=max_turns,
                                    verbose=2,
                                    dissimilarity='precomputed'),
        d_goal)

    PCA = dim_reduction(
        'Truncated SVD',
        decomposition.TruncatedSVD(n_components=target_dim),
        xs
    )

    original = dim_reduction(
        'Original Embeddings 300d',
        Identity(),
        xs
    )

    original_small = dim_reduction(
        'Original Embeddings 50d',
        Identity(),
        xs_small
    )

    SpectralEmbedding = dim_reduction(
        'SpectralEmbedding',
        manifold.SpectralEmbedding(n_components=target_dim,
                                   n_neighbors=n_neighbors),
        xs)
    tSNE = dim_reduction(
        'tSNE',
        manifold.TSNE(n_components=target_dim, init='pca', random_state=0),
        xs)

    methods = [HessianLLE]

    res = {}
    for method in methods:
        try:
            x = method.method.fit_transform(method.data)

            res[method.name] = {}

            res[method.name]['men'] = semantic_similarity(method.name, men, words, x)
            res[method.name]['simlex'] = semantic_similarity(method.name, simlex, words, x)

            print("Model: {}\t result:{}".format(method.name, res))
        except Exception as e:
            print(e)

    with open('semantic_similarity1.json', 'w') as fd:
        json.dump(res, fd, indent=4, sort_keys=True)
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

