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
from sklearn import datasets as sk_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
sys.path.append('../')


import multidimensional
import multidimensional.common
import multidimensional.mds
import multidimensional.point_filters
import multidimensional.radius_updates
import multidimensional.datagen.shapes as datagen
import multidimensional.smacof

import config


EXPERIMENT_NAME = 'clusters_3d_3e-1'

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
    data_type = 'real'
    dim = 4
    distance = 'euclidean'
    npoints = 569
    n_neighbors = 30
    noise_std = 0
    target_dim = 1
    point_filter = (multidimensional
                    .point_filters
                    .FixedStochasticFilter(keep_percent=.5))
    radius_update = (multidimensional
                     .radius_updates
                     .AdaRadiusHalving(tolerance=1e-5))
    radius_barrier = 1e-3
    explore_dim_percent = 1
    starting_radius = 4
    max_turns = 10000


def evaluate(x_embedded, labels):
    kf = KFold(n_splits=10, shuffle=False, random_state=7)

    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score

    metrics_mapper = np.array

    metrics_l = [('uw_f1',[]),
                 ('w_f1',[]), 
                 ('uw_rec',[]),
                 ('w_rec',[]),
                 ('uw_prec',[]),
                 ('w_prec',[]),
                 ('uw_acc',[]),
                 ('w_acc',[])]
    metric_dic = dict(metrics_l)

    fold_cnt = 0
    
    for tr_ind, te_ind in kf.split(labels):

        X_train = x_embedded[tr_ind]
        Y_train = labels[tr_ind]
        X_test = x_embedded[te_ind]
        Y_test = labels[te_ind]

        knn = KNeighborsClassifier(n_neighbors=1, 
              weights='uniform', leaf_size=30, 
              p=2, metric='minkowski', metric_params=None, n_jobs=8)
        
        knn.fit(X_train, Y_train) 

        est_labels = knn.predict(X_test)

        # print "Actual: {}".format(Y_test)
        # print "Predic: {}".format(est_labels)
        fold_cnt += 1
        # print "Fold: {}".format(fold_cnt)
        uw_f1 = f1_score(Y_test, est_labels, average='macro')
        w_f1 = f1_score(Y_test, est_labels, average='micro')

        uw_rec = recall_score(Y_test, est_labels, average='macro')
        w_rec = recall_score(Y_test, est_labels, average='micro')

        uw_prec = precision_score(Y_test, est_labels, average='macro')
        w_prec = precision_score(Y_test, est_labels, average='micro')
        # print recall_score(Y_test, est_labels, average='weighted')
        w_acc = accuracy_score(Y_test, est_labels)
        cmat = confusion_matrix(Y_test, est_labels)
        uw_acc = (cmat.diagonal()/(1.0*cmat.sum(axis=1))).mean()

        metrics_l = [('uw_f1',uw_f1),
                     ('w_f1',w_f1), 
                     ('uw_rec',uw_rec),
                     ('w_rec',w_rec),
                     ('uw_prec',uw_prec),
                     ('w_prec',w_prec),
                     ('uw_acc',uw_acc),
                     ('w_acc',w_acc)]
        
        for k,v in metrics_l:
            metric_dic[k].append(v)

    metrics = {}

    for k,v in metric_dic.items():
        metrics[k] = np.mean(v)
        print "{}: \t {} +-({})".format(k, np.mean(v), np.std(v))
    return metrics


@ex.automain
def experiment(
        data_type, dim, distance, npoints, n_neighbors,
        noise_std, target_dim, point_filter, radius_update, radius_barrier,
        explore_dim_percent, starting_radius, max_turns, _run):

    xs, y = sk_data.load_iris(return_X_y=True) 
    xs, d_goal, color = (datagen.DataBuilder()
                         .with_dim(dim)
                         .with_distance(distance)
                         .with_noise(noise_std)
                         .with_npoints(npoints)
                         .with_neighbors(n_neighbors)
                         .with_points(xs)
                         .build())
    color = y
    dim_reduction = namedtuple('dim_reduction', 'name method data')
    MDS_proposed = dim_reduction(
        'MDS (proposed)',
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

    PCA = dim_reduction(
        'Truncated SVD',
        decomposition.TruncatedSVD(n_components=target_dim),
        xs
    )

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
                                    dissimilarity='precomputed',
                                    history_path=EXPERIMENT_NAME + '_mds_smacof'),
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

    methods = [MDS_proposed, mds, PCA, Isomap, LLE, HessianLLE, ModifiedLLE, LTSA]
    #methods = [MDS_proposed, mds]
    fig = plt.figure(figsize=(20, 20))

    #plt.suptitle("Learning %s with %i points, %.3f noise"
    #             % (data_type, npoints, noise_std), fontsize=14)
    ax = fig.add_subplot(331, projection='3d', aspect=1)
    ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2], c=color, cmap=plt.cm.Spectral)
    plt.title("Original Manifold", fontsize=32)
    for i, method in enumerate(methods):
        print("Running {}".format(methods[i].name))
        try:
            t0 = time.time()
            x = methods[i].method.fit_transform(methods[i].data)
            t1 = time.time()
            ax = fig.add_subplot("33{}".format(i + 2), aspect=1)
            # Plot the 2 dimensions.
            ax.scatter(x[:, 0], x[:, 0], c=color, cmap=plt.cm.Spectral)
            
            plt.title(methods[i].name + "(%.2g sec)" % (t1-t0), fontsize=32)
            #ax.xaxis.set_major_formatter(NullFormatter())
            #ax.yaxis.set_major_formatter(NullFormatter())
            plt.axis('tight')
            metrics = evaluate(x, y)
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

    history = MDS_proposed.method.history_observer.history
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

    history = mds.method.history_observer.history
    for i, error in enumerate(history['error']):
        _run.log_scalar('smacof.mse.error', error, i + 1)
    for i, radius in enumerate(history['radius']):
        _run.log_scalar('smacof.step', radius, i + 1)

    return MDS_proposed.method.history_observer.history['error'][-1]

