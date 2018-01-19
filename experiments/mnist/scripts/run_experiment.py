"""Run experiments for the mnist dataset"""

from collections import namedtuple

import os
import sys
import time

# import matplotlib.pyplot as plt
# from matplotlib.ticker import NullFormatter

import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn import manifold, decomposition
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

# this is main module path
sys.path.append('../../../')

# import multidimensional
# import multidimensional.common
# import multidimensional.mds
# import multidimensional.point_filters
# import multidimensional.radius_updates
# import multidimensional.datagen.shapes as datagen

# import config


EXPERIMENT_NAME = 'MNIST_test_data_clustering'

ex = Experiment(EXPERIMENT_NAME)

def trial_man_learn(target_dim):
    res = manifold.TSNE(n_components=target_dim, 
                        init='pca', 
                        random_state=0)
    return res

@ex.config
def mnist_convergence_config():
    target_dim = 50 
    initial_dim = 784 
    n_samples = 10000 
    data = []
    labels = []
    n_folds = 10
    knn_algo = 'brute'  # {'auto', 'ball_tree', 'kd_tree', 'brute'}
    n_neighbors = 1

@ex.automain
def exp_main(target_dim, initial_dim, n_samples, data, labels,
             knn_algo, n_folds, n_neighbors):

    print ("Running Experiment for Target Dimensions: {}"
        "".format(target_dim))

    method = decomposition.TruncatedSVD(n_components=target_dim)

    x_embedded = method.fit_transform(data)

    print "This is the format of the output: {}".format(
                                                x_embedded.shape)

    kf = KFold(n_splits=10, shuffle=False, random_state=7)
    
    for tr_ind, te_ind in kf.split(labels):

        X_train = data[tr_ind]
        Y_train = labels[tr_ind]
        X_test = data[te_ind]
        Y_test = labels[te_ind]

        knn = KNeighborsClassifier(n_neighbors=n_neighbors, 
              weights='uniform', algorithm=knn_algo, leaf_size=30, 
              p=2, metric='minkowski', metric_params=None, n_jobs=8)
        
        knn.fit(X_train, Y_train) 

        est_labels = knn.predict(X_test)

        print "Actual: {}".format(Y_test)
        print "Predic: {}".format(est_labels)




