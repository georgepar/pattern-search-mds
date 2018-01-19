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

@ex.automain
def exp_main(target_dim, initial_dim, n_samples, data):
    print ("Running Experiment for Target Dimensions: {}"
        "".format(target_dim))

    method = decomposition.TruncatedSVD(n_components=target_dim)

    x = method.fit_transform(data)

    print "This is the format of the output: {}".format(x.shape)

