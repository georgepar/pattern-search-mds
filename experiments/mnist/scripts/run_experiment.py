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

import multidimensional
import multidimensional.common
import multidimensional.mds
import multidimensional.point_filters
import multidimensional.radius_updates
import multidimensional.datagen.shapes as datagen

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
    method_name = None 
    method_func = None 

@ex.automain
def exp_main(target_dim, initial_dim, n_samples, data, labels,
             knn_algo, n_folds, n_neighbors,
             method_name, method_func):

    print ("Running Experiment with Smples: {}, for Target Dimensions:"" {} by using Method: {}"
        "".format(n_samples, target_dim, method_name))

    method = method_func

    x_embedded = method.fit_transform(data)

    print "This is the format of the output: {}".format(
                                                x_embedded.shape)

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

        knn = KNeighborsClassifier(n_neighbors=n_neighbors, 
              weights='uniform', algorithm=knn_algo, leaf_size=30, 
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




