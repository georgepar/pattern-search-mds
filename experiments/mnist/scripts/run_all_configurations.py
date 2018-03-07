from run_experiment import ex
from read_data_and_convert import convert_MNIST_dataset_to_numpys as loader 
from sacred.observers import MongoObserver
from sklearn import manifold, decomposition
import sys 
import json

sys.path.append('../../../')
import multidimensional
import multidimensional.common
import multidimensional.mds
import multidimensional.point_filters
import multidimensional.radius_updates
import multidimensional.datagen.shapes as datagen



#target_dimensions = [3,5,20,50,75,100,150,200]

target_dimensions = [20]

max_turns = 1000000
method_n_comp = 12
exp_samples = 10000
point_filter = (multidimensional
                .point_filters
                .FixedStochasticFilter(keep_percent=1))
radius_update = (multidimensional
                 .radius_updates
                 .AdaRadiusHalving(tolerance=1e-3))
radius_barrier = 1e-3
explore_dim_percent = 1
starting_radius = 8
 
def safe_run_experiment(target_dim, exp_images,
    exp_labels, n_samples, initial_dim, m_name, m_func):
    try:
        exp = ex.run( config_updates = {'target_dim' : target_dim,
                                        'data': exp_images,
                                        'initial_dim': initial_dim,
                                        'n_samples': n_samples,
                                        'labels': exp_labels, 
                                        'n_folds': 10,
                                        'knn_algo': 'brute',
                                        'n_neighbors': 1,
                                        'method_name': m_name, 
                                        'method_func': m_func })
        return exp.result
    except Exception as e:
        print '\n'+'='*30
        print ('Learning Method: {} Run Time Error'
            ''.format(m_name))
        print '='*30 + '\n'
        print e
        return {}

def run_all_methods_for_some_dimensions(target_dim, exp_images,
    exp_labels, n_samples, initial_dim):

    methods_l = [
    ('MDS (proposed)',
     multidimensional.mds.MDS(
     target_dim,
     point_filter,
     radius_update,
     starting_radius=starting_radius,
     radius_barrier=radius_barrier,
     max_turns=max_turns,
     explore_dim_percent=explore_dim_percent,
     keep_history=False,
     history_color=None,
     history_path=None,
     dissimilarities='euclidean')),

    ('Truncated SVD',
    decomposition.TruncatedSVD(n_components=target_dim)),

    ('LLE', 
    manifold.LocallyLinearEmbedding(method_n_comp, target_dim,
            eigen_solver='auto',method='standard',n_jobs=8)),

    ('Isomap', manifold.Isomap(method_n_comp, target_dim)),

    ('ModifiedLLE',
    manifold.LocallyLinearEmbedding(method_n_comp,
                                    target_dim,
                                    eigen_solver='auto',
                                    method='modified',n_jobs=8) ),

    ('SpectralEmbedding',
        manifold.SpectralEmbedding(n_components=target_dim,
                                   n_neighbors=method_n_comp,
                                   n_jobs=8)),

    # ('tSNE',
    # manifold.TSNE(n_components=target_dim, 
    #               init='pca', random_state=0)),

    ('LTSA',
    manifold.LocallyLinearEmbedding(method_n_comp,
                                    target_dim,
                                    eigen_solver='auto',
                                    method='ltsa',n_jobs=8)),
    ('MDS SMACOF',
     manifold.MDS(n_components=target_dim,
                  n_init=1,
                  max_iter=max_turns,
                  verbose=2,
                  dissimilarity='euclidean')),
    ]

    results = {}
    for m_name, m_func in methods_l:
        res = safe_run_experiment(target_dim, exp_images,
                                  exp_labels, n_samples, initial_dim, 
                                  m_name, m_func)
        results[m_name] = res
    return results
         

if __name__ == '__main__':
    
    mnistdata = loader('../dataset', '../converted_dataset')
    train_images, train_labels, test_images, test_labels = mnistdata

    n_samples, initial_dim = test_images.shape
    n_samples = exp_samples
    exp_images = test_images[:n_samples]
    exp_labels = test_labels[:n_samples]

    result = {}

    for target_dim in target_dimensions:

        res = run_all_methods_for_some_dimensions(
            target_dim, exp_images, exp_labels, n_samples, initial_dim)

        result[target_dim] = res

    with open("mnist_result.json", 'w') as f:
        json.dump(result, f, indent=4, sort_keys=True)
