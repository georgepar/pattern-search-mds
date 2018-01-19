from run_experiment import ex
from read_data_and_convert import convert_MNIST_dataset_to_numpys as loader 
from sacred.observers import MongoObserver
from sklearn import manifold, decomposition
import sys 

sys.path.append('../../../')


target_dimensions = [3,5,20,50,75,100,150,200,300,500]

target_dimensions = [5]

max_turns = 1000000
method_n_comp = 12
exp_samples = 1000

def safe_run_experiment(target_dim, exp_images,
    exp_labels, n_samples, initial_dim, m_name, m_func):
    try:
        ex.run( config_updates = {'target_dim' : target_dim,
                                  'data': exp_images,
                                  'initial_dim': initial_dim,
                                  'n_samples': n_samples,
                                  'labels': exp_labels, 
                                  'n_folds': 10,
                                  'knn_algo': 'brute',
                                  'n_neighbors': 1,
                                  'method_name': m_name, 
                                  'method_func': m_func }) 
    except Exception as e:
        print '\n'+'='*30
        print ('Learning Method: {} Run Time Error'
            ''.format(m_name))
        print '='*30 + '\n'
    

def run_all_methods_for_some_dimensions(target_dim, exp_images,
    exp_labels, n_samples, initial_dim):

    methods_l = [

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
                                    method='ltsa',n_jobs=8))
    
    ]

    for m_name, m_func in methods_l:
        safe_run_experiment(target_dim, exp_images,
                            exp_labels, n_samples, initial_dim, 
                            m_name, m_func)
         

if __name__ == '__main__':
    
    mnistdata = loader('../dataset', '../converted_dataset')
    train_images, train_labels, test_images, test_labels = mnistdata

    n_samples, initial_dim = test_images.shape
    n_samples = exp_samples
    exp_images = test_images[:n_samples]
    exp_labels = test_labels[:n_samples]

    for target_dim in target_dimensions:

        run_all_methods_for_some_dimensions(target_dim, exp_images,
                                exp_labels, n_samples, initial_dim)

         