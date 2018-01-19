from run_experiment import ex
from read_data_and_convert import convert_MNIST_dataset_to_numpys as loader 
from sacred.observers import MongoObserver

target_dimensions = [20,50,75,100,150,200,300,500]
target_dimensions = [20]

if __name__ == '__main__':
    
    mnistdata = loader('../dataset', '../converted_dataset')
    train_images, train_labels, test_images, test_labels = mnistdata

    n_samples, initial_dim = test_images.shape
    n_samples = 100
    exp_images = test_images[:n_samples]

    for target_dim in target_dimensions:
        # ex.observers.append(MongoObserver.create())

        ex.run( config_updates = {'target_dim' : target_dim,
                                  'data': exp_images,
                                  'initial_dim': initial_dim,
                                  'n_samples': n_samples} )   