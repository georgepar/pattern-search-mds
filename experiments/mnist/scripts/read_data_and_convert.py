"""!
\brief This script has the purpose of downloading all required MNIST
data and converting them to numpy matrices that can be easily 
manipulated from other scripts. THe same applies to both images and
labels available.
\author Efthymios Tzinis"""

import argparse
from mnist import MNIST
import numpy as np
import os 

def safe_mkdirs(path):
    """! Makes recursively all the directory in input path

    \param path (@a str) A path to create all the directories

    \throws IOError If it fails to create recursive directories
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            raise IOError(
                ("Failed to create recursive directories: "
                " {}".format(path)
                )
            )

def load_MNIST_data(MNIST_dir):
    """!
    \brief Loads MNIST data using python-mnist library from the given
    MNIST_dir"""

    mndata = MNIST(MNIST_dir)
    mndata.gz=True
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    if (not len(test_images) == len(test_labels) or 
        not len(test_images) == 10000):
        raise ValueError("MNIST test data loaded should be 10000")

    if (not len(train_images) == len(train_labels) or 
        not len(train_images) == 60000):
        raise ValueError("MNIST test data loaded should be 10000")

    return train_images, train_labels, test_images, test_labels

def convert_data_2_numpy_arrays(mndata):
    """Convert data from list of lists and array.array to numpy"""

    converted_data = [np.array(x,dtype=np.uint8) for x in mndata]
    return converted_data

def convert_MNIST_dataset_to_numpys(MNIST_dir, converted_MNIST_dir):
    """!
    \brief Converts the data in mnist images and labels which are in 
    gunzip formats:
    t10k-images-idx3-ubyte.gz
    t10k-labels-idx1-ubyte.gz
    train-images-idx3-ubyte.gz
    train-labels-idx1-ubyte.gz
    """
    safe_mkdirs(converted_MNIST_dir)

    mndata = load_MNIST_data(MNIST_dir)

    converted_data = convert_data_2_numpy_arrays(mndata)
    train_images, train_labels, test_images, test_labels=converted_data

    print train_images.dtype, train_images.shape
    print train_labels.dtype, train_labels.shape

def get_args():
    """! Command line parser """
    parser = argparse.ArgumentParser(
        description="""MNIST Dataset Reader and Converter. Warning be
        careful to <pip install python-mnist> but without mnist 
        package. Thus you can simply <pip uninstall mnist>.""" )
    parser.add_argument("--MNIST_dir", type=str, 
        help="""Path that MNIST data are stored""", 
        default='../dataset/')
    parser.add_argument("--converted_MNIST_dir", type=str, 
        help=("The Designated dirpath where the numpy 2D matrices "
        "and the respective labels would be stored. If this folder is "
        "not existent will be created."), 
        default='../converted_dataset')
    args = parser.parse_args()
    return args 

if __name__ == "__main__":
    args = get_args()
    convert_MNIST_dataset_to_numpys(args.MNIST_dir, 
                                    args.converted_MNIST_dir)