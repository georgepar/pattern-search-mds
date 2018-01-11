"""!
\brief This script has the purpose of downloading all required MNIST
data and converting them to numpy matrices that can be easily 
manipulated from other scripts. THe same applies to both images and
labels available.
\author Efthymios Tzinis"""

import argparse
from mnist import MNIST
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

    mndata = MNIST(MNIST_dir)
    mndata.gz=True
    images, labels = mndata.load_training()

    print type(images)
    print type(labels)

    print images[0]
    print len(labels)

def get_args():
    """! Command line parser """
    parser = argparse.ArgumentParser(
        description='MNIST Dataset Reader and Converter' )
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