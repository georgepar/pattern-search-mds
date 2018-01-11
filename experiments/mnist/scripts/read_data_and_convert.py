"""!
\brief This script has the purpose of downloading all required MNIST
data and converting them to numpy matrices that can be easily 
manipulated from other scripts. THe same applies to both images and
labels available.
\author Efthymios Tzinis"""

import argparse
import os 



def convert_MNIST_dataset_to_numpys(MNIST_dir, converted_MNIST_dir):
    

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
