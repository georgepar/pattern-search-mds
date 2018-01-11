"""!
\brief This script has the purpose of downloading all required MNIST
data and converting them to numpy matrices that can be easily 
manipulated from other scripts. THe same applies to both images and
labels available.
\author Efthymios Tzinis"""

import os 

def get_args():
    """! Command line parser """
    parser = argparse.ArgumentParser(
        description='MNIST Dataset Reader and Converter' )
    parser.add_argument("--MNIST_dir", type=str, 
        help="""Path that MNIST data are stored""", 
        default='../dataset/')
    parser.add_argument("--customer_name", type=str, 
        help="""The name of the customer or dataset that you want to 
        create the plots.""", 
        default='iqms')
    parser.add_argument("--plots_configuration_list", type=str, 
        help="""Which plots do you want to produce for the specific
        customer.""", default=["overall_emotion_agent_customer",
        "overall_behavior_agent_customer"], nargs='+',
        choices=["overall_emotion_agent_customer",
        "overall_behavior_agent_customer"])
    parser.add_argument("--size", type=int, 
        help="""How many samples do you want to have in the randomly
        created dummy dataset.""", default=100)
    args = parser.parse_args()
    return args 

if __name__ == "__main__":
    args = get_args()
    data_dic = create_random_dataset(size = args.size) 
    plot_paths = create_all_plots_for_customer(
        result_dir=args.result_dir, 
        customer=args.customer_name, data_dic=data_dic,
        plots_configuration_list=args.plots_configuration_list)
    print "Plots for customer: <{}> are created in: {}".format(
        args.customer_name, plot_paths)