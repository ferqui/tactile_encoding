import argparse
import sys


sys.argv=['']

parser = argparse.ArgumentParser()

# Experiment name
parser.add_argument('-experiment_name',
                    type=str,
                    default="spike_classifier",
                    help='Name of this experiment.')
# Training needed or not
parser.add_argument('-do_training',
                    type=bool,
                    default=True,
                    help='If set to False, test only will be performed.')
# Make some statistics for training
parser.add_argument('-training_statistics',
                    type=bool,
                    default=True,
                    help='If set to True, multiple trainings will be performed (with use_seed consequently set to False).')
# Number or repetitions for training statistics
parser.add_argument('-repetitions',
                    type=int,
                    default=10,
                    help='Number of trainings to be performed for statistical evaluation.')
# Number or tests for statistics
parser.add_argument('-n_test',
                    type=int,
                    default=10,
                    help='Number of tests to be performed for statistical evaluation.')
# Number of epochs
parser.add_argument('-nb_epochs',
                    type=int,
                    default=30,
                    help='Number of training epochs.')
# ID of the NNI experiment to refer to
parser.add_argument('-experiment_id',
                    type=str,
                    default="vpeqjlkr",
                    help='ID of the NNI experiment whose results are to be used.')
# Specify availability of optimization db
parser.add_argument('-nni_db_available',
                    type=bool,
                    default=False,
                    help='Specify if the database from NNI optimization is available or not.')
# ID of the NNI trial providing the best test accuracy
parser.add_argument('-best_test_id',
                    type=str,
                    default="euX7c",
                    help='ID of the NNI trial that gave the highest test accuracy.')
# Save the weights (to be re-used right after the training to test) or not
parser.add_argument('-save_weights',
                    type=bool,
                    default=True,
                    help='Weights can be saved to be loaded after training and used for test.')
# Save figures
parser.add_argument('-save_fig',
                    type=bool,
                    default=True,
                    help='Save or not the plots produced during training and test.')
# Store the weights 
parser.add_argument('-store_weights',
                    type=bool,
                    default=True,
                    help='Weights can be stored with specific, unique name.')
# Path of weights to perform test only (if do_training is False)
parser.add_argument('-trained_layers_path',
                    type=str,
                    default="./results/layers/optimized/spike_classifier/fix_len_noisy_temp_jitter/vpeqjlkr_ref.pt", #"./NNI/results/layers/fix_len_noisy_temp_jitter/vpeqjlkr.pt",
                    help='Path of the weights to be loaded to perform test only (given do_training is set to False).')
# (maximum) GPU memory fraction to be allocated
parser.add_argument('-gpu_mem_frac',
                    type=float,
                    default=0.3,
                    help='The maximum GPU memory fraction to be used by this experiment.')
# Which GPU is actually "visible"
parser.add_argument('-visible_gpus',
                    type=int,
                    default=[0],
                    help='GPU index to be used for the experiment.')
# Set seed usage
parser.add_argument('-use_seed',
                    type=bool,
                    default=True,
                    help='Set if a seed is to be used or not.')

args = parser.parse_args()

settings = vars(args)