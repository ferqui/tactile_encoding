import argparse


parser = argparse.ArgumentParser()

# Experiment name
parser.add_argument('-experiment_name',
                    type=str,
                    default="GR_Braille_classifier",
                    help='Name of this experiment.')
# ID of the NNI experiment to refer to
parser.add_argument('-experiment_id',
                    type=str,
                    default="vpeqjlkr",
                    help='ID of the NNI experiment whose results are to be used.')
# ID of the NNI trial providing the best test accuracy
parser.add_argument('-best_test_id',
                    type=str,
                    default="euX7c",
                    help='ID of the NNI trial that gave the highest test accuracy.')
# Path of weights to perform test only (if do_training is False)
parser.add_argument('-trained_layers_path',
                    type=str,
                    default="./results/layers/optimized/spike_classifier/fix_len_noisy_temp_jitter/vpeqjlkr_ref.pt",
                    help='Path of the weights to be loaded to perform test only (given do_training is set to False).')
# Auto-selection of GPU
parser.add_argument('-auto_gpu',
                    type=bool,
                    default=False,
                    help='Enable or not auto-selection of GPU to use.')
# Manual selection of GPU
parser.add_argument('-manual_gpu_idx',
                    type=int,
                    default=0,
                    help='Set which GPU to use.')
# (maximum) GPU memory fraction to be allocated
parser.add_argument('-gpu_mem_frac',
                    type=float,
                    default=0.3,
                    help='The maximum GPU memory fraction to be used by this experiment.')
# Set seed usage
parser.add_argument('-use_seed',
                    type=bool,
                    default=True,
                    help='Set if a seed is to be used or not.')
# Enable mutliple seeds
parser.add_argument('-multi_seed',
                    type=bool,
                    default=False,
                    help='Set if more than one seed is to be used or not.')
# Number of seeds
parser.add_argument('-n_seed',
                    type=int,
                    default=10,
                    help='Set the number of seeds if more than one is needed.')
# Specify if running for debug
parser.add_argument('-debugging',
                    type=bool,
                    default=False,
                    help='Set if the run is to debug the code or not.')
# Save heatmap
parser.add_argument('-save_hm',
                    type=bool,
                    default=True,
                    help='Save or not the heatmap produced for behaviour classification.')

args = parser.parse_args()

settings = vars(args)