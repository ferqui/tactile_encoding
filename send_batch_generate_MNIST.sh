#!/bin/bash
#
# adapted from https://github.com/rug-cit-hpc/cluster_course/blob/master/advanced_course/1.7_jobarray/solution/jobscript.sh
#
# exec_parallel_cluster.sh provides the ability to run multiple simulation at once, by launching individual jobs on the RUG peregrine cluster.
# if you want to run only one simulation e.g for plotting directly call your script.
#
# the command line arguments are <python executable path> <exec home path> <python file to run>
# dont use relative pathes
# the script will span as many jobs on the cluster as iterations definded below
#

# standart for normal python
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

## please adapt the following to your need:

# set with how many cpu cores it should be running
#SBATCH --cpus-per-task=6
# from to (incl) range of iterations, max is 1000 total
#SBATCH --array=1-4
# memory requirement - check on your PC first if possible
#SBATCH --mem=12GB
# max execution time
#SBATCH --time=3-00:00:00
# choose subcluster (short: up to 30 min, regular: up to 10 days - 128GB RAM?, himem: up tp 10 days - 1TB RAM?)
#SBATCH --partition=regular
# job name
#SBATCH --job-name=generate_mnist_dataset

## Most importantly check your python script on pg-interactive.hpc.rug.nl first (with reduced load) if you environment works fine.

source ../texel/texel/bin/activate

python run_generate_MNIST_dataset.py --idx_job_array ${SLURM_JOB_ID} --home_dataset './dataset/'