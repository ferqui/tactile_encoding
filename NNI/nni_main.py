#!/usr/bin/env python3
"""
Defines the configuration for an NNI experiment.

Fra, Vittorio,
Politecnico di Torino,
EDA Group,
Torino, Italy.

Kaiser, Jakob
Institute/University
Group
Place

Muller-Cleve, Simon F.,
Istituto Italiano di Tecnologia - IIT,
Event-driven perception in robotics - EDPR,
Genova, Italy.
"""


import argparse

from nni.experiment import *


search_space = {
    'nb_hidden': {'_type': 'quniform', '_value': [50, 450, 50]}, # use it as an int in the script
    'lr': {'_type': 'choice', '_value': [0.0001, 0.001, 0.01, 0.1]},
    'fwd_weights_std': {'_type': 'quniform', '_value': [0.1, 3, 0.1]},
    'rec_weights_std': {'_type': 'quniform', '_value': [0.01, 0.5, 0.01]},
    'tau_mem': {'_type': 'choice', '_value': [1e-3, 2e-3, 5e-3, 10e-3, 20e-3, 30e-3, 40e-3, 50e-3]},
    'tau_syn': {'_type': 'choice', '_value': [1e-3, 2e-3, 5e-3, 10e-3, 20e-3, 30e-3, 40e-3, 50e-3]},
    #'scale': {'_type': 'quniform', '_value': [5, 20, 5]}, # use it as an int in the script
    #'batch_size': {'_type': 'choice', '_value': [8, 16, 32, 64, 128, 256]},
}
searchspace_filename = "train_spike_classifier_searchspace"
searchspace_path = "./searchspaces/{}.json".format(searchspace_filename)
with open(searchspace_path, "w") as write_searchspace:
    json.dump(search_space, write_searchspace)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Name for the experiment
    parser.add_argument('-exp_name',
                        type=str,
                        default='spike_classifier',
                        help='Name for the starting experiment.')

    # Maximum number of trials
    parser.add_argument('-exp_trials',
                        type=int,
                        default=1000,
                        help='Number of trials for the starting experiment.')

    # Maximum time
    parser.add_argument('-exp_time',
                        type=str,
                        default='10d',
                        help='Maximum duration of the starting experiment.')

    # How many (if any) GPUs are available
    parser.add_argument('-exp_gpu_number',
                        type=int,
                        default=2,
                        help='How many GPUs to use for the starting experiment.')

    # Which GPU to use
    parser.add_argument('-exp_gpu_sel',
                        type=int,
                        default=[0,1],
                        help='GPU index to be used for the experiment.')

    # How many trials at the same time
    parser.add_argument('-exp_concurrency',
                        type=int,
                        default=1,
                        help='Concurrency for the starting experiment.')
    
    # Max trials per GPU
    parser.add_argument('-max_per_gpu',
                        type=int,
                        default=5,
                        help='Maximum number of trials per GPU.')

    # What script to use for the experiment
    parser.add_argument('-script',
                        type=str,
                        default='train_spike_classifier.py',
                        help='Path to trainings script.')

    # Which port to use
    parser.add_argument('-port',
                        type=int,
                        default=8080,
                        help='Port number for the starting experiment.')

    args = parser.parse_args()

    
    config = ExperimentConfig(
        experiment_name=args.exp_name,
        experiment_working_directory="~/nni-experiments/{}".format(
            args.exp_name),
        trial_command=f"python3 {args.script}",
        trial_code_directory="./",
        search_space=search_space,
        tuner=AlgorithmConfig(name="Anneal",
                              class_args={"optimize_mode": "maximize"}),
        assessor=AlgorithmConfig(name="Medianstop",
                                 class_args=({'optimize_mode': 'maximize',
                                              'start_step': 10})),
        #assessor=AlgorithmConfig(name="Curvefitting",
        #                         class_args=({'epoch_num': 300,
        #                                      'start_step': 10,
        #                                      'threshold': 0.9,
        #                                      'gap': 1})),
        tuner_gpu_indices=args.exp_gpu_sel,
        max_trial_number=args.exp_trials,
        max_experiment_duration=args.exp_time,
        trial_concurrency=args.exp_concurrency,
        training_service=LocalConfig(trial_gpu_number=args.exp_gpu_number,
                                     max_trial_number_per_gpu=args.max_per_gpu,
                                     use_active_gpu=True)
    )

    experiment = Experiment(config)

    experiment.run(args.port)

    # Stop through input
    input('Press any key to stop the experiment.')

    # Stop at the end
    experiment.stop()
