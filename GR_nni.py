# NNi uses an experiment to manage the HPO process.
# The experiment config defines how to train the models and how to explore the search space.

# To retrieve the experiment results:
# nnictl view [experiment_name] --port=[port] --experiment_dir=[EXPERIMENT_DIR]
#
# For example:
# nnictl view zk29xumi --port=8080 --experiment_dir=<EXPERIMENT_DIR>

import argparse
import os
import time

from nni.experiment import ExperimentConfig, AlgorithmConfig, LocalConfig, Experiment, RemoteConfig, RemoteMachineConfig

search_space = {
    'gr': {'_type': 'loguniform',
           '_value': [0.01, 1]},
    'reg_spikes': {'_type': 'loguniform',
           '_value': [0.004, 0.01]},
    'reg_neurons': {'_type': 'loguniform',
           '_value': [0.000001, 0.01]},
    'shared_params': {'_type': 'choice',
           '_value': [True, False]}
}

initial_search_space = {
    'gr': {'_type': 'choice',
           '_value': [0.5714285714285714]},
    'reg_spikes': {'_type': 'choice',
           '_value': [0.004]},
    'reg_neurons': {'_type': 'choice',
           '_value': [0.000001]},
    'shared_params': {'_type': 'choice',
           '_value': [True]}
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Name for the experiment
    parser.add_argument('-exp_name',
                        type=str,
                        default=os.path.splitext(os.path.basename(__file__))[0],
                        help='Name for the starting experiment.')

    # Maximum number of trials
    parser.add_argument('-exp_trials',
                        type=int,
                        default=100,
                        help='Number of trials for the starting experiment.')

    # Maximum time
    parser.add_argument('-exp_time',
                        type=str,
                        default='2d',
                        help='Maximum duration of the starting experiment.')

    # How many (if any) GPUs to use
    parser.add_argument('-exp_gpu_number',
                        type=int,
                        default=1,
                        help='How many GPUs to use for the starting experiment.')

    # Which GPU to use
    parser.add_argument('-exp_gpu_sel',
                        type=int,
                        default=0,
                        help='GPU index to be used for the experiment.')

    # How many trials at the same time
    parser.add_argument('-exp_concurrency',
                        type=int,
                        default=1,
                        help='Concurrency for the starting experiment.')

    # What script to use for the experiment
    parser.add_argument('-script',
                        type=str,
                        default='GR.py',
                        help='Path to trainings script.')

    # Which port to use
    parser.add_argument('-port',
                        type=int,
                        default=8080,
                        help='Port number for the starting experiment.')

    # How many epochs per trial
    parser.add_argument('-n_epochs',
                        type=int,
                        default=300,
                        help='Port number for the starting experiment.')

    parser.add_argument('-batch_size',
                        type=int,
                        default=300,
                        help='Experiment batch size.')

    parser.add_argument('--ALIF',
                        action='store_true',
                        help="Use ALIF neurons instead of MN")

    parser.add_argument('-tuner',
                        type=str,
                        default='Anneal',
                        choices=['GridSearch', 'Anneal'],
                        help='Tuner algorithm')

    parser.add_argument('-location',
                        type=str,
                        default=None,
                        help=' If not None, this is used to specify the input folder from which data is loaded.')

    args = parser.parse_args()

    timestr = time.strftime("%Y%m%d-%H%M%S")

    training_service = LocalConfig(trial_gpu_number=args.exp_gpu_number,
                                   max_trial_number_per_gpu=args.exp_concurrency,
                                   use_active_gpu=True)

    if args.ALIF:
        trial_command = f"python3 {args.script} --norm 10 --batch_size 128 --train --log --nb_epochs {args.n_epochs} --nni --ALIF"
    else:
        trial_command = f"python3 {args.script} --norm 10 --batch_size 128 --train --log --nb_epochs {args.n_epochs} --nni "

    config = ExperimentConfig(
        experiment_name=args.script,
        experiment_working_directory="~/nni-experiments/{}".format(os.path.splitext(args.script)[0]),
        trial_command=trial_command,
        trial_code_directory="./",
        search_space=initial_search_space,
        tuner=AlgorithmConfig(name=args.tuner,  # "Anneal",
                              class_args={"optimize_mode": "maximize"}),
        assessor=AlgorithmConfig(name="Medianstop",
                                 # early stopping: Stop if the hyperparameter set performs worse than median at any step.
                                 class_args=({'optimize_mode': 'maximize',
                                              'start_step': 10})),
        tuner_gpu_indices=args.exp_gpu_sel,
        max_trial_number=args.exp_trials,
        max_experiment_duration=args.exp_time,
        trial_concurrency=1,
        training_service=training_service
    )

    experiment = Experiment(config)
#     experiment.run(args.port)
    experiment.run(args.port, wait_completion=False)
    while len(experiment.list_trial_jobs())==0:
        pass

    print('Waiting 1 minute to update')
    time.sleep(60)
    experiment.update_search_space(search_space)
    experiment.update_trial_concurrency(args.exp_concurrency)

    print('Waiting 1 minute to stop')
    time.sleep(60)

#     # Stop through input
#     input('Press any key to stop the experiment.')

    # Stop at the end
    id = experiment.id
    experiment.stop()
    print(id)

    print('Waiting 1 minuto to resume')
    time.sleep(60)
    Experiment.resume(id, port=args.port, wait_completion=True)
