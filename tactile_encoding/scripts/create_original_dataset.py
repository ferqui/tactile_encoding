"""
Creates data according to the paper "A Generalized Linear 
Integrate-and-Fire Neural Model Produces Diverse Spiking Behaviors"
by Stefan Mihalas and Ernst Niebur.

Fra, Vittorio,
Politecnico di Torino,
EDA Group,
Torino, Italy.

Muller-Cleve, Simon F.,
Istituto Italiano di Tecnologia - IIT,
Event-driven perception in robotics - EDPR,
Genova, Italy.
"""


import numpy as np
from tqdm import tqdm

from tactile_encoding.utils.functions import fix_time, fix_time_only, original
from tactile_encoding.utils.utils import create_directory

OFFSET = 0.1
NOISE = 0.2
JITTER = 10
NB_TRIALS = 3


if __name__ == '__main__':
    # create output folder
    create_directory('./data')
    create_directory('./data/original_mn_output')
    ################
    ### original ###
    ################
    print('\nCreating original data.')
    original()

    # # single
    # print('\nCreating original data with offset.')
    # original(offset=OFFSET, noise=NOISE, jitter=JITTER, add_offset=True,
    #          add_noise=False, add_jitter=False)

    # print('\nCreating noisy original data.')
    # original(offset=OFFSET, noise=NOISE, jitter=JITTER,
    #          add_offset=False, add_noise=True, add_jitter=False)

    # print('\nCreating original data with temporal jitter.')
    # original(offset=OFFSET, noise=NOISE, jitter=JITTER,
    #          add_offset=False, add_noise=False, add_jitter=True)

    # # combination of two
    # print('\nCreating noisy original data with offset.')
    # original(offset=OFFSET, noise=NOISE, jitter=JITTER,
    #          add_offset=True, add_noise=True, add_jitter=False)

    # print('\nCreating original data with temporal jitter and offset.')
    # original(offset=OFFSET, noise=NOISE, jitter=JITTER,
    #          add_offset=True, add_noise=False, add_jitter=True)

    # print('\nCreating noisy original data with temporal jitter.')
    # original(offset=OFFSET, noise=NOISE, jitter=JITTER,
    #          add_offset=False, add_noise=True, add_jitter=True)

    # # combination of three
    # print('\nCreating noisy original data with temporal jitter and offset.')
    # original(offset=OFFSET, noise=NOISE, jitter=JITTER,
    #          add_offset=True, add_noise=True, add_jitter=True)

    # # ###########################
    # # ### fix (1000ms) length ###
    # # ###########################
    # print('\nCreating 1000ms data.')
    # # much faster, 'cause current profile only copied
    # fix_time_only(max_trials=NB_TRIALS)

    # # single
    # print('\nCreating 1000ms data with offset.')
    # fix_time(max_trials=NB_TRIALS, offset=OFFSET, noise=NOISE, jitter=JITTER,
    #          add_offset=True, add_noise=False, add_jitter=False)

    # print('\nCreating noisy 1000ms data.')
    # fix_time(max_trials=NB_TRIALS, offset=OFFSET, noise=NOISE, jitter=JITTER,
    #          add_offset=False, add_noise=True, add_jitter=False)

    # print('\nCreating 1000ms data with temporal jitter.')
    # fix_time(max_trials=NB_TRIALS, offset=OFFSET, noise=NOISE, jitter=JITTER,
    #          add_offset=False, add_noise=False, add_jitter=True)

    # # combination of two
    # print('\nCreating noisy 1000ms data with offset.')
    # fix_time(max_trials=NB_TRIALS, offset=OFFSET, noise=NOISE, jitter=JITTER,
    #          add_offset=True, add_noise=True, add_jitter=False)

    # print('\nCreating 1000ms data with temporal jitter and offset.')
    # fix_time(max_trials=NB_TRIALS, offset=OFFSET, noise=NOISE, jitter=JITTER,
    #          add_offset=True, add_noise=False, add_jitter=True)

    # print('\nCreating noisy 1000ms data with temporal jitter.')
    # fix_time(max_trials=NB_TRIALS, offset=OFFSET, noise=NOISE, jitter=JITTER,
    #          add_offset=False, add_noise=True, add_jitter=True)

    # # combination of three
    # print('\nCreating noisy 1000ms data with temporal jitter and offset.')
    # fix_time(max_trials=NB_TRIALS, offset=OFFSET, noise=NOISE, jitter=JITTER,
    #          add_offset=True, add_noise=True, add_jitter=True)

    # print('\nFinished with data creation.')

    # # # ###################
    # # # # Parameter sweep #
    # # # ###################
    step_size = 0.05
    noise_levels = np.round(np.arange(0, 1+step_size, step_size), 5)
    offset_levels = noise_levels

    for offset_counter, offset in tqdm(enumerate(offset_levels), position=0, leave=True, total=len(offset_levels)):
        for noise_counter, noise in tqdm(enumerate(noise_levels), position=1, leave=True, total=len(noise_levels)):
            # TODO to clean up tqdm we would need to hand over bar object... let's keep it for now
            fix_time(max_trials=NB_TRIALS, offset=offset, noise=noise,
                     jitter=10, add_offset=True, add_noise=True, add_jitter=True)
