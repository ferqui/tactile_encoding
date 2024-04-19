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


from tqdm import tqdm

from tactile_encoding.utils.functions import fix_time, fix_time_only, original

if __name__ == '__main__':
    ################
    ### original ###
    ################
    print('\nCreating original data.')
    original()

    # single
    print('\nCreating original data with offset.')
    original(offset=0.1, noise=0.1, jitter=10, add_offset=True,
             add_noise=False, add_jitter=False)

    print('\nCreating noisy original data.')
    original(offset=0.1, noise=0.1, jitter=10,
             add_offset=False, add_noise=True, add_jitter=False)

    print('\nCreating original data with temporal jitter.')
    original(offset=0.1, noise=0.1, jitter=10,
             add_offset=False, add_noise=False, add_jitter=True)

    # combination of two
    print('\nCreating noisy original data with offset.')
    original(offset=0.1, noise=0.1, jitter=10,
             add_offset=True, add_noise=True, add_jitter=False)

    print('\nCreating original data with temporal jitter and offset.')
    original(offset=0.1, noise=0.1, jitter=10,
             add_offset=True, add_noise=False, add_jitter=True)

    print('\nCreating noisy original data with temporal jitter.')
    original(offset=0.1, noise=0.1, jitter=10,
             add_offset=False, add_noise=True, add_jitter=True)

    # combination of three
    print('\nCreating noisy original data with temporal jitter and offset.')
    original(offset=0.1, noise=0.1, jitter=10,
             add_offset=True, add_noise=True, add_jitter=True)

    ###########################
    ### fix (1000ms) length ###
    ###########################
    print('\nCreating 1000ms data.')
    # much faster, 'cause current profile only copied
    fix_time_only(max_trials=100)

    # single
    print('\nCreating 1000ms data with offset.')
    fix_time(max_trials=100, offset=0.1, noise=0.1, jitter=10,
             add_offset=True, add_noise=False, add_jitter=False)

    print('\nCreating noisy 1000ms data.')
    fix_time(max_trials=100, offset=0.1, noise=0.1, jitter=10,
             add_offset=False, add_noise=True, add_jitter=False)

    print('\nCreating 1000ms data with temporal jitter.')
    fix_time(max_trials=100, offset=0.1, noise=0.1, jitter=10,
             add_offset=False, add_noise=False, add_jitter=True)

    # combination of two
    print('\nCreating noisy 1000ms data with offset.')
    fix_time(max_trials=100, offset=0.1, noise=0.1, jitter=10,
             add_offset=True, add_noise=True, add_jitter=False)

    print('\nCreating 1000ms data with temporal jitter and offset.')
    fix_time(max_trials=100, offset=0.1, noise=0.1, jitter=10,
             add_offset=True, add_noise=False, add_jitter=True)

    print('\nCreating noisy 1000ms data with temporal jitter.')
    fix_time(max_trials=100, offset=0.1, noise=0.1, jitter=10,
             add_offset=False, add_noise=True, add_jitter=True)

    # combination of three
    print('\nCreating noisy 1000ms data with temporal jitter and offset.')
    fix_time(max_trials=100, offset=0.1, noise=0.1, jitter=10,
             add_offset=True, add_noise=True, add_jitter=True)

    print('\nFinished with data creation.')

    # # ###################
    # # # Parameter sweep #
    # # ###################

    # noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    # offset_levels = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

    # for offset_counter, offset in tqdm(enumerate(offset_levels), position=0, leave=False):
    #     for noise_counter, noise in tqdm(enumerate(noise_levels), position=1, leave=False):
    #         print(
    #             f"Working on {len(noise_levels)*offset_counter+noise_counter+1} of {len(noise_levels)*len(offset_levels)}")
    #         fix_time(max_trials=100, offset=offset, noise=noise,
    #                  jitter=10, add_offset=True, add_noise=True, add_jitter=True)
