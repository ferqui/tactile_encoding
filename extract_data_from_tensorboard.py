from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt
import json
from models import MN_neuron_sp
import os
import numpy as np
import seaborn as sns
import glob
import json
from tbparse import SummaryReader
from pathlib import Path
sns.set_style("white")



def extract_data_from_tensorboard(folder_source,folder_dest,model_name,plot=False,name_source=''):
    print(folder_source)
    scalars = SummaryReader(folder_source, pivot=True).scalars
    scalars['type'] = ['scalars' for i in range(len(scalars))]
    hparams = SummaryReader(folder_source, pivot=True).hparams
    hparams['type'] = ['hparams' for i in range(len(hparams))]
    df_scalars = scalars.melt(id_vars=['step','type'],var_name="variable", value_name="value")
    df_hparams = hparams.melt(id_vars=['type'],var_name="variable", value_name="value")
    df_hparams['step'] = [len(scalars) for i in range(len(df_hparams))]
    df = pd.concat([df_scalars,df_hparams])
    if plot:
        fig1, ax1 = plt.subplots(len(df['variable'].unique()), 1, sharex=True, figsize=(8, 8))
        colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(df['variable'].unique())))
    dict = {}
    for v_idx, variable in enumerate(df['variable'].unique()):
        sel = df[df['variable'] == variable]
        if (sel['type'] == 'scalars').any():
            dict[variable] = np.mean(sel['value'][sel['step'] > 0.9 * sel['step'].max()].values)
            if plot:
                ax1[v_idx].plot(sel['step'], sel['value'], label=variable, color=colors[v_idx])

        else:
            dict[variable] = sel['value'].values[0]
    # if plot:
    #     fig1.tight_layout()
    #     # plt.show()

    # plt.plot(sel['step'],sel['value'])
    try:
        seed = '_'+str(int(dict['seed']))
        if 'no_train_weights' in dict.keys():
            weights_trained = ['_nw' if bool(dict['no_train_weights']) == True else '_w'][0]
        else:
            weights_trained = ''
        json.dump(dict, open(os.path.join(folder_dest, f'{model_name}{weights_trained}{seed}.json'), 'w'))
        return f'{model_name}{weights_trained}{seed}.json'

    except:
        print(folder_source)
        print(dict)
        return None
        # raise ValueError
## create main
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("extract_data_from_tensorboard")
    parser.add_argument("--folder_source", type=str, default='runs_gr_Braille_converging/', help="folder_source")
    parser.add_argument("--folder_dest", type=str, default='./MN_params_Braille_w_and_nw/', help="folder_dest")
    parser.add_argument("--folder_Vittorio", type=str, default='./MN_params/', help="folder_dest")
    parser.add_argument("--model_name", type=str, default='GR_Braille', help="model_name")
    parser.add_argument("--plot", type=bool, default=False, help="plot")
    parser.add_argument("--name_source", type=str, default='', help="name_source")
    args = parser.parse_args()
    runs = glob.glob(args.folder_source + '*')
    folder_dest_path = Path(args.folder_dest)
    folder_dest_path.mkdir(exist_ok=True)
    bash_script = 'sbatch ./tactile_encoding2/send_batch_DSP.sh --debug_plot --load_range=MNIST_compressed --noise=0.01,0.01 --seed_n=100'
    coll = ''
    coll_vittorio = ''
    for run in runs:
        print(run)
        model_json = extract_data_from_tensorboard(run,args.folder_dest,args.model_name,args.plot,args.name_source)
        print(model_json)
        print('-'*30)
        if model_json is not None:
            coll += f'{model_json.replace(".json","")},'
            coll_vittorio += f'{args.folder_Vittorio}{model_json},'

    coll = coll[:-1]
    bash_script += f' --load_neuron={coll}'
    print('DSP bash script:')
    print(bash_script)
    print('Dataset creation for Vittorio:')
    bash_script = 'sbatch ./tactile_encoding2/send_batch_GR_opt_multiseed_simulate_gpushort.sh --path_to_optimal_model='
    coll_vittorio = coll_vittorio[:-1]
    bash_script += f'{coll_vittorio}'
    print(bash_script)
    plt.show()




