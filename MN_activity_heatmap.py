import datetime
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sn

from NNI.utils.utils import create_directory


def MN_activity_heatmap(activity_df, lbl_string, save_fig=False, path_to_save=None):

    if save_fig and path_to_save==None:
        raise ValueError('The path where to save the heatmap is not given.')

    experiment_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # For the activity classification:
    labels_mapping = {
        'A': "Tonic spiking",
        'B': "Class 1",
        'C': "Spike frequency adaptation",
        'D': "Phasic spiking",
        'E': "Accommodation",
        'F': "Threshold variability",
        'G': "Rebound spike",
        'H': "Class 2",
        'I': "Integrator",
        'J': "Input bistability",
        'K': "Hyperpolarizing spiking",
        'L': "Hyperpolarizing bursting",
        'M': "Tonic bursting",
        'N': "Phasic bursting",
        'O': "Rebound burst",
        'P': "Mixed mode",
        'Q': "Afterpotentials",
        'R': "Basal bistability",
        'S': "Preferred frequency",
        'T': "Spike latency",
    }
    grouped = activity_df[["Letter","Probabilities"]].groupby("Letter", as_index=False).mean()
    classified_activity_df = pd.DataFrame(index=range(len(lbl_string)), columns=range(len(list(labels_mapping.values()))))
    for ii in range(len(lbl_string)):
        for jj in range(len(list(labels_mapping.keys()))):
            classified_activity_df.iloc[ii,jj] = float(grouped[grouped["Letter"]==lbl_string[ii]]["Probabilities"].item()[-1][jj])
    classified_activity_df = classified_activity_df.apply(pd.to_numeric, errors='coerce')
    plt.figure(figsize=(16, 12))
    sn.heatmap(classified_activity_df.T,
               annot=True,
               fmt='.2f',
               cbar=False,
               square=False,
               cmap="YlOrBr"
               )
    plt.xticks(ticks=[ii+0.5 for ii in range(27)],labels=lbl_string, rotation=0)
    plt.yticks(ticks=[ii+0.5 for ii in range(20)],labels=labels_mapping.values(), rotation=0)
    plt.tight_layout()
    if save_fig:
        create_directory(path_to_save)
        plt.savefig(os.path.join(path_to_save,f"MN_heatmap{experiment_datetime}.png"), dpi=300)
        plt.savefig(os.path.join(path_to_save,f"MN_heatmap{experiment_datetime}.pdf"), dpi=300)
        plt.close()
    else:
        plt.show()


activity = pd.read_pickle("./results/activity_classification/MN_activity/MN_output_Braille/GR_braille_w_0_eval_20240116_105004.pkl")
save_path = "./"
lbl_string = ['Space', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

MN_activity_heatmap(activity, lbl_string, save_fig=True, path_to_save=save_path)
