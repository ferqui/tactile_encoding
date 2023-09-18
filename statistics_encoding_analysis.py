import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import numpy as np
from pathlib import Path

folder_data = 'results/statistics_encoding_analysis'
output_folder = Path('./results')
sns.set_style('whitegrid')
list_dir = os.listdir(folder_data)
list_cm = []
list_gain = []
list_sigma = []
dict_data = {'gain': [], 'sigma': [], 'off_values': [], 'type': []}

for exp in list_dir:
    filename = os.getcwd()+'/'+folder_data+'/'+exp+'/output_data.pickle'
    with open(filename, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    cm = data['confusion_matrix']
    gain = data['parameters']['gain']
    sigma = data['parameters']['sigma']

    list_cm.append(cm)
    list_gain.append(gain)
    list_sigma.append(sigma)
    tot_trials = np.sum(cm)
    trueA_predC = cm[0,1]/tot_trials

    dict_data['off_values'].append(trueA_predC)
    dict_data['type'].append('trueA_predC')
    dict_data['gain'].append(gain)
    dict_data['sigma'].append(sigma)

    trueC_predA = cm[1,0]/tot_trials
    dict_data['off_values'].append(trueC_predA)
    dict_data['type'].append('trueC_predA')
    dict_data['gain'].append(gain)
    dict_data['sigma'].append(sigma)

fig, ax = plt.subplots(1,1)
sns.lineplot(data=pd.DataFrame(dict_data), x='gain', y='off_values', hue='sigma', style='type',
             markers=True, ax=ax)
ax.set_ylabel('Fract off diagonal')
fig.savefig(output_folder.joinpath('Off_diagonal_values.pdf'), format='pdf')
plt.show()

fig, ax = plt.subplots(1,1)
sns.lineplot(data=pd.DataFrame(dict_data), x='gain', y='off_values', hue='sigma',
             markers=True, ax=ax)
ax.set_ylabel('Fract off diagonal')
fig.savefig(output_folder.joinpath('Off_diagonal_values_merged.pdf'), format='pdf')
plt.show()

list_unique_sigma = list(np.unique(dict_data['sigma']))
list_unique_gain = list(np.unique(dict_data['gain']))

sns.set_style('white')
fig, axs = plt.subplots(len(list_unique_sigma), len(list_unique_gain))
for i, cm in enumerate(list_cm):
    gain = list_gain[i]
    sigma = list_sigma[i]
    i_gain = list_unique_gain.index(gain)
    i_sigma = list_unique_sigma.index(sigma)
    hm=sns.heatmap(cm/tot_trials, ax=axs[i_sigma, i_gain], annot=True, cbar=False, vmin=0, vmax=1,
                xticklabels=['P_A','P_C'], yticklabels=['T_A','T_C'])
    hm.set_xticklabels(hm.get_xticklabels(), rotation=0)

for y, sigma in enumerate(list_unique_sigma):
    axs[y,0].set_ylabel('Sigma={}'.format(np.round(sigma, 2)))

for x, gain in enumerate(list_unique_gain):
    axs[0,x].set_title('Gain={}'.format(np.round(gain, 2)))

fig.set_size_inches(20,10)
plt.subplots_adjust(hspace=0.3, wspace=0.4, left=.1)
fig.savefig(output_folder.joinpath('Confusion_matrix.pdf'), format='pdf')
plt.show()