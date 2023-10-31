# import tensorboard as tb
import matplotlib.pyplot as plt
import numpy as np
import torch
import re
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import os

# experiment = tb.data.experimental.ExperimentFromDev('6eOIgzYGQH6wLVNpO6eyBg')
# df = experiment.get_scalars()
# filter = df[df['run']=='Apr30_20-53-07_fwn-nb4-129-125-7-241Stimampli_MN_classA2B']
# df.drop(filter.index,inplace=True)
# run_list =  df['run'].unique()
MNclasses = {
    'A2B': {'a':0,'A1':0,'A2': 0},
    'C2J': {'a':5,'A1':0,'A2': 0},
    'K': {'a':30,'A1':0,'A2': 0},
    'L': {'a':30,'A1':10,'A2': -0.6},
    'M2O': {'a':5,'A1':10,'A2': -0.6},
    'P2Q': {'a':5,'A1':5,'A2': -0.3},
    'R': {'a':0,'A1':8,'A2': -0.1},
    'S': {'a':5,'A1':-3,'A2': 0.5},
    'T': {'a':-80,'A1':0,'A2': 0},
    
}
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14
plt.rcParams["figure.facecolor"] = "w"

plt.rc('font', family='georgia', size=SMALL_SIZE)
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
path = 'experiments/results'
folder_high = ''
if folder_high == '':
    print('No folder high specified, using last one')
    folder_high = np.sort(os.listdir(path))[-1]
    print(folder_high)
sweeps = ['ampli_neg']#,'freqs','slopes']
window_width = 1
avg_width = 10
debug_plot = True
# classes = os.listdir(folder)
folder_high = os.path.join(path,folder_high)
MI_coll = {}
colors = ['#BCE498','#7689A9','#FFE0AA','#E297AF']
sns.set_palette(colors)
seaborn_coll = {'MI':[],'Accuracy':[],'Class':[],'sweep':[],'measure':[],'type':[],'seed':[],'step':[]}
fig1 = plt.figure(figsize=(8,8))
gs = fig1.add_gridspec(4,3)
ax1 = fig1.add_subplot(gs[:3,:])
for MNclass in os.listdir(folder_high):
    MI_coll[MNclass] = {}
    range_type = [mystr.split('_')[0] for mystr in os.listdir(os.path.join(folder_high,MNclass))]
    range_sweep = [mystr.split('_')[1] for mystr in os.listdir(os.path.join(folder_high,MNclass))]
    for stim_type in os.listdir(os.path.join(folder_high,MNclass)):
        MI_coll[MNclass][stim_type] = {}
        for sweep in os.listdir(os.path.join(folder_high,MNclass,stim_type)):
            MI_coll[MNclass][stim_type][sweep] = {}
            for measure in os.listdir(os.path.join(folder_high,MNclass,stim_type,sweep)):
                MI_coll[MNclass][stim_type][sweep][measure] = {}
                for seed in os.listdir(os.path.join(folder_high,MNclass,stim_type,sweep,measure)):
                    print(MNclass + ' ' + stim_type + ' ' + sweep + ' ' + measure + seed)
                    MI = torch.load(os.path.join(folder_high,MNclass,stim_type,sweep,measure,seed,'MI.pt'))
                    acc = torch.load(os.path.join(folder_high,MNclass,stim_type,sweep,measure,seed,'Accuracy.pt'))
                    MI_coll[MNclass][stim_type][sweep][measure][seed] = MI

                    for step in range(len(MI)):
                        seaborn_coll['step'].append(step)
                        seaborn_coll['MI'].append(MI[step])
                        seaborn_coll['Accuracy'].append(acc[step]*100)
                        seaborn_coll['Class'].append(MNclass)
                        seaborn_coll['sweep'].append(sweep)
                        seaborn_coll['measure'].append(measure)
                        seaborn_coll['type'].append(stim_type)
                        seaborn_coll['seed'].append(seed)
amplitude = torch.load('stimuli/Braille_amplitude_data.pt')
frequency = torch.load('stimuli/Braille_frequency_data.pt')
slope = torch.load('stimuli/Braille_slope_data.pt')

seaborn_pd = pd.DataFrame.from_dict(seaborn_coll)
seaborn_pd = seaborn_pd[seaborn_pd['step']>=40]

sns.boxplot(x='type',y='MI',hue='Class',data=seaborn_pd,ax=ax1)
plt.xticks([0,1,2],['Amplitude','Frequency','Slope'])
plt.ylabel('Accuracy (%)')
plt.xlabel('Stimulus type')
# plt.ylim([0,3])
plt.title(f'MI for different stimulus types (over {len(os.listdir(os.path.join(folder_high,MNclass,stim_type,sweep,measure)))}  seeds and 10 last epochs)')
# plt.text(4, 3.322 - 0.15, 'max MI = 3.322 bits', horizontalalignment='center', verticalalignment='center')
# plt.plot([-1,6 + 1], [3.322, 3.322], 'k--')
# ax = plt.gca()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2 = fig1.add_subplot(gs[3,0])
ax3 = fig1.add_subplot(gs[3,1])
ax4 = fig1.add_subplot(gs[3,2])
ax2.plot(amplitude[0],color='k')
# ax2.set_title('Amplitude')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xticks([0,1000])
ax2.set_xticklabels([0,1])
ax2.set_xlabel('Time (s)')
ax3.plot(frequency[0],color='k',alpha=0.01)
# ax3.set_title('Frequency')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_xticks([0,1000])
ax3.set_xticklabels([0,1])
ax3.set_xlabel('Time (s)')
ax4.plot(slope[0],color='k')
# ax4.set_title('Slope')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.set_xticks([0,1000])
ax4.set_xticklabels([0,1])
ax4.set_xlabel('Time (s)')
letters = ['A', 'B', 'C', 'D']
for a in [ax1, ax2, ax3, ax4]:
    a.text(-0.0, 1.1, letters.pop(0), transform=a.transAxes,
           size=20, weight='bold')
# plt.legend(['Braille Trained','MNIST Trained','Tonic Naive','Adaptive Naive'])
plt.tight_layout()
fig1.savefig('Figures/MI_stimulus_types.png')
fig1.savefig('Figures/MI_stimulus_types.svg')

plt.show()
