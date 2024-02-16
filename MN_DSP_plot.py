# import tensorboard as tb
import matplotlib.pyplot as plt
import numpy as np
import torch
import re
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import os
import glob

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
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
plt.rc('font', family='georgia', size=SMALL_SIZE)
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
path = 'experiments/results/'
path = ''
# folder_high = 'experiments/results/06Jan2024_15-53-06'
# folder_high = '15Jan2024_22-02-20'
folder_high = 'MN_DSP_ampli_cnoise_20b_100s'
if folder_high == '':

    list_of_files = glob.glob(path+'*')  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    print('No folder high specified, using last one')
    folder_high = latest_file
    print(latest_file)
else:
    folder_high = glob.glob(path+folder_high)[0]
    # * means all if need specific format then *.csv
    print(folder_high)

sweeps = ['ampli_neg']#,'freqs','slopes']
window_width = 1
avg_width = 10
debug_plot = True
# classes = os.listdir(folder)
# folder_high = os.path.join(folder_high)
print(folder_high)
MI_coll = {}
colors = ['#BCE498','#7689A9','#FFE0AA','#E297AF']
sns.set_palette(colors)
seaborn_coll = {'MI':[],'Accuracy':[],'Class':[],'sweep':[],'measure':[],'type':[],'seed':[],'step':[],'train':[]}
#select only directories
folders = [f for f in os.listdir(folder_high) if os.path.isdir(os.path.join(folder_high, f))]
# folder_w = sorted([f for f in folders if 'w' in f], key=lambda x: int(re.findall(r'\d+', x)[0]))
# folder_nw = sorted([f for f in folders if 'nw' in f], key=lambda x: int(re.findall(r'\d+', x)[0]))
# folders = folder_w + folder_nw
# print(folders)
for MNclass in folders:
    MI_coll[MNclass] = {}
    range_type = [mystr.split('_')[0] for mystr in os.listdir(os.path.join(folder_high,MNclass))]
    # range_sweep = [mystr.split('_')[1] for mystr in os.listdir(os.path.join(folder_high,MNclass))]
    for stim_type in os.listdir(os.path.join(folder_high,MNclass)):
        MI_coll[MNclass][stim_type] = {}
        for sweep in os.listdir(os.path.join(folder_high,MNclass,stim_type)):
            MI_coll[MNclass][stim_type][sweep] = {}
            for measure in os.listdir(os.path.join(folder_high,MNclass,stim_type,sweep)):
                MI_coll[MNclass][stim_type][sweep][measure] = {}
                for seed in os.listdir(os.path.join(folder_high,MNclass,stim_type,sweep,measure)):
                    try:
                        MI = torch.load(os.path.join(folder_high,MNclass,stim_type,sweep,measure,seed,'MI.pt'))
                        print(os.path.join(folder_high,MNclass,stim_type,sweep,measure,seed,'MI.pt'))
                        acc = torch.load(os.path.join(folder_high,MNclass,stim_type,sweep,measure,seed,'Accuracy.pt'))
                        MI_coll[MNclass][stim_type][sweep][measure][seed] = MI

                        for step in range(len(MI)):
                            name = MNclass.split('_')
                            if len(name) > 1:

                                if 'w'==name[2]:
                                    seaborn_coll['train'].append('w')
                                elif 'nw'==name[2]:
                                    seaborn_coll['train'].append('nw')
                                else:
                                    seaborn_coll['train'].append('w')
                            else:
                                seaborn_coll['train'].append('w')

                            seaborn_coll['step'].append(step)
                            seaborn_coll['MI'].append(MI[step])
                            seaborn_coll['Accuracy'].append(acc[step] * 100)
                            seaborn_coll['Class'].append('_'.join(name[:-1]))
                            seaborn_coll['sweep'].append(sweep)
                            seaborn_coll['measure'].append(measure)
                            stim_type_split = stim_type.split('_')
                            seaborn_coll['type'].append(stim_type)
                            # if 'compressed' in stim_type:
                            #     seaborn_coll['type'].append('_'.join(stim_type_split[2:]))
                            # else:
                            #     seaborn_coll['type'].append('_'.join(stim_type.split('_')[1:]))
                            seaborn_coll['seed'].append(seed)
                    except FileNotFoundError:
                        print('File not found',os.path.join(folder_high,MNclass,stim_type,sweep,measure,seed,'MI.pt'))
# for type in ['opt','worse']:
#     amplitude = torch.load(f'{folder_high}/Braille_amplitude_{type}_data.pt')
#     # frequency = torch.load(f'{folder_high}/Braille_frequency_{type}_data.pt')
#     # slope = torch.load(f'{folder_high}/Braille_slope_{type}_data.pt')

seaborn_pd = pd.DataFrame.from_dict(seaborn_coll)
seaborn_pd = seaborn_pd[seaborn_pd['train'] == 'w']
sns.lineplot(x='step',y='Accuracy',hue='Class',style='type',data=seaborn_pd)
sns.despine(offset=10, trim=True)
plt.ylabel('Accuracy (%)')
plt.xlabel('Epoch')
plt.legend(fancybox=True, framealpha=0.5)
plt.tight_layout()
plt.savefig('Figures/MI_stimulus_training.pdf')
fig1,ax1 = plt.subplots(1,1,figsize=(6,4))
# gs = fig1.add_gridspec(4,3)
# ax1 = fig1.add_subplot(gs[:3,:])
seaborn_pd = seaborn_pd[seaborn_pd['step']>=len(MI)-10]
# sns.catplot(
#     data=seaborn_pd, x='type', y='Accuracy', hue='train',
#     kind='box'
# )

sns.boxplot(data=seaborn_pd, x='type', y='Accuracy', hue='Class')
plt.plot([-0.1,0.1], [100/50, 100/50], 'k')
plt.plot([0.9,1.1], [100/10, 100/10], 'k')
# plt.text(0.1, 100/50 + 3, 'Chance Level (2%)', horizontalalignment='center', verticalalignment='center')
# sns.boxplot(x="type", y="Accuracy", hue='Class', data=seaborn_pd,
#                  showcaps=False,boxprops={'facecolor':'None'},
#                  showfliers=False,whiskerprops={'linewidth':0})
# g = sns.FacetGrid(seaborn_pd, col="train")
# g.map(sns.boxplot,'type','Accuracy',hue='Class')
# plt.xticks([0,1,2],['Amplitude','Frequency','Slope'])
#add a entry to the legend

plt.ylabel('Accuracy (%)')
plt.xlabel('Stimulus type')
plt.ylim([0,100])
# seeds_n = len(os.listdir(os.path.join(folder_high,MNclass,stim_type,sweep,measure)))
# plt.title(f'MI for different stimulus types (over {seeds_n}  seeds and 10 last epochs)')
# # plt.text(4, 3.322 - 0.15, 'max MI = 3.322 bits', horizontalalignment='center', verticalalignment='center')
# # plt.plot([-1,6 + 1], [3.322, 3.322], 'k--')
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
print(labels)
from matplotlib.lines import Line2D
# handles.append(Line2D([0], [0], color='k', linestyle='-'))
# labels = ['MNIST Trained','MNIST compressed Trained','Braille Trained', 'Chance Level']
# handles.append(mpl.patches.Patch(color='white', label='Chance level'))
# plt.legend(handles=handles,labels=labels,fancybox=True, framealpha=0.5)
sns.despine(offset=10, trim=True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# ax2 = fig1.add_subplot(gs[3,0])
# ax3 = fig1.add_subplot(gs[3,1])
# ax4 = fig1.add_subplot(gs[3,2])
# ax2.plot(amplitude[0],color='k')
# # ax2.set_title('Amplitude')
# ax2.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.set_xticks([0,1000])
# ax2.set_xticklabels([0,1])
# ax2.set_xlabel('Time (s)')
# # ax3.plot(frequency[0],color='k',alpha=0.01)
# # ax3.set_title('Frequency')
# ax3.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.set_xticks([0,1000])
# ax3.set_xticklabels([0,1])
# ax3.set_xlabel('Time (s)')
# # ax4.plot(slope[0],color='k')
# # ax4.set_title('Slope')
# ax4.spines['top'].set_visible(False)
# ax4.spines['right'].set_visible(False)
# ax4.set_xticks([0,1000])
# ax4.set_xticklabels([0,1])
# ax4.set_xlabel('Time (s)')
# letters = ['A', 'B', 'C', 'D']
# for a in [ax1, ax2, ax3, ax4]:
#     a.text(-0.0, 1.1, letters.pop(0), transform=a.transAxes,
#            size=20, weight='bold')
# # plt.legend(['Braille Trained','MNIST Trained','Tonic Naive','Adaptive Naive'])
ax1.set_title(f'Accuracy (over 10 last epochs), seeds = {np.unique(np.array(seaborn_coll["seed"])).shape[0]}')
fig1.tight_layout()
fig1.savefig('Figures/MI_stimulus_boxplot.pdf')
# fig1.savefig('Figures/MI_stimulus_types.svg')

plt.show()
