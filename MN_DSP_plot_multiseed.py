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
path = ''
folder_high = 'DSP_multiseed'
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
seaborn_coll = {'MI':[],'Accuracy':[],'Class':[],'sweep':[],'measure':[],'type':[],'seed_stim':[],'step':[],'train':[],'seed_model':[]}
#select only directories
folders = [f for f in os.listdir(folder_high) if os.path.isdir(os.path.join(folder_high, f))]
# folder_w = sorted([f for f in folders if 'w' in f], key=lambda x: int(re.findall(r'\d+', x)[0]))
# folder_nw = sorted([f for f in folders if 'nw' in f], key=lambda x: int(re.findall(r'\d+', x)[0]))
# folders = folder_w + folder_nw
for MNclass in folders:
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
                    try:
                        seed_model = re.findall(r'\d+', MNclass)[0]
                        class_neuron = MNclass.replace(seed_model,'')
                    except:
                        seed_model = ''
                        class_neuron = MNclass
                    try:
                        MI = torch.load(os.path.join(folder_high,MNclass,stim_type,sweep,measure,seed,'MI.pt'))
                        acc = torch.load(os.path.join(folder_high,MNclass,stim_type,sweep,measure,seed,'Accuracy.pt'))
                        MI_coll[MNclass][stim_type][sweep][measure][seed] = MI

                        for step in range(len(MI)):
                            # try:
                            #     if 'w'==name[2]:
                            #         seaborn_coll['train'].append('w')
                            #     else:
                            #         seaborn_coll['train'].append('nw')
                            # except:
                            seaborn_coll['train'].append('nw')
                            seaborn_coll['step'].append(step)
                            seaborn_coll['MI'].append(MI[step])
                            seaborn_coll['Accuracy'].append(acc[step]*100)
                            seaborn_coll['Class'].append(class_neuron)
                            seaborn_coll['sweep'].append(sweep)
                            seaborn_coll['measure'].append(measure)
                            try:
                                if 'Tonic' == MNclass:
                                    class_cleaned = 'Braille'
                                elif 'Adaptive' == MNclass:
                                    class_cleaned = 'Braille'
                                else:
                                    mylist = MNclass.replace(seed_model,'').split('_')[1:]
                                    if len(mylist) > 0:
                                        class_cleaned = '_'.join(mylist)
                                        if 'w' in class_cleaned:
                                            class_cleaned = class_cleaned.replace('w_','')
                                        elif 'nw' in class_cleaned:
                                            class_cleaned = class_cleaned.replace('nw_','')
                                        print(stim_type)
                                print(class_cleaned)
                                stim_type_cut=stim_type.replace(class_cleaned,'')
                                print(stim_type_cut)
                            except:
                                print(stim_type_cut)
                            print('-'*30)
                            seaborn_coll['type'].append(stim_type_cut)
                            seaborn_coll['seed_stim'].append(seed)
                            seaborn_coll['seed_model'].append(seed_model)
                    except FileNotFoundError:
                        print('File not found',os.path.join(folder_high,MNclass,stim_type,sweep,measure,seed,'MI.pt'))
# for type in ['opt','worse']:
#     amplitude = torch.load(f'{folder_high}/Braille_amplitude_{type}_data.pt')
#     # frequency = torch.load(f'{folder_high}/Braille_frequency_{type}_data.pt')
#     # slope = torch.load(f'{folder_high}/Braille_slope_{type}_data.pt')

seaborn_pd = pd.DataFrame.from_dict(seaborn_coll)
sns.lineplot(x='step',y='Accuracy',hue='Class',style='type',size='train',data=seaborn_pd)
fig1 = plt.figure(figsize=(8,8))
gs = fig1.add_gridspec(4,3)
ax1 = fig1.add_subplot(gs[:3,:])
seaborn_pd = seaborn_pd[seaborn_pd['step']>=len(MI)-10]
sns.catplot(
    data=seaborn_pd, x='type', y='Accuracy', hue='Class',
    kind='box'
)
# g = sns.FacetGrid(seaborn_pd, col="train")
# g.map(sns.boxplot,'type','Accuracy',hue='Class')
# plt.xticks([0,1,2],['Amplitude','Frequency','Slope'])
plt.ylabel('Accuracy (%)')
plt.xlabel('Stimulus type')
plt.ylim([0,100])
# seeds_n = len(os.listdir(os.path.join(folder_high,MNclass,stim_type,sweep,measure)))
# plt.title(f'MI for different stimulus types (over {seeds_n}  seeds and 10 last epochs)')
# # plt.text(4, 3.322 - 0.15, 'max MI = 3.322 bits', horizontalalignment='center', verticalalignment='center')
# # plt.plot([-1,6 + 1], [3.322, 3.322], 'k--')
ax = plt.gca()
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
# plt.tight_layout()
# fig1.savefig('Figures/MI_stimulus_types.png')
# fig1.savefig('Figures/MI_stimulus_types.svg')

plt.show()
