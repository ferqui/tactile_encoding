# import tensorboard as tb
import matplotlib.pyplot as plt
import numpy as np
import torch
import re
import matplotlib as mpl
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
# df['class'] = ''
# df['sweep'] = ''
# for MNclass in MNclasses.keys():
#     filter_with_class = df[df['run'].str.contains('class'+MNclass)]
#     df['class'][filter_with_class.index] = MNclass
# for sweep in ['ampli','freqs']:
#     filter_with_sweep = df[df['run'].str.contains(sweep)]
#     df['sweep'][filter_with_sweep.index] = sweep
#
# MI_filter = df[df['tag']=='MI']
# window_width = 1
# sweep_filter = MI_filter[MI_filter['sweep']=='freqs']
# avg_data = []
# std_data = []
# for MNclass in MNclasses.keys():
#     filter_with_class = sweep_filter[sweep_filter['class']==MNclass]
#     avg_data.append(np.mean(np.array(filter_with_class['value'][250:])))
#     std_data.append(np.std(np.array(filter_with_class['value'][250:])))
#
#     cumsum_vec = np.cumsum(np.array(filter_with_class['value']))
#     ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
#     window_width_filter = filter_with_class[filter_with_class['step']>=window_width]
#     plt.plot(window_width_filter['step'],ma_vec,label=MNclass)
# plt.title('MI freq (window ' + str(window_width) +' epochs)')
# plt.ylabel('MI (bits)')
# plt.xlabel('epoch')
# plt.legend()
#
# plt.show()
# plt.bar(MNclasses.keys(),avg_data,yerr=std_data)
# plt.ylabel('MI (bits)')
# plt.xlabel('epoch')
# plt.title('MI freq (last 50 epochs)')
# plt.show()
#
# value_filter = df[df['tag']=='MSE/train']
# window_width = 1
# sweep_filter = value_filter[value_filter['sweep']=='freqs']
# avg_data = []
# std_data = []
# for MNclass in MNclasses.keys():
#     filter_with_class = sweep_filter[sweep_filter['class']==MNclass]
#     avg_data.append(np.mean(np.array(filter_with_class['value'][250:])))
#     std_data.append(np.std(np.array(filter_with_class['value'][250:])))
#
#     cumsum_vec = np.cumsum(np.array(filter_with_class['value']))
#     ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
#     window_width_filter = filter_with_class[filter_with_class['step']>=window_width]
#     plt.plot(window_width_filter['step'],ma_vec,label=MNclass)
# plt.title('MSE train freq (window ' + str(window_width) +' epochs)')
# plt.ylabel('MSE')
# plt.xlabel('epoch')
# plt.legend()
# plt.show()
#
# value_filter = df[df['tag']=='MSE/train']
# window_width = 1
# sweep_filter = value_filter[value_filter['sweep']=='ampli']
# avg_data = []
# std_data = []
# for MNclass in MNclasses.keys():
#     filter_with_class = sweep_filter[sweep_filter['class']==MNclass]
#     avg_data.append(np.mean(np.array(filter_with_class['value'][250:])))
#     std_data.append(np.std(np.array(filter_with_class['value'][250:])))
#
#     cumsum_vec = np.cumsum(np.array(filter_with_class['value']))
#     ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
#     window_width_filter = filter_with_class[filter_with_class['step']>=window_width]
#     plt.plot(window_width_filter['step'],ma_vec,label=MNclass)
# plt.title('MSE train ampli (window ' + str(window_width) +' epochs)')
# plt.ylabel('MSE')
# plt.xlabel('epoch')
# plt.legend()
# plt.show()
# plt.plot(df['step'][df['sweep']=='ampli'],df['MI'][df['sweep']=='ampli'])
# print('ciao')
import os
# folder = 'experiments/results/12May2023_11-03-46'
# folder = 'experiments/results/15May2023_13-52-17/spikes'
# folder = 'experiments/results/15May2023_14-37-58/histogram_IT'
# folder = 'experiments/results/15May2023_15-32-16/histogram_IT'
# folder = 'experiments/results/15May2023_15-40-18/histogram_isi_IT'
# folder = 'experiments/results/15May2023_17-34-41/count_hist'
# folder_high = 'experiments/results/04Sep2023_09-47-01'
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
for MNclass in os.listdir(folder_high):
    MI_coll[MNclass] = {}
    for stim_type in os.listdir(os.path.join(folder_high,MNclass)):
        MI_coll[MNclass][stim_type] = {}
        for sweep in os.listdir(os.path.join(folder_high,MNclass,stim_type)):
            MI_coll[MNclass][stim_type][sweep] = {}
            for measure in os.listdir(os.path.join(folder_high,MNclass,stim_type,sweep)):
                print(MNclass + ' ' + stim_type + ' ' + sweep + ' ' + measure)
                MI = torch.load(os.path.join(folder_high,MNclass,stim_type,sweep,measure,'MI.pt'))
                MI_coll[MNclass][stim_type][sweep][measure] = MI
colors_classes = ['Greens','Blues','Reds','Purples']
hatches = ['/','\\','|','+','x','o','O','.']
rep_color = []
for measure in os.listdir(os.path.join(folder_high,MNclass,stim_type,sweep)):
    fig,ax = plt.subplots()
    labels = []
    idx_coll = []
    for MNclass_idx,MNclass in enumerate(MI_coll.keys()):
        colors = plt.cm.get_cmap(colors_classes[MNclass_idx],10)(np.linspace(0.5,1,len(MI_coll[MNclass].keys())))
        rep_color.append([MNclass,colors[1]])
        for stim_type_idx,stim_type in enumerate(MI_coll[MNclass].keys()):
            for sweep_idx,sweep in enumerate(MI_coll[MNclass][stim_type].keys()):
                MI = MI_coll[MNclass][stim_type][sweep][measure]
                avg_last = np.mean(np.array(MI[-50:]))
                print('sweep_idx',sweep_idx)
                print('stim_type_idx',stim_type_idx)
                print('MNclass_idx',MNclass_idx)
                idx = sweep_idx + stim_type_idx*(len(MI_coll.keys()))*len(MI_coll[MNclass][stim_type].keys())*1.1 + MNclass_idx*len(MI_coll[MNclass][stim_type].keys())
                print(idx)
                idx_coll.append(idx)
                ax.bar(idx,avg_last,color=colors[1], fill=True, hatch=hatches[stim_type_idx])
                ax.text(idx-0.2,avg_last+0.01,str(avg_last)[:4])
                labels.append(stim_type)
                # MI = MI_coll[MNclass][stim_type][sweep][measure]
                # cumsum_vec = np.cumsum(np.array(MI))
                # ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
                # window_width_filter = MI[window_width:]
                # avg_vec = np.convolve(window_width_filter, np.ones(avg_width), 'valid') / avg_width
                # ax.plot(avg_vec,label=MNclass + ' ' + stim_type)
    plt.plot([-1,idx+1],[3.322,3.322],'k--')
    from matplotlib.lines import Line2D
    custom_lines = []
    custom_label = []
    for c in range(len(rep_color)):
        custom_lines.append(Line2D([0], [0], color=rep_color[c][1], lw=4))
        custom_label.append(rep_color[c][0])
    for c in range(len(MI_coll[MNclass].keys())):
        custom_lines.append(mpl.patches.Patch(facecolor='#DCDCDC', hatch=hatches[c]))
        custom_label.append(list(MI_coll[MNclass].keys())[c])
    plt.legend(custom_lines, custom_label,ncol=2,loc='center',title='Neuron type | Stimulus type')
    #remove box
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # plt.legend(rep_color)
    plt.text(4, 3.322 - 0.15, 'max MI = 3.322 bits', horizontalalignment='center', verticalalignment='center')
    plt.title(measure)
    plt.ylabel('MI (bits)')
    # plt.xlabel('epoch')
    # plt.legend()
    # plt.xticks(idx_coll,labels,rotation=45)
    plt.xticks([])
    # plt.yticks((list(plt.yticks()[0]) + [3.322]),(list(plt.yticks()[0]) + ['max']))
    plt.xlim([-0.5,idx+0.5])
    plt.tight_layout()
    plt.show()
#
# for measure in os.listdir(folder_high):
#     folder = os.path.join(folder_high,measure)
#     classes = [notfile for notfile in os.listdir(folder) if os.path.isdir(folder+'/'+notfile)]
#     classes = sorted(classes)
#     MI_coll = {}
#     fig3,ax3 = plt.subplots()
#     for MNclass in classes:
#         MI_coll[MNclass] = {}
#         stim_types = os.listdir(os.path.join(folder,MNclass))
#
#         for stim_type in stim_types:
#             MI_coll[MNclass][stim_type] = {}
#             sweeps = os.listdir(folder + '/' + MNclass + '/' + stim_type)
#             for sweep in sweeps:
#                 MI_coll[MNclass][stim_type][sweep] = {}
#                 print(MNclass + ' ' + stim_type + ' ' + sweep)
#                 try:
#                     MI = torch.load(folder + '/' + MNclass + '/' + stim_type + '/' + sweep + '/MI.pt')
#                 except FileNotFoundError:
#                     MI = [np.NAN for i in range(300)]
#                 cumsum_vec = np.cumsum(np.array(MI))
#                 ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
#                 MI_coll[MNclass][stim_type][sweep]['mean'] = np.mean(np.array(MI[avg_width:]))
#                 MI_coll[MNclass][stim_type][sweep]['std'] = np.std(np.array(MI[avg_width:]))
#
#                 MI_coll[MNclass][stim_type][sweep]['number'] = re.findall(r'\b\d+.\d{1,}|\d+\b', sweep)
#                 ax3.plot(np.arange(len(MI))[window_width:], ma_vec, label="MI " + MNclass + ' ' + stim_type + ' ' + sweep)
#
#     for MN_class in MI_coll.keys():
#         print('MN_class',MN_class)
#         for stim_type in MI_coll[MN_class].keys():
#             print('stim_type',stim_type)
#             tmp = {}
#             last_coll = []
#             for sweep in MI_coll[MN_class][stim_type].keys():
#                 print('sweep',sweep)
#                 try:
#                     last_coll.append(float(MI_coll[MN_class][stim_type][sweep]['number'][-1]))
#                 except IndexError:
#                     print('IndexError')
#             last_coll = np.argsort(last_coll)
#             for colls in np.array(list(MI_coll[MN_class][stim_type].keys()))[last_coll]:
#                 tmp[colls] = MI_coll[MN_class][stim_type][colls]
#             MI_coll[MN_class][stim_type] = tmp
#     plt.title('MI (window ' + str(window_width) +' epochs)')
#     plt.ylabel('MI (bits)')
#     plt.xlabel('epoch')
#
#     # plt.legend()
#     colors = plt.cm.get_cmap('viridis',10)(np.linspace(0.4,0.8,3))
#     stim_list = MI_coll[classes[0]].keys()
#
#     for stim_type in stim_types:
#         plt.figure()
#         avg_data = []
#         std_data = []
#         x_bar = []
#         for MNclass_idx,MNclass in enumerate(classes):
#             sweeps = MI_coll[MNclass][stim_type].keys()
#             for sweep_idx,sweep in enumerate(sweeps):
#                 avg_data = MI_coll[MNclass][stim_type][sweep]['mean']
#                 std_data = MI_coll[MNclass][stim_type][sweep]['std']
#                 x_bar = sweep_idx + MNclass_idx*(len(sweeps)+1)
#                 if MNclass_idx == 0:
#                     sweep = re.findall(r'\b\d+.\d{1,}|\d+\b', sweep)
#                     try:
#                         labb = str([float(sweep[0]),float(sweep[-1])])
#                     except:
#                         print('ciao')
#                 else:
#                     labb = None
#                 plt.bar(x_bar,height = avg_data,yerr=std_data,color=colors[sweep_idx],label=labb)
#         center = np.floor(len(sweeps) / 2)
#         ticks = [center + i * (len(sweeps) + 1) for i in range(len(classes))]
#         labels = [MNclass for MNclass in classes]
#         plt.ylabel('MI (bits)')
#         plt.xlabel('classes')
#         plt.xticks(ticks,labels)
#         plt.legend()
#         plt.legend()
#         plt.title('MI ' + stim_type + ' (last ' + str(avg_width) +' epochs), measure ' + measure)
#         plt.ylim([0,3])
#         plt.savefig(folder + '/MI_' + stim_type + measure + '.png',format='png')
#     colors = plt.cm.get_cmap('magma', 10)(np.linspace(0.2, 0.8, 3))
#     fig1,ax = plt.subplots(ncols=5,nrows=2)#subplot_kw=dict(projection='polar'),figsize=(10,10))
#     for MN_class_idx, MN_class in enumerate(classes):
#         row = int(np.floor(MN_class_idx / 5))
#
#         col = MN_class_idx % 5
#         print('row: ' + str(row) + ' col: ' + str(col) + ' MN_class_idx: ' + str(MN_class_idx) + ' MN_class: ' + MN_class)
#         angles = [n / float(len(MI_coll[MN_class].keys())) * 2 * 3.14 for n in range(len(MI_coll[MN_class].keys()))]
#         angles += angles[:1]
#         # ax[row, col].set_theta_offset(3.14 / 2)
#         # ax[row, col].set_theta_direction(-1)
#         # ax[row, col].set_rlabel_position(0)
#         MI_elements = []
#         for stim_type in MI_coll[MN_class].keys():
#             stim_sweep = list(MI_coll[MN_class][stim_type].keys())[0]
#             MI_elements_sweep = []
#             for stim_sweep_idx,stim_sweep in enumerate(MI_coll[MN_class][stim_type].keys()):
#                 MI_elements_sweep.append(MI_coll[MN_class][stim_type][stim_sweep]['mean'])
#             MI_elements.append(MI_elements_sweep)
#
#         for i in range(1):
#             tmp_list = []
#             # print(colors[i])
#             for stim_idx in range(len(MI_coll[MN_class].keys())):
#                 tmp_list.append(MI_elements[stim_idx][i])
#
#             ax[row, col].bar(angles, np.append(tmp_list, tmp_list[0]),
#                       color=colors[i],
#                       linewidth=2, linestyle='solid', label=i, alpha=0.6)
#
#         # rows = int(np.ceil(total / 5))
#         # ax[row,col].fill(angles, np.append(MI_elements, MI_elements[0]),
#         #                   color=colors[MN_class_idx],alpha=1)
#         ax[row,col].set_xticks(angles[:-1])
#         ax[row,col].set_xticklabels(MI_coll[MN_class].keys(), color='grey', size=8)
#         ax[row,col].set_title(MN_class)
#         ax[row,col].legend()
#         ax[row,col].set_ylim([0,3])
#     # fig1.tight_layout()
#     fig1.suptitle('MI ' + measure)
#     fig1.savefig(folder + '/' + measure + '_MI_bars.png',format='png')

plt.show()

        # print('ciao')