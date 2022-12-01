"""

"""
# import os.path
import os
import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
sns.set_style('whitegrid')
curr = os.getcwd()
from torch.utils.tensorboard import SummaryWriter

def plot_the_data(folder_name,range_values,results_dir,params,writer = None,list_IC = [],letters = []):
    if len(list_IC) == 0:
        list_IC = os.listdir(os.path.join(results_dir,folder_name)) # list initial conditions
    if 'figures' in list_IC:
        list_IC.remove('figures')

    loss = {'value':np.array([]),'epoch':np.array([]),'param':np.array([]),'range':np.array([]), 'ic':np.array([])}
    mi = {'value':np.array([]),'epoch':np.array([]),'param':np.array([]),'stable_value':np.array([]),
          'stable_param':np.array([]),'range':np.array([]),'stable_range':np.array([]), 'ic':np.array([]), 'ic_stable':np.array([])}
    spikecount = {'mean':np.array([]),'epoch':np.array([]),'param':np.array([]),'range':np.array([]), 'ic':np.array([])}
    # Generate folder with output figures:
    if not (os.path.isdir(os.path.join(results_dir,folder_name,'figures'))):
        os.mkdir(os.path.join(results_dir,folder_name,'figures'))

    for par_ix,param in enumerate(params):
        for ran_ix, range_v in enumerate(range_values):
            for ic in list_IC:
                folder = os.path.join(results_dir,folder_name,ic,param,range_v)
                # print(param)
                with open(os.path.join(folder,'Loss.pickle'),'rb') as f:
                    tmp = pickle.load(f)
                    loss['value'] = np.append(loss['value'],tmp[:,0]/tmp[:,0].max())
                    loss['epoch'] = np.append(loss['epoch'],tmp[:,1])
                    range_here = range(len(tmp[:,0]))
                    loss['param'] = np.append(loss['param'],np.array([param for i in range_here]))
                    loss['range'] = np.append(loss['range'],np.array([range_values[ran_ix] for i in range_here]))
                    loss['ic'] = np.append(loss['ic'], np.array([ic for i in range_here]))
                with open(os.path.join(folder,'MI.pickle'),'rb') as f:
                    tmp = pickle.load(f)
                    mi['value'] = np.append(mi['value'],tmp[:,0])
                    mi['epoch'] = np.append(mi['epoch'],tmp[:,1])
                    mi['param'] = np.append(mi['param'],np.array([param for i in range(len(tmp[:,0]))]))
                    mi['range'] = np.append(mi['range'],np.array([range_values[ran_ix] for i in range(len(tmp[:,0]))]))
                    mi['ic'] = np.append(mi['ic'], np.array([ic for i in range(len(tmp[:,0]))]))
                    max = len(tmp[:,0])
                    mi['stable_value'] = np.append(mi['stable_value'],tmp[int(max*0.95):max,0])
                    mi['stable_param'] = np.append(mi['stable_param'],np.array([param for i in range(int(max*0.95),max)]))
                    mi['stable_range'] = np.append(mi['stable_range'],np.array([range_values[ran_ix] for i in range(int(max*0.95),max)]))
                    mi['ic_stable'] = np.append(mi['ic_stable'], np.array([ic for i in range(int(max*0.95),max)]))
                with open(os.path.join(folder,'spikecount.pickle'),'rb') as f:
                    tmp = pickle.load(f)
                    spikecount['mean'] = np.append(spikecount['mean'],tmp['mean'])
                    spikecount['epoch'] = np.append(spikecount['epoch'],range(len(tmp['mean'])))
                    spikecount['param'] = np.append(spikecount['param'],np.array([param for i in range(len(tmp['mean']))]))
                    spikecount['range'] = np.append(spikecount['range'],np.array([range_values[ran_ix] for i in range(len(tmp['mean']))]))
                    spikecount['ic'] = np.append(spikecount['ic'], np.array([ic for i in range(len(tmp['mean']))]))
    col_wrap = len(list_IC)*(len(list_IC)<3) + 3*(len(list_IC)>=3)
    print('Converted data to be seaborn compatible. Starting creating the plots...')
    fig0,axis0 = plt.subplots(nrows = 1, ncols = 1)
    cmap = sns.color_palette(n_colors=len(params))
    cmap_hist = sns.color_palette('viridis',n_colors=len(range_values))
    fig_folder = 'figures'
    sns.lineplot(data=pd.DataFrame(loss),x='epoch',y='value',hue = 'param',size = 'range',palette=cmap,ax=axis0)
    axis0.set_title('Loss')
    fig0.savefig(os.path.join(results_dir,folder_name,fig_folder,'Loss_sns.pdf'), format='pdf')
    print('Created Loss plot.')

    fig1,axis1 = plt.subplots(nrows = 1, ncols = 1)
    sns.lineplot(data=mi,x='epoch',y='value',hue = 'param',size = 'range',palette=cmap,ax=axis1)
    axis1.set_title('Mutual Information')
    axis1.set_ylabel('Mutual Information (Bits)')
    fig1.savefig(os.path.join(results_dir,folder_name,fig_folder,'MI_sns.pdf'), format='pdf')
    print('Created MI plot.')



    # Sort by initial conditions:
    fig3,axis3 = plt.subplots(nrows = 1, ncols = 1)
    fig3 = sns.relplot(data=pd.DataFrame(loss),x='epoch',y='value', col='ic', hue = 'param',size = 'range', kind='line',
                height=4, col_wrap=3,palette = cmap)

    fig3.savefig(os.path.join(results_dir,folder_name,fig_folder,'Loss_sns_ics.pdf'), format='pdf')
    print('Created Loss plot for classes.')

    fig4,axis4 = plt.subplots(nrows = 1, ncols = 1)
    mi_allsamples = mi.copy()
    mi_allsamples.pop('stable_value')
    mi_allsamples.pop('stable_param')
    mi_allsamples.pop('stable_range')
    mi_allsamples.pop('ic_stable')
    fig4 = sns.relplot(data=mi_allsamples,x='epoch',y='value', col='ic', hue = 'param',size = 'range', kind='line',
                height=4, col_wrap=col_wrap,palette = cmap)
    axis4.set_title('Mutual Information')
    fig4.savefig(os.path.join(results_dir,folder_name,fig_folder,'MI_sns_ics.pdf'), format='pdf')
    print('Created MI plot for classes.')

    #

    fig5,axis5 = plt.subplots(nrows = 1, ncols = 1)
    mi_finalsamples = mi.copy()
    mi_finalsamples.pop('epoch')
    mi_finalsamples.pop('value')
    mi_finalsamples.pop('param')
    mi_finalsamples.pop('range')
    mi_finalsamples.pop('ic')
    fig5 = sns.catplot(data=pd.DataFrame(mi_finalsamples),x='stable_param',y='stable_value', col='ic_stable', hue = 'stable_range',
                kind='bar', height=4, col_wrap=col_wrap,palette = cmap_hist)
    fig5.savefig(os.path.join(results_dir,folder_name,fig_folder,'MI_hist_sns_ics.pdf'), format='pdf')
    # f_and_a = []

    fig6 = sns.relplot(data=pd.DataFrame(spikecount), x='epoch', y='mean', col='ic', hue='param', size='range', kind='line',
                       height=4, col_wrap=col_wrap, palette=cmap,alpha = 1)

    fig6.savefig(os.path.join(results_dir, folder_name, fig_folder, 'Spikecount_sns_ics.pdf'), format='pdf')
    print('Created Spike count plot for classes.')

    # for i in range(len(list_IC)):
    #     f_and_a.append(plt.subplots(nrows=1,ncols=1))
    for MN_class_idx,MN_class in enumerate(list_IC):
        print(MN_class)
        fig2,axis2 = plt.subplots(nrows = 1, ncols = 1)
        indexes = mi_finalsamples['ic_stable'] == MN_class
        tmp = {}
        for key in mi_finalsamples:
            tmp[key] = mi_finalsamples[key][indexes]
        sns.barplot(data=tmp,x='stable_param',y='stable_value',hue = 'stable_range',ax=axis2,palette = cmap_hist)
        axis2.set_title('Mutual Information')
        axis2.set_ylabel('Mutual Information (Bits)')
        axis2.set_title(MN_class)
        # f_and_a[MN_class_idx][0].savefig(os.path.join(results_dir,folder_name,MN_class,'MI_hist_sns.pdf'), format='pdf')
        writer.add_figure('MI_histograms', fig2,global_step=MN_class_idx)
        plt.show()
    print('Created MI histogram for classes.')
    #
    result_plots = {
        'Loss_sns':[fig0,axis0],
        'MI_sns':[fig1,axis1],
    #     # 'Loss_sns_ics':[fig3,axis3],
    #     # 'MI_sns_ics':[fig4,axis4],
    #     # 'MI_hist_sns_ics':[fig5,axis5]
    }
    #
    if writer:
        for key in result_plots:
            result_plots[key][1].set_title(
                str(key))
            writer.add_figure(key, result_plots[key][0])
        print('Wrote to tensorboard.')


if __name__ == "__main__":
    writer = SummaryWriter(comment="final_data_classes")
    folder_name = '30Nov2022_10-21-47/[]'
    range_values = ['[-10, 10]']#, '[-100, 100]', '[-1000, 1000]']
    results_dir = os.path.join(curr, 'results')
    params = ['gain','a', 'A1', 'A2', 'b', 'G', 'k1', 'k2']
    list_IC = ['C2J']
    plot_the_data(folder_name,range_values,results_dir,params,writer,list_IC = list_IC)
    plt.show()