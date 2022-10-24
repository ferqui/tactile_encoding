"""

"""
# import os.path
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
sns.set_style('whitegrid')
curr = os.getcwd()
folder_name = '21Oct2022_10-00-17'
range_values = ['[-10, 10]','[-100, 100]','[-1000, 1000]']
results_dir = os.path.join(curr,'results')
params = ['a', 'A1', 'A2','b','G','k1','k2']


loss = {'value':np.array([]),'epoch':np.array([]),'param':np.array([]),'range':np.array([])}
mi = {'value':np.array([]),'epoch':np.array([]),'param':np.array([]),'stable_value':np.array([]),
      'stable_param':np.array([]),'range':np.array([]),'stable_range':np.array([])}

for par_ix,param in enumerate(params):
    for ran_ix, range_v in enumerate(range_values):
        # print(param)
        with open(os.path.join(results_dir,folder_name,param,range_v,'Loss.pickle'),'rb')as f:
            tmp = pickle.load(f)
            loss['value'] = np.append(loss['value'],tmp[:,0]/tmp[:,0].max())
            loss['epoch'] = np.append(loss['epoch'],tmp[:,1])
            range_here = range(len(tmp[:,0]))
            loss['param'] = np.append(loss['param'],np.array([param for i in range_here]))
            loss['range'] = np.append(loss['range'],np.array([range_values[ran_ix] for i in range_here]))
        with open(os.path.join(results_dir,folder_name,param,range_v,'MI.pickle'),'rb') as f:
            tmp = pickle.load(f)
            mi['value'] = np.append(mi['value'],tmp[:,0])
            mi['epoch'] = np.append(mi['epoch'],tmp[:,1])
            mi['param'] = np.append(mi['param'],np.array([param for i in range(len(tmp[:,0]))]))
            mi['range'] = np.append(mi['range'],np.array([range_values[ran_ix] for i in range(len(tmp[:,0]))]))
            max = len(tmp[:,0])
            mi['stable_value'] = np.append(mi['stable_value'],tmp[int(max*0.95):max,0])
            mi['stable_param'] = np.append(mi['stable_param'],np.array([param for i in range(int(max*0.95),max)]))
            mi['stable_range'] = np.append(mi['stable_range'],np.array([range_values[ran_ix] for i in range(int(max*0.95),max)]))

cmap = sns.color_palette()
# g = sns.FacetGrid(loss,col = 'range',row = 'value',hue = 'param')
# g.map(sns.lineplot, x='epoch',y='value')
sns.lineplot(loss,x='epoch',y='value',hue = 'param',size = 'range',color=cmap)
plt.title('Loss')
plt.savefig(os.path.join(results_dir,folder_name,'Loss_sns.pdf'), format='pdf')
plt.figure()
sns.lineplot(mi,x='epoch',y='value',hue = 'param',size = 'range',color=cmap)
plt.title('Mutual Information')
plt.ylabel('Mutual Information (Bits)')
plt.savefig(os.path.join(results_dir,folder_name,'MI_sns.pdf'), format='pdf')
plt.figure()
sns.set_palette("husl", 3)
sns.barplot(mi,x='stable_param',y='stable_value',hue = 'stable_range')
plt.title('Mutual Information')
plt.ylabel('Mutual Information (Bits)')
plt.savefig(os.path.join(results_dir,folder_name,'MI_hist_sns.pdf'), format='pdf')
plt.show()


# fig = plt.figure(figsize=(8, 5))
# gs = gridspec.GridSpec(1, 1)
# gs.update(wspace=0.5, hspace=0.3, left=None, right=None)
# axs = [fig.add_subplot(gs[0, 0])]
# for param in folder.keys():
#     axs[0].plot(loss[param], label=param)
# axs[0].legend()
# axs[0].set_title('Loss')
# axs[0].set_xlabel('Training Epochs')
# fig.savefig('results/figures/Loss.pdf', format='pdf')
#
# fig = plt.figure(figsize=(8, 5))
# gs = gridspec.GridSpec(1, 1)
# gs.update(wspace=0.5, hspace=0.3, left=None, right=None)
# axs = [fig.add_subplot(gs[0, 0])]
# for param in folder.keys():
#     axs[0].plot(mi[param], label=param)
# axs[0].legend()
# axs[0].set_title('MI')
# fig.savefig('results/figures/MI.pdf', format='pdf')
# axs[0].set_title('MI')
# axs[0].set_xlabel('Test Epochs')
# fig.savefig('results/figures/MI.pdf', format='pdf')

plt.show()

