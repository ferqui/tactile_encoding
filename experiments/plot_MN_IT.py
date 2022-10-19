"""

"""
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec

sns.set_style('whitegrid')

results_dir = 'results/'
folder = {'a': '18Oct2022_09-55-48',
          'A1': '18Oct2022_09-55-48',
          'A2': '18Oct2022_09-55-48'}

loss = dict.fromkeys(folder.keys(), None)
mi = dict.fromkeys(folder.keys(), None)

for param in folder.keys():
    print(param)
    with open('results/'+param+'/'+folder[param]+'/Loss.pickle','rb') as f:
        loss[param] = pickle.load(f)

    with open('results/'+param+'/'+folder[param]+'/MI.pickle','rb') as f:
        mi[param] = pickle.load(f)

cmap = sns.color_palette()

fig = plt.figure(figsize=(8, 5))
gs = gridspec.GridSpec(1, 1)
gs.update(wspace=0.5, hspace=0.3, left=None, right=None)
axs = [fig.add_subplot(gs[0, 0])]
for param in folder.keys():
    axs[0].plot(loss[param], label=param)
axs[0].legend()
axs[0].set_title('Loss')
axs[0].set_xlabel('Training Epochs')
fig.savefig('results/figures/Loss.pdf', format='pdf')

fig = plt.figure(figsize=(8, 5))
gs = gridspec.GridSpec(1, 1)
gs.update(wspace=0.5, hspace=0.3, left=None, right=None)
axs = [fig.add_subplot(gs[0, 0])]
for param in folder.keys():
    axs[0].plot(mi[param], label=param)
axs[0].legend()
axs[0].set_title('MI')
fig.savefig('results/figures/MI.pdf', format='pdf')
axs[0].set_title('MI')
axs[0].set_xlabel('Test Epochs')
fig.savefig('results/figures/MI.pdf', format='pdf')

plt.show()

