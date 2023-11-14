from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt
import json
from models import MN_neuron_sp
import os
import numpy as np
import seaborn as sns
import glob

sns.set_style("darkgrid")
path = 'runs/'
folder_high = 'Nov13_09-25-46_v100gpu24GR_MNIST/events.out.tfevents.1699863946.v100gpu24.999429.0'
if folder_high == '':

    list_of_files = glob.glob(path+'*')  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    # print(latest_file)
    print('No folder high specified, using last one')
    folder_high = latest_file
    # print(latest_file)
    file = glob.glob(folder_high+'/*')[0]
    # * means all if need specific format then *.csv
    print(file)
else:
    file = glob.glob(path+folder_high)[0]
    # * means all if need specific format then *.csv
    print(file)
ea = event_accumulator.EventAccumulator(file)
ea.Reload()
print(ea.scalars.Keys())
print(ea.scalars.Items('a'))
dict = {'wall_time':[],'step':[],'value':[],'variable':[]}
df_coll = []

for key in ea.scalars.Keys():
    df = pd.DataFrame(ea.Scalars(key))
    df['variable'] = [key for i in range(len(df))]
    df_coll.append(df)
df = pd.concat(df_coll)
df.to_pickle('GR_MNIST_gradient_madness.pkl')
print(df)
# ax = plt.gca()
tmp = np.array(df['variable'])
import re
parameters = ['A1','A2','a','b','G','R1','R2']
grad_guys = ['A1_grad','A2_grad','a_grad','b_grad','G_grad','R1_grad','R2_grad']+['Loss/GR']
for param in grad_guys:

    df_gr_sel_param = df[df['variable']==param]['value'].values
    print(param,np.where(np.isnan(df_gr_sel_param))[0][0])
raise ValueError
colors = plt.get_cmap('tab20')(np.linspace(0,1,len(parameters)))
for g_idx,grad_guy in enumerate(grad_guys):
    print(grad_guy)
    sel = [grad_guy in tmp[i] for i in range(len(tmp))]
    df_grad = df[sel]
    df_grad_sel = df_grad[df_grad['step']>0]
    # df_grad['step'] = df_grad['step'].astype(int)
    sns.lineplot(data=df_grad_sel,x='step',y='value',hue='variable',color=colors[g_idx])
# plt.show()
plt.figure()
sel = ['GR' in tmp[i] for i in range(len(tmp))]
df_gr = df[sel]
df_gr_sel = df_gr[df_gr['step']>0]




# df_gr['step'] = df_gr['step'].astype(int)
sns.lineplot(x='step',y='value',hue='variable',data=df_gr_sel)
# plt.ylim([0,1e-5])
# plt.xlim([0,120])
plt.figure()
sel = ['test' in tmp[i] for i in range(len(tmp))]
df_gr = df[sel]
# df_gr_sel = df_gr[df_gr['step']<=80]
# df_gr['step'] = df_gr['step'].astype(int)
sns.lineplot(x='step',y='value',hue='variable',data=df_gr)
# plt.xlim([0,80])
plt.figure()
sel = ['spk' in tmp[i] for i in range(len(tmp))]
df_gr = df[sel]
# df_gr_sel = df_gr[df_gr['step']<=80]
# df_gr['step'] = df_gr['step'].astype(int)
sns.lineplot(x='step',y='value',hue='variable',data=df_gr)
plt.show()
print('ciao')


fig1, ax1 = plt.subplots(len(df['variable'].unique()),1,sharex=True,figsize=(8,8))
colors = plt.get_cmap('tab20')(np.linspace(0,1,len(df['variable'].unique())))
breakpoint_epoch = 231
dict = {}
for v_idx,variable in enumerate(df['variable'].unique()):
    sel = df[df['variable']==variable]
    # sel = sel[sel['step']<=breakpoint_epoch]
    ax1[v_idx].plot(sel['step'],sel['value'],label=variable,color=colors[v_idx])
    # dict[variable] = sel[sel['step']==breakpoint_epoch]['value'].values[0]
# print(dict)
# #save dict in json
# with open('GR_MNIST_gradient_madness.json','w') as f:
#     json.dump(dict,f)
    # ax1[v_idx].set_title(variable)
# sel = df[df['variable']=='a']
fig1.tight_layout()
# plt.plot(sel['step'],sel['value'])
plt.show()

