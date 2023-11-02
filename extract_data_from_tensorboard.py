from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt
import json
from models import MN_neuron_sp
import numpy as np
ea = event_accumulator.EventAccumulator('runs/Oct31_15-33-06_fwn-nb4-129-125-7-241GR_MNIST/events.out.tfevents.1698762786.fwn-nb4-129-125-7-241.996692.0')
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
print(df)
# ax = plt.gca()
fig1, ax1 = plt.subplots(len(df['variable'].unique()),1,sharex=True,figsize=(8,8))
colors = plt.get_cmap('tab20')(np.linspace(0,1,len(df['variable'].unique())))
breakpoint_epoch = 231
dict = {}
for v_idx,variable in enumerate(df['variable'].unique()):
    sel = df[df['variable']==variable]
    sel = sel[sel['step']<=breakpoint_epoch]
    ax1[v_idx].plot(sel['step'],sel['value'],label=variable,color=colors[v_idx])
    dict[variable] = sel[sel['step']==breakpoint_epoch]['value'].values[0]
print(dict)
#save dict in json
with open('GR_MNIST_opt.json','w') as f:
    json.dump(dict,f)
    # ax1[v_idx].set_title(variable)
# sel = df[df['variable']=='a']
fig1.tight_layout()
# plt.plot(sel['step'],sel['value'])
plt.show()

