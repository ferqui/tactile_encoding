import os
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

axis_label_size = 18
axis_ticks_size = 14
dimension_label_size = axis_label_size*1
dimension_ticks_size = axis_ticks_size*1
cbar_label_size = 16
cbar_ticks_size = 14
legend_title_size = 'x-large'
legend_entry_size = 14

# init dataloading (load the last trial)
file_storage_found = False
idx_file_storage = 1
while not file_storage_found:
    file_storage_path = './results/experiment_' + str(idx_file_storage) + '.pkl'
    if os.path.isfile(file_storage_path):
        idx_file_storage += 1
    else:
        file_storage_path = './results/experiment_' + str(idx_file_storage-1) + '.pkl'
        file_storage_found = True

# put id here if NOT last should be loaded
# file_storage_path = './results/experiment_' + str(idx_file_storage) + '.pkl'
file_storage_path = './results/experiment_50_100.pkl'

with open(file_storage_path, 'rb') as f:
    data = pickle.load(f)

# seperate best individual and data
best_individual = data[1::2]
data = data[::2]
# select only data from best individual
all_data = []
data_best_individual = []
for counter, individual in enumerate(data):
    all_data = np.append(all_data, individual)
    data_best_individual.append(individual[best_individual[counter]])
    
df_plot = pd.DataFrame.from_records(all_data)
test = df_plot['fitness']
fig = go.Figure(layout=dict(width=2000, height=500, margin=go.layout.Margin(l=40, r=20, b=20, t=50, pad=0)),
                data=go.Parcoords(
    line=dict(color=df_plot['fitness'],
                colorscale=px.colors.diverging.Portland,
                showscale=True,
                cmin=10,
                cmax=90,
                colorbar={"title": dict(text="Accuracy (%)", side="right", font=dict(size=cbar_label_size)),
                          "tickvals": np.arange(10, 90+1, 10),
                          "tickfont": dict(size=cbar_ticks_size)}),
    dimensions=list([
        dict(tickvals=np.arange(np.nanmin(df_plot['a']),
                                np.nanmax(
            df_plot['a'])*1.05,
            15),
            label="a", values=df_plot['a']),
        dict(tickvals=np.arange(np.nanmin(df_plot['A1']),
                                np.nanmax(
            df_plot['A1'])*1.05,
            1),
            label='A1', values=df_plot['A1']),
        dict(tickvals=[1, 2, 4, 8],
             label='A2', values=df_plot['A2']),
        dict(tickvals=np.arange(np.nanmin(df_plot['A2']),
                                np.nanmax(
            df_plot['A2'])*1.05,
            10),
            label='A2', values=df_plot['A2'])#,
        # dict(tickvals=[1, 2, 5, 10],
        #      label='tau_ratio', values=df_plot['parameters.tau_ratio']),
        # dict(tickvals=np.arange(np.nanmin(df_plot['parameters.fwd_weight_scale']),
        #                         np.nanmax(
        #     df_plot['parameters.fwd_weight_scale'])*1.05,
        #     1),
        #     label='fwd_weight_scale', values=df_plot['parameters.fwd_weight_scale']),
        # dict(tickvals=np.arange(np.nanmin(df_plot['parameters.weight_scale_factor']),
        #                         np.nanmax(
        #     df_plot['parameters.weight_scale_factor'])*1.05,
        #     15e-3),
        #     tickformat=".0s",
        #     label='weight_scale_factor', values=df_plot['parameters.weight_scale_factor']),
        # dict(tickvals=np.arange(np.nanmin(df_plot['parameters.reg_spikes']),
        #                         np.nanmax(
        #     df_plot['parameters.reg_spikes'])*1.05,
        #     1e-3),
        #     tickformat=".0s",
        #     label='reg_spikes', values=df_plot['parameters.reg_spikes']),
        # dict(tickvals=np.arange(np.nanmin(df_plot['parameters.reg_neurons']),
        #                         np.nanmax(
        #     df_plot['parameters.reg_neurons'])*1.05,
        #     2e-6),
        #     tickformat=".0s",
        #     label='reg_neurons', values=df_plot['parameters.reg_neurons']),
        # dict(range=[10, 90],
        #      tickvals=np.arange(10, 90+1, 20),
        #      label='Accuracy (%)', values=df_plot['default'])
    ]),
    labelfont=dict(size=dimension_label_size),
    tickfont=dict(size=dimension_ticks_size)
)
)
fig.show()