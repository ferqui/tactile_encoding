import pickle
from tactile_encoding.parameters.ideal_params import neuron_parameters
classes = list(neuron_parameters.keys())
# load data
data_path = './data/original_mn_output'
filename = 'data_encoding_original'
with open(f"{data_path}/{filename}.pkl", 'rb') as infile:
    data = pickle.load(infile)
nb_repetitions = 10
new_data = []
labels = []
for k, class_data in enumerate(data):
    for i in range(nb_repetitions):
        new_data.append(class_data)
        labels.append(classes[k])
filename = f'data_encoding_original_{nb_repetitions}_repetitions'
with open(f"{data_path}/{filename}.pkl", 'wb') as infile:
    pickle.dump(new_data, infile, protocol=pickle.HIGHEST_PROTOCOL)
filename = f'label_encoding_original_{nb_repetitions}_repetitions'
with open(f"{data_path}/{filename}.pkl", 'wb') as infile:
    pickle.dump(labels, infile, protocol=pickle.HIGHEST_PROTOCOL)