import json
import os
import matplotlib.pyplot as plt
import numpy as np
def get_params(path):
    dict = {}
    for file in os.listdir(path):
        if file.endswith('.json'):
            with (open(os.path.join(path, file), 'r') as f):
                data = json.load(f)
                name = file.split('-')[-1].replace('.json','')
                values = np.array([data[i][2] for i in range(len(data))])
                dict[name] = float(values[-1])

                #plt.plot(dict[name]/np.abs(dict[name]).max(),label=name)
    return dict

MN_Braille = os.path.join('params_training','MN_Braille')
MN_MNIST = os.path.join('params_training','MN_MNIST')
MN_params = 'MN_params'
MN_Braille_dict = get_params(MN_Braille)
MN_MNIST_dict = get_params(MN_MNIST)
try:
    os.mkdir(MN_params)
except FileExistsError:
    pass
json.dump(MN_Braille_dict, open(os.path.join(MN_params,'Braille.json'), 'w'))
json.dump(MN_MNIST_dict, open(os.path.join(MN_params,'MNIST.json'), 'w'))