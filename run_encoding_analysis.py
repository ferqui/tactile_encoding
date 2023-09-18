import os
import numpy as np

if __name__ == "__main__":
    list_gain = np.linspace(0.1,1, 10)
    list_sigma = np.linspace(0,0.5, 5)
    for sigma in list_sigma:
        for gain in list_gain:
            print('Run with')
            print('gain:', gain)
            print('sigma:', sigma)
            os.system('python3 encoding_analysis.py --gain={} --sigma={}'.format(gain, sigma))