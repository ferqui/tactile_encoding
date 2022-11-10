from tbparse import SummaryReader
import os
import pandas as pd
import matplotlib.pyplot as plt

basedir = os.getcwd()

file = '/home/p301974/Projects/telluride_2022_async/tactile_encoding_NMF/runs/Nov10_11-25-19_fwn-nb4-129-125-35-231'
reader = SummaryReader(file)
df = reader.images
image = df.loc[1, 'value']
plt.imshow(image)
plt.show()
print('ciao')