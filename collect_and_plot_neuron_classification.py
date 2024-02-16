import glob
import pickle as pkl
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
folder = 'activity_*.pkl'
files = glob.glob(folder)
print(files)
dfs = []
for file in files:
    with open(file, 'rb') as f:
        dict_tmp = pkl.load(f)
        df = pd.DataFrame(dict_tmp)
        dfs.append(df)
df = pd.concat(dfs)
df = df.sort_values(by=['Probabilities'],ascending=False)
plt.figure(figsize=(20,4))

g = sn.barplot(x='Behaviour', y='Probabilities',hue = 'dataset',data=df,palette='Set2')#,order=activity_new.sort_values(by=['Probabilities'], ascending=False).set_index('Behaviour').index)
g.set_xticklabels(g.get_xticklabels(), rotation = 45,horizontalalignment='right')
# g.get_legend().remove()

plt.title('Behaviours vs Datasets')
plt.tight_layout()
plt.savefig('Probabilities_Behaviours_vs_dataset',transparent= False)