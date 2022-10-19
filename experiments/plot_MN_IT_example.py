import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from matplotlib import gridspec
#sns.set_style('whitegrid')

with open('../results/controlled_stim/Loss.pickle', 'rb') as f:
    loss = pickle.load(f)

with open('../results/controlled_stim/MI.pickle', 'rb') as f:
    MI = pickle.load(f)

with open('../results/controlled_stim/H_matrix.pickle', 'rb') as f:
    H_matrix = pickle.load(f)

with open('../results/controlled_stim/V_matrix.pickle', 'rb') as f:
    V_matrix = pickle.load(f)

with open('../results/controlled_stim/W.pickle', 'rb') as f:
    W = pickle.load(f)

with open('../results/controlled_stim/Out_nmf.pickle', 'rb') as f:
    out_nmf = pickle.load(f)

with open('../results/controlled_stim/Out_classifier.pickle', 'rb') as f:
    out_classifier = pickle.load(f)

with open('../results/controlled_stim/w_classifier.pickle', 'rb') as f:
    w_classifier = pickle.load(f)

with open('../results/controlled_stim/pdf_x1x2.pickle', 'rb') as f:
    pdf_x1x2 = pickle.load(f)


fig = plt.figure()
plt.plot(loss)
plt.title('Loss')
plt.xlabel('Training Epochs')

fig = plt.figure()
plt.plot(MI)
plt.title('MI')
plt.xlabel('Test Epochs')

fig, axs = plt.subplots(1,2)
axs[0].imshow(H_matrix, aspect='auto',interpolation='nearest')
axs[0].set_title('H NMF')
axs[0].set_xlabel('Rank')
axs[0].set_ylabel('TrialxVariable')

axs[1].imshow(W, aspect='auto',interpolation='nearest')
axs[1].set_title('W')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Rank')

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
axs[0].imshow(V_matrix, aspect='auto',interpolation='nearest')
axs[1].imshow(out_nmf, aspect='auto',interpolation='nearest')
axs[0].set_xlabel('Time')
axs[1].set_xlabel('Time')
axs[0].set_title('Input NMF')
axs[1].set_title('Output NMF')
axs[0].set_ylabel('TrialxVariable')


fig = plt.figure()
plt.imshow(out_classifier, aspect='auto',interpolation='nearest')
plt.title('Output classifier')

fig = plt.figure()
plt.imshow(w_classifier, aspect='auto',interpolation='nearest')
plt.title('w classifier')

fig = plt.figure()
plt.imshow(pdf_x1x2, aspect='auto',interpolation='nearest')
plt.title('pdf_x1x2')

plt.show()
