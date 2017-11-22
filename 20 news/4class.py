import scipy.io
import numpy as np
train = scipy.io.loadmat('ng2011293x8165itrn.mat')
train_data = train['Dtrn'].tolist()[0][0][1]
train_label_matrix = train['Dtrn'].tolist()[0][0][2]
sample = {}
subset = []
label = []
for i in [1, 9, 12, 17]:
    sample_i = np.random.choice(np.where(train_label_matrix.todense()[:, i] != 0)[0], 500, replace=False)
    sample[str(i)] = sample_i
    subset = np.hstack((subset, sample_i))
    label = np.hstack((label, np.repeat(i, 500)))
train_select = train_data[subset, ]
