import scipy.io
import numpy as np
import pdb
import progressbar

train = scipy.io.loadmat('./20 news/ng2011293x8165itrn.mat')
test = scipy.io.loadmat('./20 news/ng207528x8165itst.mat')
train_data = train['Dtrn'].tolist()[0][0][1]
test_data = test['Dtst'].tolist()[0][0][1]
train_data = scipy.sparse.vstack([train_data, test_data], format='csr')

DOC_NUM = train_data.shape[0]
VOCAB_SIZE = train_data.shape[1]
TOPIC_NUM = 20
dirichlet_prior = np.full(TOPIC_NUM, 2)

bar = progressbar.ProgressBar()
q_z = [[] for x in range(DOC_NUM)]
for y in bar(range(VOCAB_SIZE)):
    for x in range(DOC_NUM):
        if train_data[x, y] > 0:
            q_z[x].append([y, train_data[x, y], np.random.dirichlet(dirichlet_prior, 1)[0]])

np.save(open('init.dat.' + str(TOPIC_NUM), 'wb'), np.asarray(q_z))

