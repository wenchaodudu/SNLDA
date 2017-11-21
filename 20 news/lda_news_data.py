## import data
import scipy.io
import numpy as np
from lda import update_variables, lda_likelihood

# test = scipy.io.loadmat('ng207528x8165itst.mat')
# test_data = test['Dtst'].tolist()[0][0][1]
# test_label_matrix = test['Dtst'].tolist()[0][0][2]
train = scipy.io.loadmat('ng2011293x8165itrn.mat')
train_data = train['Dtrn'].tolist()[0][0][1]
train_label_matrix = train['Dtrn'].tolist()[0][0][2]

sample = {}
subset = []
label = []
for i in range(20):
    sample_i = np.random.choice(np.where(train_label_matrix.todense()[:, i] != 0)[0].tolist()[0], 100, replace=False)
    sample[str(i)] = sample_i
    subset = np.hstack((subset, sample_i))
    label = np.hstack((label, np.repeat(i, 100)))
train_select = train_data[subset,]

# random sample SVM. Data matrix: X, n * p, n = 2000, 100 sample in each catogory
# n: sample size in each category; k: number of class
def random_sample(n, k, m):
    sample = []
    CC = {}
    for i in range(k):
        sample_i = np.random.choice(range(i * n, (i + 1) * n), size=m, replace=False)
        CC[i] = sample_i
        sample = np.hstack((sample, sample_i))
    sample = np.array(sample, dtype=int)
    return sample, CC

# data and parameter
from snlda import admm
# all data: train_select and label
sample_size1 = 100
class_num = 20
label_num = 10 # labeled data in each category
rho = 10.0
iter_num = 20 # maximal iter for admm
X = train_select

'''
# select 20 documents
sample_size1 = 10
class_num = 2
label_num = 5 # labeled data in each category
rho = 10.0
iter_num = 20 # maximal iter for admm
X = train_select[90:110,:]
'''

# ADMM
labeled_info = random_sample(sample_size1, class_num, label_num)
sample = labeled_info[0] # labeled data index
TOPIC_NUM = 20 # need to set it in snlda.py
# C = np.hstack((np.repeat(0, 10), np.repeat(1, 10)))
CC = labeled_info[1]
C = np.repeat(-1, class_num * sample_size1)
for i in CC.keys():
    C[CC[i]] = i
# print(C)
# print(CC)
admm_fit = admm(X, CC, C, rho, iter_num)

# linear svm prediction on training data
from sklearn import svm
labeled_data = labeled_info
clf = svm.SVC(kernel='linear')
clf.fit(X[labeled_data,], label[labeled_data])
accuracy = sum(clf.predict(X) == label) / float(class_num * sample_size1)
print 'Prediction accuracy:', round(accuracy, 4)
