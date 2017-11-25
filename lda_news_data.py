## import data
import scipy.io
import numpy as np
from lda import update_variables
import pdb

# test = scipy.io.loadmat('ng207528x8165itst.mat')
# test_data = test['Dtst'].tolist()[0][0][1]
# test_label_matrix = test['Dtst'].tolist()[0][0][2]
train = scipy.io.loadmat('./20 news/ng2011293x8165itrn.mat')
train_data = train['Dtrn'].tolist()[0][0][1]
train_label_matrix = train['Dtrn'].tolist()[0][0][2]

'''
rand_ind = np.random.choice(train_data.shape[0], size=2000, replace=False)
train_data = train_data[rand_ind]
train_label_matrix = train_label_matrix[rand_ind]
'''

sample = {}
subset = []
label = []
sample_size1 = 100
for i in [1, 9, 12, 17]:
    sample_i = np.random.choice(np.where(train_label_matrix.todense()[:, i] != 0)[0].tolist(), sample_size1, replace=False)
    sample[str(i)] = sample_i
    subset = np.hstack((subset, sample_i))
    label = np.hstack((label, np.repeat(i, sample_size1)))
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
'''
from snlda import admm
# all data: train_select and label
sample_size1 = 100
class_num = 20
label_num = 10 # labeled data in each category
rho = 10.0
iter_num = 20 # maximal iter for admm
X = train_select


# select 20 documents
sample_size1 = 10
class_num = 2
label_num = 5 # labeled data in each category
rho = 10.0
iter_num = 20 # maximal iter for admm
X = train_select[90:110,:]


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
'''

# labeled data = 5, topic_num = 20
from snlda import admm
from sklearn import svm
class_num = 4
label_num = sample_size1 / 5 # labeled data in each category
rho = 1
iter_num = 6 # maximal iter for admm
X = train_select
label1 = label

Accuracy_l, Accuracy_u = [], []
for ite in range(20):
    labeled_info = random_sample(sample_size1, class_num, label_num)
    sample = labeled_info[0]  # labeled data index
    TOPIC_NUM = 20  # need to set it in snlda.py
    # C = np.hstack((np.repeat(0, 10), np.repeat(1, 10)))
    CC = labeled_info[1]
    C = np.repeat(-1, class_num * sample_size1)
    for i in CC.keys():
        C[CC[i]] = i
    # print(C)
    # print(CC)
    admm_fit = admm(X, CC, TOPIC_NUM, C, rho, iter_num)
    labeled_data = sample
    clf = svm.SVC(kernel='linear')
    clf.fit(admm_fit[labeled_data, :], label1[labeled_data])
    predict = clf.predict(admm_fit)
    accuracy = sum(predict == label1) / float(class_num * sample_size1)
    accuracy_labeled = sum(predict[labeled_data] == label1[labeled_data]) / float(class_num * label_num)
    accuracy_unlabeled = sum(predict[-labeled_data] == label1[-labeled_data]) / float(class_num * (sample_size1-label_num))
    print 'Prediction accuracy on labeled part for iteration', ite, ': ', round(accuracy_labeled * 100, 2)
    print 'Prediction accuracy on unlabeled part for iteration', ite, ': ', round(accuracy_unlabeled * 100, 2)
    Accuracy_l.append(accuracy_labeled)
    Accuracy_u.append(accuracy_unlabeled)
print 'Accuracy_labeled:', Accuracy_l, '\n mean:', np.mean(Accuracy_l), \
    '\n Accuracy_unlabeled:', Accuracy_u, '\n mean:', np.mean(Accuracy_u)
