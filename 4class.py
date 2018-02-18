import scipy
import scipy.io
import numpy as np
import pdb
train = scipy.io.loadmat('./20 news/ng2011293x8165itrn.mat')
test = scipy.io.loadmat('./20 news/ng207528x8165itst.mat')
train_data = train['Dtrn'].tolist()[0][0][1]
test_data = test['Dtst'].tolist()[0][0][1]
train_label_matrix = train['Dtrn'].tolist()[0][0][2]
test_label_matrix = test['Dtst'].tolist()[0][0][2]

np.random.seed(309)

train_data = scipy.sparse.vstack([train_data, test_data], format='csr')
train_label_matrix = scipy.sparse.vstack([train_label_matrix, test_label_matrix], format='csr')

sample = {}
subset = []
label = []
#sample_size1 = 100
class_num = 20
q_z = np.load('init.dat.20')

'''
label2ind = dict()
label2ind[0] = 1
for x in range(1, 6):
    label2ind[x] = 2
label2ind[6] = 3
'''

#for i in [0, 15, 19]:
#for i in [1, 9, 12, 17]:
for i in range(20):
    #sample_i = np.random.choice(np.where(train_label_matrix.todense()[:, i] != 0)[0], sample_size1, replace=False)
    sample_i = np.where(train_label_matrix.todense()[:, i] != 0)[0]
    sample[str(i)] = sample_i
    subset = np.hstack((subset, sample_i))
    #label = np.hstack((label, np.repeat(i, sample_size1)))
    label = np.hstack((label, np.repeat(i, len(sample_i))))
subset = subset.astype(np.int32)
train_select = train_data[subset, ]
q_z_select = q_z[subset, ]

'''
label = np.repeat(0, class_num * sample_size1)
for x in range(class_num):
    label[x*sample_size1:(x+1)*sample_size1] = np.repeat(x, sample_size1)
'''

# random sample SVM. Data matrix: X, n * p, n = 2000, 100 sample in each catogory
# n: sample size in each category; k: number of class
def random_sample(class_num, label_num):
    sample = []
    CC = {}
    '''
    for i in range(k):
        sample_i = np.random.choice(range(i * n, (i + 1) * n), size=m, replace=False)
        CC[i] = sample_i
        sample = np.hstack((sample, sample_i))
    '''
    for i in range(class_num):
        instances = np.where(train_label_matrix.todense()[:, i] != 0)[0]
        sample_i = np.random.choice(instances, int(len(instances) * label_num), replace=False)
        CC[i] = sample_i
        sample = np.hstack((sample, sample_i))
    sample = np.array(sample, dtype=int)
    return sample, CC


# labeled data = 5, topic_num = 20
from snlda import admm, DTM
from sklearn import svm
label_num = 0.2 # labeled data in each category
rho = 10
iter_num = 10 # maximal iter for admm
X = train_select
label1 = label

Accuracy_l, Accuracy_u = [], []
for ite in range(1):
    labeled_info = random_sample(class_num, label_num)
    sample, CC = labeled_info  # labeled data index
    TOPIC_NUM = 20  # need to set it in snlda.py
    # C = np.hstack((np.repeat(0, 10), np.repeat(1, 10)))
    C = np.repeat(-1, train_data.shape[0])
    CCC = dict()
    clf_train_data = []
    for i in CC.keys():
        ll = len(CC[i])
        C[CC[i][:ll/2]] = i
        CCC[i] = CC[i][:ll/2]
        clf_train_data += CC[i][ll/2:].tolist()
    # print(C)
    # print(CC)
    #admm_fit = admm(X, CC, TOPIC_NUM, C, rho, iter_num, q_z_select)
    admm_fit = admm(X, CCC, TOPIC_NUM, C, rho, iter_num, q_z_select)
    #admm_fit = DTM(X, CCC, TOPIC_NUM, iter_num)
    labeled_data = sample
    clf = svm.SVC(kernel='linear')
    clf.fit(admm_fit[labeled_data, :], label1[labeled_data])
    #clf.fit(admm_fit[clf_train_data, :], label1[clf_train_data])
    predict = clf.predict(admm_fit)
    accuracy = sum(predict == label1) / float(X.shape[0])
    accuracy_labeled = sum(predict[labeled_data] == label1[labeled_data]) / float(X.shape[0] * label_num)
    accuracy_unlabeled = sum(predict[-labeled_data] == label1[-labeled_data]) / float(X.shape[0] * (1-label_num))
    print 'Prediction accuracy on labeled part for iteration', ite, ': ', round(accuracy_labeled * 100, 2)
    print 'Prediction accuracy on unlabeled part for iteration', ite, ': ', round(accuracy_unlabeled * 100, 2)
    Accuracy_l.append(accuracy_labeled)
    Accuracy_u.append(accuracy_unlabeled)
print 'Accuracy_labeled:', Accuracy_l, '\n mean:', np.mean(Accuracy_l), \
    '\n Accuracy_unlabeled:', Accuracy_u, '\n mean:', np.mean(Accuracy_u)
