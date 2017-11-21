## import data
import scipy.io

test = scipy.io.loadmat('ng207528x8165itst.mat')
test_data = test['Dtst'].tolist()[0][0][1]
test_label_matrix = test['Dtst'].tolist()[0][0][2]
train = scipy.io.loadmat('ng2011293x8165itrn.mat')
train_data = train['Dtrn'].tolist()[0][0][1]
train_label_matrix = train['Dtrn'].tolist()[0][0][2]

## transform label matrix to vectors
import numpy as np

sample = {}
subset = []
label = []
for i in range(20):
    sample_i = np.random.choice(np.where(train_label_matrix.todense()[:, i] != 0)[0], 100, replace=False)
    sample[str(i)] = sample_i
    subset = np.hstack((subset, sample_i))
    label = np.hstack((label, np.repeat(i, 100)))
train_select = train_data[subset,]


# random sample SVM. Data matrix: X, n * p, n = 2000, 100 sample in each catogory
# n: sample size in each category; k: number of class
def random_sample(n, k, m):
    sample = []
    for i in range(k):
        sample_i = np.random.choice(range(i * n, (i + 1) * n), size=m, replace=False)
        sample = np.hstack((sample, sample_i))
    sample = np.array(sample, dtype=int)
    return (sample)


## SVD + SVM
label_num = 20
from scipy.sparse import csc_matrix
from sklearn import svm

A = csc_matrix(train_select, dtype=float)
TOPIC_NUM = [5, 10, 20, 25, 30, 40, 50]
for topic_num in TOPIC_NUM:
    [u, s, vt] = scipy.sparse.linalg.svds(A, topic_num)
    X = np.dot(u, np.diag(s))
    S = 0
    for i in range(20):
        labeled_data = random_sample(100, 20, label_num)
        clf = svm.SVC(kernel='linear')
        clf.fit(X[labeled_data,], label[labeled_data])
        S += sum(clf.predict(X) == label) / 2000.
    print([label_num, topic_num, round(S / 20., 4)])

label_num = 15
from sklearn.decomposition import LatentDirichletAllocation as LDA

TOPIC_NUM = [5, 10, 20, 25, 30, 40, 50]
for topic_num in TOPIC_NUM:
    solution = LDA(n_components=topic_num, learning_method='batch', max_iter=60).fit_transform(train_select)
    S = 0
    for i in range(20):
        labeled_data = random_sample(100, 20, label_num)
        clf = svm.SVC(kernel='linear')
        clf.fit(X[labeled_data,], label[labeled_data])
        accuracy = sum(clf.predict(X) == label) / 2000.
        S += accuracy
    print([label_num, topic_num, round(S / 20., 4)])

## use words directly
LABEL_NUM = [5,10,15,20]
ACCURACY_WORD = []
for label_num in LABEL_NUM:
    S = 0
    for i in range(20):
        labeled_data = random_sample(100, 20, label_num)
        clf = svm.SVC(kernel='linear')
        clf.fit(train_select[labeled_data,], label[labeled_data])
        S += sum(clf.predict(train_select) == label) / 2000.
    S = S / 20.
    ACCURACY_WORD.append(round(S * 100, 2))
print(ACCURACY_WORD)
# 26.17, 42.33, 50.29, 57.59
# better than all the methods @_@