import numpy as np
import itertools
import pickle
from sne import update_theta
from lda import update_variables
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS 
from sklearn.decomposition import LatentDirichletAllocation as LDA, PCA
import progressbar
import pdb

TOPIC_NUM = 0
DOC_NUM = 0
VOCAB_SIZE = 0
prior_std = np.sqrt(2) / 2

def admm(X, W, k, C, rho, iter_num, init=None):
    # initialize variables
    TOPIC_NUM = k
    DOC_NUM, VOCAB_SIZE = X.shape
    theta_1 = np.random.normal(scale=prior_std, size=(DOC_NUM, TOPIC_NUM))
    theta_2 = np.copy(theta_1)
    '''
    theta_1 = np.zeros((DOC_NUM, TOPIC_NUM))
    theta_2 = np.zeros((DOC_NUM, TOPIC_NUM))
    '''
    q_z = [[] for x in range(DOC_NUM)]
    dirichlet_prior = np.full(TOPIC_NUM, 2)
    beta = np.random.dirichlet(np.full(VOCAB_SIZE, 2), TOPIC_NUM)
    u = np.zeros((DOC_NUM, TOPIC_NUM))
    bar = progressbar.ProgressBar()
    if init is None:
        print "Initializing."
        for y in bar(range(VOCAB_SIZE)):
            for x in range(DOC_NUM):
                if X[x, y] > 0:
                    q_z[x].append([y, X[x, y], np.random.dirichlet(dirichlet_prior, 1)[0]])
    else:
        q_z = init

    labeled = C != -1
    for it in range(iter_num):
        theta_1, q_z, beta = update_variables(X, theta_1, theta_2, q_z, beta, C, prior_std**2, rho, u, it)
        theta_2, obj = update_theta(theta_1, theta_2, W, C, 1 / (2 * prior_std**2), u, rho, it)
        u[labeled] += (theta_1[labeled] - theta_2[labeled])
        print np.linalg.norm(theta_1[labeled] - theta_2[labeled])

        '''
        if it % 5 == 4:
            solution = (theta_1 + theta_2) / 2
            low_dim = TSNE().fit_transform(solution)
            plt.scatter(low_dim[:, 0], low_dim[:, 1], c=colors)
            plt.show()
        '''
    def normalize_exp(arr):
        arr -= np.mean(arr, axis=1)[:, np.newaxis]
        arr = np.exp(arr)
        arr /= np.sum(arr, axis=1)[:, np.newaxis]
        return arr
    '''
    solution_1 = normalize_exp(theta_1)
    solution_2 = normalize_exp(theta_2)
    solution_1[labeled] = (solution_1[labeled] + solution_2[labeled]) / 2
    '''
    theta_1[labeled] = (theta_1[labeled] + theta_2[labeled]) / 2
    solution = theta_1 - np.mean(theta_1, axis=1)[:, np.newaxis]
    return theta_1

if __name__ == "__main__":
    synth = pickle.load(open('synthetic_data'))
    data = synth['data']
    clusters = synth['clusters']
    C = dict()
    CC = dict()
    labels =  np.asarray([int(clusters[x]) if np.random.uniform() < 0.2 else -1 for x in range(data.shape[0])])
    categories = set(clusters)
    for c in categories:
        if c is not None:
            C[c] = np.where(clusters==c)[0]
            CC[c] = np.where(labels==c)[0]
    color = {0: 'g', 1: 'b', 2: 'r', 3: 'y', 4: 'm', 5: 'k', 6:'c', 7:'peru', 8:'coral', 9:'gold'}
    colors = [color[x] for x in clusters]
    solution = admm(data, CC, 20, labels, 2, 4)
    #solution = LDA(n_components=20, learning_method='batch', max_iter=50, n_jobs=2).fit_transform(data)
    low_dim = PCA(n_components=2).fit_transform(solution)
    plt.scatter(low_dim[:, 0], low_dim[:, 1], c=colors)
    plt.show()




