import numpy as np
import itertools
import pickle
from sne import update_theta
from lda import update_variables, lda_likelihood
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS 
from sklearn.decomposition import LatentDirichletAllocation as LDA, PCA
import pdb
TOPIC_NUM = 20
# DOC_NUM = 0
# VOCAB_SIZE = 0
prior_std = np.sqrt(2) / 2

def admm(X, W, C, rho, iter_num):
    # initialize variables
    DOC_NUM, VOCAB_SIZE = X.shape
    theta_1 = np.random.normal(scale=prior_std, size=(DOC_NUM, TOPIC_NUM))
    theta_2 = np.copy(theta_1)
    q_z = [dict() for x in range(DOC_NUM)]
    dirichlet_prior = np.full(TOPIC_NUM, 2)
    print 'Initialing q:'
    for x in range(DOC_NUM):
        # print x
        for y in range(VOCAB_SIZE):
            if X[x, y]:
                q_z[x][y] = np.random.dirichlet(dirichlet_prior, 1)[0]
    print 'Initialing beta:'
    beta = np.random.dirichlet(np.full(VOCAB_SIZE, 2), TOPIC_NUM)
    u = np.zeros((DOC_NUM, TOPIC_NUM))
    print 'Finish initialization.'
    DELTA_THETA, OBJECTIVE_G, OBJECTIVE_NF, THETA_NORM = [], [], [], []
    for it in range(iter_num):
        theta_1, q_z, beta = update_variables(X, theta_1, theta_2, q_z, beta, rho, u, it)
        theta_2, obj, obj_g = update_theta(theta_1, theta_2, W, C, 1 / (2 * prior_std**2), u, rho, it)
        u += 0.1 * (theta_1 - theta_2)
        print '||theta_1-theta_2||:', np.linalg.norm(theta_1 - theta_2)
        print 'Objective g(x):', obj_g
        # very slow
        # ll = lda_likelihood(theta_1, X, beta.transpose())
        # print 'Objective -f(x):', -ll
        # OBJECTIVE_NF.append(-ll)
        print '||theta||:', np.linalg.norm((theta_1 + theta_2) / 2)
        DELTA_THETA.append(np.linalg.norm(theta_1 - theta_2))
        OBJECTIVE_G.append(obj_g)
        THETA_NORM.append(np.linalg.norm((theta_1 + theta_2) / 2))

        '''
        if it % 5 == 4:
            solution = (theta_1 + theta_2) / 2
            low_dim = TSNE().fit_transform(solution)
            plt.scatter(low_dim[:, 0], low_dim[:, 1], c=colors)
            plt.show()
        '''
    print 'DELTA_THETA:', DELTA_THETA, '\n g(x):', OBJECTIVE_G, '\n theta_norm', THETA_NORM
        # '\n -f(x):', OBJECTIVE_NF
    return (theta_1 + theta_2) / 2

if __name__ == "__main__":
    synth = pickle.load(open('synthetic_data'))
    data = synth['data']
    clusters = synth['clusters']
    C = dict()
    CC = dict()
    labels = np.asarray([int(clusters[x]) if np.random.uniform() < 0.2 else -1 for x in range(data.shape[0])])
    categories = set(clusters)
    for c in categories:
        if c is not None:
            C[c] = np.where(clusters==c)[0]
            CC[c] = np.where(labels==c)[0]
    TOPIC_NUM = 20
    DOC_NUM, VOCAB_SIZE = data.shape
    color = {0: 'g', 1: 'b', 2: 'r', 3: 'y'}
    colors = [color[x] for x in clusters]
    # solution = admm(data, CC, labels, 10, 5)
    # solution = LDA(n_components=20, learning_method='batch', max_iter=50, n_jobs=2).fit_transform(data)
    # low_dim = PCA(n_components=2).fit_transform(solution)
    # plt.scatter(low_dim[:, 0], low_dim[:, 1], c=colors)
    # plt.show()




