import numpy as np
import itertools
import pickle
from sne import update_theta
from lda import update_variables
import matplotlib.pyplot as plt
import pdb

TOPIC_NUM = 0
DOC_NUM = 0
VOCAB_SIZE = 0
prior_std = np.sqrt(2) / 2

def admm(X, W, C, rho, iter_num):
    # initialize variables
    theta_1 = np.random.normal(scale=prior_std, size=(DOC_NUM, TOPIC_NUM))
    theta_2 = np.copy(theta_1)
    q_z = [dict() for x in range(DOC_NUM)]
    dirichlet_prior = np.full(TOPIC_NUM, 2)
    '''
    for x in range(DOC_NUM):
        for y in range(VOCAB_SIZE):
            if X[x, y]:
                q_z[x][y] = np.random.dirichlet(dirichlet_prior, 1)[0]
    '''
    beta = np.random.dirichlet(np.full(VOCAB_SIZE, 2), TOPIC_NUM)
    u = np.zeros((DOC_NUM, TOPIC_NUM))

    for it in range(iter_num):
        #theta_1, q_z, beta = update_variables(X, theta_1, theta_2, q_z, beta, rho, u, it)
        theta_2, obj = update_theta(theta_1, theta_2, W, C, 1 / (2 * prior_std**2), u, rho, it)
        u += (theta_1 - theta_2)

    return (theta_1 + theta_2) / 2

if __name__ == "__main__":
    synth = pickle.load(open('synthetic_data'))
    data = synth['data']
    clusters = synth['clusters']
    C = dict()
    labels =  np.asarray([int(clusters[x]) if np.random.uniform() < 1 else -1 for x in range(data.shape[0])])
    categories = set(clusters)
    for c in categories:
        if c is not None:
            C[c] = np.where(clusters==c)[0]
    TOPIC_NUM = 20
    DOC_NUM, VOCAB_SIZE = data.shape
    solution = admm(data, C, labels, 2, 30)
    #U, S, V = np.linalg.svd(solution)
    


