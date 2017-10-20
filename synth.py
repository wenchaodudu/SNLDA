import numpy as np
import sys
import pdb
from scipy.sparse import lil_matrix, csr_matrix

cluster_num = 4
topic_num = 20
doc_num = 500
vocab_size = 10000
centers = np.zeros((cluster_num, topic_num))
X = lil_matrix((doc_num, vocab_size))
beta = np.random.dirichlet(np.full(vocab_size, 2), topic_num)

for x in range(cluster_num):
    centers[x] = np.random.uniform(low=-1, high=1, size=topic_num)

for x in range(doc_num):
    print x
    doc_len = np.random.poisson(500)
    cluster = np.random.choice(cluster_num, 1)[0]
    topic_dist = np.random.normal(loc=centers[cluster], scale=1)
    topic_dist = np.exp(topic_dist)
    topic_dist /= np.sum(topic_dist)
    for y in range(doc_len):
        topic = np.nonzero(np.random.multinomial(1, topic_dist))[0][0]
        word = np.nonzero(np.random.multinomial(1, beta[topic]))[0][0]
        X[x, word] += 1

np.save('synthetic_data', csr_matrix(X))
        
