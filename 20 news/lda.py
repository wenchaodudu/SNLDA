import numpy as np
import pdb
from scipy.special import digamma

iter_num = 5

np.seterr(over='raise')

def exp_proportion(theta):
    exp_theta = np.exp(theta)
    return exp_theta / np.sum(exp_theta)

def update_variables(X, theta_1, theta_2, q_z, beta, rho, u, it):
    DOC_NUM, TOPIC_NUM = X.shape
    for itt in range(iter_num):
        print "Updating topics; iter:", itt
        new_beta = np.ones(beta.shape)
        for x in range(DOC_NUM):
            # update theta
            #Ez = sum((digamma(q_z[x][w]) - digamma(np.sum(q_z[x][w]))) * X[x,w] for w in q_z[x])
            Ez = np.sum(exp_proportion(q_z[x][w]) * X[x, w] for w in q_z[x].keys())
            pi = exp_proportion(theta_1[x,])
            theta_1[x,] = (Ez - np.sum(Ez) * pi) / rho + theta_2[x,] - u[x,]

            # update q_z
            eta_hessian = np.asmatrix(pi).T * np.asmatrix(pi) - np.diag(pi)
            Sigma = eta_hessian * np.sum(Ez) - rho
            Eq = theta_1[x,] - np.log(np.sum(np.exp(theta_1[x,]))) + 1 / 2. * np.trace(eta_hessian * Sigma)
            for w in q_z[x]:
                qz = Eq + np.log(beta[:, w])
                qz = np.exp(qz)
                q_z[x][w] = qz / np.sum(qz)
                new_beta[:, w] += q_z[x][w] * X[x, w]

        # update beta
        new_beta /= np.sum(new_beta, axis=1)[:, np.newaxis]
    return theta_1, q_z, beta


def lda_likelihood(theta, X, beta):
    """
    Calculate the likelihood function for word count directly
    :param theta: DOC_NUM * TOPIC_NUM
    :param X: DOC_NUM * VOCAB_SIZE
    :param beta: VOCAB_SIZE * TOPIC_NUM
    :return: log likelihood function
    """
    ll = 0
    DOC_NUM, TOPIC_NUM = theta.shape
    VOCAB_SIZE = beta.shape[0]
    for doc_idx in range(DOC_NUM):
        theta_i = exp_proportion(theta[doc_idx, :])
        for vocab_idx in range(VOCAB_SIZE):
            p = np.dot(theta_i, beta[vocab_idx, :])
            if p != 0:
                ll += X[doc_idx, vocab_idx] * np.log(p)
        # if doc_idx % 10 == 0:
        #     print(doc_idx)
    return ll

