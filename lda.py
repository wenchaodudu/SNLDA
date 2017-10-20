import numpy as np
from scipy.special import digamma

iter_num = 20

def exp_proportion(theta):
    exp_theta = np.exp(theta)
    return exp_theta / np.sum(exp_theta)

def update_variables(X, theta_1, theta_2, q_z, beta, rho, u, it):
    DOC_NUM, TOPIC_NUM = X.shape
    for itt in range(iter_num):
        for x in range(DOC_NUM):
            # update theta
            Ez = sum((digamma(q_z[x][w]) - digamma(np.sum(q_z[x][w]))) * X[x,w] for w in q_z[x])
            pi = exp_proportion(theta_1[x])
            theta_1[x] = (Ez - np.sum(Ez) * pi) / rho + theta_2[x] - u

            # update q_z
            eta_hessian = np.asmatrix(pi).T * np.asmatrix(pi) - np.diag(pi)
            Sigma = eta_hessian * np.sum(Ez) - rho
            Eq = theta_1[x] - np.log(np.sum(np.exp(theta_1[x]))) + 1 / 2 * np.trace(eta_hessian * Sigma)
