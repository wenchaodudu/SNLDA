import numpy as np
import pdb
from scipy.special import digamma

iter_num = 10

np.seterr(over='raise')

def exp_proportion(theta):
    exp_theta = np.exp(theta)
    return exp_theta / np.sum(exp_theta)

def update_variables(X, theta_1, theta_2, q_z, beta, labels, lbda, rho, u, it):
    DOC_NUM, TOPIC_NUM = X.shape
    labeled = [x for x in range(DOC_NUM) if labels[x] != -1]
    unlabeled = [x for x in range(DOC_NUM) if labels[x] == -1]
    for itt in range(iter_num):
        print "Updating topics; iter:", itt
        new_beta = np.ones(beta.shape)
        for x in labeled + unlabeled:
            # update theta
            #Ez = np.sum(q_z[x][w] * X[x, w] for w in q_z[x])
            Ez = np.sum(tp[1] * tp[2] for tp in q_z[x])
            pi = exp_proportion(theta_1[x])
            if labels[x] != -1 and it:
                theta_1[x] = (Ez - np.sum(Ez) * pi) / rho + theta_2[x] - u[x]
            else:
                theta_1[x] = (Ez - np.sum(Ez) * pi) * lbda

            # update q_z
            eta_hessian = np.asmatrix(pi).T * np.asmatrix(pi) - np.diag(pi)
            if labels[x] != -1 and it:
                Sigma = eta_hessian * np.sum(Ez) - rho
            else:
                Sigma = eta_hessian * np.sum(Ez) - 1 / lbda
            Eq = theta_1[x] - np.log(np.sum(np.exp(theta_1[x]))) + 1 / 2 * np.trace(eta_hessian * Sigma)
            for tp in q_z[x]:
                qz = Eq + np.log(beta[:, tp[0]])
                qz = np.exp(qz)
                #q_z[x][w] = qz / np.sum(qz)
                tp[2] = qz / np.sum(qz)
                #new_beta[:, w] += q_z[x][w] * X[x, w]
                new_beta[:, tp[0]] += tp[1] * tp[2]

        # update beta
        new_beta /= np.sum(new_beta, axis=1)[:, np.newaxis]
        print np.linalg.norm(new_beta - beta)
        beta = new_beta
    return theta_1, q_z, beta

