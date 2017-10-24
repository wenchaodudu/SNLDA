import numpy as np
# from snlda import DOC_NUM, TOPIC_NUM
import random
import pdb

np.seterr(invalid='raise', divide='raise')

# update_theta(theta_1, theta_2, W, 1 / (2 * prior_std**2), u, rho, it)
'''
    theta: D * TOPIC_NUM, document topic vectors
    theta_2: D * TOPIC_NUM, the starting point for optimization
    # W: D * D adjacency matrix, 1 if in same category, -1 if in different category, 0 if unknown
    lambda: scalar, parameter of Gaussian prior
    u: D * TOPIC_NUM, penalty term of admm, 
    rho: scalar, penalty coeffcient of admm
    C: a list representing the category of each data point. e.g.: ['1','2',None,'1']

    return: updated theta
'''

def update_theta(theta0, starting, W, C, lamba, u, rho, step=.001):
    # W: a dict representing the category
    # C: a list representing the category
    # processing W and C

    theta = starting
    for _ in range(10):
        theta_old = theta.copy()
        ite = np.where(C!=-1)[0]
        random.shuffle(ite)
        for i in ite:
            dir = - gradient(i, C[i], theta, theta0, W, lamba, u, rho)
            theta[i, :] += dir * step
            print(objective(theta, theta0, W,C, lamba, u, rho))
            pdb.set_trace()
        if np.linalg.norm(theta_old - theta) < 0.01 or \
                        np.abs(objective(theta, theta0, W, C, lamba, u, rho) - objective(theta_old, theta0, W, C, lamba,
                                                                                         u, rho)) < 1:
            # print(np.linalg.norm(theta_old - theta) < 0.01)
            break
    return (theta, objective(theta, theta0, W,C, lamba, u, rho))


def objective(theta, theta0, W, C, lamda, u, rho):
    # W: a dict representing the category
    # C: a list representation
    D = len(C)
    y0 = np.sum(np.sum(np.linalg.norm(theta[i, :] - theta[j, :]) ** 2 for i in l for j in l) for l in W.values())
    def S(i):
        category = set(W.keys()) - set([C[i]])
        s = np.sum(np.sum(np.exp(-np.linalg.norm(theta[i, :] - theta[j, :]) ** 2) for j in W[l]) for l in category)
        np.log(s)
        return (np.log(s))

    y1 = np.sum(S(i) for i in range(D))
    y2 = lamda * np.linalg.norm(theta) + rho / 2 * np.linalg.norm(theta0 - theta + u)
    return (y0 + y1 + y2)


def gradient(i, ci, theta, theta0, W, lamda, u, rho):
    # W: a dict representing the category
    # ci: the category of i-th point
    category = set(W.keys()) - set([ci])
    M = {}
    for d in category:
        M[d] = np.mean(theta[W[d]], axis=0)
    g0 = 4 * np.sum((theta[i, :] - theta[j, :]) for j in W[ci])
    # print(M)
    S = np.sum(len(W[l]) * np.exp(-np.linalg.norm(theta[i, :] - M[l])**2) for l in M.keys())
    g1 = np.sum(len(W[l]) * np.exp(-np.linalg.norm(theta[i, :] - M[l])**2) * 2 * (theta[i, :] - M[l]) for l in M.keys()) / S
    g2 = lamda * 2 * theta[i, :] + rho * (theta[i, :] - theta0[i, :] - u[i, :])
    return (g0 + g1 + g2)


if __name__ == '__main__':
    # update_theta(theta0, W, lamba, u, rho)
    theta0 = np.random.rand(100, 20)
    theta = np.random.rand(100, 20)
    W = {1:list(range(40)), 2:list(range(40,80))}
    C = np.full(100,0)
    for i in range(40):
        C[i] = 1
    for i in range(40,80):
        C[i] = 2
    for i in range(80,100):
        C[i] = -1
    lamba = 100
    u = np.random.rand(100, 20)
    rho = 0.01
    G = gradient(0,1,theta, theta0, W, lamba, u, rho)
    N = update_theta(theta0, theta0, C,W,  lamba, u, rho)
