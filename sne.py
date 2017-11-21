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

step_size = .5 * 1e-3
sigma = 0.1

def update_theta(theta0, starting, W, C, lamba, u, rho, it):
    # W: a dict representing the category
    # C: a list representing the category
    # processing W and C

    theta = starting
    adagrad = np.ones(theta.shape) * 100

    global sigma
    sigma = 1. / theta0.shape[1]
    for _ in range(10):
        theta_old = theta.copy()
        ite = np.where(C!=-1)[0]
        random.shuffle(ite)
        for i in ite:
            dir = - gradient(i, C[i], theta, theta0, W, C, lamba, u, rho)
            theta[i, :] += np.divide(dir, adagrad[i])
            adagrad[i] = np.sqrt(adagrad[i]**2 + dir**2)
        '''
        if np.linalg.norm(theta_old - theta) < 0.01 or \
                        np.abs(objective(theta, theta0, W, C, lamba, u, rho) - objective(theta_old, theta0, W, C, lamba,
                                                                                         u, rho)) < 1:
            # print(np.linalg.norm(theta_old - theta) < 0.01)
            break
        '''
        objective(theta, theta0, W, C, lamba, u, rho)
    return (theta, objective(theta, theta0, W,C, lamba, u, rho))


def objective(theta, theta0, W, C, lamda, u, rho):
    # W: a dict representing the category
    # C: a list representation
    '''
    D = len(C)
    y0 = np.sum(np.sum(np.linalg.norm(theta[i, :] - theta[j, :]) ** 2 for i in l for j in l) for l in W.values())
    def S(i):
        category = set(W.keys()) - set([C[i]])
        s = np.sum(np.sum(np.exp(-np.linalg.norm(theta[i, :] - theta[j, :]) ** 2) for j in W[l]) for l in category)
        np.log(s)
        return (np.log(s))

    y1 = np.sum(S(i) for i in range(D))
    y2 = lamda * np.linalg.norm(theta)**2 + rho / 2 * np.linalg.norm(theta0 - theta + u)**2
    return (y0 + y1 + y2)
    '''
    denom = np.zeros(len(C))
    labeled = C != -1
    total = 0
    for x in range(len(C)):
        if labeled[x]:
            denom[x] = np.sum(np.exp(-np.linalg.norm(theta[x] - theta[labeled], axis=1)**2 * sigma)) - 1 # excluding the exponential of distance to self
    for cat_list in W.values():
        for x in cat_list:
            total += np.sum(np.linalg.norm(theta[x] - theta[cat_list], axis=1)**2 * sigma)
            total += np.log(denom[x]) * (len(cat_list) - 1)
    total += np.linalg.norm(theta)**2
    print total
    total += rho / 2 * np.linalg.norm(theta0 - theta + u)**2
    return total
    

def gradient(i, ci, theta, theta0, W, C, lamda, u, rho):
    # W: a dict representing the category
    # C: labels of documents
    # ci: the category of i-th point
    '''
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
    '''
    p_ij = np.zeros(len(C))
    p_ji = np.zeros(len(C))
    labeled = C != -1
    p_ij_denom = np.sum(np.exp(-np.linalg.norm(theta[i] - theta[labeled], axis=1)**2 * sigma)) - 1
    same_label = W[ci]
    for x in range(len(C)):
        if labeled[x] and x != i:
            dist = np.exp(-np.linalg.norm(theta[i] - theta[x])**2 * sigma)
            try:
                p_ij[x] = dist / p_ij_denom
            except:
                pdb.set_trace()
            try:
                p_ji[x] = dist / (np.sum(np.exp(-np.linalg.norm(theta[x] - theta[labeled], axis=1)**2 * sigma)) - 1)
            except FloatingPointError:
                print dist, (np.sum(np.exp(-np.linalg.norm(theta[x] - theta[labeled], axis=1)**2 * sigma)) - 1)
    grad = np.zeros(len(theta[0]))
    grad += 2 * np.sum(np.multiply(theta[i] - theta[same_label], 2 - p_ji[same_label][:, np.newaxis]), axis=0)
    grad -= 2 * (len(same_label) - 1) * np.sum(np.multiply(theta[i] - theta[labeled], p_ij[labeled][:, np.newaxis]), axis=0)
    for key in W:
        same_label_j = W[key]
        count = len(same_label_j) - 1 if key != ci else len(same_label_j) - 2
        grad -= 2 * count * np.sum(np.multiply(theta[i] - theta[same_label_j], p_ji[same_label_j][:, np.newaxis]), axis=0)
    grad += 2 * lamda * theta[i]
    grad += rho * (theta[i] - theta0[i] - u[i])
    return grad


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
    G = gradient(0,1,theta, theta0, W,C, lamba, u, rho)
    N = update_theta(theta0, theta0, W, C,  lamba, u, rho, 0)
