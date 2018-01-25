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
weight = 1

def update_theta(theta0, starting, W, C, lamba, u, rho, it):
    # W: a dict representing the category
    # C: a list representing the category
    # processing W and C

    theta = starting
    adagrad = np.ones(theta.shape) * 500
    gamma = 0.2
    best_theta = np.zeros(theta.shape)
    best_obj = 0

    global sigma
    sigma = 1. / theta0.shape[1]
    #sigma = .1
    iter_num = 10 if it == 0 else 10
    for _ in range(iter_num):
        ite = np.where(C!=-1)[0]
        random.shuffle(ite)
        grad = np.zeros(theta.shape)
        y_last = np.zeros(theta.shape)
        for i in ite:
            grad[i] = - gradient(i, C[i], theta, theta0, W, C, lamba, u, rho, it)
            theta[i, :] += np.divide(grad[i], adagrad[i])
            adagrad[i] = np.sqrt(adagrad[i]**2 + grad[i]**2)
        '''
        y = theta + np.divide(grad, adagrad)
        theta = (1 - gamma) * y + gamma * y_last
        y_last = y
        theta += np.divide(grad, adagrad)
        adagrad = np.sqrt(adagrad**2 + grad**2)
        '''
        obj, g = objective(theta, theta0, W, C, lamba, u, rho, it)
        if obj > best_obj:
            best_theta = theta
            best_obj = obj
    return (theta, obj, g)


def objective(theta, theta0, W, C, lamda, u, rho, it):
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
    translation = theta - np.mean(theta, axis=1)[:, np.newaxis] 
    for x in range(len(C)):
        if labeled[x]:
            denom[x] = np.sum(np.exp(-np.linalg.norm(translation[x] - translation[labeled], axis=1)**2 * sigma)) - 1 # excluding the exponential of distance to self
    for cat_list in W.values():
        for x in cat_list:
            total += np.sum(np.linalg.norm(translation[x] - translation[cat_list], axis=1)**2 * sigma)
            total += np.log(denom[x]) * (len(cat_list) - 1)
    total *= weight
    total += np.linalg.norm(translation[labeled])**2
    print total
    g = total
    if it:
        total += rho / 2 * np.linalg.norm(theta0[labeled] - theta[labeled] + u[labeled])**2
    return total, g

def gradient(i, ci, theta, theta0, W, C, lamda, u, rho, it):
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
    translation = theta - np.mean(theta, axis=1)[:, np.newaxis] 
    TOPIC_NUM = theta.shape[1]
    A = (np.diag(np.ones(TOPIC_NUM)) * (TOPIC_NUM + 1) - np.ones((TOPIC_NUM, TOPIC_NUM))) / TOPIC_NUM
    A = np.dot(A, A)
    p_ij_denom = np.sum(np.exp(-np.linalg.norm(translation[i] - translation[labeled], axis=1)**2 * sigma)) - 1
    same_label = W[ci]

    def dot_gradient(ind, lst):
        return np.dot(theta[ind] - theta[lst], A)

    for x in range(len(C)):
        if labeled[x] and x != i:
            dist = np.exp(-np.linalg.norm(translation[i] - translation[x])**2 * sigma)
            p_ij[x] = dist / p_ij_denom
            p_ji[x] = dist / (np.sum(np.exp(-np.linalg.norm(translation[x] - translation[labeled], axis=1)**2 * sigma)) - 1)
    grad = np.zeros(len(theta[0]))
    grad += 2 * np.dot(2 - p_ji[same_label], dot_gradient(i, same_label))
    grad -= 2 * (len(same_label) - 1) * np.dot(p_ij[labeled], dot_gradient(i, labeled))
    for key in W:
        same_label_j = W[key]
        count = len(same_label_j) - 1 if key != ci else len(same_label_j) - 2
        grad -= 2 * count * np.dot(p_ji[same_label_j], dot_gradient(i, same_label_j))
    grad *= weight
    grad += 2 * lamda * np.dot(theta[i], A)
    #grad += 2 * lamda * theta[i]
    if it:
        grad += rho * (theta[i] - theta0[i] - u[i])
    return grad

if __name__ == '__main__':
    theta0 = np.random.rand(2000, 20)
    theta = np.random.rand(2000, 20)
    X = np.zeros((2000, 20))
    Y = np.zeros((2000, 20))
    #W = {1:list(range(40)), 2:list(range(40,80))}
    W = {}
    C = np.full(2000,-1)
    '''
    for i in range(40):
        C[i] = 1
    for i in range(40,80):
        C[i] = 2
    for i in range(80,100):
        C[i] = -1
    '''
    c_num = 50
    for i in range(4):
        C[i*c_num:(i+1)*c_num] = i
        W[i] = range(i*c_num, (i+1)*c_num)
        X[i*c_num:(i+1)*c_num, i*4:(i+1)*4] = 1
    lamba = .5
    u = np.random.rand(2000, 20)
    rho = 0
    print objective(X, X, W, C, lamba, u, rho)
    print objective(Y, Y, W, C, lamba, u, rho)
    G = gradient(0,1,theta, theta0, W,C, lamba, u, rho)
    theta, obj = update_theta(theta, theta0, W, C,  lamba, u, rho, 0)
    prob_theta = np.exp(theta)
    prob_theta /= np.sum(prob_theta, axis=1)[:, np.newaxis]
    pdb.set_trace()
