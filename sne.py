import numpy as np
# from snlda import DOC_NUM, TOPIC_NUM
import random
# update_theta(theta_1, theta_2, W, 1 / (2 * prior_std**2), u, rho, it)
'''
    theta: D * TOPIC_NUM, document topic vectors
    W: D * D adjacency matrix, 1 if in same category, -1 if in different category, 0 if unknown
    lambda: scalar, parameter of Gaussian prior
    u: D * TOPIC_NUM, penalty term of admm, 
    rho: scalar, penalty coeffcient of admm

    return: updated theta
'''

def update_theta(starting, theta0, W, lamba, u, rho):
    D = np.shape(W)[0]
    theta = starting
    for _ in range(1000):
        theta_old = theta.copy()
        ite = list(range(D))
        random.shuffle(ite)
        for i in ite:
            dir = gradient(i,theta,theta0,W,lamba,u,rho)
            update_sub(i, theta, dir)
            # theta[i,:] += dir * 0.001
        print(objective(theta,theta0,W,lamba,u,rho),objective(theta_old,theta0,W,lamba,u,rho))
        if np.linalg.norm(theta_old-theta) < 0.01 or \
            np.abs(objective(theta,theta0,W,lamba,u,rho)-objective(theta_old,theta0,W,lamba,u,rho)) < 1:
            # print(np.linalg.norm(theta_old-theta) < 0.01)
            break
    return(theta,objective(theta,theta0,W,lamba,u,rho))

def update_sub(i, theta, direction, step=-0.001):
    # update the i-th dimension
    theta[i,:] += direction*step
    return

def objective(theta,theta0,W,lamda,u,rho):
    def d(i,j):
        return(distance(theta[i,:], theta[j,:]))
    D = np.shape(W)[0]
    y0 = 0
    for i in range(D):
        S = 0
        for j in list(range(i))+list(range(i+1,D)):
            S += np.exp(-d(i, j)**2)
        # print('S=:', S)
        for j in list(range(i))+list(range(i+1,D)):
            # y0 += -pow(d(i,j),2)*(W[i,j]==1)-np.log(sum(np.exp(-pow(d(i,j)*(W[i,j]==-1),2))))
            y0 += -d(i, j)**2 * (W[i, j] == 1) - np.log(S)
    y1 = lamda * np.linalg.norm(theta)
    y2 = rho/2 * np.linalg.norm(theta0-theta+u)
    return(y0 + y1 + y2)

def gradient(i,theta,theta0,W,lamda,u,rho):
    # gradient with respect to theta on i-th element
    D, L = np.shape(theta)
    g0 = np.zeros(L)
    S = 0
    for j in list(range(i)) + list(range(i + 1, D)):
        S += np.exp(-distance(theta[i,:],theta[j,:])**2)
    for j in list(range(i))+list(range(i+1,D)):
        g0 += (2-np.exp(-distance(theta[i,:],theta[j,:])**2)/S) * g_delta(theta[i,:],theta[j,:])
    g1 = lamda * 2 * theta[i,:]
    g2 = rho * (theta[i,:]-theta0[i,:]-u[i,:])
    return(g0 + g1 + g2)

def distance(x1,x2):
    return(np.linalg.norm(x1-x2))

def g_delta(x1,x2):
    # partial d(x1,x2)/partial(x1)
    return(2*(x1-x2))


if __name__=='__main__':
    # update_theta(theta0, W, lamba, u, rho)
    theta0 = np.random.rand(100,20)
    theta = np.random.rand(100, 20)
    W = np.zeros((100,100))
    for i in range(50):
        for j in range(50):
            W[i,j] = 1
        for j in range(50,100):
            W[i,j] = -1
    for i in range(50,100):
        for j in range(50):
            W[i,j] = -1
        for j in range(50,100):
            W[i,j] = 1
    lamba = 100
    u = np.random.rand(100,20)
    rho = 0.01
    # G = gradient(0,theta,theta0, W, lamba, u, rho)
    N = update_theta(theta0, theta0, W, lamba, u, rho)
    # print(N)

