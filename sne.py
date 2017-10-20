import numpy as np
from snlda import DOC_NUM, TOPIC_NUM
import random

'''
    theta: D * TOPIC_NUM, document topic vectors
    W: D * D adjacency matrix, 1 if in same category, -1 if in different category, 0 if unknown
    lambda: scalar, parameter of Gaussian prior
    u: D * TOPIC_NUM, penalty term of admm, 
    rho: scalar, penalty coeffcient of admm

    return: updated theta
'''

def update_theta(theta0, W, lamba, u, rho):
    D = np.shape(W)[0]
    theta = theta0
    for _ in range(1000):
        theta_old = theta
        shuffle = random.shuffle(range(D))
        for i in shuffle:
            dir = gradient(i,theta,theta0,W,lamba,u,rho)
            theta = update_sub(i,theta,dir)
        if np.linalg.norm(theta_old-theta) < 0.01 or \
            np.abs(objective(theta,theta0,W,lamba,u,rho)-objective(theta_old,theta0,W,lamba,u,rho)) < 0.01:
            break
    return(theta,objective(theta,theta0,W,lamba,u,rho))

def update_sub(i, theta, direction, step=0.1):
    # update the i-th dimension
    theta[i,:] = theta[i,:] + direction*step
    return(theta)

def objective(theta,theta0,W,lamda,u,rho):
    def d(i,j):
        return(distance(theta[i,:], theta[j,:]))
    D = np.shape(W)[0]
    y0 = 0
    for i in range(D):
        for j in list(range(i))+list(range(i+1,D)):
            # y0 += -pow(d(i,j),2)*(W[i,j]==1)-np.log(sum(np.exp(-pow(d(i,j)*(W[i,j]==-1),2))))
            y0 += -pow(d(i, j), 2) * (W[i, j] == 1) - np.log(sum(np.exp(-pow(d(i, j), 2))))
    y1 = lamda * np.linalg.norm(theta)
    y2 = - rho/2 * np.linalg.norm(theta0-theta+u)
    return(y0 + y1 + y2)

def gradient(i,theta,theta0,W,lamda,u,rho):
    # gradient with respect to theta on i-th element
    D, L = np.shape(theta)
    g0 = np.zeros(L)
    for j in list(range(i))+list(range(i+1,D)):
        g0 += (2-np.exp(-pow(distance(theta[i,:],theta[j,:]),2))/sum(np.exp(-pow(
            distance(theta[i,:],theta[j,:]), 2)))) * g_delta(theta[i,:],theta[j,:])
    g1 = lamda * 2 * theta[i,:]
    g2 = - rho * (theta[i,:]-theta0[i,:]-u[i,:])
    return(g0+g1+g2)

def distance(x1,x2):
    return(np.linalg.norm(x1-x2))
def g_delta(x1,x2):
    # partial d(x1,x2)/partial(x1)
    return(2*(x1-x2))
