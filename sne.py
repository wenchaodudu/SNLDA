import numpy as np
from snlda import DOC_NUM, TOPIC_NUM

'''
    theta_1: D * TOPIC_NUM, document topic vectors returned from f
    theta_2: D * TOPIC_NUM, document topic vectors returned from g in the last iteration
    W: D * D adjacency matrix, 1 if in same category, -1 if in different category, 0 if unknown
    lambda: scalar, parameter of Gaussian prior
    u: D * TOPIC_NUM, penalty term of admm, 
    rho: scalar, penalty coeffcient of admm
    it: scalar, iteration count

    return: updated theta
'''

def update_theta(theta_1, theta_2, W, lamba, u, rho, it):
    pass

