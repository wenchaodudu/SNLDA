import numpy as np
from snlda import DOC_NUM, TOPIC_NUM

'''
    theta: D * TOPIC_NUM, document topic vectors
    W: D * D adjacency matrix, 1 if in same category, -1 if in different category, 0 if unknown
    lambda: scalar, parameter of Gaussian prior
    u: D * TOPIC_NUM, penalty term of admm, 

    return: updated theta
'''

def update_theta(theta, W, lamba, u):
    pass

