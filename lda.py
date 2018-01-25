import numpy as np
import pdb
from scipy.special import digamma

iter_num = 10

np.seterr(over='raise')

def exp_proportion(theta):
    exp_theta = np.exp(theta)
    return exp_theta / np.sum(exp_theta)

def update_variables(X, theta_1, theta_2, q_z, beta, labels, lbda, rho, u, it):
    DOC_NUM, VOCAB_SIZE = X.shape
    labeled = [x for x in range(DOC_NUM) if labels[x] != -1]
    unlabeled = [x for x in range(DOC_NUM) if labels[x] == -1]
    total = 0
    for itt in range(iter_num):
        print "Updating topics; iter:", itt
        new_beta = np.ones(beta.shape)
        for x in labeled + unlabeled:
            weight = 1 if labels[x] != -1 else 1
            # update theta
            #Ez = np.sum(q_z[x][w] * X[x, w] for w in q_z[x])
            Ez = np.sum(tp[1] * tp[2] for tp in q_z[x])
            pi = exp_proportion(theta_1[x])
            #if labels[x] != -1 and it:
            if labels[x] != -1:
                theta_1[x] = (Ez - np.sum(Ez) * pi) / rho + theta_2[x] - u[x]
            else:
                theta_1[x] = (Ez - np.sum(Ez) * pi) * lbda

            # update q_z
            eta_hessian = np.asmatrix(pi).T * np.asmatrix(pi) - np.diag(pi)
            #if labels[x] != -1 and it:
            if labels[x] != -1:
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
                new_beta[:, tp[0]] += weight * tp[1] * tp[2]

        # update beta
        new_beta /= np.sum(new_beta, axis=1)[:, np.newaxis]
        print np.linalg.norm(new_beta - beta)
        beta = new_beta
            
    return theta_1, q_z, beta

def dtm_update(X, W, topic_num, theta, phi, it_num, dic):
    DOC_NUM, VOCAB_SIZE = X.shape
    gamma = 0.1

    def Q(theta):
        total = 0
        for x in range(DOC_NUM):
            '''
            for y in range(VOCAB_SIZE):
            '''
            for y in dic[x]:
                post = np.multiply(theta[x, :], phi[:, y])
                post /= np.sum(post)
                total += X[x, y] * np.dot(post, np.log(theta[x, :]) + np.log(phi[:, y]))
        return total

    def R(theta):
        dist = np.zeros((DOC_NUM, DOC_NUM))
        for x in range(DOC_NUM):
            dist[x] = np.linalg.norm(theta[x] - theta, axis=1)**2
        return np.sum(dist) / np.sum(np.multiply(W, dist))
                        
    for _ in range(it_num):
        print _
        new_phi = np.ones(phi.shape) * .01
        for x in range(DOC_NUM):
            '''
            for y in range(VOCAB_SIZE):
            '''
            for y in dic[x]:
                post = np.multiply(theta[x, :], phi[:, y])
                post /= np.sum(post)
                new_phi[:, y] += post * X[x, y]
        new_phi /= np.sum(new_phi, axis=1)[:, np.newaxis]
        phi = new_phi
        theta_1 = np.copy(theta)

        dist = np.zeros((DOC_NUM, DOC_NUM))
        for x in range(DOC_NUM):
            dist[x] = np.linalg.norm(theta[x] - theta, axis=1)**2

        alpha = R(theta)
        for k in range(topic_num):
            for x in range(DOC_NUM):
                beta = (DOC_NUM * theta_1[x, k] + alpha * np.dot(W[x, :], theta_1[:, k])) / (np.sum(theta_1[:, k]) + alpha * np.sum(W[x, :]) * theta_1[x, k])
                if theta_1[x, k] > 0 and beta > 1 / theta_1[x, k]:
                    beta = 0.99 / theta_1[x, k]
                '''
                if beta * theta_1[x, k] == 1:
                    theta_1[x] = .01
                    theta_1[x, k] = 1 - .01 * (topic_num - 1)
                '''
                coef = (1 - beta * theta_1[x, k]) / (1 - theta_1[x, k])
                theta_1[x] *= coef
                theta_1[x, k] /= coef
                theta_1[x, k] *= beta

        if np.any(theta_1 <= 0):
            theta_1 += .01
            theta_1 /= np.sum(theta_1, axis=1)[:, np.newaxis]

        Q1 = Q(theta)
        Q2 = Q(theta_1)
        if Q2 > Q1:
            theta = theta_1
            print Q2
        else:
            print "Q1 not improved"
            theta_2 = np.ones(theta_1.shape) * .01
            for x in range(DOC_NUM):
                #for y in range(VOCAB_SIZE):
                for y in dic[x]:
                    post = np.multiply(theta_1[x, :], phi[:, y])
                    post /= np.sum(post)
                    theta_2[x] += post * X[x, y]
            theta_2 /= np.sum(theta_2, axis=1)[:, np.newaxis]
            if np.any(theta_2 <= 0):
                pdb.set_trace()
            theta_3 = np.copy(theta_1)
            s = 0.
            while True:
                print s
                theta_3 += gamma * (theta_2 - theta_1)                
                s += gamma
                Q1 = Q(theta_3)
                Q2 = Q(theta)
                if Q1 > Q2 and R(theta_3) > R(theta):
                    theta = theta_3
                    print Q1
                    break
                elif s >= 3 * gamma:
                    gamma /= 3
                    break

    return theta, phi

