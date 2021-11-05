import numpy as np
import matplotlib.pyplot as plt
Q= 6
N=5
A=2

i=1
gamma =.92
step_size=0.5
iterations =100
def softmax_intialmemory(phi):
    alpha = np.exp(phi)
    alpha /= np.sum(alpha)
    q0 = np.random.choice(Q,p=alpha )
    return alpha, q0
phi= np.ones(Q)

alpha ,q0 = softmax_intialmemory(phi)
print("initial memory state",q0)
#print(alpha)

def softmax_action(chi,q0):

    x = np.exp(chi)
    xi =(x)/np.sum(x[:,None],axis=0)
    w=(xi.T)
    a = np.random.choice(A,p=xi[:,q0].T)
    #p=xi[q0,:].T
    #a = np.random.choice(Q,p=[0.1, 0, 0.3, 0.4, 0,0.2] )
    return xi,a,w
chi= np.ones((A,Q))

xi,a,w = softmax_action(chi,q0)
print(a)
def dirichlet_sample(alphas):


    r = np.random.standard_gamma(alphas)
    r /= r.sum(-1).reshape(-1, 1)
    return r


if __name__ == "__main__":
    alphas1 = np.array([[200,1,800,2,3],[200,1,800,2,3],[200,1,2,800,3],[200,1,2,3,800],[200,1,2,4,800]])
    alphas2=  np.array([[800,200,1,2,3],[800,2,200,1,3],[800,1,12,200,2],[800,1,4,1,200],[800,4,2,1,200]])

    transition_probablity1 = dirichlet_sample(alphas1)
    transition_probablity2 = dirichlet_sample(alphas2)
    print(transition_probablity1)

    transitionMatrix= np.dstack((transition_probablity2,transition_probablity1)) #order is i,j,a


P1= transitionMatrix
