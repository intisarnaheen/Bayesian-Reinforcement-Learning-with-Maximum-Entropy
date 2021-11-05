import numpy as np
import gym
import gym.spaces
import tensorflow as tf
env = gym.make('NChain-v0')
N= 5 # No of physical states
Q=6 #no of memory states
A =2 #no_of_actions
intial_state=1
gamma= .995
def dirichlet_sample(alphas):


    r = np.random.standard_gamma(alphas)
    r /= r.sum(-1).reshape(-1, 1)
    return r


if __name__ == "__main__":
    alphas1 = np.array([[200,0,800,0,0],[200,0,800,0,0],[200,0,0,800,0],[200,0,0,0,800],[200,0,0,0,800]])
    alphas2=  np.array([[800,200,0,0,0],[800,0,200,0,0],[800,0,0,200,0],[800,0,0,0,200],[800,0,0,0,200]])

    transition_probablity1 = dirichlet_sample(alphas1)
    transition_probablity2 = dirichlet_sample(alphas2)
    #print("dirichlet_sample1:",transition_probablity1)
    #print()
    #print("dirichlet_sample2:",transition_probablity2)
    #print()
    T= np.dstack((transition_probablity2,transition_probablity1))
    #print(T)

next_state =np.random.choice(N,p=[0.1, 0.3, 0.3, 0.1,0.2] )
print("the next state is:",next_state)
def softmax_intialmemory(phi):
    alpha = np.exp(phi)
    alpha /= np.sum(alpha)
    q0 = np.random.choice(Q,p=[0.1, 0, 0.3, 0.4, 0,0.2] )
    return alpha, q0
phi= np.ones(Q)


def softmax_action(chi):

    xi = np.exp(chi)
    xi /= np.sum(xi)

    a = np.random.choice(Q,p=[0.2, 0.1, 0.3, 0.3, 0,0.1] )
    return xi,a
chi= np.ones((Q,A))

def softmax_transition(shi):
    eta = np.exp(shi)
    eta /= np.sum(eta)
    next_memory_state = np.random.choice(Q,p=[0.1, 0, 0.3, 0.4, 0,0.2] )
    return eta, next_memory_state
shi= np.ones((Q,Q,N,N,A))


alpha, q0 = softmax_intialmemory(phi)
print(alpha)
print("initial_memorystate",q0)

xi, a = softmax_action(chi)
#print("actiondistribution",a)
#print(xi)
#print(chi)

eta, next_memory_state = softmax_transition(shi)
#print(eta)
#print("next_memory_state also", next_memory_state)

#print(shi)

#print(np.zeros((5, 6, 7, 8, 9))[0, :, 4].shape)

r = np.ones((N,N,A))
#print(eta)
#print("next_megemory_state also", next_memory_state)
A_a= np.einsum("qa,ija,pqija -> pqij", xi,T,eta)
print(A_a)
b_phi= np.einsum("qa,ija,ija -> qi", xi,T,r)
print(b_phi.shape)
#u = np.zeros((N,Q))
#v= np.zeros((N,Q))

#for current_state in range(N):
    #for current_memory_state in range(Q):
        #v[current_state,current_memory_state]= np.sum(xi,np.sum(np.multiply(T,(r+gamma*np.sum(np.dot( eta ,v[next_state,next_memory_state]))))))
#d=  np.linalg.inv(gamma*A_a)       #print(v)

#print(d)

#u = np.dot(np.linalg.inv(np.identity(4) - gamma*A_a), b_phi)
