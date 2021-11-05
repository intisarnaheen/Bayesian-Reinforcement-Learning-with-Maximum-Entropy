import numpy as np
import matplotlib.pyplot as plt
Q= 6
N=5
A=2
step_size=0.5
iterations =2
gamma =.92
al = 0.05


for _ in range (iterations):
    def dirichlet_sample(alphas):
        r = np.random.standard_gamma(alphas)
        r /= r.sum(-1).reshape(-1, 1)
        return r

    if __name__ == "__main__":
        alphas1 = np.array([[200,1,800,2,3],[200,1,800,2,3],[200,1,2,800,3],[200,1,2,3,800],[200,1,2,4,800]])
        alphas2=  np.array([[800,200,1,2,3],[800,2,200,1,3],[800,1,12,200,2],[800,1,4,1,200],[800,4,2,1,200]])

        transition_probablity1 = dirichlet_sample(alphas1)
        transition_probablity2 = dirichlet_sample(alphas2)

        transitionMatrix= np.dstack((transition_probablity2,transition_probablity1)) #order is i,j,a

    P= transitionMatrix
    #print(P.shape)

    i0=1


    #print("initial memory state",q_0)

    trajectory = 2
    t=1
    gradient_phi=[]
    gradient_chi=[]
    gradient_shi=[]
    time = []
    value_reward =[]
    while t<trajectory:
        def softmax_intialmemory(phi):
            alpha = np.exp(phi)
            alpha /= np.sum(alpha)
            q_0 = np.random.choice(Q,p=alpha )
            return alpha, q_0
        phi= np.ones(Q)

        alpha ,q_0 = softmax_intialmemory(phi)
        i =i0
        q0 = q_0
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
        #print("action",a)

        j_next= np.random.choice(N,p=P[i,:,a].T)
        #print("next_state",j_next)

        def softmax_transition(shi,q0,i,j_next,a):
            et = np.exp(shi)
            eta =(et)/np.sum(et, keepdims=True,axis=3)
            q_next = np.random.choice(Q,p=eta[i,q0,j_next,:,a].T)
            return eta, q_next
        shi= np.ones((N,Q,N,Q,A)) # Dimension(i,q0,j,q_next,a)


        eta, q_next = softmax_transition(shi,q0,i,j_next,a)
        #print("next memory state",q_next)

        c = np.array([2, 0, 0, 0, 10])
        r_new = np.tile(c, (5,1))
        R=np.dstack((r_new,r_new))

        A_a = np.einsum("aq,ija,iqjpa -> iqjp", xi,P,eta)
        A_new = A_a.reshape((N*Q,N*Q))
        #print(A_new)
        b_new = np.einsum("aq,ija,ija -> iq", xi,P,R).reshape((N*Q))
        u = np.linalg.solve(np.eye(N*Q) - gamma * A_new, b_new).reshape((N, Q))
        V=u.T
        utility_matrix = np
        def update_utility(utility_matrix,i,j_next,q0,q_next,r, step_size, gamma):
            u = utility_matrix[q0,i]
            u_t1 = utility_matrix[q_next,j_next]
            utility_matrix[q0,i]= utility_matrix + step_size * (r + gamma * u_t1 - u)

            return utility_matrix,utility_matrix_one

        utility_matrix,utility_matrix_one = update_utility(utility_matrix, i,j_next, r, step_size, gamma)

        print(utility_matrix)

        t= t+1






    '''Return the updated utility matrix

    @param utility_matrix the matrix before the update
    @param observation the state observed at t
    @param new_observation the state observed at t+1
    @param reward the reward observed after the action
    @param alpha the step size (learning rate)
    @param gamma the discount factor
    @return the updated utility matrix

    u = utility_matrix[observation[0], observation[1]]
    u_t1 = utility_matrix[new_observation[0], new_observation[1]]
    utility_matrix[observation[0], observation[1]] += \
        alpha * (reward + gamma * u_t1 - u)
    return utility_matrix
    '''
