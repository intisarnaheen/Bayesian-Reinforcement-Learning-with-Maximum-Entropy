import numpy as np
import matplotlib.pyplot as plt


Q= 6
N=12
A=4
step_size=0.5
iterations =10
gamma =.92
al = 0.05
i =0
phi = np.ones(Q)
chi= np.ones((Q,A))
shi= np.ones((N,Q,N,Q,A)) # Dimension(i,q0,j,q_next,a)
i_0 =0

def softmax_intialmemory(phi):
    alpha = np.exp(phi)
    alpha /= np.sum(alpha)
    q0 = np.random.choice(Q,p=alpha )
    #print("in function ",q_0)
    return alpha, q0

def softmax_action (chi,q0):

    x = np.exp(chi)
    xi =(x)/np.sum(x,keepdims=True, axis=1) #(et, keepdims=True,axis=3)
    w=(xi.T)
    #print(xi)
    a = np.random.choice(A,p=xi[q0,:].T)
    #p=xi[q0,:].T
    #a = np.random.choice(Q,p=[0.1, 0, 0.3, 0.4, 0,0.2] )
    return xi,a,w

def dirichlet_sample(alphas):
    r = np.random.standard_gamma(alphas)
    r /= r.sum(-1).reshape(-1, 1)
    return r

def softmax_transition(shi,q0,i,j_next,a):
    et = np.exp(shi)
    eta =(et)/np.sum(et, keepdims=True,axis=3)
    q_next = np.random.choice(Q,p=eta[i,q0,j_next,:,a].T)
    return eta, q_next

alpha ,q0 = softmax_intialmemory(phi)
xi,a,w = softmax_action(chi,q0)



alphas0 = np.array([[900,100,2,1,1,1,2,5,1,1,1,1],[100,800,100,2,1,4,1,6,3,4,2,3],[1,100,800,100,3,1,1,1,1,2,4,2],[1,1,2,1,1,1,1,1,2,1,2,1],[800,1,2,2,200,1,1,1,1,1,2,3],[1,1,1,3,1,1,1,1,2,1,1,1],[1,1,800,1,2,3,100,100,2,3,1,1],[1,1,1,2,1,1,1,1,1,1,1,1],[1,1,2,3,800,1,2,1,100,100,3,1],[1,1,2,3,1,1,2,1,100,800,100,1],[1,1,2,1,2,0,800,1,1,100,1,100],[1,1,2,1,1,2,1,800,1,1,100,100]])
alphas1 = np.array([[900,1,2,1,100,1,2,5,1,1,1,1],[800,200,1,2,1,4,1,6,3,4,2,3],[1,800,100,2,3,1,100,1,1,2,4,2],[1,1,2,1,1,1,1,1,2,1,2,1],[100,1,2,2,800,1,1,1,100,1,2,3],[1,1,2,3,1,1,1,1,2,1,1,1],[100,1,100,1,2,3,800,1,2,3,100,1],[1,1,1,2,1,1,1,1,1,1,1,1],[1,1,2,3,100,1,2,900,1,2,3,1],[1,1,2,3,1,1,2,800,200,1,2,1],[1,1,2,1,2,0,100,1,1,800,100,1],[1,1,2,1,1,2,1,100,1,1,800,100]])
alphas2 = np.array([[100,100,2,1,800,1,2,5,1,1,1,1],[100,800,100,2,1,4,1,6,3,4,2,3],[1,100,1,100,3,1,800,1,1,2,4,2],[1,1,2,1,1,1,1,1,2,1,2,1],[1,1,2,2,200,1,1,1,800,1,2,3],[1,1,2,3,1,1,1,1,2,1,1,1],[1,1,1,1,2,3,100,100,2,3,800,1],[1,1,1,2,1,1,1,1,1,1,1,1],[1,1,2,3,1,1,2,1,900,100,2,3],[1,1,2,3,1,1,2,1,100,800,100,2],[1,1,2,1,2,1,1,1,1,100,800,100],[1,1,2,1,1,2,1,1,1,1,100,900]])
alphas3 = np.array([[100,800,2,1,100,1,2,5,1,1,1,1],[1,200,800,2,1,4,1,6,3,4,2,3],[1,1,100,800,3,1,100,1,1,2,4,2],[1,1,2,1,1,1,1,1,2,1,2,1],[100,1,2,2,800,1,1,1,100,1,2,3],[1,1,2,3,1,1,1,1,2,1,1,1],[1,1,100,1,2,3,1,800,2,3,100,1],[1,1,1,2,1,1,1,1,1,1,1,1],[1,1,2,3,100,1,2,1,100,800,2,3],[1,1,2,3,1,1,2,1,1,200,800,2],[1,1,2,1,2,1,100,1,2,3,100,800],[1,1,2,1,1,2,1,100,1,1,1,900]])


transition_probablity0 = dirichlet_sample(alphas0)
transition_probablity1 = dirichlet_sample(alphas1)
transition_probablity2 = dirichlet_sample(alphas2)
transition_probablity3 = dirichlet_sample(alphas3)
transitionMatrix= np.dstack((transition_probablity0,transition_probablity1,transition_probablity2,transition_probablity3)) #order is i,j,a

#print(transitionMatrix.shape)

P= transitionMatrix

j_next= np.random.choice(N,p=P[i,:,a])




eta, q_next = softmax_transition(shi,q0,i,j_next,a)
for _ in range (iterations):

    #P= transitionMatrix
    #print(P.shape)

    #i0=1


    alpha ,q0= softmax_intialmemory(phi)


    #print("initial memory state",q_0)
    trajectory =100

    t=1
    gradient_phi=[]
    gradient_chi=[]
    gradient_shi=[]
    time = []
    value_reward =[]
    while t<trajectory:



        xi,a,w = softmax_action(chi,q0)

        # FIXME: P[i,:,a].T value never changes

        j_next= np.random.choice(N,p=P[i,:,a])
        #print("next state" , j_next)
        eta, q_next = softmax_transition(shi,q0,i,j_next,a)

        #print("action",a)

        #print("next_state",j_next)


        #print("next memory state",q_next)

        c = np.array([-0.04, -0.04, -0.04,  +1.0,
                      -0.04,   0.0, -0.04,  -1.0,
                      -0.04, -0.04, -0.04, -0.04])


        r_new = np.tile(c, (12,1))
        R=np.dstack((r_new,r_new,r_new,r_new))


        #for i5 in range (N):
            #for q6 in range (Q):
                #for j_next_1 in range (N):
                #for q_next_1 in range (Q):
                        #A_a = np.sum (chi[q6,a]*P[i,j_next_1,a]*eta[i5,q6,j_next_1,q_next_1,a])
                        #A_new[(i5-1)*Q+q6,(j_next_1-1)*Q+q_next_1] = A_a

        #print("A matrix",A_new)

        #b_new = np.zeros((N-1)*Q+Q)
        #for i6 in range (N):
            #for q7 in range (Q):
                #B_b = np.sum(np.sum(chi[q7,a]*P[i6,j_next,a]*R[i,j_next,a]))
                #b_new[(i6-1)*Q+q7] = B_b
        #print("B matrix",b_new)
        A_a = np.einsum("qa,ija,iqjpa -> iqjp", xi,P,eta)
        A_new = A_a.reshape((N*Q,N*Q))
        #print(A_new)
        b_new = np.einsum("qa,ija,ija -> iq", xi,P,R).reshape((N*Q))
        u = np.linalg.solve(np.eye(N*Q) - gamma * A_new, b_new).reshape((N, Q))
        V=u.T
        #print(V)
        # Ekhane dekho FOr loop ta
        for i1 in range(N-1):
            for q1 in range(Q):
                u_one = u[i1,q1] + step_size * (R[i,j_next,a] + gamma * u[i1+1,q1] - u[i1,q1])
                u[i1,q1] = u_one
                v=u.T
        #print("value",v)
        value_function = u[i,q0]
        #b_xi_final =np.zeros((N*Q))

        b_xi_final =np.zeros((N*Q))
        for i2 in range (N):
            b_xi_one =np.sum(P[i2,j_next,a]* (R[i2,j_next,a] + (gamma * np.sum(eta[i2,q0,j_next,q_next,a]*u[j_next,q_next]))))
            b_xi_two =np.sum(xi[q0,a]*(b_xi_one))
            b_xi_three= (b_xi_one-b_xi_two)
            b_xi_four = xi[q0,a]*(b_xi_three)
            b_xi_final[(i2-1)*Q+q0] = b_xi_four

        #print(b_xi_final)
        dell_xi_init = np.linalg.solve(np.eye(N*Q) - gamma * A_new, b_xi_final).reshape((N, Q))
        #print(dell_xi_init)
        b_shi_final = np.zeros((N*Q))
        b_shi_one = u[j_next,q0]-np.sum(eta[i,q0,j_next,q_next,a]*u[j_next,q0])
        b_shi_two =gamma*P[i,j_next,a]*eta[i,q0,j_next,q_next,a]*(b_shi_one)
        b_shi_final[(i-1)*Q+q0]  = b_shi_two
        dell_shi_init = np.linalg.solve(np.eye(N*Q) - gamma * A_new, b_shi_final).reshape((N, Q))
        #print(dell_shi_init)
        dell_phi=np.zeros(Q)
        for q2 in range(Q):
            dell_phi_init= alpha*(u[i,q2]- np.sum(alpha*u[i,q0]))
        dell_phi =dell_phi_init[i_0]
        #print('gradient1',dell_phi)
        phi = phi+ (1/t)*dell_phi
        dell_xi_one = np.sum(alpha* dell_xi_init,axis=1)
        dell_xi = dell_xi_one[i_0]
        #print ("gradient",dell_xi_final)
        chi[q0,a] = chi[q0,a]+ (1/t)*dell_xi
        #print("action",a,"memory state",q0,"Add:",(1/t)*dell_xi)
        #print(chi)
        dell_shi_one = np.sum(alpha* dell_shi_init,axis=1)
        dell_shi = dell_shi_one[i_0]
        shi[i,q0,j_next,q_next,a] = shi[i,q0,j_next,q_next,a]+ (1/t)*dell_shi


        i = j_next
        q0 = q_next

        t = t+1
        gradient_phi.append(dell_phi)
        gradient_chi.append(dell_xi)
        gradient_shi.append(dell_shi)
        time.append(t)
        value_reward.append(value_function)


#plt.show()
    #print("value update",v)
#print(b_xi_final)
    #phi = phi+ al*beta_phi
    #print(phi)
    #chi = chi+ al*beta_chi
    #print(chi)
    #shi = shi+ al*beta_shi
#print(shi)

#print("value", v)
print(chi)
#print(v)
iterations_reward = []
policy_reward =[]
i =0
i_optimal =0
ipochs = 50
q0_updated= np.random.choice(Q,p=alpha )
cum_reward =0
c_count = 0
optimal_cum_reward = 0
for _ in range (ipochs):
    for c_count in range (100):
           a_updated = np.random.choice(A,p=xi[q0_updated,:].T)

           P_updated=P #np.load("T.npy")
           # FIXME: p=P_updated[i,:,a_updated].T - never changes
           j_updated= np.random.choice(N,p=P_updated[i,:,a_updated])
           #print("updated state" ,j_updated)
           q_next_updated = np.random.choice(Q,p=eta[i,q0_updated,j_updated,:,a_updated].T)
           current_reward = R [i,j_updated,a_updated]
           a_optimal = 1
           j_optimal = np.random.choice(N,p=P_updated[i_optimal,:,a_optimal])
           #print("optimal next state", j_optimal)
           optimal_reward = R [i_optimal,j_optimal,a_optimal]
           #print(current_reward)
           cum_reward = cum_reward + current_reward #* (gamma**c_count)
           optimal_cum_reward =optimal_cum_reward + optimal_reward #*(gamma**c_count)
           i = j_updated
           i_optimal = j_optimal
           q0_updated = q_next_updated
           c_count = c_count +1
           iterations_reward.append(cum_reward)
           policy_reward.append(optimal_cum_reward)


print(cum_reward)
print(optimal_cum_reward)
