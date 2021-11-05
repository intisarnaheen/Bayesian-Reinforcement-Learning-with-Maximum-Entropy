import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline

Q= 6
N=5
A=2
step_size=0.5
iterations =10
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
    print(P.shape)

    i0=1

    def softmax_intialmemory(phi):
        alpha = np.exp(phi)
        alpha /= np.sum(alpha)
        q_0 = np.random.choice(Q,p=alpha )
        return alpha, q_0
    phi= np.ones(Q)

    alpha ,q_0 = softmax_intialmemory(phi)
    #print("initial memory state",q_0)

    trajectory = 200
    t=1
    gradient_phi=[]
    gradient_chi=[]
    gradient_shi=[]
    time = []
    value_reward =[]
    while t<trajectory:
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
        chi= np.ones((A,Q)) # q,a

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


        A_new = np.zeros((N*Q,N*Q))

        for i in range (N):
            for q0 in range (Q):
                for j_next in range (N):
                    for q_next in range (Q):
                        A_a = np.sum (chi[a,q0]*P[i,j_next,a]*eta[i,q0,j_next,q_next,a])
                        A_new[(i-1)*Q+q0,(j_next-1)*Q+q_next] = A_a

        #print("A matrix",A_new)

        b_new = np.zeros((N-1)*Q+Q)
        for i in range (N):
            for q0 in range (Q):
                B_b = np.sum(np.sum(chi[a,q0]*P[i,j_next,a]*R[i,j_next,a]))
                b_new[(i-1)*Q+q0] = B_b
        #print("B matrix",b_new)
        u = np.linalg.solve(np.eye(N*Q) - gamma * A_new, b_new).reshape((N, Q))
        V=u.T
        for i in range(N):
            for q0 in range(Q):
                u_one = u[i,q0] + step_size * (R[i,j_next,a] + gamma * u[j_next,q_next] - u[i,q0])
                u[i,q0] = u_one
                v=u.T
        value_function = u[i,q0]
        #print("valeue:",value_function)
        b_xi_final =np.zeros((N*Q))
        for i in range (N):
            b_xi_one =np.sum(P[i,j_next,a]* R[i,j_next,a] + np.sum(eta[i,q0,j_next,q_next,a]*u[j_next,q0]))
            b_xi_two =np.sum(xi[a,q0]*b_xi_one)
            b_xi_three= np.sum(P[i,j_next,a]* R[i,j_next,a] + np.sum(eta[i,q0,j_next,q_next,a]*u[j_next,q_next]))
            b_xi_four = xi[a,q0]*(b_xi_one-b_xi_two)
            b_xi_final[(i-1)*Q+q0] = b_xi_four
        #print(b_xi_final)

        dell_xi_init = np.linalg.solve(np.eye(N*Q) - gamma * A_new, b_xi_final).reshape((N, Q))
        #print(dell_xi_init)
        b_shi_final = np.zeros((N*Q))
        b_shi_one = u[j_next,q_next]-np.sum(eta[i,q0,j_next,q_next,a]*u[j_next,q0])
        b_shi_two =gamma*P[i,j_next,a]*eta[i,q0,j_next,q_next,a]*(b_shi_one)
        b_shi_final[(i-1)*Q+q0]  = b_shi_two
        #print(b_shi_final)
        dell_shi_init = np.linalg.solve(np.eye(N*Q) - gamma * A_new, b_shi_final).reshape((N, Q))
        #print(dell_shi_init)
        dell_phi=np.zeros(Q)
        for q0 in range(Q):
            dell_phi= alpha*(u[i,q_next]- np.sum(alpha*u[i,q0]))
        #print(dell_phi)
        phi = phi+ (1/(t+1))*(dell_phi)
        #print(phi)
        dell_xi_one = np.sum(alpha* dell_xi_init,axis=1)


        dell_xi = dell_xi_one[i]
        #print("gradient",dell_xi_final)
        chi = chi + (1/t)*dell_xi
        #print(chi)
        dell_shi_one = np.sum(alpha* dell_shi_init,axis=1)
        #print (chi)
        dell_shi = dell_shi_one[i]
        #print(dell_shi_final)
        shi = shi+ (1/(t+1))*(dell_shi)


        t=t+1
        gradient_phi.append(dell_phi)
        gradient_chi.append(dell_xi)
        gradient_shi.append(dell_shi)
        time.append(t)
        value_reward.append(value_function)

plt.plot(time,gradient_phi, 'r')
plt.plot(time,gradient_chi, 'b' )
plt.plot(time,gradient_shi, 'g')
plt.show()
plt.plot(value_reward, 'b')
plt.show()

    #print("value update",v)
