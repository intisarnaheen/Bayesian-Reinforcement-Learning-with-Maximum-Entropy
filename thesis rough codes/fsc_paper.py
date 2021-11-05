import numpy as np

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
    xi =(x)/np.sum(x,keepdims=True, axis=1) #(et, keepdims=True,axis=3)
    w=(xi.T)
    print(xi)
    a = np.random.choice(A,p=xi[q0,:].T)
    #p=xi[q0,:].T
    #a = np.random.choice(Q,p=[0.1, 0, 0.3, 0.4, 0,0.2] )
    return xi,a,w
chi= np.ones((Q,A))

xi,a,w = softmax_action(chi,q0)
#print(a)
#print(chi)
print("action",a)


def dirichlet_sample(alphas):


    r = np.random.standard_gamma(alphas)
    r /= r.sum(-1).reshape(-1, 1)
    return r


if __name__ == "__main__":
    alphas1 = np.array([[200,0,800,0,0],[200,0,800,0,0],[200,0,0,800,0],[200,0,0,0,800],[200,0,0,0,800]])
    alphas2=  np.array([[800,200,0,0,0],[800,0,200,0,0],[800,0,0,200,0],[800,0,0,0,200],[800,0,0,0,200]])

    transition_probablity1 = dirichlet_sample(alphas1)
    transition_probablity2 = dirichlet_sample(alphas2)

    transitionMatrix= np.dstack((transition_probablity2,transition_probablity1)) #order is i,j,a


P1= transitionMatrix

j_next1= np.random.choice(N,p=P1[i,:,a].T)
print("next_state",j_next1)


def softmax_transition(shi,q0,i,j_next1,a):
    et = np.exp(shi)
    eta =(et)/np.sum(et, keepdims=True,axis=3)
    q_next = np.random.choice(Q,p=eta[i,q0,j_next1,:,a].T)
    return eta, q_next
shi= np.ones((N,Q,N,Q,A)) # Dimension(q0,i,q_next,j)


eta, q_next = softmax_transition(shi,q0,i,j_next1,a)
print("next memory state",q_next)
#print(eta)


c = np.array([2, 0, 0, 0, 10])
r_new = np.tile(c, (5,1))
R1=np.dstack((r_new,r_new))

#for _ in range (iterations):
P=P1
alpha ,q0 = softmax_intialmemory(phi)
trajectory = 100
t=2
    #while t<trajectory:
current_hybrid_state=(i,q0)
xi,a,w = softmax_action(chi,q0)
j_next = j_next1
eta, q_next = softmax_transition(shi,q0,i,j_next1,a)
R = R1

A_a = np.einsum("aq,ija,iqjpa -> iqjp", xi,P,eta)
A_new = A_a.reshape((N*Q,N*Q))
        #print(A_new)
b_new = np.einsum("aq,ija,ija -> iq", xi,P,R).reshape((N*Q))
u = np.linalg.solve(np.eye(N*Q) - gamma * A_new, b_new).reshape((N, Q))
V=u.T
for i in range(N):
    for q0 in range(Q):
        u_one = u[i,q0] + step_size * (R[i,j_next,a] + gamma * u[j_next,q_next] - u[i,q0])
        u[i,q0] = u_one
v=u.T
print("value update",v)
b_xi_final =np.zeros((N-1)*Q+Q)
for i in range (N):
    b_xi_one =np.sum(P[i,j_next,a]* R[i,j_next,a] + np.sum(eta[i,q0,j_next,q_next,a]*u[j_next,q0]))
    b_xi_two =np.sum(xi[a,q0]*b_xi_one)
    b_xi_three= np.sum(P[i,j_next,a]* R[i,j_next,a] + np.sum(eta[i,q0,j_next,q_next,a]*u[j_next,q_next]))
    b_xi_four = xi[a,q0]*(b_xi_one-b_xi_two)
    b_xi_final[(i-1)*Q+q0] = b_xi_four
print(b_xi_final)
dell_xi_init = np.linalg.solve(np.eye(N*Q) - gamma * A_new, b_xi_final).reshape((N, Q))
#print(dell_xi_init)
b_shi_final = np.zeros((N*Q))
b_shi_one = u[j_next,q_next]-np.sum(eta[i,q0,j_next,q_next,a]*u[j_next,q0])
b_shi_two =gamma*P[i,j_next,a]*eta[i,q0,j_next,q_next,a]*(b_shi_one)
b_shi_final[(i-1)*Q+q0]  = b_shi_two
dell_shi_init = np.linalg.solve(np.eye(N*Q) - gamma * A_new, b_shi_final).reshape((N, Q))
#print(dell_shi_init)
dell_phi=np.zeros(Q)
for q0 in range(Q):
    dell_phi= alpha*(u[i,q_next]- np.sum(alpha*u[i,q0]))
#print(dell_phi)
phi = phi+ .01*dell_phi
dell_xi_one = np.sum(alpha* dell_xi_init,axis=1)
dell_xi_final = dell_xi_one[i]
print ("gradient",dell_xi_final)
chi = chi+ (1/t)*dell_xi_final
#print(chi)
dell_shi_one = np.sum(alpha* dell_shi_init,axis=1)
dell_shi_final = dell_shi_one[i]
shi = shi+ (1/t)*dell_shi_final
#print(shi.shape)
