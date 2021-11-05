
def softmax_action(chi,q0):

    zeta = np.exp(chi)
    zeta /= np.sum(zeta)
    a=np.empty(A, dtype= np.intp)

    for q0 in range(Q):
        a[q0] = np.random.choice(A,p=zeta[q0])

    return zeta,a
chi= np.ones((Q,A))

zeta,a= softmax_action(chi,q0)
print(a)

j =np.random.choice(N,p=[0.1, 0.3, 0.3, 0.1,0.2] )
#a = np.random.choice(Q, p=[0.1, 0, 0.3, 0.4, 0,0.2])

def softmax_transition(shi):
    eta = np.exp(shi)
    eta /= np.sum(eta)
    for q0 in range(Q):
            for i in range(N):
                for j_next in range(N):
                    for a in range(A):
                        q_next[q0,i,j_next,a]= np.random.choice(Q, p= eta[q0,i,j_next,a])
     and None


    return eta, q
shi= np.ones((Q,N,N,A,Q))


eta, q = softmax_transition(shi)
print(q)

''' and None

'''
b_xi_final =np.zeros((N-1)*Q+Q)
for i in range (N):
    b_one=R[:,None,:,:]+gamma*(np.sum(eta*u[None,None,:,:,None],axis=3))
    b_two = np.sum(t[:,None,:,:]*b_one,axis=2)
    b_three= np.sum(w[None,:,:]*b_two,axis=2)
    b_final= w[None,:,:]*(b_two - b_three[:,:,None])
    b_xi_final = b_final[(i-1)*Q]
    print(b_xi_final.shape)



























b_xi_final =np.zeros((N-1)*Q+Q)
for i in range (N):
    b_one=R[:,None,:,:]+gamma*(np.sum(eta*u[None,None,:,:,None],axis=3))
    b_two = np.sum(t[:,None,:,:]*b_one,axis=2)
    b_three= np.sum(w[None,:,:]*b_two,axis=2)
    b_final= w[None,:,:]*(b_two - b_three[:,:,None])
    b_final_one= b_final[i,q0,a]
    b_xi_final[(i-1)*Q+q0] = b_final_one
print(b_xi_final)

dell_xi = np.linalg.solve(np.eye(N*Q) - gamma * A_new, b_xi_final).reshape((N, Q))
print(dell_xi.shape)

#print(b_xi_final)
#b_x_x = b_xi_final.reshape((Q*N))
#print(b_x_x.shape)

b_shai_final_one = np.zeros((N*Q))
b_shi_one =np.sum(eta*u[None,None,:,:,None],axis=3)
b_shi_two= u[None,None,:,:,None]- b_shi_one[:,:,:,None,:]
b_shi_three= gamma*w[None,:,None,None,:]*t[:,None,:,None,:]*eta*b_shi_two
print(b_shi_three.shape)

b_shi_final=b_shi_three[i,q0,j_next,q_next,a]
print(b_shi_final)
b_shai_final_one[(i-1)*Q+q0]  = b_shi_final
print(b_shai_final_one)
A_a_xi_final =np.zeros(N*Q)
for q0 in range(Q):
    A_a_xi= np.sum(w[None,:,None,None,:]*t[:,None,:,None,:]*eta,axis=4).reshape((N*Q,N*Q))
    #print(A_a_xi.shape)

A_a_xi_final = A_a_xi[(i-1)*Q+q0]
#print(A_a_xi_final.shape)


A_a_xi_1 = np.einsum("aq,ija,iqjpa -> iq", xi,t,eta)
#print(A_a_xi_1)

#print(b_xi_final.shape)
#print(dell_phi)





b_xi=R[:,None,:,:] +gamma* np.einsum("iqjpa,jp -> iqja", eta,u)
b_one = np.einsum("iqja,ija -> iqa", b_xi,t)
b_two= np.einsum("aq,iqa -> iq",xi,b_one)
b_final = w*(b_one-b_two[:,:,None])

#b_final = w[None,:,:]*(b_one-b_two[:,:,None])
#print(b_final)

#print(b_xi.shape)
#print(b_xi)
#rint(b_one.shape)
#print(b_two.shape)
#print(b_final.shape)
b_xi_final = b_final[(i-1)*Q+q0]
#print(b_xi_final)
#b_x_x = b_xi_final.reshape((Q*N))
#print(b_x_x.shape)










A_a_xi = np.einsum("aq,ija,iqjpa -> iq", xi,t,eta)
print(A_a_xi.shape)


b_xi_final =np.zeros((N*Q))
for i1 in range (N):
    b_xi_one =np.sum(P[i1,j_next,a]* R[i1,j_next,a] + np.sum(eta[i1,q0,j_next,q_next,a]*u[j_next,q0]))
    b_xi_two =np.sum(xi[a,q0]*b_xi_one)
    b_xi_three= np.sum(P[i1,j_next,a]* R[i1,j_next,a] + np.sum(eta[i1,q0,j_next,q_next,a]*u[j_next,q_next]))
    b_xi_four = xi[a,q0]*(b_xi_one-b_xi_two)
    b_xi_final[(i1-1)*Q+q0] = b_xi_four
