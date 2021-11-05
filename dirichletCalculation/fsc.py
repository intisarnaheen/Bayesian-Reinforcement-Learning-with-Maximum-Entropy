import numpy as np

num_memory_states =10
N= 5
intial_state = 1
num_actions =2

def dirichlet_sample(alphas):


    r = np.random.standard_gamma(alphas)
    r /= r.sum(-1).reshape(-1, 1)
    return r


if __name__ == "__main__":
    alphas1 = np.array([[200,0,800,0,0],[200,0,800,0,0],[200,0,0,800,0],[200,0,0,0,800],[200,0,0,0,800]])
    alphas2=  np.array([[800,200,0,0,0],[800,0,200,0,0],[800,0,0,200,0],[800,0,0,0,200],[800,0,0,0,200]])

    transition_probablity1 = dirichlet_sample(alphas1)
    transition_probablity2 = dirichlet_sample(alphas2)
    print("dirichlet_sample1:")
    print(transition_probablity1)
    print("dirichlet_sample2:")
    print(transition_probablity2)
    T= np.dstack((transition_probablity2,transition_probablity1))
    print(T)

next_state =np.random.choice(N,p=[0.1, 0.3, 0.3, 0.1,0.2] )
print(next_state)
#b= transition_probablity
#print(b)
#c= np.random.multinomial(len(b), b)
#d= np.random.choice(b)


#start of implementation
