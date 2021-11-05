import numpy as np

def return_state_utility(v, t, u, reward, gamma):

    action_array = np.zeros(2)
    for action in range(0, 2):
        #prob * (reward + discount_factor * V[next_state])
        action_array = np.sum(np.multiply(np.dot(v, t[:,:,action]),reward)+ gamma * np.multiply(u, np.dot(v, t[:,:,action])))
        #action_array[action] = np.sum(np.multiply(np.dot(v, T[:,:,action]),reward)+ gamma * np.multiply(u, np.dot(v, T[:,:,action])))
        #action_array = np.sum(reward+ gamma * np.multiply(u, np.dot(v, T[:,:,action])))
        #action_array[action] = np.sum(np.multiply(u, np.dot(v, T[:,:,action])))
        #a_s= action_array.shape
        #print(a_s)
        #a_dim = action_array.ndim
        #print(a_dim)
    return np.max(action_array)
    #return reward + gamma * np.max(action_array)
def main():
    #Change as you want
    tot_states = 5
    gamma = 0.8 #Discount factor
    iteration = 0 #Iteration counter
    epsilon = 0.01 #Stopping criteria small value

    #List containing the data for each iteation
    graph_list = list()

    #Transition matrix loaded from file (It is too big to write here)
    transitionMatrix1 = np.array([[0.2,0.8,0,0,0],[0.2,0,0.8,0,0],[0.2,0,0,0.8,0],[0.2,0,0,0,0.8],[0.2,0,0,0,0.8]])
    #t1 = transitionMatrix1.transpose()
    #print()
    transitionMatrix2 = np.array([[0.8,0.2,0,0,0],[0.8,0,0.2,0,0],[0.8,0,0,0.2,0],[0.8,0,0,0,0.2],[0.8,0,0,0,0.2]])
    #t2 = transitionMatrix2.transpose()
    #d = np.concatenate((t1, t2), axis =1)
    #d  = np.concatenate((transitionMatrix2, transitionMatrix1),axis=1)
    b= np.dstack((transitionMatrix2,transitionMatrix1))
    #print(b)
    t=(b.T).T

    xxx= b[:,:,1]
    print("transitio for action",xxx)




    #t_s= T.shape
    #t_dim = T.ndim
    #print(t_s)
    #print(t_dim)
    #np.
    #concatenate((a, b), axis=0)
    #print(T)
    #Reward vector

    r = np.array([2, 0, 0, 0, 10])

    #Utility vectors
    u = np.array([0.0, 0.0, 0.0,0.0,0.0])
    u1 = np.array([0.0, 0.0, 0.0,0.0,0.0])

    while True:
        delta = 0
        u = u1.copy()
        iteration += 1
        graph_list.append(u)
        for s in range(tot_states):
            reward = r[s]
            v = np.zeros((1,tot_states))
            v[0,s] = 1.0
            u1[s] = return_state_utility(v, t, u, reward, gamma)
            delta = max(delta, np.abs(u1[s] - u[s])) #Stopping criteria
        if delta < epsilon * (1 - gamma) / gamma:
                print("=================== FINAL RESULT ==================")
                print("Iterations: " + str(iteration))
                print("Delta: " + str(delta))
                print("Gamma: " + str(gamma))
                print("Epsilon: " + str(epsilon))
                print("===================================================")
                print(u[0:5])
                print("===================================================")
                break

if __name__ == "__main__":
    main()
