import numpy as np
import gym
import gym.spaces

env = gym.make('NChain-v0')
#print(env.action_space)
#print(env.observation_space)
#print(env.observation_space.high)
num_memory_states=6
num_actions=2
num_states=5
initial_state=1
def one_hotencoding():
    onehot_encoded = list()
    for q in range (num_memory_states):
        vector= [0] * (num_memory_states)
        vector[q]=1
        onehot_encoded.append(vector)
    return onehot_encoded
    print(onehot_encoded)

phi =np.zeros((6,6))
print(phi)

matrix= np.transpose(onehot_encoding)
print(matrix)
b=np.exp(phi,matrix)
c=b/np.sum(b)
print(c)
