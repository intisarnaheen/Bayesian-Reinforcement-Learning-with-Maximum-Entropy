import numpy as np
import random as rm

states = ["1","2","3","4","5"]
transitionMatrix1 = np.array([[0.2,0.8,0,0,0],[0.2,0,0.8,0,0],[0.2,0,0,0.8,0],[0.2,0,0,0,0.8],[0.2,0,0,0,0.8]])
transitionMatrix2 = np.array([[0.8,0.2,0,0,0],[0.8,0,0.2,0,0],[0.8,0,0,0.2,0],[0.8,0,0,0,0.2],[0.8,0,0,0,0.2]])
transitionMatrix = np.concatenate((transitionMatrix1, transitionMatrix2), axis =0)
#np.concatenate((a, b), axis=0)
#print(transitionMatrix)

action = np.random.choice(5, 1, p=[0.8,0.2,0,0,0])

print(action)
