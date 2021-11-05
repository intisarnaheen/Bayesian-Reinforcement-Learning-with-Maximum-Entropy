import numpy as np
import random as rm

states = ["1","2","3","4","5"]
transitionMatrix1 = np.array([[0.2,0.8,0,0,0],[0.2,0,0.8,0,0],[0.2,0,0,0.8,0],[0.2,0,0,0,0.8],[0.2,0,0,0,0.8]])
transitionMatrix2 = np.array([[0.8,0.2,0,0,0],[0.8,0,0.2,0,0],[0.8,0,0,0.2,0],[0.8,0,0,0,0.2],[0.8,0,0,0,0.2]])
#transitionMatrix = np.concatenate((transitionMatrix2, transitionMatrix1),axis=0)

v= np.dstack((transitionMatrix2,transitionMatrix1))
#print(v[1,:,:])
b=(v.T).T
print(b[:,:,0])
#print(transitionMatrix)
#np.concatenate((a, b), axis=0)
#print(transitionMatrix)
#data = data.reshape((data.shape[0], data.shape[1], 1))
#T = transitionMatrix.reshape((transitionMatrix2.shape[1], transitionMatrix1.shape[0], 2))
#T= transitionMatrix.reshape(5,5,2)
#print(T[:,:,1])
#print(T[:,:,0])
#np.save('T.npy', T)


#f= [[[0.2 0.8 0 0 0]
 #[0.2 0 0.8 0 0]
 #[0.2 0 0 0.8 0]
 #[0.2 0 0 0 0.8]
 #[0.2 0 0 0 0.8]]
 #[[0.8 0.2 0 0 0]
 #[0.8 0 0.2 0 0]
 #[0.8 0 0 0.2 0]
 #[0.8 0 0 0 0.2]
 #[0.8 0 0 0 0.2]]]

#f_s= f.shape
#f_dim = f.ndim
#print(f_s)
#print(f_dim)
