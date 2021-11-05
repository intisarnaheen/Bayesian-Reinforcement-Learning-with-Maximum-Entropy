current_state=1
current_memory_state=1
current_hybrid_state=(current_state,current_memory_state)
print(current_hybrid_state)

b_chi = np.zeros(N*Q)
for current_state in range(N):
    b_chi [(current_state-1)Q+next_memory_state] = np.sum(np.multiply(T ,(r+ gamma * np.sum(eta*V[next_state,next_memory_state]))))
