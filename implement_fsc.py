import gym
import gym.spaces
env = gym.make('NChain-v0')
num_states =5 #defines number of memory states
initial_state=0
next_state=0




def softmax_action(env,memory_state,observation,θ_action):
    prob_list=np.exp(θ_action[memory_state,observation]) ##clipping the softmax function to prevent saturation
    den=np.sum(prob_list)
    prob=[prob_list[a]/den for a in range(θ_action.shape[2])]
    prob=prob/np.sum(prob)
    action_probs=prob.reshape(env.number_of_actions())
    #print(action_probs)
    action=np.random.multinomial(len(action_probs),action_probs).argmax()
    return action,action_probs
def softmax_transition(env,num_states,observation,memory_state,γ_transition):
    #import pdb;pdb.set_trace()
    prob_list=np.exp(γ_transition[observation,memory_state])
    den=np.sum(prob_list)
    prob=[prob_list[a]/den for a in range(γ_transition.shape[2])]
    prob=prob/np.sum(prob)
    transition_prob=prob.reshape(num_states)
    #print("Trainsition Prob",transition_prob)
    next_memory_state=np.random.multinomial(len(transition_prob),transition_prob).argmax()
    return next_memory_state,transition_prob
