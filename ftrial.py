
import gym
import gym.spaces
#from gym import spaces
env = gym.make('CartPole-v0')

def softmax_action(env,memory_state,observation,θ_action):
    prob_list=np.exp(θ_action[memory_state,env.observation]) ##clipping the softmax function to prevent saturation
    #print(memory_state)
    den=np.sum(prob_list)
    prob=[prob_list[a]/den for a in range(θ_action.shape[2])]
    prob=prob/np.sum(prob)
    action_probs=prob.reshape(env.number_of_actions())
    print(action_probs)
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


θ=np.zeros(shape=(num_states,env.number_of_observations(),env.number_of_actions()))
γ=np.zeros(shape=(env.number_of_observations(), num_states,num_states))
iterations=1000
bar=pyprind.ProgBar(iterations)
memory_state=0
observation_probs=np.array([0.5,0.5])
observation=np.random.multinomial(len(observation_probs),observation_probs).argmax()
reward=0
for i in range(iterations):
    bar.update()

    z_θ=0
    z_ϕ=0
    Δ_θ=0
    Δ_ϕ=0
    β=0.95
    α=0.05
    #memory_state=0
    T=1000
    t=0
    #action=0
    #observation,reward,_=env.step(action)
    cum_rewards=0
    scale=1.0
    while t<T:

        next_memory_state,_=softmax_transition(env,num_states,observation,memory_state,γ)

        action,_=softmax_action(env,next_memory_state,observation,θ)

        z_ϕ=β*z_ϕ+grad_log_boltzman_transition(observation,memory_state,γ)

        z_θ=β*z_θ+grad_log_boltzman_policy(next_memory_state,observation,action,θ)

        ##alternative way
        #z_ϕ=β*z_ϕ+grad_log_transition(env,observation,memory_state,num_states,γ)

        #z_θ=β*z_θ+grad_log_policy(env,memory_state,observation,action,θ)



        Δ_θ=Δ_θ+(1/(t+1))*(reward*z_θ-Δ_θ)

        Δ_ϕ=Δ_ϕ+(1/(t+1)) * (reward*z_ϕ-Δ_ϕ)

        observation,reward,_=env.step(action)
        #print("Memory_State",memory_state)
        cum_rewards+=scale*reward
        scale*=β

        memory_state = next_memory_state

        t+=1
    iteration_reward.append(cum_rewards)
    #print("Iterations :",i)

    θ=θ+α*Δ_θ
    γ=γ+α*Δ_ϕ
    θ=np.clip(θ,a_min=-30,a_max=30)
    γ=np.clip(γ,a_min=-30,a_max=30)
print(bar)
