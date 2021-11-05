import gym
import gym.spaces
env = gym.make('NChain-v0')
print(env.action_space)
print(env.observation_space)
