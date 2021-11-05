import gym
from gym import spaces
from gym.utils import seeding
import gym.spaces
#env = gym.make('NChain-v0')

class NChainEnv(gym.Env):
    def __init__(self, n=5, slip=0.2, small=2,large=10,left_action=0.8):
        self.n = n
        self.slip = slip  # probability of 'slipping' an action
        self.small = small  # payout for 'backwards' action
        self.large = large  # payout at end of chain for 'forwards' action
        self.state = 0  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2)
        #self.action = actions
        self.left_action = left_action
        self.observation_space = spaces.Discrete(self.n)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        #Randomly selecting an action for each state
        action = np.random.choice(5, 1, p=[0.8,0.2,0,0,0])
        if action ==0:
         if self.np_random.rand() < self.slip:
          # 'backwards': go back to the beginning, get small reward
            reward = self.small
            self.state = 0
         elif self.state < self.n - 1:  # 'forwards': go up along the chain
            reward = 0
            self.state += 1
         else:  # 'forwards': stay at the end of the chain, collect large reward
            reward = self.large
        if action ==1 :

         if self.np_random.rand() < self.left_action:
            # 'backwards': go back to the beginning, get small reward
              reward = self.small
              self.state = 0
         elif self.state < self.n - 1:  # 'forwards': go up along the chain
              reward = 0
              self.state += 1
         else:  # 'forwards': stay at the end of the chain, collect large reward
              reward = self.large



        done = False
        return self.state, reward, done, {}

    def reset(self):
        self.state = 0
        return self.state
if __name__ == "__init__":
    main()
