import numpy as np
import random
reward_count = 0
moved_right_count = 0 #the number of time the state moves to the right
moved_left_count = 0 #the number of time the state moves to the start
stayed_count = 0 #the number of times the state stays the same
p1 = [10, 0, 10, 10, 0] #initial probability of moving right
p2 = 80 #probability of moving right given p1
p3 = 20
state = 1
count = 100 #number of total iterations
reward = 0
for i in range(count):
    r1 = random.randint(1,100)
    r2 = random.randint(1,100)#pick a random number between 1 and 100
    if r1 <= p1[state-1]: #if the random number is <= our first probability, we check for the second one
         #we again pick a random number for the probability p2
        if (r2 <= p2 and state < 5): #if it checks and we are not at the last state, we move right
            state = state + 1
            print ("Moved right, and the current state is: ", state, "\tIteration number: ", i+1)
            moved_right_count = moved_right_count + 1
            reward = 0
            reward_count = reward_count+reward
            print ("Reward in this state is :", reward , "\tIteration number:", i+1)
        elif (r2 <= p2 and state ==5):
            print ("Reached the final state ", state, "\tIteration number: ", i+1)
            stayed_count = stayed_count + 1
            reward = +10
            reward_count = reward_count+reward
            print ("Reward in the final state is :", reward , "\tIteration number:", i+1)
        else:
            state = 1
            print ("Went back to initial state, iteration number: ", i+1)
            moved_left_count = moved_left_count + 1
            reward = +2
            reward_count = reward_count+reward
            print ("Went back to initial state, so Reward in this state is :", reward , "\tIteration number:", i+1)
    else:
        if (r2 <= p3 and state < 5): #if it checks and we are not at the last state, we move right
            state = state + 1
            print ("Moved right, and the current state is: ", state, "\tIteration number: ", i+1)
            moved_right_count = moved_right_count + 1
            reward = 0
            reward_count = reward_count+reward
            print ("Reward in this state is :", reward , "\tIteration number:", i+1)
        elif (r2 <= p2 and state ==5):
            print ("Reached the final state ", state, "\tIteration number: ", i+1)
            stayed_count = stayed_count + 1
            reward = +10
            reward_count = reward_count+reward
            print ("Reward in the final state is :", reward , "\tIteration number:", i+1)
        else:
            state = 1
            print ("Went back to initial state, iteration number: ", i+1)
            moved_left_count = moved_left_count + 1
            reward = +2
            reward_count = reward_count+reward
            print ("Went back to initial state, so Reward in this state is :", reward , "\tIteration number:", i+1)
        #state = 1
        #print ("Went back to initial state, iteration number: ", i+1)
        #stayed_count = stayed_count + 1
print("Total reward:", reward_count )
print ("Stayed in the same state: ", stayed_count, " times")
print ("Moved right: ", moved_right_count, " times")
print ("Moved left: ", moved_left_count, " times")
