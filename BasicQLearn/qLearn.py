# Josh's Script for a basic qTable lookup QLearning algorithm
# Here we utilize gym_simplegrid for our gridworld. It looks nice 
import gym 
import gym_simplegrid
import numpy as np
import random

# Custom reward map 
my_reward_map = {
        b'E': -1.0,
        b'S': -1.0,
        b'W': -10.0,
        b'G': 10.0,
    }
#SimpleGrid-8v8-v0 is a great approximator from simplegrid. 
env = gym.make('SimpleGrid-8x8-v0', reward_map=my_reward_map)
env.metadata["render_fps"] = 60
observation = env.reset()

#Initialize my state map and variables
complete = 0
gamma = 0.1
alpha = 0.95
epsi = 0.1

#Classic qTable setup setting everything to 0
qTab = np.zeros([env.observation_space.n, env.action_space.n])
state = observation

#For my sanity, I only ran it for 10 iterations, but it runs much faster when you only visualize the end result
while complete < 100:
    done = 0
    while done == 0:
        #Time to do the Q learning. 
        # generate a value. If less than epsilon value, we get crazy. 
        if random.uniform(0,1) < epsi:
            #random action
            action = env.action_space.sample()
        else:
            #Get my actions
            qList = qTab[state]
            #Returns random and breaks ties for max values
            action = np.random.choice(np.flatnonzero(qList == qList.max()))
        #STEP
        observation, reward, done, info = env.step(action) 
        
        #Get my current qVal
        curVal = qTab[state][action]
        #Get the max value of the next state
        nextMax = np.max(qTab[observation])

        #This is an update step
        newVal = (1 - alpha) * curVal + alpha * (reward + gamma * nextMax)
        qTab[state][action] = newVal
        state = observation

        if(complete > 90):
            env.render()
        if done:
            observation = env.reset()
        if reward == 10:
            complete += 1
env.close()
