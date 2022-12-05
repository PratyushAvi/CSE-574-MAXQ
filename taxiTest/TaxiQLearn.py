# Josh's code for the taxi problem basic Q Table
#
#
import gym 
import numpy as np
import random

env = gym.make('Taxi-v3')
observation = env.reset()

#Initialize my state map and variables
complete = 0
gamma = 0.1
alpha = 0.95
epsi = 0.1
#Set my Q Table + initial state
qTab = np.zeros([env.observation_space.n, env.action_space.n])
state = observation

# 100 iterations
while complete < 100:
    done = 0
    while done == 0:
        #Time to do the Q learning. 
        # If below Epsilon, random otherwise get the optimal action
        if random.uniform(0,1) < epsi:
            action = env.action_space.sample()
        else:
            qList = qTab[state]
            action = np.random.choice(np.flatnonzero(qList == qList.max()))
        #STEP
        observation, reward, done, info = env.step(action) 
        
        #Update Q Vals
        curVal = qTab[state][action]
        nextMax = np.max(qTab[observation])
        newVal = (1 - alpha) * curVal + alpha * (reward + gamma * nextMax)
        qTab[state][action] = newVal
        state = observation
        #Render for last 9 sets
        if complete > 90:
            env.render()
        if done:
            observation = env.reset()
            print("------------------------------------------------- RESET ---------------------------------------------------")
            complete += 1
env.close()
