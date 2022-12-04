import gym 
import numpy as np
import random

env = gym.make('Taxi-v3')
#env.metadata["render_fps"] = 60
observation = env.reset()

#Initialize my state map and variables
complete = 0
gamma = 0.1
alpha = 0.95
epsi = 0.1
qTab = np.zeros([env.observation_space.n, env.action_space.n])
#print(qTab)
state = observation
#State 0 is the initial state. 


while complete < 10:
    done = 0
    while done == 0:
        #Time to do the Q learning. 
        if random.uniform(0,1) < epsi:
            action = env.action_space.sample()
            #print(action)
        else:
            qList = np.asarray(qTab[state])
            action = np.argmax(np.random.random(qList.shape)*(qList==qList.max()))
            #print(action)
        observation, reward, done, info = env.step(action) 
        #print("reward")
        #print(reward)
        
        oldVal = qTab[state, action]
        nextMax = np.max(qTab[observation])

        newVal = (1 - alpha) * oldVal + alpha * (reward + gamma * nextMax)
        qTab[state, action] = newVal
        state = observation
        #print(qTab[state])

        if complete == 9:
            env.render()
        #step is what advances the action
        if done:
            observation = env.reset()
            complete += 1
env.close()
