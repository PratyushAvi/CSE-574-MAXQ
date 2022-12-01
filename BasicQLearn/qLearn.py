import gym 
import gym_simplegrid
import numpy as np
import random

#Initialize 8x8 environment
my_reward_map = {
        b'E': -1.0,
        b'S': -1.0,
        b'W': -5.0,
        b'G': 5.0,
    }
env = gym.make('SimpleGrid-8x8-v0', reward_map=my_reward_map, p_noise = 0)
env.metadata["render_fps"] = 60
observation = env.reset()

#Initialize my state map and variables
actions = (0,1,2,3) #L D R U
complete = 0
gamma = 0.1
alpha = 0.95
epsi = 0.1
qTab = np.zeros([env.observation_space.n, env.action_space.n])
#print(qTab)
state = 0
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
        print(qTab[state])


        env.render()
        #step is what advances the action
        if done:
            observation = env.reset()
        if reward == 5:
            complete += 1
env.close()
