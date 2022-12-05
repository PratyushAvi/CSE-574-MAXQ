# Josh's code for the taxi problem basic Q Table
#
#
import gym 
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as mpl

env = gym.make('Taxi-v3')
observation = env.reset()

#Initialize my state map and variables
complete = 0
gamma = 0.1
alpha = 0.95
epsi = 0.001
#Set my Q Table + initial state
qTab = np.zeros([env.observation_space.n, env.action_space.n])
state = observation

# 100 iterations
counter = 0

collected_data = {'success': [], 'reward': [], 'actions': []}

runs = 2000
for i in range(runs):
    print(f"{i}/{runs}", end='\r')
    done = 0
    steps = 0
    total_reward = 0
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
        
        steps += 1
        total_reward += reward

        #Update Q Vals
        curVal = qTab[state][action]
        nextMax = np.max(qTab[observation])
        newVal = (1 - alpha) * curVal + alpha * (reward + gamma * nextMax)
        qTab[state][action] = newVal
        state = observation
        #Render for last 9 sets
        if i == runs-1:
            env.render()
            print(list(env.decode(env.s)))
            if done:
                print(steps)

        if done:
            # data collection
            x = list(env.decode(env.s))
            # if success
            if (x[2] == x[3]):
                counter += 1
                collected_data['success'].append(1)
            else:
                collected_data['success'].append(0)
            collected_data['reward'].append(total_reward)
            collected_data['actions'].append(steps)

            # print("------------------------------------------------- RESET ---------------------------------------------------")
            complete += 1
            observation = env.reset()

# pandas
df = pd.DataFrame.from_dict(collected_data)
print(df)
print(counter/runs*100, "percent successes")
env.close()


#Josh moment for plots and data and stuff
dfsub1 = df[["actions"]]
dfsub1 = dfsub1.plot.line(legend=False, title="Q Learn Actions")
dfsub1.set_xlabel("Epochs")
dfsub1.set_ylabel("Actions")
dfsub2 = df[["reward"]]
dfsub2 = dfsub2.plot.line(legend=False, title="Q Learn Reward")
dfsub2.set_xlabel("Epochs")
dfsub2.set_ylabel("Rewards")
mpl.show()
