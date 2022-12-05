import gym 
import gym_simplegrid
import numpy as np
import random
import pandas as pd

from maxQagent import MaxQAgent
from maxQagent import Actions

env = gym.make('Taxi-v3')
observation = env.reset()

#Initialize my state map and variables
complete = 0
gamma = 1
alpha = 0.95
epsi = 0.001
#Set my Q Table + initial state
qTab = np.zeros([env.observation_space.n, env.action_space.n])
state = observation

agent = MaxQAgent(env, alpha, gamma, epsi)

collected_data = {'success': [], 'reward': [], 'actions': []}

runs = 5000

counter = 1
for i in range(runs):
    print(f"{i}/{runs}", end='\r')
    agent.reset()
    steps = agent.maxq(Actions.ROOT, env.s, (i == runs-1))
    if (i == runs-1):
        print(steps)

        
    x = list(env.decode(env.s))
    # if success (compare passenger position and destination)
    if (x[2] == x[3]):
        counter += 1
        collected_data['success'].append(1)
    else:
        collected_data['success'].append(0)
    collected_data['reward'].append(agent.total_reward)
    collected_data['actions'].append(steps)

# pandas
df = pd.DataFrame.from_dict(collected_data)
print(df)

print(counter/runs*100, "percent successes")
env.close()


'''
NOTES:

    Must abstract problem:
        1. Get Passenger
        2. Dropoff Passenger
    
    env.decode(observation) gives me state info
        taxi-row, taxi-column, passenger-location, passender-destination 

'''