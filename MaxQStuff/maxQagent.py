import collections
import numpy as np
import random
from enum import IntEnum

class Actions(IntEnum):
    SOUTH = 0
    NORTH = 1
    EAST = 2
    WEST = 3
    PICKUP = 4
    DROPOFF = 5
    
    GO_TO_SRC = 6
    GO_TO_DST = 7
    GET_PASSENGER = 8
    PUT_PASSENGER = 9
    ROOT = 10


class MaxQAgent:
    def __init__(self, env, alpha, gamma, epsilon):
        self.env = env
        self.next_state = self.env.s
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.v = collections.defaultdict(lambda: 0)
        # self.v_copy = self.v.copy()
        self.c = collections.defaultdict(lambda: 0)
        self.done = False

        self.actions = (Actions.SOUTH, Actions.NORTH, Actions.EAST, Actions.WEST, Actions.PICKUP, Actions.DROPOFF) # south, north, east, west, pickup, dropoff
        
        self.graph = {
            # go to source
            Actions.GO_TO_SRC : (Actions.SOUTH, Actions.NORTH, Actions.EAST, Actions.WEST),
            # go to destination
            Actions.GO_TO_DST : (Actions.SOUTH, Actions.NORTH, Actions.EAST, Actions.WEST),
            # get passenger
            Actions.GET_PASSENGER : (Actions.GO_TO_SRC, Actions.PICKUP),
            # put passenger
            Actions.PUT_PASSENGER : (Actions.GO_TO_DST, Actions.DROPOFF),
            # max root : over-arching goal
            Actions.ROOT : (Actions.GET_PASSENGER, Actions.PUT_PASSENGER)
        }
        # primitive actions
        for a in self.actions:
            self.graph[a] = ()
        # print(self.graph)

    def isPrimitive(self, max_node):
        return max_node in self.actions
    
    def isTerminal(self, max_node, done):
        taxi_row, taxi_col, pass_pos, destination = list(self.env.decode(self.env.s))
        RGYB = [(0, 0), (0, 4), (4, 0), (4, 3)]
        if done:
            return True
        elif max_node == Actions.ROOT:
            return done
        elif self.isPrimitive(max_node):
            return True
        elif max_node == Actions.GO_TO_SRC:
            return pass_pos < 4 and (taxi_row, taxi_col) == RGYB[pass_pos]
        elif max_node == Actions.GO_TO_DST:
            return pass_pos > 3 and (taxi_row, taxi_col) == RGYB[destination]
        elif max_node == Actions.GET_PASSENGER:
            return pass_pos > 3
        elif max_node == Actions.PUT_PASSENGER:
            return pass_pos < 4

    def evaluateMaxNode(self, max_node, state):
        if self.isPrimitive(max_node):
            return self.v[(max_node, state)]
        else:
            umm = []
            for j in self.graph[max_node]:
                self.v[(j, state)] = self.evaluateMaxNode(j, state)
            # for j in self.graph[max_node]:
                # umm.append(self.v[(j, state)] + self.c[(max_node, state, j)])
                umm.append(self.v[(j, state)])
            return self.v[(np.argmax(umm), state)]
            
    def greedyAction(self, max_node, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            umm = []
            umm_vals = []
            for j in self.graph[max_node]:
                if self.isPrimitive(j) or not self.isTerminal(j, self.done):
                    umm.append(j)
                    umm_vals.append(self.v[(j, state)] + self.c[(max_node, state, j)])
            return umm[np.argmax(umm_vals)]

    def maxq(self, max_node, state, render):
        if self.done:
            return 0
        self.done = False
        if self.isPrimitive(max_node):
            self.next_state, reward, self.done, info = self.env.step(max_node)
            # self.next_state = observation
            # self.done = done
            self.v[(max_node, state)] += self.alpha * (reward - self.v[(max_node, state)])
            if render:
                self.env.render()
                print(list(self.env.decode(self.env.s)))
            return 1
        else:
            count = 0
            while not self.isTerminal(max_node, self.done):
                action = self.greedyAction(max_node, state)
                n = self.maxq(action, state, render)
                # self.v_copy = self.v.copy()
                self.c[(max_node, state, action)] += self.alpha * ((self.gamma ** n) * self.evaluateMaxNode(max_node, self.next_state) - self.c[(max_node, state, action)])
                count += n
                state = self.next_state
            return count
    
    def reset(self):
        self.env.reset()
        self.done = False
        self.next_state = self.env.s