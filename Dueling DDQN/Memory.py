import numpy as np
from collections import deque

class Memory:
    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.states = deque(maxlen = self.mem_size)
        self.actions = deque(maxlen = self.mem_size)
        self.next_states = deque(maxlen = self.mem_size)
        self.rewards = deque(maxlen = self.mem_size)
        self.dones = deque(maxlen = self.mem_size)
        self.mem_size  = 0

        
    




    def generate_batches(self, batch_size, shuffle =  False):

        batch = np.random.choice(self.mem_size, batch_size, replace=False)


        return np.vstack(np.array(self.states)[batch]),\
            np.vstack(np.array(self.actions)[batch]),\
            np.vstack(np.array(self.next_states)[batch]),\
            np.vstack(np.array(self.rewards)[batch]),\
            np.vstack(np.array(self.dones)[batch])
    


    def store_memory(self, state, action, next_states, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_states)
        self.rewards.append(reward)
        self.dones.append(done)
        self.mem_size += 1

    def clear_memory(self):
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.dones = []
