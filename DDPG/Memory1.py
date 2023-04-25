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
        self.mem_counter = 0


        
    




    def generate_batches(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        
        #print('BATCH', batch)
        return np.squeeze(np.array(self.states)[batch]),\
            np.squeeze(np.array(self.actions)[batch]),\
            np.squeeze(np.array(self.next_states)[batch]),\
            np.squeeze(np.array(self.rewards)[batch]),\
            np.squeeze(np.array(self.dones)[batch])
    


    def store_memory(self, state, action, next_states, reward, done):
        
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_states)
        self.rewards.append(reward)
        self.dones.append(done)
        self.mem_counter += 1


    def clear_memory(self):
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.dones = []
