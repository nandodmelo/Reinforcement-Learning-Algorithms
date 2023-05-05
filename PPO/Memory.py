import numpy as np

class Memory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size
    




    def generate_batches(self, shuffle =  False):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)

        if shuffle == True:
        
            np.random.shuffle(indices)


        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return np.vstack(np.array(self.states)),\
            np.vstack(np.array(self.actions)),\
            np.vstack(np.array(self.probs)),\
            np.vstack(np.array(self.vals)),\
            np.vstack(np.array(self.rewards)),\
            np.vstack(np.array(self.dones)),\
            batches
    


    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []