import numpy as np

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def generate_batches(self):


        return np.vstack(np.array(self.states)), np.vstack(np.array(self.actions)),np.vstack(np.array(self.rewards))

    


    def store_memory(self, state, action, reward, action_size):
        action_onehot = np.zeros([action_size])
        action_onehot[action] = 1
        self.actions.append(action_onehot)
        self.states.append(state)
        self.rewards.append(reward)


    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
