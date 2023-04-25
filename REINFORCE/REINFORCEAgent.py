from collections import deque

from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Dense
import random
import numpy as np
import gym
from Memory import Memory
from Model import MLP

env = gym.make('LunarLander-v2')
state_size = env.observation_space.shape[0]

action_size = env.action_space.n

episodes = 1000
memory_size = 1000
batch_size = 32
class REINFORCEagent:
    def __init__(self, state_size, action_size, model):
        #state_size é a dimensão dos dados utilizadas para descrever o estado.
        self.state_size = state_size

        #Número de ações possíveis.
        self.action_size = action_size

        #Deque nada mais é que uma lista com tamanho fixo, que quando adicionamos um valor tiramos outro.
        self.memory = deque(maxlen=memory_size)

        #Parâmetros gamma (fator de desconto para recompensas futuras)
        #episilon é a taxa que nosso agente irá explorar o ambiente, ou seja, tomar ações aleatórias, buscando aprende.
        #episilon_decay é a taxa de decaimento do nosso episilon(exploração), logo com o passar do tempo ele diminui a exploração.
        #episilon_min taxa de exploração mínima.
        #learning_rate, taxa de aprendizado da nossa rede neural
        #model nada mais é o nosso modelo.
        self.gamma = 0.95
        self.episilon = 0.99
        self.episilon_decay = 0.995
        self.episilon_min = 0.2
        self.Model = model(self.state_size, self.action_size)
        
        self.memory = Memory(self.batch_size, action =self.action_size)
        
    
    def discount_rewards(self, reward):
        # Compute the gamma-discounted rewards over an episode
        gamma = 0.99    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            if reward[i] != 0: # reset the sum, since this was a game boundary (pong specific!)
                running_add = 0
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= np.std(discounted_r) # divide by standard deviation
        return discounted_r
        #até o compile 
    
    def remember(self, state, action, next_state, reward,  done):
        self.memory.store_memory(state, action, next_state, reward, done)

    def act(self, state):

        prediction = self.Model(np.expand_dims(state, axis=0)).numpy()

        action = np.random.choice(self.action_size, p=np.squeeze(prediction))

        return action, prediction

        

    def replay(self):
        #pega um sample 
        state, actions, probs, next_state, rewards, done, _ =  self.memory.generate_batches()

        discounted_r = self.discount_rewards(self.rewards)
        
        self.Model.fit(state, actions, sample_weight = discounted_r, epochs = 4)

        self.memory.clear_memory()



    def save_model(self, name):
        self.model.save_weights(name)
        
    def load_model(self, name):
        self.model.load_weights('REINFORCE')

    #random sample with batch size from memory

    #for loop in sample 



agent = REINFORCEagent(state_size, action_size)

done = False
acc_reward = 0
for epi in range(episodes):
    state = env.reset()
    state = np.reshape(state[0], [1, state_size])


    for time in range(5000):
        action = agent.act(state)
        
        next_state, reward, done, terminated, _ =  env.step(action)

        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        #print('SHAPe', state.shape, next_state.shape)
        agent.remember(state, action, next_state, reward, done)

        state = next_state
        
        if done:
            agent.copy_network()
            acc_reward += time
            print('episode: {}/{}, score: {}, e: {:.2}, acc_rew: {}'.format(epi, episodes, time,  acc_reward/(epi+1) ))
            break

    agent.replay()
    print('epi', epi)
    if epi % 50 == 0 and epi > 0:
        agent.save_model('DQN' + 'Weights' + '{:04d}'.format(epi) + '.hdf5')