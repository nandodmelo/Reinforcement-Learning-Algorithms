from collections import deque

from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Dense
import random
import numpy as np
import gym
from Memory import Memory

from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam, RMSprop
from Model import NN
env = gym.make('LunarLander-v2')
state_size = env.observation_space.shape[0]
print('------', state_size)
action_size = env.action_space.n

episodes = 1000
memory_size = 1000
batch_size = 32
class REINFORCEagent:
	def __init__(self, state_size, action_size):
		#state_size é a dimensão dos dados utilizadas para descrever o estado.
		self.state_size = state_size

		#Número de ações possíveis.
		self.action_size = action_size

		#Deque nada mais é que uma lista com tamanho fixo, que quando adicionamos um valor tiramos outro.
		self.memory = deque(maxlen=memory_size)

		#Parâmetros gamma (fator de desconto para recompensas futuras)

		#learning_rate, taxa de aprendizado da nossa rede neural
		#model nada mais é o nosso modelo.
		self.gamma = 0.99
		print('aaa', self.state_size, self.action_size)
		#self.Model = NN(self.state_size, self.action_size)
		self.Model = self.Model.MLP()
		
				
		self.memory = Memory()

	
	
		
	
	def discount_rewards(self, reward):
		# Compute the gamma-discounted rewards over an episode

		G_total = 0
		G = np.zeros_like(reward)
		for i in reversed(range(0, len(reward))):
			G_total = G_total*self.gamma + reward[i]
			
			G[i] = G_total


		mean = np.mean(G)
		std = np.std(G)
		G = (G -mean)/std

		return G

		

		return discounted_r
		#até o compile 
	
	def remember(self, state, action,  reward):
		self.memory.store_memory(state, action,reward, self.action_size)

	def act(self, state):

		prediction = self.Model(state).numpy()

		action = np.random.choice(self.action_size, p=np.squeeze(prediction))

		return action

		

	def replay(self):
		#pega um sample 
		
		state, actions, rewards =  self.memory.generate_batches()

		discounted_r = self.discount_rewards(rewards)
		
		self.Model.fit(state, actions, sample_weight = discounted_r, epochs = 4, verbose=0)

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
		agent.remember(state, action, reward)

		state = next_state
		
		if done:
			
			acc_reward += time
			print('episode: {}/{}, score: {}, acc_rew: {}'.format(epi, episodes, time,  acc_reward/(epi+1) ))
			break

	agent.replay()
	if epi % 50 == 0 and epi > 0:
		agent.save_model('DQN' + 'Weights' + '{:04d}'.format(epi) + '.hdf5')