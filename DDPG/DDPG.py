from collections import deque

from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Dense
import random
import numpy as np
import gym
from Memory1 import Memory
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam, RMSprop
from Model import NN
env = gym.make('BipedalWalker-v3')
state_size = env.observation_space.shape[0]
print('------', state_size)
action_size = env.action_space.shape[0]
episodes = 1000
memory_size = 1000
batch_size = 32
class DDPGagent:
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
		
		self.Model = NN(self.state_size, self.action_size)
		self.Actor = self.Model.Actor()
		self.Actor_target = self.Model.Actor()

		self.Critic = self.Model.Critic()
		self.Critic_target = self.Model.Critic()
		
		
		
				
		self.memory = Memory(10000)

	
	
		
	
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

		#até o compile 
	
	def remember(self, state, action, next_state, reward,  done):
		self.memory.store_memory(state, action, next_state, reward, done)

	def act(self, state):

		action = self.Actor(state).numpy()
		action = np.random.normal(0, 0.1, len(action)) +  action

		#action = np.random.choice(self.action_size, p=np.squeeze(prediction))

		return action

	def polyak_update(self, tau = 0.001):
		for (target, main) in zip(self.Actor_target.variables, self.Actor.variables):
			target.assign(main * tau + target * (1 - tau))
		for (target, main) in zip(self.Critic_target.variables, self.Critic.variables):
			target.assign(main * tau + target * (1 - tau))

	def replay(self):
		#pega um sample 
		batch_size = 64

		state, action, next_state,  reward, done =  self.memory.generate_batches(batch_size = batch_size)
		#print('SHAPEE',np.shape(state), np.shape(action), np.shape(reward), np.shape(next_state))

		state = tf.convert_to_tensor(state, dtype=tf.float32)
		next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
		reward = tf.convert_to_tensor(reward, dtype=tf.float32)
		action = tf.convert_to_tensor(action, dtype=tf.float32)
		##Critic
		loss_Q = tf.keras.losses.MeanSquaredError()
		
		policy_t = self.Actor_target(next_state).numpy()

		with tf.GradientTape() as tape:
			##Calculando y do Critic e atualizando o Critic

			critic_nn = self.Critic([state, policy_t], training=True)
			y = np.squeeze(reward) + self.gamma*(self.Critic_target([next_state, policy_t], training=True))*np.squeeze(done)
			critic_loss = loss_Q(y, critic_nn)


		gradients = tape.gradient(critic_loss, self.Critic.trainable_variables)

		##Atualiza os pesos do modelo
		self.Critic.optimizer.apply_gradients(zip(gradients, self.Critic.trainable_variables))

		##Actor
		with tf.GradientTape() as tape:
			##Calculando y do Critic e atualizando o Critic
			policy_nn = self.Actor(state, training=True)	
			critic_nn = self.Critic([state, policy_nn], training=True)
			actor_loss = -tf.math.reduce_mean(critic_nn)

			

		gradients = tape.gradient(actor_loss, self.Actor.trainable_variables)

			# Atualiza os pesos do modelo
		self.Actor.optimizer.apply_gradients(zip(gradients, self.Actor.trainable_variables))

		#Update by polyak #Média Model Exponencial	
		self.polyak_update()









	#def save_model(self, name):
		#self.model.save_weights(name)
		
	#def load_model(self, name):
		#self.model.load_weights('A2C')

	#random sample with batch size from memory

	#for loop in sample 



agent = DDPGagent(state_size, action_size)

done = False
acc_reward = 0
for epi in range(episodes):
	state = env.reset()
	state = np.reshape(state[0], [1, state_size])


	for time in range(5000):
		action = agent.act(state)
		#print('ACAO', action)
		next_state, reward, done, terminated, _ =  env.step(action[0])
		#print('STATE----',time)
		reward = reward if not done else -10
		next_state = np.reshape(next_state, [1, state_size])
		agent.remember(state, action, next_state, reward, done)

		state = next_state
		#@print('done', done, terminated)
		if done :
			
			acc_reward += time
			print('episode: {}/{}, score: {}, acc_rew: {}'.format(epi, episodes, time,  acc_reward/(epi+1) ))
			break

	agent.replay()
	if epi % 50 == 0 and epi > 0:
		agent.save_model('DQN' + 'Weights' + '{:04d}'.format(epi) + '.hdf5')