import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from Models import ActorNetworkContinuous, CriticNetworkContinuous
from tensorboardX import SummaryWriter
from datetime import datetime
import os
import json
import copy
from Memory import Memory
import time
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import backend as K
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
class Agent:

	def __init__(self, actions, state, epsilon = 1e-8,  gamma = 0.99, policy_clip = 0.2, epochs = 10, lr_annealing= True, lr  = 3e-4, optimizer = Adam, model = 'MLP', batch_size = 64,dir = 'models/'):
		self.gamma = gamma
		self.policy_clip = policy_clip
		self.epochs = epochs
		self.dir = dir
		self.epsilon = epsilon
		self.annealing  = lr_annealing
		self.action_space = actions
		self.state_space = state
		self.lr =  lr
		self.optimizer = optimizer
		self.model = model
		self.log = datetime.now().strftime("%Y_%m_%d_%H_%M") + self.model
		self.batch_size =  batch_size
		self.LOSS_CLIPPING = 0.2
		self.ENTROPY_LOSS = 1e-3
		self.memory = Memory(self.batch_size)

		self.Actor = ActorNetworkContinuous(self.state_space, self.action_space , optimizer, lr,   self.annealing , model = "MLP" )
		
		self.Critic = CriticNetworkContinuous(self.state_space, self.action_space ,optimizer, lr,   self.annealing , model = "MLP" )
		
	def create_writer(self, initial_balance, normalize_value, train_episodes):
		self.replay_count = 0
		self.writer = SummaryWriter('runs/' + self.log)

		# Create folder to save models
		if not os.path.exists(self.log):
			os.makedirs(self.log)

		self.start_training_log(initial_balance, normalize_value, train_episodes)
			
	def start_training_log(self, initial_balance, normalize_value, train_episodes):      
		# save training parameters to Parameters.json file for future
		current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
		params = {
			"training start": current_date,
			"initial balance": initial_balance,
			"training episodes": train_episodes,
			"lr annealing": self.annealing,
			"lr": self.lr,
			"epochs": self.epochs,
			"normalize value": normalize_value,
			"model": self.model,
			"saving time": "",
			"Actor name": "",
			"Critic name": "",
		}
		with open(self.log+"/Parameters.json", "w") as write_file:
			json.dump(params, write_file, indent=4)

	#gae lambda, passar no GAE

	def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.9, normalize=True):
		deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
		deltas = np.stack(deltas)
		gaes = copy.deepcopy(deltas)
		for t in reversed(range(len(deltas) - 1)):
			gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

		if normalize:
			gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
		return np.vstack(gaes)
	

	def replay(self):

		state, actions, probs, next_state, rewards, done, batches =  self.memory.generate_batches()


		#na implantação do Phil ele pega o valor em x pra value e x+1 pra next_value, no detalhada, ele usa o values e da predict só pro ultimo ponto (len + 1), entendo que essa abordagem é melhor, calcula tudo de uma vez, é um cálculo rápido e diminui a complexidade
		#de colocar uma condição if + for, no caso do tutorial do Phil, ele calcula o next_value no choose action, no nosso caso, calculamos aqui
		value = self.Critic.Critic(state).numpy()
		next_value = self.Critic.Critic(next_state).numpy()

		advt = self.get_gaes(rewards, done, np.squeeze(value), np.squeeze(next_value))
		
		for _ in range(0, self.epochs):
			for i in batches:
				with tf.GradientTape(persistent=True) as tape:
					states = state[i]
					old_prob = tf.convert_to_tensor(probs[i], dtype=tf.float32)
					

					mu_sigma = self.Actor.Actor(state)
					mu = mu_sigma[:, 0:self.action_space]
					new_prob_ = mu_sigma[:, self.action_space:]
					probs_ =  tfp.distributions.Normal(mu, scale=new_prob_)
					mu = probs_.sample()
					new_prob = probs_.log_prob(mu)
					#print('Newlogprob', np.shape(tf.reduce_sum(new_prob,axis=0)), np.shape(old_prob))

					#print('TOTAL LOSS12', new_prob)

					
					ratio = tf.math.exp(tf.reduce_sum(new_prob,axis=1, keepdims=True) - old_prob)
					p1 = ratio * advt[i]
					
					p2 = tf.clip_by_value(ratio, 1 - self.LOSS_CLIPPING, 1 + self.LOSS_CLIPPING) * advt[i]
					#print('TOTAL LOSS132', np.shape(p2))

					actor_loss = -tf.math.reduce_mean(tf.math.minimum(p1, p2))

					#Calculo da Entropia para uma distribuição normal/gaussiana
					#https://pytorch.org/docs/stable/_modules/torch/distributions/normal.html#Normal.entropy
					entropy = np.sum(0.5 + 0.5 * np.log(2 * np.pi) + np.log(new_prob_),axis=0)
					
					#entropy = -tf.math.reduce_mean(mu * (tf.math.log(((self.action_limits(mu, -1, 1) + 1) / 2)+0.0001)))
					
					entropy = self.ENTROPY_LOSS * np.mean(entropy)
				
					total_loss = actor_loss - entropy
					#print('ENTRPY', np.shape(total_loss))
					#print('TOTAL LOSS', actor_loss, entropy, 'Total', total_loss)
					##Critic

					critic_value = self.Critic.Critic(states)
					critic_value = tf.squeeze(critic_value, 1)
					returns = tf.squeeze(advt[i], 1) + value[i]
					#print('----',value[i],critic_value )
					critic_loss = tf.math.reduce_mean(keras.losses.MSE(critic_value, returns))
					#print(critic_loss, 'loss')
					


					

				actor_params = self.Actor.Actor.trainable_variables
				actor_grads = tape.gradient(total_loss, actor_params)

				self.Actor.Actor.optimizer.apply_gradients(
							zip(actor_grads, actor_params))
				
				critic_params = self.Critic.Critic.trainable_variables
				critic_grads = tape.gradient(critic_loss, critic_params)

				self.Critic.Critic.optimizer.apply_gradients(
							zip(critic_grads, critic_params))	



		#self.replay_count += 1
		self.memory.clear_memory()






	def save_model(self, reward):

		self.Actor.save_weights(f"Actor_{self.model}_{reward}.h5")
		self.Critic.save_weights(f"Critic_{self.model}_{reward}.h5")

	def load_model(self, reward):

		self.Actor.load_weights(self.log, f"Actor_{self.model}_{reward}.h5")
		self.Critic.load_weights(self.log, f"Critic_{self.model}_{reward}.h5")


	def act(self, state):
		
		#prediction = self.Actor.actor_predict(state)


		#print('SHAPESTATE', np.shape(np.expand_dims(state, axis=0)))
		mu_sigma = self.Actor.Actor(state)
		mu = mu_sigma[:, 0:self.action_space]
		sigma = mu_sigma[:, self.action_space:]
		#print('shapo', np.shape(mu), np.shape(sigma))
		probs =  tfp.distributions.Normal(mu, scale=sigma)
		#$print('1',mu,'2', sigma,'3', probs,'4', probs.sample())
		mu = probs.sample()
		
		#print('PROBS', probs, mu)
		# probs =  tfp.distributions.MultivariateNormalDiag(mu, scale_diag=sigma)
		# mu = probs.sample()
		# print('CAL', np.shape(probs), np.shape(mu))
		return mu, tf.reduce_sum(probs.log_prob(mu), axis=1, keepdims=True)

	def store_transition(self, state, action, probs, vals, reward, done):
		
		self.memory.store_memory(state, action, probs, vals, reward, done)
	#def store memory

	def action_limits(self, action, min, max):
		return np.squeeze(np.clip(action, min, max))


	def reward_scaling(self, reward):
		# Compute the gamma-discounted rewards over an episode
		G_total = 0
		G = np.zeros_like(reward)
		for i in reversed(range(0, len(reward))):
			G_total = G_total*self.gamma + reward[i]
			
			G[i] = G_total


		# mean = np.mean(G)
		# std = np.std(G)
		# G_discounted = (G -mean)/std
				
		scaled_reward = reward[-1] /  np.sqrt(np.var(G) + self.epsilon)

		return scaled_reward
