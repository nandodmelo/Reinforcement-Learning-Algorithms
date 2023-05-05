import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from ModelTF2 import ActorNetwork, CriticNetwork
from tensorboardX import SummaryWriter
from datetime import datetime
import os
import json
import copy
from Memory import Memory
import tensorflow_probability as tfp
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tensorflow.keras import backend as K
class Agent:

	def __init__(self, actions, state, gamma = 0.99, policy_clip = 0.2, epochs = 4, lr_annealing= True, lr  = 3e-4, optimizer = Adam, model = 'MLP', batch_size = 32,dir = 'models/'):
		self.gamma = gamma
		self.policy_clip = policy_clip
		self.epochs = epochs
		self.dir = dir
		self.annealing  = lr_annealing
		self.action_space = actions
		self.state_space = state
		self.lr =  lr
		self.optimizer = optimizer
		self.model = model
		self.log = datetime.now().strftime("%Y_%m_%d_%H_%M") + self.model
		self.batch_size =  batch_size
		self.LOSS_CLIPPING = 0.2
		self.ENTROPY_LOSS = 0.01
		self.memory = Memory(self.batch_size)

		self.Actor = ActorNetwork(self.state_space, self.action_space , optimizer, lr,   self.annealing , model = "MLP" )
		
		self.Critic = CriticNetwork(self.state_space, self.action_space ,optimizer, lr,   self.annealing , model = "MLP" )
		
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
		#print('ACTIONSSS',actions, rewards, len(actions), len(rewards))

		#na implantação do Phil ele pega o valor em x pra value e x+1 pra next_value, no detalhada, ele usa o values e da predict só pro ultimo ponto (len + 1), entendo que essa abordagem é melhor, calcula tudo de uma vez, é um cálculo rápido e diminui a complexidade
		#de colocar uma condição if + for, no caso do tutorial do Phil, ele calcula o next_value no choose action, no nosso caso, calculamos aqui
		value = self.Critic.Critic(state).numpy()
		next_value = self.Critic.Critic(next_state).numpy()

		advt = self.get_gaes(rewards, done, np.squeeze(value), np.squeeze(next_value))
		print('CRITICCC',np.shape(probs), np.shape(batches),  np.shape(state))
		for i in batches:
			with tf.GradientTape(persistent=True) as tape:

				states = tf.convert_to_tensor(state[i], dtype=tf.float32)
				prob = tf.convert_to_tensor(probs[i], dtype=tf.float32)
				action = tf.convert_to_tensor(actions[i], dtype=tf.float32)

				y_pred = self.Actor.Actor(states)

				dist = tfp.distributions.Categorical(prob)
				old_prob = dist.log_prob(action)

				dist = tfp.distributions.Categorical(y_pred)
				new_prob = dist.log_prob(action)
				
				ratio = tf.math.exp(new_prob - old_prob)
				
				p1 = ratio * advt[i]
				p2 = tf.clip_by_value(ratio, 1 - self.LOSS_CLIPPING, 1 + self.LOSS_CLIPPING) * advt[i]

				actor_loss = -tf.math.reduce_mean(tf.math.minimum(p1, p2))

				entropy = -(y_pred * tf.math.log(y_pred + 1e-10))
				entropy = self.ENTROPY_LOSS * K.mean(entropy)
				
				total_loss = actor_loss - entropy
				##Critic
				critic_value = self.Critic.Critic(states)
				critic_value = tf.squeeze(critic_value, 1)
				returns = tf.squeeze(advt[i], 1) + value[i]

				critic_loss = keras.losses.MSE(critic_value, returns)
				print('actor', np.shape(total_loss), total_loss, 'Critic', np.shape(critic_loss))


				

			actor_params = self.Actor.Actor.trainable_variables
			actor_grads = tape.gradient(total_loss, actor_params)

			self.Actor.Actor.optimizer.apply_gradients(
						zip(actor_grads, actor_params))
			
			critic_params = self.Critic.Critic.trainable_variables
			critic_grads = tape.gradient(critic_loss, critic_params)

			self.Critic.Critic.optimizer.apply_gradients(
						zip(critic_grads, critic_params))	

		self.memory.clear_memory()








	def save_model(self, reward):

		self.Actor.save_weights(f"Actor_{self.model}_{reward}.h5")
		self.Critic.save_weights(f"Critic_{self.model}_{reward}.h5")

	def load_model(self, reward):

		self.Actor.load_weights(self.log, f"Actor_{self.model}_{reward}.h5")
		self.Critic.load_weights(self.log, f"Critic_{self.model}_{reward}.h5")


	def act(self, state):
		
		#prediction = self.Actor.actor_predict(state)


		
		prediction = self.Actor.Actor(np.expand_dims(state, axis=0)).numpy()
	   # end = time.time()
		#np.testing.assert_array_equal(  np.squeeze(prediction), np.squeeze(pred2))
		#print('predict2', end-start)
		#start = time.time()
		action = np.random.choice(self.action_space, p=np.squeeze(prediction))
		#end = time.time()
		#print('action', end- start)

		return action, np.squeeze(prediction)

	def store_transition(self, state, action, probs, vals, reward, done):
		self.memory.store_memory(state, action, probs, vals, reward, done)
	#def store memory



