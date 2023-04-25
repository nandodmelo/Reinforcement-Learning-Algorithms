import numpy as np
import gym
from collections import deque
import random
from scipy.special import softmax

random.seed(10)

class PortEnv(gym.Env):

	def __init__(self, df, cash=10000, assets=[], lookback_window_size = 50):
		self.df = df
		self.total_steps = len(self.df)
		self.cash = cash
		self.assets  = assets
		self.weights = np.zeros(len(self.assets))
		self.quants = np.zeros(len(self.assets))
		self.prices = np.zeros(len(self.assets))
		self.market_state = dict.fromkeys(self.assets)
		self.lookback_window_size = lookback_window_size

		self.action_space = gym.spaces.Box(low=-1,
							   high=1, shape=([4]),
							   dtype=np.float32)
		#self.df_normalized = 
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.lookback_window_size, 61), dtype=np.float32)

		self.market = deque(maxlen = self.lookback_window_size)

		self.order = deque(maxlen = self.lookback_window_size)




	def reset(self, env_steps_size = 0):
		self.balance = self.cash
		self.net_worth = self.cash
		self.prev_net_worth = self.cash
		self.weights = np.zeros(len(self.assets))
		self.quants = np.zeros(len(self.assets))
		self.prices = np.zeros(len(self.assets))

		if env_steps_size > 0:
			self.start_step = random.randint(self.lookback_window_size, len(self.df) - env_steps_size-1)
			self.end_step = self.start_step + env_steps_size
		

		for i in reversed(range(0, self.lookback_window_size)):
			self.order.append([self.net_worth/self.cash] +
							  [number for number in self.quants] +
							  [number for number in self.weights])

		self.quants[0] = self.balance/self.df[0,2,0]
		for j in range(0,len(self.assets)):
			
			#print('---', j)
			self.market_state[str(j)] = deque(maxlen=self.lookback_window_size)
				 
			for i in reversed(range(self.lookback_window_size)):
				
				current_step = self.start_step - i
				#print('Asset', self.assets[j],self.df[current_step, :,j] )
				self.market_state[str(j)].append(self.df[current_step, :,j])


		state = np.concatenate(([self.market_state[str(x)] for x in range(0,len(self.assets))]), axis=1) 
		state = np.concatenate((state, self.order) , axis=1)
		state = np.float32(state)
		#print(state.shape)
		return state
	

	def _next_observation(self):
		
		
		#Nesse passo, ele atualiza o estado com o ponto mais recent que foi utilizado em step, por exemplo, no Step ele pega o ponto seguinte após o market history, logo se o market history vai até t, no step ele pega o ponto t+1, no next observation, ele da append desse ponto. Porém como tamanho máximo é 10, quando ele da append
		# ele perde o ponto mais antigo, e o novo é adicionado. No step, ele adiciona +1 no self.current_step, devido a isso a janela vai andando.

		for j in range(0, len(self.assets)):
			
			
		   
			self.market_state[str(j)].append(self.df[self.start_step, :, j])
			
			
			
		
		obs = np.concatenate(([self.market_state[str(x)] for x in range(0,len(self.assets))]), axis=1) 
		obs = np.concatenate((obs, self.order) , axis=1)
		obs = np.float32(obs)
		return obs


	def step(self, action):

		self.start_step += 1
		#print('networth',self.net_worth )

		self.prev_net_worth = self.net_worth

		#Pega os preços no dia atual
		prices = np.array([self.df[self.start_step,2,x] for x in range(0,len(self.assets))])
		np.set_printoptions(suppress = True,  formatter = {'float_kind':'{:f}'.format})
		#print('prices', prices )
		#Calcula o valor do portfólio no dia, ou seja ele pega as quantidades compradas no dia anterior e atualiza o valor do portfólio considerando o dia de hj (D+1)
		self.net_worth = np.dot(self.quants, prices)
		#print('self.net_worth11', self.net_worth )
		#Transforma de uma tanh pra uma softmax, ficando dentro do range 0 e 1, possibilitando comprar ações de maneira que o portfolio some 1
		prediction = softmax(action)
		
		#Compra as ações baseado nos pesos devolvidos pelo modelo (action/prediction)

		self.quants = [self.net_worth*prediction[x]/prices[x] for x in range(0,len(self.assets))]


		self.order.append([self.net_worth/self.cash] + 
									[number/sum(self.quants) for number in self.quants] + prediction.tolist())

		reward = np.log(self.net_worth/self.prev_net_worth)
		#print('quants1', self.quants)
		if self.net_worth <= self.cash/2:
			done = True
		else:
			done = False

		obs = self._next_observation() 

		info = {}
		return obs, reward, done, info


	def render(self):
		print(f'Step: {self.start_step}, Net Worth: {self.net_worth}')


