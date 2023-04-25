from collections import deque

from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Dense
import random
import numpy as np
import gym

env = gym.make('CartPole-v0')
state_size = env.observation_space.shape[0]
print('------', state_size)
action_size = env.action_space.n

episodes = 1000
memory_size = 1000
batch_size = 32
class DQNagent:
    def __init__(self, state_size, action_size):
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
        self.learning_rate = 0.001
        self.model = self.model()
        
    
    def model(self):
        #criamos aqui um modelo sequencial, com 128, 64 e 32 neurônicos, nosso input size é nosso estado, e nosso output o número de ações possíveis.
        #A saída é linear, porque não estamos prevendo uma ação em si, mas a ação com o maior Q-Valor, e tomamos a ação com maior Q-Valor.
        model = Sequential()
        model.add(Dense(128, input_dim = self.state_size, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(optimizer=Adam(learning_rate = self.learning_rate), loss="mse")

        return model
        #até o compile 
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.episilon:
            return random.randrange(self.action_size)
        act_v = self.model.predict(state, verbose=0)
        return np.argmax(act_v[0])

        

    def replay(self, batch_size):

        batch  =  random.sample(self.memory, batch_size)
        

        for  state, action, reward, next_state, done in batch:
            target = reward

            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0]))
            
            target_f  = self.model.predict(state, verbose=0)
            #   print('-o-o-o-o-o-o-o-o-o',target_f, target_f[0], action)
            target_f[0][action] =  target 
            #print('---------',target_f)

            self.model.fit(state, target_f, epochs = 1, verbose=0)

        if self.episilon > self.episilon_min:
            self.episilon = self.episilon * self.episilon_decay

    def save_model(self, name):
        self.model.save_weights(name)
        
    def load_model(self, name):
        self.model.load_weights('DQN')

    #random sample with batch size from memory

    #for loop in sample 



agent = DQNagent(state_size, action_size)

done = False
acc_reward = 0
for epi in range(episodes):
    state = env.reset()
    print('BATCH--', epi, state)
    state = np.reshape(state[0], [1, state_size])


    for time in range(5000):
        action = agent.act(state)
        next_state, reward, done, terminated, _ =  env.step(action)
        print('----', time, next_state, reward, done, terminated, _)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state, action, reward, next_state, done)

        state = next_state
        
        if done:
            acc_reward += time
            print('episode: {}/{}, score: {}, e: {:.2}, acc_rew: {}'.format(epi, episodes, time, agent.episilon, acc_reward/(epi+1) ))
            break

        
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
    
    if epi % 50 == 0:
        agent.save_model('DQN' + 'Weights' + '{:04d}'.format(epi) + '.hdf5')