import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from Models import ActorNetwork, CriticNetwork
from tensorboardX import SummaryWriter
from datetime import datetime
import os
import json
import copy
from Memory import Memory
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)
    

    def replay(self):

        state, actions, probs, next_state, rewards, done, _ =  self.memory.generate_batches()
        #print('ACTIONSSS',actions, rewards, len(actions), len(rewards))

        #na implantação do Phil ele pega o valor em x pra value e x+1 pra next_value, no detalhada, ele usa o values e da predict só pro ultimo ponto (len + 1), entendo que essa abordagem é melhor, calcula tudo de uma vez, é um cálculo rápido e diminui a complexidade
        #de colocar uma condição if + for, no caso do tutorial do Phil, ele calcula o next_value no choose action, no nosso caso, calculamos aqui
        value = self.Critic.critic_predict(state)
        next_value = self.Critic.critic_predict(next_state)

        advt, target = self.get_gaes(rewards, done, np.squeeze(value), np.squeeze(next_value))
        print('TARGET', np.shape(target))
        y_true = np.hstack([advt, probs, actions])
        
        # training Actor and Critic networks
        a_loss = self.Actor.Actor.fit(state, y_true, epochs=self.epochs, verbose=0, shuffle=True, batch_size=self.batch_size)
        c_loss = self.Critic.Critic.fit(state, target, epochs=self.epochs, verbose=0, shuffle=True, batch_size=self.batch_size)

        #self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
        #self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
        #self.replay_count += 1
        self.memory.clear_memory()

        return np.sum(a_loss.history['loss']), np.sum(c_loss.history['loss'])






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

        return action, prediction

    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)
    #def store memory



