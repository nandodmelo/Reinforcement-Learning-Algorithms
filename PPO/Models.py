import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D#, LSTM
from tensorflow.compat.v1.keras.layers import CuDNNLSTM as LSTM # only for GPU
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam


class ActorNetwork():

    def __init__(self, state_shape, action_space, optimizer,lr = 3e-4, lr_annealing= False,  model = "MLP" ):
        X_input = Input(state_shape)
        self.action_space = action_space
        self.model = model
             

        

              
        if self.model=="CNN":
            X = Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh")(X_input)
            X = MaxPooling1D(pool_size=2)(X)
            X = Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh")(X)
            X = MaxPooling1D(pool_size=2)(X)
            X = Flatten()(X)
            X = Dense(64, activation="relu")(X)
            output = Dense(self.action_space, activation="tanh")(X)

        elif self.model=="MLP":
            
            X = Dense(64, activation="tanh", kernel_initializer = initializers.Orthogonal(gain=np.sqrt(2)), bias_initializer=initializers.Zeros())(X_input)
            X = Dense(64, activation="tanh", kernel_initializer = initializers.Orthogonal(gain=np.sqrt(2)), bias_initializer=initializers.Zeros())(X)
            output = Dense(self.action_space, activation="softmax")(X)


        

        self.Actor = Model(inputs = X_input, outputs = output)
        self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(lr=lr))
        print(self.Actor.summary())



    def ppo_loss(self, y_true, y_pred):
        # Defined in https://arxiv.org/abs/1707.06347
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.01
        
        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))
        
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)
        
        total_loss = actor_loss - entropy

        return total_loss
    
    def actor_predict(self, state):
        #print('---+++111', type(state))
        return self.Actor.predict(np.expand_dims(state, axis=0), verbose=0)



class CriticNetwork():

    def __init__(self, state_shape, action_space, optimizer ,lr = 3e-4, lr_annealing= True, model = "MLP" ):
        X_input = Input(state_shape)
        self.action_space = action_space
        self.model = model
        self.annealing  = lr_annealing

        
              
        if self.model=="CNN":
            X = Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh")(X_input)
            X = MaxPooling1D(pool_size=2)(X)
            X = Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh")(X)
            X = MaxPooling1D(pool_size=2)(X)
            X = Flatten()(X)
            X = Dense(64, activation="relu")(X)
            output = Dense(self.action_space, activation="linear")(X)

        else:
            
            X = Dense(64, activation="tanh", kernel_initializer = initializers.Orthogonal(gain=np.sqrt(2)), bias_initializer=initializers.Zeros())(X_input)
            X = Dense(64, activation="tanh", kernel_initializer = initializers.Orthogonal(gain=np.sqrt(2)), bias_initializer=initializers.Zeros())(X)
            output = Dense(1, activation="linear")(X)


        self.Critic = Model(inputs = X_input, outputs = output)
        self.Critic.compile(loss=self.critic_loss, optimizer=optimizer(lr=lr))

    def critic_loss(self, y_true, y_pred):
        value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
        return value_loss
    
    def critic_predict(self, state):
        return self.Critic.predict(state, verbose=0)