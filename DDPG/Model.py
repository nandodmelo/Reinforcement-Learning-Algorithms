from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Concatenate
from keras.optimizers import Adam, RMSprop
import tensorflow as tf

class NN():
    def __init__(self, state_shape, action_space):
        
        self.state_shape = state_shape
        self.action_space = action_space
        
    

        # 'Dense' is the basic form of a neural network layer
        # Input Layer of state size(4) and Hidden Layer with 512 nodes
    def Actor(self):
        X_input = Input(self.state_shape)
        X = Dense(256, activation="relu")(X_input)

            # Hidden layer with 256 nodes
        X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
            
            # Hidden layer with 64 nodes
        X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

            # Output Layer with # of actions: 2 nodes (left, right)
        #X = Dense(self.action_space, activation="softmax", kernel_initializer='he_uniform')(X)

        X = Dense(self.action_space, activation="tanh", kernel_initializer='he_uniform')(X)
        model = Model(inputs = X_input, outputs = X, name='CartPole_DDQN_model')
        model.compile(optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01))
 

        return model
    def Reduce_mean_loss(self,  y_true, y_pred):
        # Defined in https://arxiv.org/abs/1707.06347
        total_loss = -tf.math.reduce_mean(y_true)
        return total_loss
    
    
    def Critic(self):
        #State Input
        state_input = Input(self.state_shape)
        # Hidden layer with 256 nodes
        X_state = Dense(32, activation="relu", kernel_initializer='he_uniform')(state_input)

        #Action input            
        action_input = Input(self.action_space)


            # Output Layer with # of actions: 2 nodes (left, right)
        X_action = Dense(32, activation="relu", kernel_initializer='he_uniform')(action_input)
        concat = Concatenate()([X_state, X_action])


        out = Dense(256, activation="relu", kernel_initializer='he_uniform')(concat)
        out = Dense(256, activation="relu", kernel_initializer='he_uniform')(out)



        X = Dense(1, kernel_initializer='he_uniform')(out)
        model = Model(inputs = [state_input, action_input], outputs = X)
        model.compile(optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01))
 

        return model

        

    
