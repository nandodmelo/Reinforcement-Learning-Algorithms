from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam, RMSprop

class NN():
    def __init__(self, state_shape, action_space):
        
        self.state_shape = state_shape
        self.action_space = action_space
        
    

        # 'Dense' is the basic form of a neural network layer
        # Input Layer of state size(4) and Hidden Layer with 512 nodes
    def MLP(self,A2C_type = ''):
        X_input = Input(self.state_shape)
        X = Dense(256, activation="relu")(X_input)

            # Hidden layer with 256 nodes
        X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
            
            # Hidden layer with 64 nodes
        X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

            # Output Layer with # of actions: 2 nodes (left, right)
        X = Dense(self.action_space, activation="softmax", kernel_initializer='he_uniform')(X)

        if A2C_type == 'Actor':
            X = Dense(self.action_space, activation="softmax", kernel_initializer='he_uniform')(X)
            model = Model(inputs = X_input, outputs = X, name='CartPole_DDQN_model')
            model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01))
        else:
            X = Dense(1, kernel_initializer='he_uniform')(X)
            model = Model(inputs = X_input, outputs = X, name='CartPole_DDQN_model')
            model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01))        

        return model

        

    
