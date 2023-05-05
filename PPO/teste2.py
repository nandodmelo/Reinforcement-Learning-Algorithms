import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import tensorflow as tf
probs = [[0.49992284, 0.5000772 ],
 [0.50596946, 0.49403054],
 [0.5000592,  0.49994084],
 [0.49376667, 0.50623333],
 [0.49986768, 0.5001324 ]]
actions = [1,0,0,1,0]


probs = tf.convert_to_tensor(probs)
actions = tf.convert_to_tensor(actions)

dist = tfp.distributions.Categorical(probs)
print(dist.log_prob(actions), probs*actions)