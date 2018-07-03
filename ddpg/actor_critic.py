import tensorflow as tf
import numpy as np
from utils import neural_network

class ActorCritic:

    def __init__(self,
                  hidden_size,
                  num_layers,
                  obs_dims,
                  action_dims,
                  ):

        self.obs_pl = tf.placeholder(tf.float32, shape=(None,obs_dims), name='obs_pl')
        self.action_pl = tf.placeholder(tf.float32, shape=(None, action_dims), name='action_pl')

        with tf.variable_scope('actor') as vs:
            # Weight initializers - layer 1
            magnitude = np.sqrt(float(obs_dims))
            initializers = [tf.random_uniform_initializer(minval=-1/magnitude,maxval=1/magnitude)]
            # hidden layers
            magnitude = np.sqrt(float(hidden_size))
            initializers += [tf.random_uniform_initializer(minval=-1/magnitude,maxval=1/magnitude)]*num_layers

            self.actor = tf.tanh(neural_network(self.obs_pl,
                                            [hidden_size]*num_layers + [action_dims],
                                            name='actor',
                                            initializers=initializers
                                            ))

        # to train critic
        with tf.variable_scope('critic') as vs:
            # Weight initializers - layer 1
            magnitude = np.sqrt(float(obs_dims))
            initializer = tf.random_uniform_initializer(minval=-1/magnitude,maxval=1/magnitude)
            self.critic_lr1 = tf.nn.relu(tf.layers.batch_normalization(
                                            tf.layers.dense(inputs=self.obs_pl,
                                                units=hidden_size,
                                                reuse=None,
                                                kernel_initializer=initializer,
                                                name='critic_obs_layer',
                                                )))

            # hidden layers
            magnitude = np.sqrt(float(hidden_size))
            initializers = [tf.random_uniform_initializer(minval=-1/magnitude,maxval=1/magnitude)]*(num_layers)
            self.critic = neural_network(tf.concat([self.critic_lr1,self.action_pl],1),
                                            [hidden_size]*(num_layers-1) + [1],
                                            name='critic',
                                            reuse=None,
                                            initializers=initializers
                                            )
            # to train actor
            self.actor_training_signal = tf.gradients(self.critic, self.action_pl, stop_gradients=[self.obs_pl])
