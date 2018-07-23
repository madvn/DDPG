import tensorflow as tf
import numpy as np
from utils import neural_network

class ActorCritic:

    def __init__(self,
                  hidden_layers,
                  obs_dims,
                  action_dims,
                  action_gain,
                  ):

        # placeholders for observation and action
        self.obs_pl = tf.placeholder(tf.float32, shape=(None,obs_dims), name='obs_pl')
        self.action_pl = tf.placeholder(tf.float32, shape=(None, action_dims), name='action_pl')

        # designing the actor network
        with tf.variable_scope('actor') as vs:
            # Weight initializers - layer 1
            magnitude = np.sqrt(obs_dims)
            initializers = [tf.random_uniform_initializer(minval=-1/magnitude,maxval=1/magnitude)]
            # Weight initializers - hidden layers
            magnitudes = np.sqrt(hidden_layers)
            initializers += [tf.random_uniform_initializer(minval=-1/magnitudes[i],maxval=1/magnitudes[i]) for i in range(len(hidden_layers))]

            # creating a feed-forward neural network with observation as input and tanh activated action_dim outputs
            self.actor = action_gain*tf.tanh(neural_network(self.obs_pl,
                                            hidden_layers + [action_dims],
                                            name='actor',
                                            initializers=initializers
                                            ))

        # to train critic
        with tf.variable_scope('critic') as vs:
            # Weight initializers - layer 1
            magnitude = np.sqrt(obs_dims)
            initializer = tf.random_uniform_initializer(minval=-1/magnitude,maxval=1/magnitude)

            # creating just one layer of the critic first with
                #1. obsevation as input and
                #2. batch_normalization
            self.critic_lr1 = tf.nn.relu(tf.layers.batch_normalization(
                                            tf.layers.dense(inputs=self.obs_pl,
                                                units=hidden_layers[0],
                                                reuse=None,
                                                kernel_initializer=initializer,
                                                name='critic_obs_layer',
                                                )))

            # Weight initializers - hidden layers
            magnitudes = np.sqrt(hidden_layers)
            initializers = [tf.random_uniform_initializer(minval=-1/magnitudes[i],maxval=1/magnitudes[i]) for i in range(len(hidden_layers))]

            # concatenating the action input to the output of layer 1 critic as input to layer 2 of critic
            self.critic = neural_network(tf.concat([self.critic_lr1,self.action_pl],1),
                                            hidden_layers[1:] + [1],
                                            name='critic',
                                            reuse=None,
                                            initializers=initializers
                                            )
            # partial derivative of critic with respect to actions - to train the actor
            self.actor_training_signal = tf.gradients(self.critic, self.action_pl, stop_gradients=[self.obs_pl])
