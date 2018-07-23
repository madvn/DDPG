import tensorflow as tf
import numpy as np

def neural_network(
            input,
            layer_sizes,
            activation=tf.nn.relu,
            name='',
            initializers=None,
            reuse=None
):
    '''Utility function to create a neural network'''
    for i,size in enumerate(layer_sizes):
        input = tf.layers.dense(inputs=input,
                                 units=size,
                                 reuse=reuse,
                                 kernel_initializer=initializers[i],
                                 name=name+'_'+str(i))

        if i < len(layer_sizes)-1:
            input = activation(input)

    return input
