from train import *

# parameters
env_name = 'Pendulum-v0'
num_episodes = 2
logs_dir = './logs'
print_freq = 1000

ddpg_params = {}
ddpg_params['hidden_size'] = 400
ddpg_params['num_layers'] = 2
ddpg_params['obs_dims'] = 3
ddpg_params['action_dims'] = 1
ddpg_params['actor_lr'] = 10e-4
ddpg_params['critic_lr'] = 10e-3
ddpg_params['gamma'] = 0.99
ddpg_params['tau'] = 0.001
ddpg_params['batch_size'] = 64
ddpg_params['buffer_size'] = 1e6
#ddpg_params['hidden_activation'] = tf.nn.relu
#ddpg_params['action_activation'] = tf.tanh
#ddpg_params['critic_activation'] = None # linear

train(ddpg_params,
         env_name,
         num_episodes = num_episodes,
         logs_dir = logs_dir,
         print_freq = print_freq,
         movie_save_dir='./results',
         movie_name='trained'
 )
