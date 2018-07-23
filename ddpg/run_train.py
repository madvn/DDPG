from train import *
import pprint

# parameters
env_name = 'Pendulum-v0'
num_episodes = 500
logs_dir = './logs'
print_freq = 50
update_frequency = 1

ddpg_params = {}
ddpg_params['hidden_layers'] = [400, 300]
ddpg_params['actor_lr'] = 0.0001
ddpg_params['critic_lr'] = 0.001
ddpg_params['gamma'] = 0.99
ddpg_params['tau'] = 0.001
ddpg_params['batch_size'] = 64
ddpg_params['buffer_size'] = 10000
#ddpg_params['hidden_activation'] = tf.nn.relu
#ddpg_params['action_activation'] = tf.tanh
#ddpg_params['critic_activation'] = None # linear

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(ddpg_params)

train(ddpg_params,
         env_name,
         num_episodes = num_episodes,
         logs_dir = logs_dir,
         print_freq = print_freq,
         update_frequency = update_frequency,
         movie_save_dir='./results',
         movie_name='trained'
 )
