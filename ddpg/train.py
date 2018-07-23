import os
import gym
import numpy as np
import tensorflow as tf
from ddpg import DDPG

# from https://github.com/openai/baselines/blob/master/baselines/ddpg/
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

def train(ddpg_params,
             env_name,
             num_episodes = 1000,
             logs_dir = './logs',
             print_freq = 20,
             update_frequency = 5,
             movie_save_dir = None,
             movie_name = 'movie',
):

    # Create environment
    env = gym.make(env_name)

    # env params
    action_gain = env.action_space.high[0] # assuming symmetric action space
    assert env.action_space.high[0]==-env.action_space.low[0], "Action space assumption of being symmetric i.e. [-a,a] is not met"
    obs_dims = env.observation_space.shape[0]
    action_dims = env.action_space.shape[0]

    # DDPG network env related params
    ddpg_params['action_gain'] = action_gain
    ddpg_params['obs_dims'] = obs_dims
    ddpg_params['action_dims'] = action_dims

    # creating DDPG agent
    learner = DDPG(**ddpg_params)

    # exploration
    actor_noise = OrnsteinUhlenbeckActionNoise(np.zeros(ddpg_params['action_dims']))

    with tf.Session() as sess:
        if not os.path.exists(logs_dir):
            os.mkdir(logs_dir)
        writer = tf.summary.FileWriter(logs_dir, graph=tf.get_default_graph())

        for e in range(num_episodes):
            obs = env.reset()
            done = False

            buffer = []
            episode_reward = 0.
            episode_q = 0.

            # episode
            while done is False:
                a,q = learner.act_critc(sess, obs)
                a = a.flatten() + actor_noise().flatten()
                a = np.clip(a, -action_gain, action_gain)

                new_obs, r, done, _ = env.step(a)

                episode_reward += r
                episode_q += q[0][0]

                learner.append_buffer(np.concatenate(tuple([obs, a, [r], [done], new_obs.flatten()])))

                obs = new_obs

            # train
            if e % update_frequency == 0:
                # train params of main
                c_loss_summary = learner.train_critic(sess)
                a_loss = learner.train_actor(sess)

                # update params of target
                learner.update_target_critic(sess)
                learner.update_target_actor(sess)

            # summaries
            reward_summary = tf.Summary(value=[
                                    tf.Summary.Value(tag="episode_total_reward", simple_value=episode_reward),
                                    ])
            writer.add_summary(reward_summary,e)

            #episode_q_summary = tf.Summary(value=[
            #                        tf.Summary.Value(tag="episode_total_q", simple_value=episode_q),
            #                        ])
            #writer.add_summary(episode_q_summary,e)
            #try:
                # trying because training may not happen until there is enough samples
            #    summary = c_loss_summary[1]
            #    writer.add_summary(summary,e)
            #except:
            #    pass

            # print to console periodically
            if e % print_freq == 0 or e == num_episodes-1:
                #print('Episode # {} - Episode total. reward = {} - Episode avg. Q = {}'.format(str(e),str(episode_reward),str(episode_q)))
                print('Episode # {} - Episode total. reward = {}'.format(str(e),str(episode_reward)))

        # check if movie needs to be saved
        if movie_save_dir:
            import matplotlib
            matplotlib.use('Agg') # To not display but save movie
            import matplotlib.pyplot as plt
            import matplotlib.animation as manimation

            if not os.path.exists(movie_save_dir):
                os.mkdir(movie_save_dir)

            # setting up movie writer
            fig = plt.figure(figsize=[3, 3])
            ax = fig.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            FFMpegWriter = manimation.writers['ffmpeg']
            mov_writer = FFMpegWriter(fps=25)
            mov_writer.setup(
                            fig,
                            os.path.join(movie_save_dir,'{}.mp4'.format(movie_name))
                            )

            # simulate once at the end and save movie
            obs = env.reset()
            done = False
            episode_r = 0.

            # execute one episode and capture movie
            while done is False:
                a,q = learner.act_critc(sess, obs)
                a = a.flatten() + actor_noise().flatten()

                obs, r, done, _ = env.step(a)
                print(r, episode_r)
                episode_r += r

                frame = env.render(mode='rgb_array')
                ax.cla()
                ax.imshow(frame)
                mov_writer.grab_frame()

            # finish and close
            mov_writer.finish()
            plt.close('all')


if __name__ == "__main__":
    print('\nPut this, or something similar, in a separate file called run_train.py to avoid git conflicts')
    print("""
        from train import *
        import pprint

        # parameters
        env_name = 'Pendulum-v0'
        num_episodes = 50000
        logs_dir = './logs'
        print_freq = 10
        update_frequency = 2

        ddpg_params = {}
        ddpg_params['hidden_layers'] = [400, 300]
        #ddpg_params['num_layers'] = 2
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

        """
    )
