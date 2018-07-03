import tensorflow as tf
import numpy as np

from actor_critic import ActorCritic
from replay_buffer import ReplayBuffer

class DDPG:

    def __init__(self,
                  hidden_size,
                  num_layers,
                  obs_dims,
                  action_dims,
                  actor_lr,
                  critic_lr,
                  gamma,
                  tau,
                  batch_size=1000,
                  buffer_size=10000,
    ):
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size)

        # creating networks
        with tf.variable_scope('ddpg'):
            # main actor-critic
            with tf.variable_scope('main') as vs:
                self.main = ActorCritic(hidden_size,
                                              num_layers,
                                              obs_dims,
                                              action_dims,
                                              )
                vs.reuse_variables()

            # target actor-critic
            with tf.variable_scope('target') as vs:
                self.target = ActorCritic(hidden_size,
                                              num_layers,
                                              obs_dims,
                                              action_dims,
                                              )
                vs.reuse_variables()

        # get trainable variables
        self.main_params = self._params('ddpg/main')
        self.target_params = self._params('ddpg/target')
        self.main_actor_params = self._params('ddpg/main/actor')
        self.main_critic_params = self._params('ddpg/main/critic')
        self.target_actor_params = self._params('ddpg/target/actor')
        self.target_critic_params = self._params('ddpg/target/critic')

        '''# prep minibatch to train
        total_dims = self.obs_dims + self.action_dims + 1 + 1 + self.obs_dims # s,a,r,t,s'
        self.sampled_batch = tf.placeholder(tf.float32, shape=[self.batch_size, total_dims])
        self.obs_batch = self.sampled_batch[:,self.obs_dims]
        self.rewards_batch = self.sampled_batch[:,self.obs_dims+self.action_dims:self.obs_dims+self.action_dims+1]
        self.done_batch = self.sampled_batch[:,-self.obs_dims-1:-self.obs_dims]
        self.inverted_done_batch = tf.cast(tf.logical_not(tf.cast(self.done_batch, tf.bool)), tf.float32)'''

        with tf.name_scope('train_main'):
            with tf.name_scope('critic'):
                # train main critic
                self.rewards_pl = tf.placeholder(tf.float32, shape=[1,None], name='rewards_pl')
                self.inverted_done_pl = tf.placeholder(tf.float32, shape=[None,1], name='inverted_done_pl')
                self.main_critic_desired = tf.transpose(self.rewards_pl) + tf.multiply(self.inverted_done_pl, self.gamma*(self.target.critic))
                self.main_critic_loss = tf.losses.mean_squared_error(self.main_critic_desired, self.main.critic)
                self.main_critic_optimizer = tf.train.AdamOptimizer(self.critic_lr).minimize(self.main_critic_loss)

            with tf.name_scope('actor'):
                # train main actor - gradient provided by critic
                self.main_action_gradient_pl = tf.placeholder(tf.float32, shape=[None,self.action_dims], name='action_gradient_pl')
                self.combined_gradients = tf.gradients(self.main.actor, self.main_actor_params, -self.main_action_gradient_pl)
                self.main_actor_gradients = [tf.div(self.combined_gradients[i],self.batch_size) for i in range(len(self.combined_gradients))]
                self.main_actor_optimizer = tf.train.AdamOptimizer(self.actor_lr).apply_gradients(zip(
                                                                                               self.main_actor_gradients,
                                                                                               self.main_actor_params))

        # update target nets
        self.init_target_params = [tf.assign(self.target_params[i], self.main_params[i]) for i in range(len(self.main_params))]
        with tf.name_scope('train_target'):
            self.update_target_actor_params = [self.target_actor_params[i].assign(tf.multiply(self.main_actor_params[i], self.tau)+
                                                                            tf.multiply(self.target_actor_params[i], 1.-self.tau)
                                                                            )
                                                    for i in range(len(self.target_actor_params))]
            self.update_target_critic_params = [self.target_critic_params[i].assign(tf.multiply(self.main_critic_params[i], self.tau)+
                                                                            tf.multiply(self.target_critic_params[i], 1.-self.tau)
                                                                            )
                                                    for i in range(len(self.target_critic_params))]

        # housekeeping
        self.vars_inited = False
        tf.summary.scalar("main_critic_loss", self.main_critic_loss)
        self.merged_summaries = tf.summary.merge_all()

    def act(self, sess, obs):
        ''' actor output for one observation'''
        if not self.vars_inited:
            sess.run(tf.global_variables_initializer())
            sess.run(self.init_target_params)
            self.vars_inited = True

        return sess.run(self.main.actor,feed_dict={self.main.obs_pl: np.reshape(obs, [-1, self.obs_dims])})

    def act_critc(self, sess, obs):
        ''' actor and critic output for observation'''
        if not self.vars_inited:
            sess.run(tf.global_variables_initializer())
            sess.run(self.init_target_params)
            self.vars_inited = True

        actor_out = sess.run(self.main.actor, feed_dict={self.main.obs_pl: np.reshape(obs, [-1, self.obs_dims])})
        critic_out = sess.run(self.main.critic,feed_dict={self.main.obs_pl: np.reshape(obs, [-1, self.obs_dims]),
                                                           self.main.action_pl: actor_out})
        return actor_out, critic_out


    def update_target_critic(self, sess):
        ''' run op to update target critic'''
        if not self.vars_inited:
            sess.run(tf.global_variables_initializer())
            sess.run(self.init_target_params)
            self.vars_inited = True

        sess.run(self.update_target_critic_params)

    def update_target_actor(self, sess):
        ''' run op to update target actor'''
        if not self.vars_inited:
            sess.run(tf.global_variables_initializer())
            sess.run(self.init_target_params)
            self.vars_inited = True

        sess.run(self.update_target_actor_params)

    def train_critic(self, sess):
        ''' update critic and return action gradients to update actor'''
        if not self.vars_inited:
            sess.run(tf.global_variables_initializer())
            sess.run(self.init_target_params)
            self.vars_inited = True

        if self.buffer.size() >= self.batch_size:
            batch = self.buffer.sample_batch(self.batch_size)
            batch = self._preprocess_batch(batch)

            #print('\n\n',np.shape(batch['next_s']),'\n\n')
            #desired_critic_out = sess.run(self.main_critic_desired, feed_dict={self.target.obs_pl: batch['s']})
            target_act = sess.run(self.target.actor, feed_dict={self.target.obs_pl: batch['next_s']})
            _, loss, summary = sess.run([self.main_critic_optimizer, self.main_critic_loss, self.merged_summaries], feed_dict={
                                                        self.target.obs_pl: batch['next_s'],
                                                        self.target.action_pl: target_act,
                                                        self.main.obs_pl: batch['s'],
                                                        self.rewards_pl: batch['r'],
                                                        self.main.action_pl: batch['a'],
                                                        self.inverted_done_pl: np.transpose(np.logical_not(batch['t']).astype('float32'))
                                                        })
            return loss,summary
        else:
            return 100.

    def train_actor(self, sess):
        ''' update main actor params'''
        if not self.vars_inited:
            sess.run(tf.global_variables_initializer())
            sess.run(self.init_target_params)
            self.vars_inited = True

        if self.buffer.size() >= self.batch_size:
            batch = self.buffer.sample_batch(self.batch_size)
            batch = self._preprocess_batch(batch)

            main_batch_actions = sess.run(self.main.actor, feed_dict={self.main.obs_pl: batch['s']})
            main_action_grad = sess.run(self.main.actor_training_signal, feed_dict={self.main.obs_pl: batch['s'],
                                                                                    self.main.action_pl: main_batch_actions})[0]
            loss = sess.run(self.main_actor_optimizer, feed_dict = {self.main.obs_pl: batch['s'],
                                                        self.main.action_pl: main_batch_actions,
                                                        self.main_action_gradient_pl: main_action_grad,
                                                        })
            return loss
        else:
            return 100.

    def add_episode_to_buffer(self, buffer):
        ''' add a list of experiences to buffer'''
        for experience in buffer:
            self.buffer.add(experience)

    def append_buffer(self,experience):
        ''' add one experience to buffer'''
        self.buffer.add(experience)

    def _params(self,scope):
        ''' returns all trainable vars in given scope'''
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        return vars

    def _global_vars(self, scope):
        ''' returns all vars in scope'''
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        return vars

    def _preprocess_batch(self,batch):
        ''' converts sample from buffer to dict for easy feed_dicting'''
        batch = np.transpose(batch)
        processed_batch = {}
        processed_batch['s'] = batch[:,:self.obs_dims]
        processed_batch['a'] = batch[:,self.obs_dims:self.obs_dims+self.action_dims]
        processed_batch['r'] = [batch[:,self.obs_dims+self.action_dims]]
        processed_batch['t'] = [batch[:,-self.obs_dims-1]]
        processed_batch['next_s'] = batch[:,-self.obs_dims:]
        return processed_batch

if __name__ == '__main__':
    print('\n************* Compile test of DDPG *************')
    # parameters
    params = {}
    params['hidden_size'] = 200
    params['num_layers'] = 2
    params['obs_dims'] = 4
    params['action_dims'] = 1
    params['actor_lr'] = 1e-3
    params['critic_lr'] = 1e-3
    params['gamma'] = 0.98
    params['tau'] = 0.1
    params['batch_size'] = 1000
    params['buffer_size'] = 10000

    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(params)

    print('\nCreating object')
    ddpg = DDPG(**params)
    print('\nDone!\n')

    '''# inting session
    self.sess = tf.get_default_session()
    if self.sess is None:
        self.sess = tf.InteractiveSession()
    '''
