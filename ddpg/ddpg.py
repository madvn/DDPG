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
            # creating the main actor and main critic networks
            with tf.variable_scope('main') as vs:
                self.main = ActorCritic(hidden_size,
                                              num_layers,
                                              obs_dims,
                                              action_dims,
                                              )
                vs.reuse_variables()

            # creating the target actor and target critic networks
            with tf.variable_scope('target') as vs:
                self.target = ActorCritic(hidden_size,
                                              num_layers,
                                              obs_dims,
                                              action_dims,
                                              )
                vs.reuse_variables()

        # get trainable variables from all 4 networks
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
            # train main critic
            with tf.name_scope('critic'):
                # placeholders for rewards and not(done) from minibatch sampled from ReplayBuffer
                self.rewards_pl = tf.placeholder(tf.float32, shape=[1,None], name='rewards_pl')
                self.inverted_done_pl = tf.placeholder(tf.float32, shape=[None,1], name='inverted_done_pl')

                # desired output of main critic Q(s_t) = r + gamma * target_critic(s_t+1)
                # target_critic is computed using s_t+1 and target_actor(s_t+1)
                # note that s_t and s_t+1 come from minibatch sampled from ReplayBuffer
                self.main_critic_desired = tf.transpose(self.rewards_pl) + tf.multiply(self.inverted_done_pl, self.gamma*(self.target.critic))

                # loss is simply root mean squared error of desired and actual output
                # actual output of main_critic is computed from s_t and actions from sampled minibatch as opposed to main_actor(s_t)
                self.main_critic_loss = tf.losses.mean_squared_error(self.main_critic_desired, self.main.critic)
                self.main_critic_optimizer = tf.train.AdamOptimizer(self.critic_lr).minimize(self.main_critic_loss)

            # train main actor - gradient provided by critic
            with tf.name_scope('actor'):
                # placeholder for incoming gradient from critic i.e. gradient of critic w.r.t. action
                self.main_action_gradient_pl = tf.placeholder(tf.float32, shape=[None,self.action_dims], name='action_gradient_pl')

                # find gradient of action w.r.t. to actor_parameters with incoming gradients from critic as initial value
                # note minus sign in incoming gradients from critic
                self.combined_gradients = tf.gradients(self.main.actor, self.main_actor_params, -self.main_action_gradient_pl)

                # averaging across gradients from sampled minibatch
                self.main_actor_gradients = [tf.div(self.combined_gradients[i],self.batch_size) for i in range(len(self.combined_gradients))]

                # applying gradients to actor parameters
                self.main_actor_optimizer = tf.train.AdamOptimizer(self.actor_lr).apply_gradients(zip(
                                                                                               self.main_actor_gradients,
                                                                                               self.main_actor_params))

        # initialize target actor and critic networks to be same as main actor and critic networks
        self.init_target_params = [tf.assign(self.target_params[i], self.main_params[i]) for i in range(len(self.main_params))]

        with tf.name_scope('train_target'):
            # updating parameters of target critic
            # target_critic_params <- (1-tau)*target_critic_params + tau*main_critic_params
            self.update_target_critic_params = [self.target_critic_params[i].assign(tf.multiply(self.main_critic_params[i], self.tau)+
                                                                            tf.multiply(self.target_critic_params[i], 1.-self.tau)
                                                                            )
                                                    for i in range(len(self.target_critic_params))]

            # updating target actor params
            # target_actor_params <- (1-tau)*target_actor_params + tau*main_actor_params
            self.update_target_actor_params = [self.target_actor_params[i].assign(tf.multiply(self.main_actor_params[i], self.tau)+
                                                                            tf.multiply(self.target_actor_params[i], 1.-self.tau)
                                                                            )
                                                    for i in range(len(self.target_actor_params))]

        # housekeeping
        self.vars_inited = False
        tf.summary.scalar("main_critic_loss", self.main_critic_loss)
        self.merged_summaries = tf.summary.merge_all()

    def act(self, sess, obs):
        ''' actor output for one observation'''
        # initializing variables only once
        if not self.vars_inited:
            sess.run(tf.global_variables_initializer())
            sess.run(self.init_target_params)
            self.vars_inited = True

        # just get output of main actor given the observation, s
        return sess.run(self.main.actor,feed_dict={self.main.obs_pl: np.reshape(obs, [-1, self.obs_dims])})

    def act_critc(self, sess, obs):
        ''' actor and critic output for observation'''
        # initializing variables only once
        if not self.vars_inited:
            sess.run(tf.global_variables_initializer())
            sess.run(self.init_target_params)
            self.vars_inited = True

        # get output of main actor and main critic given an observation
        actor_out = sess.run(self.main.actor, feed_dict={self.main.obs_pl: np.reshape(obs, [-1, self.obs_dims])})
        critic_out = sess.run(self.main.critic,feed_dict={self.main.obs_pl: np.reshape(obs, [-1, self.obs_dims]),
                                                           self.main.action_pl: actor_out})
        return actor_out, critic_out


    def update_target_critic(self, sess):
        ''' run op to update target critic'''
        # initializing variables only once
        if not self.vars_inited:
            sess.run(tf.global_variables_initializer())
            sess.run(self.init_target_params)
            self.vars_inited = True

        # update target critic parameters
        sess.run(self.update_target_critic_params)

    def update_target_actor(self, sess):
        ''' run op to update target actor'''
        # initializing variables only once
        if not self.vars_inited:
            sess.run(tf.global_variables_initializer())
            sess.run(self.init_target_params)
            self.vars_inited = True

        # update target actor parameters
        sess.run(self.update_target_actor_params)

    def train_critic(self, sess):
        ''' update critic and return action gradients to update actor'''
        # initializing variables only once
        if not self.vars_inited:
            sess.run(tf.global_variables_initializer())
            sess.run(self.init_target_params)
            self.vars_inited = True

        # train main critic using a minibatch sampled from ReplayBuffer
        if self.buffer.size() >= self.batch_size:
            batch = self.buffer.sample_batch(self.batch_size)
            batch = self._preprocess_batch(batch)

            # items used:
                #1. target_actor_output for s_t+1 in minibatch
                #2. target_critic output for s_t+1 and target_actor_output from 1
                #3. main_critic_output using s_t and a_t from minibatch
                #4. rewards from minibatch
                #5. not(done from minibatch)
            #
            target_act = sess.run(self.target.actor, feed_dict={self.target.obs_pl: batch['next_s']}) #1 above
            _, loss, summary = sess.run([self.main_critic_optimizer, self.main_critic_loss, self.merged_summaries], feed_dict={
                                                        self.target.obs_pl: batch['next_s'],          #2
                                                        self.target.action_pl: target_act,            #2
                                                        self.main.obs_pl: batch['s'],                 #3
                                                        self.main.action_pl: batch['a'],              #3
                                                        self.rewards_pl: batch['r'],                  #4
                                                        self.inverted_done_pl: np.transpose(np.logical_not(batch['t']).astype('float32')) #5
                                                        })
            return loss,summary
        else:
            return 100. # NOT SURE OF THIS IS OKAY

    def train_actor(self, sess):
        ''' update main actor params'''
        # initializing variables only once
        if not self.vars_inited:
            sess.run(tf.global_variables_initializer())
            sess.run(self.init_target_params)
            self.vars_inited = True

        # train main actor using a minibatch sampled from ReplayBuffer
        # this is a different sample from what was used to train the critic - hope that's okay
        if self.buffer.size() >= self.batch_size:
            batch = self.buffer.sample_batch(self.batch_size)
            batch = self._preprocess_batch(batch)

            # items used:
                #1. main_actor_output using s_t from sampled minibatch
                #2. action_gradients from main_critic given 1. and s_t from minibatch
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
