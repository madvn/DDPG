import numpy as np
import tensorflow as tf
from tensorflow import keras

from actor_critic import Actor, Critic


class DDPG:
    def __init__(
        self,
        obs_dim,
        action_dim,
        action_gain,
        actor_learning_rate=0.0001,
        critic_learning_rate=0.001,
        gamma=0.99,
        tau=0.001,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        # make main networks
        self.actor = Actor(obs_dim, action_dim, action_gain, actor_learning_rate)
        self.critic = Critic(obs_dim, action_dim, critic_learning_rate)

        # make target networks
        self.target_actor = Actor(obs_dim, action_dim, action_gain)
        self.target_actor.model.set_weights(self.actor.model.get_weights())
        self.target_critic = Critic(obs_dim, action_dim)
        self.target_critic.model.set_weights(self.critic.model.get_weights())

    def act(self, obs):
        return self.actor.act([[obs]])[0]

    def update_networks(self, batch):
        """ runs all updates from provided training batch """
        batch = self._prep_batch(batch)

        self.update_critic(batch)
        self.update_actor(batch)
        self.target_actor.model = self.update_target(
            self.actor.model, self.target_actor.model
        )
        self.target_critic.model = self.update_target(
            self.critic.model, self.target_critic.model
        )

    def update_critic(self, batch):
        """ minimize td-loss from target """
        # td estimate based on targets' behavior
        target_future_actions = self.target_actor.act(batch["next_s"])
        target_future_qs = self.target_critic.estimate_q(
            batch["next_s"], target_future_actions
        )
        target_current_qs = batch["r"] + self.gamma * target_future_qs

        # main critic's td estimate and loss
        with tf.GradientTape() as tape:
            model_vars = self.critic.model.trainable_variables
            tape.watch(model_vars)
            main_current_qs = self.critic.model([batch["s"], batch["a"]])
            loss = keras.losses.mse(target_current_qs, main_current_qs)

        # update main critic
        dloss_dcrit = tape.gradient(loss, model_vars)
        self.critic.optimizer.apply_gradients(zip(dloss_dcrit, model_vars))

    def update_actor(self, batch):
        """ dq_dtheta = dq_da * da_dtheta"""
        # first, finding dq_da
        with tf.GradientTape() as tape:
            a = self.actor.model(batch["s"])
            q = self.critic.model([batch["s"], a])
        dq_da = tape.gradient(q, a)

        # second, finding da_dtheta
        with tf.GradientTape() as tape:
            model_vars = self.actor.model.trainable_variables
            tape.watch(model_vars)
            a = self.actor.model(batch["s"])
        da_dtheta = tape.gradient(a, model_vars, output_gradients=-dq_da)

        # updating the model
        self.actor.optimizer.apply_gradients(zip(da_dtheta, model_vars))

    def update_target(self, main_model, target_model):
        """ target = tau*main + (1-tau)*target """
        new_weights = list(map(lambda x: self.tau * x, main_model.get_weights()))
        new_weights = list(
            map(
                lambda x: x[0] + (1 - self.tau) * x[1],
                list(zip(new_weights, target_model.get_weights())),
            )
        )
        target_model.set_weights(new_weights)
        return target_model

    def _prep_batch(self, batch):
        """ converts sample from buffer to dict for easy feed_dicting"""
        # batch = np.transpose(batch)
        processed_batch = {}
        processed_batch["s"] = batch[0]
        processed_batch["a"] = batch[1]
        processed_batch["r"] = batch[2]
        processed_batch["t"] = batch[3]
        processed_batch["next_s"] = batch[4]
        return processed_batch


if __name__ == "__main__":
    learner = DDPG(4, 1, 2)

    # testing target net == main net
    obs = np.ones(4)
    assert learner.actor.act([[obs]]) == learner.target_actor.act(
        [[obs]]
    ), "ERROR: Target and main are not the same nets"

    # testing action
    obs = np.ones(4)
    action = learner.act(obs)
    print("Action for ones: ", action)

    # testing update
    print("Updating... ", end="")
    # creating random [s, a, r, t, s'] batch with 64 samples
    batch = [np.random.rand(64, i) for i in [4, 1, 1, 1, 4]]
    learner.update_networks(batch)
    print("Done!")
