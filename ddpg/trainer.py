import numpy as np
import gym
import matplotlib.pyplot as plt

from ddpg import DDPG
from replay_buffer import ReplayBuffer


def train(
    env,
    actor_learning_rate=0.0001,
    critic_learning_rate=0.001,
    gamma=0.99,
    tau=0.001,
    max_episodes=100,
    buffer_size=1000000,
    batch_size=64,
    plot_flag=True,
):
    # setup learner
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_gain = env.action_space.high
    learner = DDPG(
        obs_dim,
        action_dim,
        action_gain,
        actor_learning_rate=0.0001,
        critic_learning_rate=0.001,
        gamma=0.99,
        tau=0.001,
    )

    # setup buffer
    buffer = ReplayBuffer(buffer_size)
    epi_rwds = []

    # train
    for e in range(max_episodes):
        epi_rwd = 0
        s = env.reset()
        done = False

        # run episode and train
        while not done:
            # act and get feedback
            a = learner.act(s)
            next_s, r, done, _ = env.step(a)

            # record feedback and train
            epi_rwd += r
            buffer.add([s, a, r, done, next_s])
            if buffer.size() >= batch_size:
                batch = buffer.sample_batch(batch_size)
                learner.update_networks(batch)

            s = next_s

        print("Episode # {} | Total reward = {}".format(e + 1, epi_rwd))
        epi_rwds.append(epi_rwd)

    # plot training curve
    if plot_flag:
        plt.plot(epi_rwds)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.savefig("./training_curve.png")
        plt.show()


if __name__ == "__main__":
    env = gym.make("Pendulum-v0")
    train(env)
