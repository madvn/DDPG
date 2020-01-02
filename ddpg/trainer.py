import os
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
    max_episodes=500,
    buffer_size=1000000,
    batch_size=64,
    plot_flag=True,
    verbose=True,
    save_dir="./",
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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

        if verbose or (e + 1) % 50 == 0:
            print("Episode # {} | Total reward = {}".format(e + 1, epi_rwd))
        epi_rwds.append(epi_rwd)

    # save trained model
    learner.save_model(save_dir)

    # plot training curve
    if plot_flag:
        plt.plot(epi_rwds)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.savefig(os.path.joiin(save_dir, "training_curve.png"))
        plt.show()

    return epi_rwds


if __name__ == "__main__":
    env = gym.make("Pendulum-v0")
    save_dir = "./results/pendulum"

    # running multiple runs
    for r in range(10):
        this_save_dir = os.path.join(save_dir, "run_{}".format(r))
        print("\nRun # {}. Saving to {}".format(r, this_save_dir))

        epi_rwds = train(env, plot_flag=False, save_dir=this_save_dir)
        plt.plot(epi_rwds, alpha=0.7)

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(save_dir, "./pendulum_training_curves.png"))
