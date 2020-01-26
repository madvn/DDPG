# DDPG

Implementing algorithm from

Lillicrap, Timothy P., Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. "Continuous control with deep reinforcement learning." arXiv preprint arXiv:1509.02971 (2015).
https://arxiv.org/abs/1509.02971

Modify and/or run [ddpg/trainer.py](https://github.com/madvn/DDPG/blob/master/ddpg/trainer.py)

Dependencies: Tensorflow r2.0, numpy, matplotlib, gym


#### TODO

- [x] noiseless evals every X training episodes
- [ ] parametrize network architecture
- [ ] better hyperparams for Pendulum
- [ ] more tasks


### Results

From 10 runs of [ddpg/trainer.py](https://github.com/madvn/DDPG/blob/master/ddpg/trainer.py) as is

<img src="https://github.com/madvn/DDPG/blob/master/ddpg/results/pendulum/pendulum_training_curves.png"/>
