"""
Sample reimplementation of Kernel-based Reinforcement Learning

Shinnosuke Usami, 2020
susami@andrew.cmu.edu
"""

import argparse
import random
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import gym

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--bandwidth', '-b', type=float, default=0.02)
    p.add_argument('--n_episodes', '-e', type=int, default=1000)
    p.add_argument('--gamma', '-g', type=float, default=0.1)
    p.add_argument('--n_iter', '-i', type=int, default=50)
    p.add_argument('--n_samples', '-s', type=int, default=1000)
    p.add_argument('--render', '-r', action='store_true', default=False)

    return p.parse_args()

def collect_sample_transitions(env, n_samples, render=False):
    """This function is a random sample transition per action until it falls.
       It works in the CartPole problem because it earns a reward every time the
       pole stays up on the cart.

    Args:
        env: gym environment
        n_samples: the required number of samples
        render: to render the gym environment or not

    Returns:
        sample_trans: the tensor of the sample (n_samples, n_actions, 2, d_states)
    """
    # in cartpole, 2 actions: (left, right)
    n_actions = env.action_space.n

    # number of times the action was taken
    cnt_action = []
    for i in range(n_actions):
        cnt_action.append(0)

    # in cartpole, dimension of states: (c_pos, c_vel, p_angle, p_vel_tip)
    d_states = len(env.observation_space.high)

    # tensor to hold data, 2 is for the transition start state and end state
    sample_trans = np.zeros([n_samples, n_actions, 2, d_states]) # (1000, 2, 2, 4)

    # reward per action within the observation transition (x -> y)
    sample_reward = np.zeros([n_samples, n_actions])

    # average number of steps the env can go without the pole falling
    avg_random_steps = 0

    # number of episodes the simulator executed
    n_episodes = 0

    # loop until the action exceeds the number of n_samples
    while cnt_action[0] < n_samples or cnt_action[1] < n_samples:
        # start of observation which is x in the paper
        obs_x = env.reset()

        # number of steps to make sure because it is random
        max_steps = 100
        for step in range(max_steps):
            if render:
                env.render()

            # there is no policy so we take a random action
            action = env.action_space.sample()

            # define obs_y as the end of the transition like in the paper
            obs_y, r, done, _ = env.step(action)

            # do note collect transition if the action has appeared more than
            # the preferred number of samples
            if   (action == 0 and cnt_action[0] < n_samples) \
              or (action == 1 and cnt_action[1] < n_samples):
                sample_trans[cnt_action[action], action, 0, :] = obs_x
                sample_trans[cnt_action[action], action, 1, :] = obs_y
                sample_reward[cnt_action[action], action] = r
                cnt_action[action] += 1
                obs_x = obs_y

            if done:
                sample_reward[cnt_action[action] - 1, action] = 0
                avg_random_steps = avg_random_steps + step + 1
                break

        n_episodes += 1

    avg_random_steps /= (n_episodes - 1)
    print("average number of random steps until falling: ", avg_random_steps)
    print("total episodes (does not matter): ", n_episodes)
    print("The above should multiply up to aprox. 2000 because we are \
           collecting 1000 samples per action")

    return sample_trans, sample_reward

def nn_kernel(S, x, bandwidth):
    """ Computes the nearest neighbor kernel

    Args:
        S (n_samples, d_states): multiple start data points(x)
        x (d_states): single destination data point (y) with multidimensions

    Returns:
        weights (n_samples): basically, further the data point x is from
          S, the weighting decreases. Closer the data point x is from S,
          the weighting increases.
    """
    # number of samples
    m, _ = S.shape
    weights = np.exp(np.power(np.linalg.norm(S - x, axis=1) / bandwidth, 2) / (-2))
    weights = weights / np.sum(weights)
    return weights

def O_action(x, y, bandwidth):
    """ Computes O per action

    Args:
        x (n_samples, d_states): samples of x for specific action
        y (n_samples, d_states): samples of y for specific action
        bandwidth: the bandwidth for kernel

    Returns:
        O_action (n_samples, n_samples): for each action, we get to know the
          weighting of the sample from x to y which is "how close is x and y?"
          So it is like a force from x to y when taking action.
    """
    # n_samples
    m, _ = x.shape
    O_action = np.zeros([m, m])

    # iter through samples
    for i in range(m):
        O_action[i, :] = nn_kernel(x, y[i], bandwidth)

    return O_action

def create_O(sample_trans, bandwidth):
    """ Computes O which is a tensor which weights x->y for each actions.

    Args:
        sample_trans: the tensor of the sample (n_samples, n_actions, 2(x->y), d_states)

    Returns:
        O (n_samples, n_actions, n_samples): for n_actions, each n_samples(x)->
          n_samples(y) means "how close is the samples?"
    """
    m, M, _, _ = sample_trans.shape
    # O holds tensor (n_samples, n_actions, n_samples)
    O = np.zeros([m, M, m])
    for i in range(M):
        O[:, i, :] = O_action(sample_trans[:, i, 0, :],
                              sample_trans[:, i, 1, :],
                              bandwidth)
    return O

def create_kbrl_table(R, O, gamma, n_iter):
    """Create the kbrl table

    Args:
        gamma: discount rate
        R: m(n_samples)xM(n_actions) reward matrix
        O: mxMxm tensor k(xs, ys'^a) at each location

    Returns:
        J m(n_samples)xM(n_actions)
    """
    J_prev = np.zeros(R.shape)
    for i in range(n_iter):
        J = np.tensordot(O, R + gamma * J_prev, axes=1)
        J = np.amax(J, 1)
        J_prev = J
    return J

def create_OJ(bandwidth, gamma, n_iter, sample_trans, R):
    O = create_O(sample_trans, bandwidth)
    J = create_kbrl_table(R, O, gamma, n_iter)
    return O, J

def best_action_kbrl(J, R, sample_trans, gamma, bandwidth, x):
    m, M = J.shape
    kernel_x = np.zeros([m, M])
    for i in range(M):
        kernel_x[:, i] = nn_kernel(sample_trans[:, i, 0, :], x, bandwidth)
    temp = np.multiply(kernel_x, R + gamma * J)
    temp = np.sum(temp, axis=0)
    best_action = np.argmax(temp)
    return best_action

def train_kbrl():
    env = gym.make('CartPole-v0')
    args = get_args()

    # training...

    # 1. sample transitions
    n_samples = args.n_samples
    sample_trans, sample_reward = collect_sample_transitions(env, n_samples, args.render)

    gamma = args.gamma
    bandwidth = args.bandwidth
    n_iter = args.n_iter
    n_episodes = args.n_episodes

    # 2. create the kbrl table
    O, J = create_OJ(bandwidth, gamma, n_iter,
                             sample_trans, sample_reward)

    # testing...

    # 3. test the computed policy
    avg_episodes = 0
    for episode in range(n_episodes):
        obs = env.reset()
        for t in range(1000):
            env.render()
            action = best_action_kbrl(J, sample_reward, sample_trans,
                                      gamma, bandwidth, obs)
            obs, _, done, _ = env.step(action)
            if done:
                avg_episodes = avg_episodes + t + 1
                break
        print(episode, t)
    avg_episodes = avg_episodes / n_episodes
    print("resulting average episode:", avg_episodes)

train_kbrl()


