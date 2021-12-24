"""
Personal implementation of Playing Atari with Deep Reinforcement Learning, by Mnih et al. (2013)
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

Anson Ho, 2021
"""

import gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from assets.memorybuffer import MemoryBuffer
import numpy as np
import random
from collections import deque
import numpy as np
import wandb

wandb.init(project="DQN", entity="ansonwhho")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_environment(game, num_frame_stack):
    """
    Take 210 x 160 images with 128 color palette, then
    downsample into 110 x 84 image, then crop into
    84 x 84 region that captures playing area.
    
    Note: observe 4 frames at a time and with gray-scale.
    These are implemented in other functions, unlike in the 
    original paper. 
    """
    env = gym.make(game)

    # apply preprocessing using wrappers
    from gym.wrappers import ResizeObservation
    from gym.wrappers import GrayScaleObservation
    from gym.wrappers import FrameStack

    # env = GrayScaleObservation(env, True)
    env = ResizeObservation(env, 84)
    env = FrameStack(env, num_frame_stack)

    return env

class DQN(nn.Module):
    """
    DQN for approximating the Q-function
    The input is a 4 x 84 x 84 image
    """

    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(4,8,8), stride=4),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(1,4,4), stride=2),
            nn.ReLU()            
        )
        self.fc1 = nn.Sequential(
            torch.nn.Linear(2592, 256), # https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            torch.nn.Linear(256, num_actions),
            nn.ReLU()
        )
        self.loss_function = nn.MSELoss()
    
    def forward_pass(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def train_model(minibatch, discount):
    """
    Trains model based on the Bellman equation
    Applies to both the policy and target nets

    Takes a minibatch where each element is of form:
    [phi_j, a_j, r_j, phi_j+1]

    """  

    state_minibatch = [transition[0] for transition in minibatch]
    reward_minibatch = [transition[2] for transition in minibatch]
    reward_minibatch = torch.tensor(reward_minibatch)
    next_state_minibatch = [transition[3] for transition in minibatch]

    target_q_values = [target_net.forward_pass(state) for state in next_state_minibatch]
    target_q_values = torch.stack(target_q_values, dim=0)
    targets = torch.add(reward_minibatch, discount * torch.max(target_q_values, dim=1)[0])

    policy_values = [policy_net.forward_pass(state) for state in state_minibatch]
    policy_values = torch.stack(policy_values, dim=0)
    predictions = torch.max(policy_values, dim=1)[0]

    # target = reward + discount * max(target_net.forward_pass(next_state))
    # prediction = max(policy_net.forward_pass(state))
    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=learning_rate)

    loss = policy_net.loss_function(predictions, targets)
    # loss = policy_net.loss_function(prediction, target)
    loss.backward()
    optimizer.step()

def epsilon_greedy(epsilon, state):
    """
    Select actions based on an 
    epsilon-greedy policy, i.e.
    explore with probability epsilon,
    exploit with probability 1 - epsilon,
    for 0 <= epsilon <= 1
    """

    # explore
    if random.uniform(0,1) < epsilon:
        action = env.action_space.sample()
    
    # exploit
    else: 
        # print("Exploit!")
        q_values = policy_net.forward_pass(state)
        action = torch.argmax(q_values).item()

    return action


if __name__=="__main__":
    
    # https://gym.openai.com/envs/#atari
    games = ["BeamRider-v0", "Breakout-v0", "Enduro-v0", "Pong-v0", "Qbert-v0", "Seaquest-v0", "SpaceInvaders-v0"]

    # for initial testing use cartpole https://gym.openai.com/envs/CartPole-v1/

    config = {
        "learning_rate": 0.001,
        "minibatch_size": 128,
        "num_episodes": 1_000,
        "max_time": 10_000,
        "discount": 0.99,
        "epsilon": 1

    }

    wandb.config = config

    episode_durations = []

    num_frame_stack = 4
    env = make_environment("CartPole-v1", num_frame_stack)
    num_actions = env.action_space.n

    # inputs for taking actions
    memory = MemoryBuffer(1_000_000)
    num_episodes = 1_000
    max_time = 10_000
    discount = 0.99

    # anneal epsilon linearly for 1 million
    # frames until 0.1, then keep fixed
    initial_epsilon = 1
    final_epsilon = 0.1
    num_anneal_frames = 1_000_000 
    decrease_per_frame = (initial_epsilon - final_epsilon) / num_anneal_frames
    frame_num = 0
    epsilon = initial_epsilon

    # training hyperparameters
    minibatch_size = 8
    learning_rate = 3e-4
    
    # initialise DQNs
    policy_net = DQN(num_actions)
    target_net = DQN(num_actions)

    for episode in range(num_episodes):
        observation = env.reset()
        state = torch.from_numpy(observation.__array__()) / 255
        state = state.permute((3,0,1,2)).unsqueeze(0)
        wandb.log({"loss": loss})

        for t in range(max_time):

            screen = env.render()
            transition = [state]
            action = epsilon_greedy(epsilon, state)
            observation, reward, done, info = env.step(action)

            # store transition minibatch
            next_state = torch.from_numpy(observation.__array__()) / 255
            next_state = next_state.permute((3,0,1,2)).unsqueeze(0)
            transition.extend([action, reward, next_state])
            memory.store_transition(transition)

            if len(memory) >= 10 * minibatch_size:
                minibatch = memory.random_sample(minibatch_size)
                train_model(minibatch, discount)

            if done:
                print("Episode {} finished after {} timesteps".format(episode+1, t+1))
                break

            # anneal epsilon
            frame_num += 1
            if frame_num <= num_anneal_frames:
                epsilon -= decrease_per_frame

    env.close()