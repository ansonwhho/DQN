"""
Personal implementation of Playing Atari with Deep Reinforcement Learning, by Mnih et al. (2013)
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

Anson Ho, 2021
"""

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import wandb
import numpy as np

from assets.memorybuffer import MemoryBuffer


WANDB = 1

if WANDB:
    wandb.init(project="DQN", entity="ansonwhho")

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
    The input is a 84 x 84 image
    Input channels = 4 to account for the stacked frames
    """

    # Note on fc1: see https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    # 32 channels x 9 width x 9 height
    # input has width 84 and height 84
    # conv1 outputs (84 - 8)/4 + 1 = 20
    # conv2 outputs (20 - 4)/2 + 1 = 9
    # so we expect 9 width and 9 height

    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.fc1 = torch.nn.Linear(32 * 9 * 9, 256)
        self.head = torch.nn.Linear(256, num_actions)
        self.loss_function = nn.MSELoss()
    
    def forward_pass(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.head(x)
        return x

def get_state(observation):
    """
    Observation is of form 
    channels x width x height x batch_size
    want the state to be of form
    channels x height x width
    """

    state = torch.from_numpy(observation.__array__())
    # state = torch.from_numpy(observation.__array__()) / 255 # will this cause vanishing gradients?
    state = state.permute((3,0,1,2))
    # print(state.shape) [1,4,84,84] after permuting

    return state

def train_model(minibatch, discount):
    """
    Trains model based on the Bellman equation
    Applies to both the policy and target nets

    Takes a minibatch where each element is of form:
    [state, action, reward, next_state]

    """  

    state_minibatch = [transition[0] for transition in minibatch]
    reward_minibatch = [transition[2] for transition in minibatch]
    reward_minibatch = torch.tensor(reward_minibatch)
    next_state_minibatch = [transition[3] for transition in minibatch]
    done_minibatch= [transition[4] for transition in minibatch]
    done_minibatch = torch.tensor(done_minibatch)


    target_q_values = [target_net.forward_pass(state)[0] for state in next_state_minibatch]
    target_q_values = torch.stack(target_q_values, dim=0)

    targets = torch.add(reward_minibatch,  (1 - done_minibatch.long()) * discount * torch.max(target_q_values, dim=1)[0])

    policy_values = [policy_net.forward_pass(state)[0] for state in state_minibatch]
    policy_values = torch.stack(policy_values, dim=0)
    predictions = torch.max(policy_values, dim=1)[0]

    # target = torch.add(reward, discount * max(target_net.forward_pass(next_state)))
    # prediction = max(policy_net.forward_pass(state))
    
    loss = policy_net.loss_function(predictions, targets)
    # loss = policy_net.loss_function(prediction, target)

    if WANDB:
        wandb.log({
            "loss": loss,
            "epsilon": epsilon,
            "avg_q_value": avg_q_value,
            "avg_reward": avg_reward
            })

    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=config["learning_rate"])
    optimizer.zero_grad()
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
        # print("exploit!")
        q_values = policy_net.forward_pass(state)[0]
        max_q_value = max(q_values)
        episode_q_values.append(max_q_value.detach().numpy())
        action = torch.argmax(q_values).item()

    return action


if __name__=="__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "learning_rate": 3e-4,
        "minibatch_size": 64,
        "num_episodes": 3000,
        "max_episode_time": 10_000,
        "discount": 0.999,
        "epsilon": 1,
        "num_frame_stack": 4,
        "memory_capacity": 10_000
    }

    if WANDB:
        wandb.config = config

    # https://gym.openai.com/envs/#atari
    # for initial testing use cartpole https://gym.openai.com/envs/CartPole-v1/
    games = ["BeamRider-v0", "Breakout-v0", "Enduro-v0", "Pong-v0", "Qbert-v0", "Seaquest-v0", "SpaceInvaders-v0"]
    env = make_environment("CartPole-v1", config["num_frame_stack"])
    num_actions = env.action_space.n

    # inputs for taking actions
    memory = MemoryBuffer(config["memory_capacity"])

    # anneal epsilon linearly for 1 million
    # frames until 0.1, then keep fixed
    initial_epsilon = 1
    final_epsilon = 0.1
    num_anneal_frames = 10_000
    decrease_per_frame = (initial_epsilon - final_epsilon) / num_anneal_frames
    frame_num = 0
    epsilon = initial_epsilon
    
    # initialise DQNs
    policy_net = DQN(num_actions)
    target_net = DQN(num_actions)

    for episode in range(config["num_episodes"]):
        observation = env.reset()
        episode_rewards = 0
        episode_q_values = []

        for t in range(config["max_episode_time"]):

            screen = env.render()

            state = get_state(observation)
            action = epsilon_greedy(epsilon, state)
            observation, reward, done, info = env.step(action)
            next_state = get_state(observation)

            # store transition minibatch
            # differs from the original paper due to the inclusion of "done"
            transition = [state, action, reward, next_state, done]
            # transition = [state, action, reward, next_state]
            memory.store_transition(transition)
            episode_rewards += reward

            if len(memory) >= config["memory_capacity"]:
                minibatch = memory.random_sample(config["minibatch_size"])
                train_model(minibatch, config["discount"])

            if done:
                print("Episode {} finished after {} timesteps".format(episode+1, t+1))
                break

            # anneal epsilon
            frame_num += 1
            if frame_num <= num_anneal_frames:
                epsilon -= decrease_per_frame

        avg_reward = episode_rewards
        avg_q_value = np.mean(episode_q_values)

    env.close()