"""
Personal implementation of Playing Atari with Deep Reinforcement Learning, by Mnih et al. (2013)
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

Anson Ho, 2021
"""

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import wandb
from assets.memorybuffer import MemoryBuffer

class ffDQN(nn.Module):
    """
    Creates a simple DQN with fully connected layers.
    This only works on simple environments, so that there
    is no need to learn from pixels

    Args
        - num_inputs depends on the environment 
        - num_outputs is typically the size of the action space
    """
    def __init__(self, num_inputs, num_outputs):
        super(ffDQN, self).__init__()
        self.fc1 = nn.Linear(num_inputs, config["l1_neurons"])
        self.fc2 = nn.Linear(config["l1_neurons"], config["l2_neurons"])
        self.fc3 = nn.Linear(config["l2_neurons"], num_outputs)
        self.loss_function = nn.MSELoss()
    
    def forward_pass(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def epsilon_greedy(epsilon, state):
    """
    Select actions based on an 
    epsilon-greedy policy
    """

    q_values = policy_net.forward_pass(state)
    max_q_value = torch.max(q_values).detach().numpy()

    # explore
    if random.uniform(0,1) < epsilon:
        action = env.action_space.sample()
    # exploit
    else: 
        action = torch.argmax(q_values).item()

    return action, max_q_value

def optimise():
    """
    Performs a single optimisation step,
    given a minibatch of transitions
    """

    if len(memory) < 10 * config["minibatch_size"]:
        return

    minibatch = memory.random_sample(config["minibatch_size"])
    state_minibatch, action_minibatch, reward_minibatch, next_state_minibatch, done_minibatch = tuple(zip(*minibatch))
    state_minibatch, action_minibatch, reward_minibatch, next_state_minibatch, done_minibatch = torch.stack(state_minibatch), torch.tensor(action_minibatch), torch.tensor(reward_minibatch), torch.stack(next_state_minibatch), torch.tensor(done_minibatch)

    # create mask for non-terminal states
    mask = done_minibatch
    non_terminal_next_states = next_state_minibatch[mask]

    # predictions and targets
    next_q_values = torch.zeros((config["minibatch_size"], config["action_space_size"]))
    next_q_values[mask] = target_net.forward_pass(non_terminal_next_states)
    # print(torch.max(next_q_values, dim=1))
    # print(torch.max(next_q_values, dim=1)[0])
    # print(config["discount"])
    targets = torch.add(reward_minibatch, config["discount"] * torch.max(next_q_values, dim=1)[0])
    # predictions = policy_net(state_minibatch).gather(1, action_minibatch).long().unsqueeze(1).squeeze(1)
    pred_q_vals = policy_net.forward_pass(state_minibatch)
    predictions = pred_q_vals.gather(1, action_minibatch.unsqueeze(0))[0]
    
    # optimise
    loss = policy_net.loss_function(predictions, targets)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    return loss

def train(render_screen=True):

    epsilon = config["initial_epsilon"]
    loss, avg_q_value, avg_reward, episode_time = 0, 0, 0, 0

    for episode in range(config["num_episodes"]):
        observation = env.reset()
        episode_q_values = []

        for t in range(config["max_episode_time"]):

            if render_screen:
                env.render()

            state = torch.tensor(observation)
            action, q_value = epsilon_greedy(epsilon, state)
            observation, reward, done, _ = env.step(action)
            next_state = torch.tensor(observation)

            memory.store_transition(state, action, reward, next_state, done)
            episode_q_values.append(q_value)
            loss = optimise()
            
            epsilon -= (config["initial_epsilon"] - config["final_epsilon"]) / config["epsilon_anneal_frames"]

            if WANDB:
                wandb.log({
                    "loss": loss,
                    "epsilon": epsilon,
                    "avg_q_value": avg_q_value,
                    "avg_reward": avg_reward,
                    "episode_time": episode_time
                })

            if done:
                print("Episode {} finished after {} timesteps".format(episode+1, t+1))
                episode_time = t + 1
                break

        avg_reward = episode_time # for cartpole, not in general
        avg_q_value = np.mean(episode_q_values)

    env.close()

if __name__ == "__main__":

    # make environment
    env = gym.make("CartPole-v1")

    config = {
        "learning_rate": 1e-5,
        "minibatch_size": 16,
        "memory_capacity": 10_000, 
        "num_episodes": 5_000,
        "max_episode_time": 10_000,
        "discount": 0.999,
        "initial_epsilon": 1,
        "final_epsilon": 0.1,
        "epsilon_anneal_frames": 100_000,
        "l1_neurons": 32, 
        "l2_neurons": 64,
        "observation_space_size": env.observation_space.shape[0],
        "action_space_size": env.action_space.n
    }

    # logging results
    WANDB = 1
    if WANDB:
        wandb.init(project="DQN-1", entity="ansonwhho", reinit=True)
        wandb.config = config

    memory = MemoryBuffer(config["memory_capacity"])
    policy_net = ffDQN(config["observation_space_size"], config["action_space_size"])
    target_net = ffDQN(config["observation_space_size"], config["action_space_size"])
    optimiser = torch.optim.SGD(policy_net.parameters(), lr=config["learning_rate"])

    train()