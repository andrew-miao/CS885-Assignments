# Author: Yanting Miao
# Modern SAC discrete, my code is different from Lecture slides
# Reference: https://arxiv.org/pdf/1812.05905.pdf
from os import stat
import gym
import numpy as np
from torch.distributions.gamma import Gamma
from torch.nn.modules.activation import Softmax
import utils.envs, utils.seed, utils.buffers, utils.torch, utils.common
import torch
import tqdm
import matplotlib.pyplot as plt
import warnings
import torch.nn as nn
from torch.distributions import Categorical
warnings.filterwarnings("ignore")

# Constants
SEEDS = [1,2,3,4,5]
t = utils.torch.TorchHelper()
DEVICE = t.device
OBS_N = 4               # State space size
ACT_N = 2               # Action space size
MINIBATCH_SIZE = 64     # How many examples to sample per train step
GAMMA = 0.99            # Discount factor in episodic reward objective
LEARNING_RATE = 5e-4    # Learning rate for Adam optimizer
TRAIN_AFTER_EPISODES = 10   # Just collect episodes for these many episodes
TRAIN_EPOCHS = 25       # Train for these many epochs every time
BUFSIZE = 10000         # Replay buffer size
EPISODES = 300          # Total number of episodes to learn over
TEST_EPISODES = 1       # Test episodes after every train episode
HIDDEN = 512            # Hidden nodes
TARGET_UPDATE_FREQ = 10 # Target network update frequency
STARTING_EPSILON = 1.0  # Starting epsilon
STEPS_MAX = 20000       # Gradually reduce epsilon over these many steps
EPSILON_END = 0.01      # At the end, keep epsilon at this value

# Global variables
EPSILON = STARTING_EPSILON
Q = None

# Create environment
# Create replay buffer
# Create network for Q(s, a)
# Create target network
# Create optimizer
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.zeros_(m.bias)

class SoftmaxActor(nn.Module):
    def __init__(self, input_dim=OBS_N, hidden_dim=HIDDEN, output_dim=ACT_N):
        super(SoftmaxActor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        # self.apply(weights_init)

    def forward(self, x):
        if isinstance(x, (np.ndarray, np.generic)):
            x = t.f(x).view(-1, OBS_N)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        prob = self.softmax(self.fc3(x))
        return prob

class QNetwork(nn.Module):
    def __init__(self, input_dim=OBS_N, hidden_dim=HIDDEN, output_dim=ACT_N):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.fc4 = nn.Linear(input_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        # self.apply(weights_init)

    def forward(self, x):
        x1 = self.relu(self.fc1(x))
        x1 = self.relu(self.fc2(x1))
        x1 = self.fc3(x1)

        x2 = self.relu(self.fc4(x))
        x2 = self.relu(self.fc5(x2))
        x2 = self.fc6(x2)
        return x1, x2

# Play an episode according to a given policy
# env: environment
# policy: function(env, state)
# render: whether to render the episode or not (default - False)
def play_episode(env, policy, render = False):
    states, actions, rewards = [], [], []
    states.append(env.reset())
    done = False
    if render: env.render()
    while not done:
        action = Categorical(policy(states[-1])).sample().item()
        actions.append(action)
        obs, reward, done, info = env.step(action)
        if render: env.render()
        states.append(obs)
        rewards.append(reward)
    return states, actions, rewards

# Play an episode according to a given policy and add 
# to a replay buffer
# env: environment
# policy: function(env, state)
def play_episode_rb(env, policy, buf):
    states, actions, rewards = [], [], []
    states.append(env.reset())
    done = False
    while not done:
        action = Categorical(policy(states[-1])).sample().item()
        actions.append(action)
        obs, reward, done, info = env.step(action)
        buf.add(states[-1], action, reward, obs, done)
        states.append(obs)
        rewards.append(reward)
    return states, actions, rewards

def create_everything(seed):
    utils.seed.seed(seed)
    env = gym.make("CartPole-v0")
    env.seed(seed)
    test_env = gym.make("CartPole-v0")
    test_env.seed(10+seed)
    buf = utils.buffers.ReplayBuffer(BUFSIZE)

    critic = QNetwork().to(DEVICE)
    critic_target = QNetwork().to(DEVICE)
    actor = SoftmaxActor().to(DEVICE)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr = LEARNING_RATE)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    return env, test_env, buf, critic, critic_target, actor, critic_optimizer, actor_optimizer

# Update a target network using a source network
def update(target, source):
    rou = 0.005
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(rou * p.data + (1 - rou) * tp.data)

# Update networks
def update_networks(epi, buf, critic, critic_target, actor, critic_optimizer, actor_optimizer, LAMBDA):
    state, action, reward, next_state, done = buf.sample(MINIBATCH_SIZE, t)
    # sample next action from pi
    next_probs = actor(next_state)
    next_actions = Categorical(next_probs).sample()

    # update critic network
    with torch.no_grad():
        q1_target, q2_target = critic_target(next_state)
        q1_target = q1_target.gather(1, next_actions.view(-1, 1)).squeeze()
        q2_target = q2_target.gather(1, next_actions.view(-1, 1)).squeeze()
        min_q_target = torch.min(q1_target, q2_target)
        critic_target_values = reward + GAMMA * (1 - done) * (min_q_target + LAMBDA * Categorical(next_probs).entropy())
    q1_value, q2_value = critic(state)
    q1_value = q1_value.gather(1, action.view(-1, 1)).squeeze()
    q2_value = q2_value.gather(1, action.view(-1, 1)).squeeze()
    critic_criterion = nn.MSELoss()
    q1_loss = critic_criterion(q1_value, critic_target_values)
    q2_loss = critic_criterion(q2_value, critic_target_values)
    critic_loss = q1_loss + q2_loss
    critic.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # update actor network
    softmax = nn.Softmax(dim=-1)
    probs = actor(state)
    with torch.no_grad():
        q1_value, q2_value = critic(state)
        min_q = torch.min(q1_value, q2_value)
        # target_distribution = torch.sum(min_q * probs, dim=1, keepdim=True)
    actor_loss = torch.mean(probs * (LAMBDA * torch.log(probs + 1e-16) - min_q))
    actor.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    update(critic_target, critic)
    # if epi % TARGET_UPDATE_FREQ == 0:
    #     update(critic_target, critic)

# Play episodes
# Training function
def train(seed, LAMBDA):

    global EPSILON, Q
    print("Seed=%d" % seed)

    # Create environment, buffer, Q, Q target, optimizer
    env, test_env, buf, critic, critic_target, actor, critic_optimizer, actor_optimizer = create_everything(seed)

    # epsilon greedy exploration
    EPSILON = STARTING_EPSILON

    testRs = []
    last25testRs = []
    print("Training:")
    pbar = tqdm.trange(EPISODES)
    for epi in pbar:

        # Play an episode and log episodic reward
        play_episode_rb(env, actor, buf)
        
        # Train after collecting sufficient experience
        if epi >= TRAIN_AFTER_EPISODES:

            # Train for TRAIN_EPOCHS
            for tri in range(TRAIN_EPOCHS): 
                update_networks(epi, buf, critic, critic_target, actor, critic_optimizer, actor_optimizer, LAMBDA)

        # Evaluate for TEST_EPISODES number of episodes
        Rews = []
        for epj in range(TEST_EPISODES):
            _, __, R = play_episode(test_env, actor, render=False)
            Rews += [sum(R)]
        testRs += [sum(Rews)/TEST_EPISODES]

        # Update progress bar
        last25testRs += [sum(testRs[-25:])/len(testRs[-25:])]
        pbar.set_description("R25(%g)" % (last25testRs[-1]))

    # Close progress bar, environment
    pbar.close()
    print("Training finished!")
    env.close()
    test_env.close()

    return last25testRs

# Plot mean curve and (mean-std, mean+std) curve with some transparency
# Clip the curves to be between 0, 200
def plot_arrays(vars, color, label):
    mean = np.mean(vars, axis=0)
    std = np.std(vars, axis=0)
    plt.plot(range(len(mean)), mean, color=color, label=label)
    plt.fill_between(range(len(mean)), np.maximum(mean-std, 0), np.minimum(mean+std,200), color=color, alpha=0.3)

if __name__ == "__main__":

    # Train for different seeds
    curves = []
    for seed in SEEDS:
        curves += [train(seed, LAMBDA=10)]

    # Plot the curve for the given seeds
    plot_arrays(curves, 'b', 'SAC Discrete')
    plt.legend(loc='best')
    plt.savefig('SAC Discrete.png')
    plt.show()