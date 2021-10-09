# Author: Yanting Miao
# Modern SAC continuous
# Reference: https://arxiv.org/pdf/1812.05905.pdf
from os import stat
import gym
import numpy as np
from torch.distributions.gamma import Gamma
from torch.nn.modules.activation import Softmax
from torch.types import Device
import utils.envs, utils.seed, utils.buffers, utils.torch, utils.common
from utils.envs import NormalizeBoxActionWrapper
import torch
import tqdm
import matplotlib.pyplot as plt
import warnings
import torch.nn as nn
from torch.distributions import Normal
import collections
import random
warnings.filterwarnings("ignore")

# Constants
SEEDS = [1,2,3,4,5]
t = utils.torch.TorchHelper()
DEVICE = t.device

MINIBATCH_SIZE = 256   # How many examples to sample per train step
GAMMA = 0.99            # Discount factor in episodic reward objective
LEARNING_RATE = 3e-4    # Learning rate for Adam optimizer
TRAIN_AFTER_EPISODES = 10   # Just collect episodes for these many episodes
TRAIN_EPOCHS = 25       # Train for these many epochs every time
BUFSIZE = 10000         # Replay buffer size
EPISODES = 300          # Total number of episodes to learn over
TEST_EPISODES = 1       # Test episodes after every train episode
HIDDEN = 256        # Hidden nodes
TARGET_UPDATE_FREQ = 10 # Target network update frequency
STARTING_EPSILON = 1.0  # Starting epsilon
STEPS_MAX = 20000       # Gradually reduce epsilon over these many steps
EPSILON_END = 0.01      # At the end, keep epsilon at this value
LOG_STD_MAX = 2         # Log std max value
LOG_STD_MIN = -20       # Log std min value
rou = 0.005             # Hyperparameter for smoothing update Q target network
initial_alpha = 0.20     # Initial alpha

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
        nn.init.xavier_uniform_(m.weight, 1)
        nn.init.zeros_(m.bias)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        # input dim = state dim + action dim
        input_dim = state_dim + action_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.fc4 = nn.Linear(input_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.apply(weights_init)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x1 = self.relu(self.fc1(x))
        x1 = self.relu(self.fc2(x1))
        x1 = self.fc3(x1)

        x2 = self.relu(self.fc4(x))
        x2 = self.relu(self.fc5(x2))
        x2 = self.fc6(x2)
        return x1, x2

class GaussianPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, epsilon=1e-06):
        super(GaussianPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.epsilon = epsilon
        self.apply(weights_init)
    
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal = Normal(mean, std)
        raw_action = normal.rsample() # allow backpropagate when using reparameterization trick
        action = torch.tanh(raw_action)
        log_prob = normal.log_prob(raw_action) - torch.log(1 - action.pow(2) + self.epsilon)
        log_prob = torch.sum(log_prob, dim=-1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean

# Play an episode according to a given policy
# env: environment
# policy: function(env, state)
# render: whether to render the episode or not (default - False)
def play_episode(env, policy, render = False):
    states, actions, rewards = [], [], []
    states.append(env.reset())
    done = False
    if render: env.render()
    with torch.no_grad():
        while not done:
            state = states[-1]
            if isinstance(state, (np.ndarray, np.generic)):
                state = torch.from_numpy(state).view(1, -1)
            _, __, action = policy.sample(state)
            action = action.detach().cpu().numpy()
            actions.append(action)
            obs, reward, done, info = env.step(action)
            obs = torch.from_numpy(obs).view(1, -1)
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
        state = states[-1]
        if isinstance(state, (np.ndarray, np.generic)):
            state = torch.from_numpy(state).view(1, -1)
        action, _, __ = policy.sample(state)
        action = action.detach().cpu().numpy()
        actions.append(action)
        obs, reward, done, info = env.step(action)
        obs = torch.from_numpy(obs).view(1, -1)
        buf.add(state, action, reward, obs, done)
        states.append(obs)
        rewards.append(reward)
    return states, actions, rewards

# Replay buffer
class ReplayBuffer:
    
    # create replay buffer of size N
    def __init__(self, N):
        self.buf = collections.deque(maxlen = N)
    
    # add: add a transition (s, a, r, s2, d)
    def add(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))
    
    # sample: return minibatch of size n
    def sample(self, n, t):
        minibatch = random.sample(self.buf, n)
        S, A, R, S2, D = [], [], [], [], []
        
        for mb in minibatch:
            s, a, r, s2, d = mb
            a = torch.from_numpy(a)
            S += [s]; A += [a]; R += [r]; S2 += [s2]; D += [d]
        S = torch.cat(S, 0)
        S2 = torch.cat(S2, 0)
        A = torch.cat(A, 0)
        return S.to(DEVICE), A.to(DEVICE), t.f(R).to(DEVICE), S2.to(DEVICE), t.i(D).to(DEVICE)

def create_everything(seed):
    utils.seed.seed(seed)
    env = NormalizeBoxActionWrapper(gym.make('Pendulum-v0'))
    env.seed(seed)
    test_env = NormalizeBoxActionWrapper(gym.make('Pendulum-v0'))
    test_env.seed(10+seed)
    buf = ReplayBuffer(BUFSIZE)

    critic = QNetwork(env.observation_space.shape[0], env.action_space.shape[0]).to(DEVICE)
    critic_target = QNetwork(env.observation_space.shape[0], env.action_space.shape[0]).to(DEVICE)
    actor = GaussianPolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0]).to(DEVICE)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE)

    log_alpha = torch.tensor(np.log(initial_alpha), requires_grad=True).to(DEVICE)
    alpha_optim = torch.optim.Adam([log_alpha], lr=LEARNING_RATE)
    return env, test_env, buf, critic, critic_target, actor, critic_optimizer, actor_optimizer, log_alpha, alpha_optim

# Update a target network using a source network
def update(target, source):
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(rou * p.data + (1 - rou) * tp.data)

# Update networks
def update_networks(buf, critic, critic_target, actor, critic_optimizer, actor_optimizer, log_alpha, alpha_optimizer):
    state, action, reward, next_state, done = buf.sample(MINIBATCH_SIZE, t)
    alpha = log_alpha.exp()
    # update critic network
    with torch.no_grad():
        next_action, next_log_prob, _ = actor.sample(next_state)
        q1_target, q2_target = critic_target(next_state, next_action)
        min_q_target = torch.min(q1_target, q2_target)
        critic_target_values = reward + GAMMA * (1 - done) * (min_q_target - alpha.detach() * next_log_prob)
    q1_value, q2_value = critic(state, action)
    critic_criterion = nn.MSELoss()
    q1_loss = critic_criterion(q1_value, critic_target_values)
    q2_loss = critic_criterion(q2_value, critic_target_values)
    critic_loss = q1_loss + q2_loss
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # update policy network
    action, log_prob, _ = actor.sample(state)
    q1_value, q2_value = critic(state, action)
    min_q = torch.min(q1_value, q2_value)
    actor_loss = torch.mean(alpha.detach() * log_prob - min_q)
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # update alpha
    alpha_loss = (alpha *(-log_prob - (-1)).detach()).mean()
    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()
    update(critic_target, critic)

# Play episodes
# Training function
def train(seed):

    global EPSILON, Q
    print("Seed=%d" % seed)

    # Create environment, buffer, Q, Q target, optimizer
    env, test_env, buf, critic, critic_target, actor, critic_optimizer, actor_optimizer, log_alpha, alpha_optimizer = create_everything(seed)

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
                update_networks(buf, critic, critic_target, actor, critic_optimizer, actor_optimizer, log_alpha, alpha_optimizer)

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
    plt.fill_between(range(len(mean)), mean-std, mean+std, color=color, alpha=0.3)
    plt.legend(loc='upper left')
    plt.grid()
    plt.savefig('SAC Continuous.png')
    plt.show()

if __name__ == "__main__":

    # Train for different seeds
    curves = []
    for seed in SEEDS:
        curves += [train(seed)]

    # Plot the curve for the given seeds
    curves = np.array(curves)
    curves = curves.reshape(len(SEEDS), -1)
    plot_arrays(curves, 'b', 'SAC Continuous')