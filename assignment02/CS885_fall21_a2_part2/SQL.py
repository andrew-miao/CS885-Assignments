# Author: Yanting Miao
# Soft Q-learning

import gym
import numpy as np
import utils.envs, utils.seed, utils.buffers, utils.torch, utils.common
import torch
import tqdm
import matplotlib.pyplot as plt
import warnings
import torch.nn as nn
from torch.distributions import Categorical
import argparse
warnings.filterwarnings("ignore")

# Deep Q Learning
# Slide 14
# cs.uwaterloo.ca/~ppoupart/teaching/cs885-spring20/slides/cs885-lecture4b.pdf

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

class QNetwork(nn.Module):
    def __init__(self, input_dim=OBS_N, hidden_dim=HIDDEN, output_dim=ACT_N, eps=1e-08):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.eps = eps  # avoid numerical overflow issue
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)
        return out
    
    def value(self, x, alpha):
        # alpha = LAMBDA
        x = self.forward(x)
        y = alpha * torch.logsumexp(x/alpha, dim=-1)
        """
        x = torch.clamp(x / alpha, max=85)  # use clamp to avoid numerical overflow
        y = torch.exp(x)
        y = torch.sum(y, dim=-1)
        y = alpha * torch.log(y + self.eps)"""
        return y

def create_everything(seed):

    utils.seed.seed(seed)
    env = gym.make("CartPole-v0")
    env.seed(seed)
    test_env = gym.make("CartPole-v0")
    test_env.seed(10+seed)
    buf = utils.buffers.ReplayBuffer(BUFSIZE)
    Q = QNetwork().to(DEVICE)
    Qt = QNetwork().to(DEVICE)
    OPT = torch.optim.Adam(Q.parameters(), lr = LEARNING_RATE)
    return env, test_env, buf, Q, Qt, OPT

# Update a target network using a source network
def update(target, source):
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(p.data)

# Create epsilon-greedy policy
def policy(env, obs):

    global EPSILON, Q

    obs = t.f(obs).view(-1, OBS_N)  # Convert to torch tensor
    
    softmax = nn.Softmax()
    action_prob = softmax(Q(obs))
    action = Categorical(action_prob).sample().item()
    return action


# Update networks
def update_networks(epi, buf, Q, Qt, OPT, LAMBDA):
    
    # Sample a minibatch (s, a, r, s', d)
    # Each variable is a vector of corresponding values
    S, A, R, S2, D = buf.sample(MINIBATCH_SIZE, t)
    
    # Get Q(s, a) for every (s, a) in the minibatch
    qvalues = Q(S).gather(1, A.view(-1, 1)).squeeze()


    # Compute TD target
    with torch.no_grad():
        td_target = R + GAMMA * (1 - D) * Qt.value(S2, LAMBDA)


    # Detach y since it is the target. Target values should
    # be kept fixed.
    loss = torch.nn.MSELoss()(qvalues, td_target)

    # Backpropagation
    OPT.zero_grad()
    loss.backward()
    OPT.step()

    # Update target network every few steps
    if epi % TARGET_UPDATE_FREQ == 0:
        update(Qt, Q)

    return loss.item()

# Play episodes
# Training function
def train(seed, LAMBDA=10):

    global EPSILON, Q
    print("Seed=%d" % seed)

    # Create environment, buffer, Q, Q target, optimizer
    env, test_env, buf, Q, Qt, OPT = create_everything(seed)

    # epsilon greedy exploration
    EPSILON = STARTING_EPSILON

    testRs = []
    last25testRs = []
    print("Training:")
    pbar = tqdm.trange(EPISODES)
    for epi in pbar:

        # Play an episode and log episodic reward
        S, A, R = utils.envs.play_episode_rb(env, policy, buf)
        
        # Train after collecting sufficient experience
        if epi >= TRAIN_AFTER_EPISODES:

            # Train for TRAIN_EPOCHS
            for tri in range(TRAIN_EPOCHS): 
                update_networks(epi, buf, Q, Qt, OPT, LAMBDA)

        # Evaluate for TEST_EPISODES number of episodes
        Rews = []
        for epj in range(TEST_EPISODES):
            S, A, R = utils.envs.play_episode(test_env, policy, render = False)
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

def main():
    parser = argparse.ArgumentParser(description='SQL experiment, default: only test temperature=10')
    parser.add_argument('--multialpha', type=bool, default=False, help='Test multiple temperature hyperparameters')
    args = parser.parse_args()
    print('-----------------SQL experiment-----------------')
    if args.multialpha:
        LAMBDA_list = [1, 10, 100, 1000]
        curves = []
        for alpha in LAMBDA_list:
            print('---------alpha = %d-------------' % (alpha))
            for seed in SEEDS:
                curves += [train(seed, alpha)]
        torch.save(curves, 'sql_multialpha_results.pt')
    else:
        curves = []
        for seed in SEEDS:
            curves += [train(seed, 10)]
        curves = np.array(curves).reshape(len(SEEDS), -1)
        torch.save(curves, 'sql_results.pt')
    print('Results saved!')

if __name__ == "__main__":
    main()