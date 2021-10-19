import math, random
import gym
import gym_cartpolemod
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
import matplotlib.pyplot as plt


USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)



def check_state(state):

    for i in range(len(state[0])): 
        if isinstance(state[0][i], np.ndarray):
            state[0][i] = state[0][i][0]

    state = state[0]
    state = list(state)
    return state


    

from collections import deque

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)
env = gym.make('CartPoleMod-v1')

epsilon = 0.01


class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n)
        )
        
    def forward(self, x):
        return self.layers(x)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state = Variable(torch.FloatTensor(state).unsqueeze(0))
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(env.action_space.n)
        return action

model = DQN(env.observation_space.shape[0], env.action_space.n)
target_model = DQN(env.observation_space.shape[0], env.action_space.n)

def update_target(model, target_model):
    target_model.load_state_dict(model.state_dict())

update_target(model, target_model)

if USE_CUDA:
    model = model.cuda()
    target_model = target_model.cuda()
    
optimizer = optim.Adam(model.parameters())

replay_buffer = ReplayBuffer(1000)


def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
    action     = Variable(torch.LongTensor(action))
    reward     = Variable(torch.FloatTensor(reward))
    done       = Variable(torch.FloatTensor(done))

    q_values      = model(state)
    next_q_values = target_model(next_state)

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss


def plot(rewards):
    
    plt.figure(figsize=(20,5))
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.savefig('DQN.png')
    print('plotted')



batch_size = 32
gamma      = 0.9
total_episodes = 200
state_size = env.observation_space.shape[0]
episode = 0 
frame_idx = 0
losses = []
all_rewards = []
episode_reward = 0
episode = 0
state = env.reset()
state = np.reshape(state, [1, state_size])
state = check_state(state)
while True:
    action = model.act(state, epsilon)
    if(torch.is_tensor(action)): 
        action = action.tolist()
    next_state, reward, done, _ = env.step(action)
    next_state = np.reshape(next_state, [1, state_size])
    next_state = check_state(next_state)
    replay_buffer.push(state, action, reward, next_state, done) 
    state = next_state
    episode_reward += reward

    frame_idx = frame_idx + 1
    if done:
        print("Just completed episode", episode)
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        state = check_state(state)
        episode = episode + 1
        all_rewards.append(episode_reward)
        print("The reward is", episode_reward)
        episode_reward = 0
    if episode >= total_episodes: 
        break

    if frame_idx % 100 == 0:
        update_target(model, target_model)

    if len(replay_buffer) > batch_size:
        loss = compute_td_loss(batch_size)
        new_loss = loss.data.tolist()
        losses.append(new_loss)
        
plot(all_rewards)






