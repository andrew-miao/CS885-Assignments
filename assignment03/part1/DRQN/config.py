import torch

env_name = 'CartPole-v1'
gamma = 0.99
batch_size = 32
lr = 3e-04
initial_exploration = 1000
goal_score = 200
log_interval = 10
update_target = 120
replay_memory_capacity = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_epsilon = 0.05
sequence_length = 4
hidden_size = 16