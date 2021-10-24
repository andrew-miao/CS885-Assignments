from numpy import dtype
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import gamma, device, batch_size, sequence_length, hidden_size

class DRQN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DRQN, self).__init__()
        # The inputs are two integers giving the dimensions of the inputs and outputs respectively. 
        # The input dimension is the state dimention and the output dimension is the action dimension.
        # This constructor function initializes the network by creating the different layers. 
        # This function now only implements two fully connected layers. Modify this to include LSTM layer(s). 
        
        
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.fc1 = nn.Linear(self.num_inputs, 128)
        self.lstm = nn.LSTM(128, hidden_size, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, hidden=None):
        # The variable x denotes the input to the network. 
        # The hidden variable denotes the hidden state and cell state inputs to the LSTM based network. 
        # The function returns the q value and the output hidden variable information (new cell state and new hidden state) for the given input. 
        # This function now only uses the fully connected layers. Modify this to use the LSTM layer(s).          
        if len(x.size()) == 1:
            x = x.view(-1, self.num_inputs)
        out = F.relu(self.fc1(x))
        if len(out.size()) < 3:
            out = out.unsqueeze(0)
        out, hidden = self.lstm(out, hidden)
        qvalue = self.fc2(out)
        return qvalue, hidden


    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch):
        # The online_net is the variable that represents the first (current) Q network.
        # The target_net is the variable that represents the second (target) Q network.
        # The optimizer is Adam. 
        # Batch represents a mini-batch of memory. Note that the minibatch also includes the rnn state (hidden state) for the DRQN. 

        # This function takes in a mini-batch of memory, calculates the loss and trains the online network. Target network is not trained using back prop. 
        # The loss function is the mean squared TD error that takes the difference between the current q and the target q. 
        # Return the value of loss for logging purposes (optional).

        # Implement this function. Currently, temporary values to ensure that the program compiles. 
        
        obs = torch.stack(batch.state)
        next_obs = torch.stack(batch.next_state)
        actions = torch.stack(batch.action)
        actions = actions.type(torch.long).to(device)
        rewards = torch.stack(batch.reward).to(device)
        mask = torch.stack(batch.mask).to(device)
        rnn_states = torch.stack(batch.rnn_state).to(device)
        h, c = rnn_states[:, :, 0, :], rnn_states[:, :, 1, :]
        h_q, c_q = h[:, 0, :], c[:, 0, :]  # hidden state and cell state for the Q network
        # h_q, c_q = torch.zeros_like(h[:, 0, :]), torch.zeros_like(c[:, 0, :])
        h_q, c_q = torch.unsqueeze(h_q, 0).to(device), torch.unsqueeze(c_q, 0).to(device)
        h_target, c_target = h[:, 1, :], c[:, 1, :]  # hidden state and cell state for the target Q network
        # h_target, c_target = torch.zeros_like(h[:, 1, :]), torch.zeros_like(c[:, 1, :])
        h_target, c_target = torch.unsqueeze(h_target, 0).to(device), torch.unsqueeze(c_target, 0).to(device)
        with torch.no_grad():
            q_target, _ = target_net(next_obs, (h_target, c_target))
            max_q_target, _ = torch.max(q_target, dim=-1)
            q_target = rewards + gamma * mask * max_q_target

        q_value, _ = online_net(obs, (h_q, c_q))
        q_value = torch.gather(q_value, -1, actions.view(q_value.size(0), -1, 1)).squeeze()

        criterion = nn.MSELoss()
        loss = criterion(q_value, q_target)
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=1.0)
        optimizer.step()

    def get_action(self, state, hidden):
        # state represents the state variable. 
        # hidden represents the hidden state and cell state for the LSTM.
        # This function obtains the action from the DRQN. The q value needs to be obtained from the forward function and then a max needs to be computed to obtain the action from the Q values. 
        # Implement this function. 
        # Template code just returning a random action.
        with torch.no_grad():
            q_value, hidden = self.forward(state, hidden)
            action = torch.argmax(q_value, dim=-1)
            if action.size(0) == 1:
                action = action.squeeze()
            action = action.item()
        return action, hidden
