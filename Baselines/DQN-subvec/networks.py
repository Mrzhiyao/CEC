import torch
import torch.nn as nn
from torch.distributions import Categorical 
import numpy as np
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()

        self.input_norm = nn.LayerNorm(state_size)
        self.fc1 = nn.Linear(state_size, hidden_size)
        # print('self.fc1', self.fc1)
        self.fc2 = nn.Linear(hidden_size, hidden_size*2)
        # print('self.fc2', self.fc2)
        self.fc4 = nn.Linear(hidden_size*2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        # print('self.fc3', self.fc3)
        self.softmax = nn.Softmax(dim=-1)
        
    def standardize(self, tensor):
        mean = tensor.mean(dim=0, keepdim=True)
        std = tensor.std(dim=0, keepdim=True)
        return (tensor - mean) / std
    
    def forward(self, state):
        # state = self.standardize(state)
        # print('state.shape', state.shape)
        # state = self.input_norm(state)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc4(x))
        action_probs = self.softmax(self.fc3(x))
        return action_probs
    
    def evaluate(self, state, epsilon=1e-6):
        # state = self.standardize(state)
        # state = self.input_norm(state)
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities        
    
    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        # state = self.standardize(state)
        # state = self.input_norm(state)
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities
    
    def get_det_action(self, state):
        # state = self.standardize(state)
        # state = self.input_norm(state)
        action_probs = self.forward(state)
        print('action_probs', action_probs)
        dist = Categorical(action_probs)
        print('dist', dist)
        action = dist.sample().to(state.device)
        return action.detach().cpu()


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_size=256, seed=1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_norm = nn.LayerNorm(state_size)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size*2)
        self.fc4 = nn.Linear(hidden_size*2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
 
    def forward(self, state):
        # state = self.standardize(state)
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # state = self.input_norm(state)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc4(x))
        return self.fc3(x)