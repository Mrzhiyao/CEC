import gym
import random
import torch
import numpy as np

import copy
from collections import namedtuple, deque
import torch.nn as nn

import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LR_ACTOR = 0.0001
LR_CRITIC = 0.001
WEIGHT_DECAY = 0
EPSILON = 1.0
EPSILON_DECAY = 1
TAU = 1e-3
actor_node = 512
import gym
import random
import torch
import numpy as np
import copy
from collections import namedtuple, deque
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import argparse

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model for Discrete Action Space"""

    def __init__(self, state_size, action_size, seed, fc1_units=actor_node, fc2_units=actor_node):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.batch_norm = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)  # Output size = action_size
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> action probabilities"""
        x = F.relu(self.batch_norm(self.fc1(state)))
        x = F.relu(self.fc2(x))
        # Use log softmax for numerical stability
        return F.log_softmax(self.fc3(x), dim=1)

class Critic(nn.Module):
    """Critic (Value) Model for Discrete Actions"""

    def __init__(self, state_size, action_size, seed, fcs1_units=actor_node, fc2_units=actor_node):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.batch_norm = nn.BatchNorm1d(fcs1_units)
        
        # Output Q-values for all actions
        self.fc2 = nn.Linear(fcs1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)  # Output size = action_size
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build a critic (value) network that maps states -> Q-values for all actions"""
        xs = F.relu(self.batch_norm(self.fcs1(state)))
        x = F.relu(self.fc2(xs))
        return self.fc3(x)

class DDPG():
    """Interacts with and learns from the environment for discrete action space"""
    
    def __init__(self, state_dim_actor, action_dim, actor_node, lr_actor, device):
        random_seed = 0
        state_size = state_dim_actor
        action_size = action_dim
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.device = device
        print("Using: ", device)
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)


        checkpoint = torch.load('ddpg_pretain.pth', map_location=self.device)

        # 加载网络权重
        self.actor_local.load_state_dict(checkpoint['actor_local'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_local.load_state_dict(checkpoint['critic_local'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        
        # 加载优化器状态
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.epsilon = EPSILON

        # Initialize target networks same as local networks
        self.soft_update(self.actor_local, self.actor_target, 1.0)
        self.soft_update(self.critic_local, self.critic_target, 1.0)


    def get_action(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        uniform_data = [[float(elem) if isinstance(elem, np.ndarray) else float(elem) for elem in sublist] for sublist in state]

        data_array = np.array(uniform_data)
        state = torch.from_numpy(np.squeeze(data_array)).float().to(self.device)
        self.actor_local.eval()
        if state.dim() == 1:
            state = state.unsqueeze(0)  # 或者 state = state[None, 
        with torch.no_grad():
            log_probs = self.actor_local(state)
            probs = torch.exp(log_probs).cpu().data.numpy().squeeze(0)
        self.actor_local.train()
        
        # Epsilon-greedy exploration
        if add_noise and random.random() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            action = np.argmax(probs)
            
        action2 = 0
        return action, np.array(action2)

    def learn(self, step, experiences, gamma):
        """Update policy and value parameters for discrete action space"""
        states, actions, rewards, next_states, dones = experiences
        
        # Convert actions to long for indexing
        actions = actions.long()
        
        # ---------------------------- update critic ---------------------------- #
        # Get Q-values for all actions from local critic
        Q_expected = self.critic_local(states)
        # Select Q-values for the actions that were actually taken
        Q_expected = Q_expected.gather(1, actions)
        
        # Compute next actions using target actor
        with torch.no_grad():
            next_log_probs = self.actor_target(next_states)
            next_probs = torch.exp(next_log_probs)
            
            # Compute Q-values for next states using target critic
            Q_targets_next = self.critic_target(next_states)
            # Use weighted average based on action probabilities
            Q_targets_next = (next_probs * Q_targets_next).sum(dim=1, keepdim=True)
            
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        log_probs = self.actor_local(states)
        with torch.no_grad():
            # Use local critic values as baseline
            Q_values = self.critic_local(states)
            # Detach to avoid updating critic through actor
            baseline = Q_values.mean(dim=1, keepdim=True).detach()
        
        # Compute advantage
        advantage = (Q_values - baseline).detach()
        
        # Policy gradient loss
        actor_loss = -(log_probs * advantage).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     
        
        # ----------------------- update epsilon for exploration ----------------------- #
        # self.epsilon = max(EPS_MIN, self.epsilon * EPSILON_DECAY)
        self.epsilon *= EPSILON_DECAY

        return critic_loss, actor_loss

    

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



