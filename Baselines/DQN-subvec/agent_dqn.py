import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from networks import Critic, Actor
import numpy as np
import math
import copy
import random
import psutil
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import random
import torch.profiler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    """docstring for Net"""
    def __init__(self, state_dim_actor, action_dim, actor_node, device):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_dim_actor, actor_node)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(actor_node,actor_node)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(actor_node,action_dim)
        self.out.weight.data.normal_(0,0.1)
        self.to(device)  # 将整个网络移到设备


    def forward(self, x):
        if not x.is_cuda and device.type == 'cuda':
            x = x.to(device)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob


LR = 0.01
GAMMA = 0.90
EPISILO = 0.9
MEMORY_CAPACITY = 10000
Q_NETWORK_ITERATION = 100
BATCH_SIZE = 600
action_dim = 4

class DQN():
    """docstring for DQN"""
    def __init__(self, state_dim_actor, action_dim, actor_node, lr_actor, device):
        super(DQN, self).__init__()
        self.device = device
        self.eval_net, self.target_net = Net(state_dim_actor, action_dim, actor_node, device), Net(state_dim_actor, action_dim, actor_node, device)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr_actor)
        self.loss_func = nn.MSELoss()

    def get_action(self, state):

        uniform_data = [[float(elem) if isinstance(elem, np.ndarray) else float(elem) for elem in sublist] for sublist in state]

        data_array = np.array(uniform_data)
        state = torch.from_numpy(np.squeeze(data_array)).float().to(self.device)
        if np.random.randn() <= EPISILO:# greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 0)[1].data
            action = action.item()
        else: # random policy
            action = np.random.randint(0, action_dim)
            action = action 

        action2 = 0
        return action, np.array(action2)


    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self, steps, experiences):
        batch_state, batch_action, batch_reward, batch_next_state, dones = experiences
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)

        q_eval = self.eval_net(batch_state).gather(1, batch_action.long())
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if steps % 100 == 0 and steps !=0:
            checkpoint = {
                'eval_net_state_dict': self.eval_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'learn_step_counter': self.learn_step_counter
            }
            torch.save(checkpoint, 'dqn_checkpoint.pth')
            print('完整训练状态保存成功')
        return loss