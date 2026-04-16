import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from agent_sac_network import Critic, Actor
import numpy as np
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SAC(nn.Module):
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, actor_node, lr_actor, device):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super(SAC, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.device = device
        
        self.gamma = 0.99
        self.tau = 1e-2
        hidden_size = actor_node
        learning_rate = lr_actor
        self.clip_grad_param = 1

        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=device)  # 确保创建时指定设备
        self.target_entropy = torch.tensor(-action_size, device=device)  # 改为设备上的张量
        self.alpha = self.log_alpha.exp().detach()


        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate) 
                
        self.actor_local = Actor(state_size, action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)     
        
        self.critic1 = Critic(state_size, action_size, hidden_size, 2).to(device)
        self.critic2 = Critic(state_size, action_size, hidden_size, 1).to(device)
        
        assert self.critic1.parameters() != self.critic2.parameters()
        
        self.critic1_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate) 

        # 加载模型
        checkpoint = torch.load('sac_pretain.pth', map_location=self.device)
        
        self.actor_local.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        
        with torch.no_grad():
            self.log_alpha.copy_(checkpoint['log_alpha'].to(self.device))
        self.alpha = self.log_alpha.exp().detach()
        
        # 优化器状态加载
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        
        for optimizer in [self.alpha_optimizer, self.actor_optimizer, 
                          self.critic1_optimizer, self.critic2_optimizer]:
            for param in optimizer.param_groups[0]['params']:
                param.data = param.data.to(self.device)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(self.device)



    def get_action(self, state):
        uniform_data = [[float(elem) if isinstance(elem, np.ndarray) else float(elem) for elem in sublist] for sublist in state]

        data_array = np.array(uniform_data)
        state = torch.from_numpy(np.squeeze(data_array)).float().to(self.device)
        """Returns actions for given state as per current policy."""
        with torch.no_grad():
            action = self.actor_local.get_det_action(state)

        action2 = 0
        return action.numpy(), np.array(action2)

    def calc_policy_loss(self, states, alpha):
        _, action_probs, log_pis = self.actor_local.evaluate(states)

        q1 = self.critic1(states)   
        q2 = self.critic2(states)
        min_Q = torch.min(q1,q2)
        actor_loss = (action_probs * (alpha * log_pis - min_Q )).sum(1).mean()
        log_action_pi = torch.sum(log_pis * action_probs, dim=1)
        return actor_loss, log_action_pi
    
    def learn(self, step, experiences, gamma, d=1):
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, log_pis = self.calc_policy_loss(states, current_alpha.to(self.device))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        alpha_loss = -(self.log_alpha.exp() * (log_pis + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ---------------------------- update critic ---------------------------- #
        with torch.no_grad():
            _, action_probs, log_pis = self.actor_local.evaluate(next_states)
            Q_target1_next = self.critic1_target(next_states)
            Q_target2_next = self.critic2_target(next_states)
            Q_target_next = action_probs * (torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * log_pis)

            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * (1 - dones) * Q_target_next.sum(dim=1).unsqueeze(-1)) 

        # Compute critic loss
        q1 = self.critic1(states).gather(1, actions.long())
        q2 = self.critic2(states).gather(1, actions.long())
        
        critic1_loss = 0.5 * F.mse_loss(q1, Q_targets)
        critic2_loss = 0.5 * F.mse_loss(q2, Q_targets)

        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
        
        return actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(), current_alpha

    def soft_update(self, local_model , target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
