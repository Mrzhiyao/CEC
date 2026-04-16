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

class CQLSAC(nn.Module):
    """Interacts with and learns from the environment."""
    
    def __init__(self,
                        state_size,
                        action_size,
                        device
                ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super(CQLSAC, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.device = device
        
        self.gamma = 0.99
        self.tau = 1e-2
        hidden_size = 256
        learning_rate = 1e-4
        # learning_rate = 1e-3
        self.clip_grad_param = 1

        self.target_entropy = -action_size  # -dim(A)

        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate) 
        
        # CQL params
        self.with_lagrange = False
        self.temp = 1.0
        self.cql_weight = 1.0
        self.target_action_gap = 0.0
        self.cql_log_alpha = torch.zeros(1, requires_grad=True)
        self.cql_alpha_optimizer = optim.Adam(params=[self.cql_log_alpha], lr=learning_rate) 
        
        # Actor Network 

        self.actor_local = Actor(state_size, action_size, hidden_size).to(device)
        # self.actor_local.load_state_dict(torch.load("E:/offline-rl/RL/trained_models/CQL-SAC-discreteCQL-test299.0.pth"))
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)     
        
        # Critic Network (w/ Target Network)

        self.critic1 = Critic(state_size, action_size, hidden_size, 2).to(device)
        self.critic2 = Critic(state_size, action_size, hidden_size, 1).to(device)
        
        assert self.critic1.parameters() != self.critic2.parameters()
        
        self.critic1_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate) 
        self.softmax = nn.Softmax(dim=-1)

    
    def get_action(self, state, eval=False):
        """Returns actions for given state as per current policy."""
        # print('ssss',state)
        # flattened_data = [
        #     item.item() if isinstance(item, np.ndarray) else item
        #     for sublist in state for item in sublist
        # ]
        uniform_data = [[float(elem) if isinstance(elem, np.ndarray) else float(elem) for elem in sublist] for sublist in state]
        # print(uniform_data)
        # flattened_data = [item.item() if isinstance(item, np.ndarray) else item for sublist in data for item in sublist]
        # data_out = [[item.item() if isinstance(item, np.ndarray) else item for item in sublist] for sublist in data]

        # 转换为NumPy数组
        # data_array = np.array(uniform_data)
        # mean_val = np.mean(data_array)
        # std_val = np.std(data_array)
        # standardized_array = (data_array - mean_val) / std_val

        # state = torch.from_numpy(np.squeeze(standardized_array)).float().to(self.device)
        # print('state_true',state.shape)
        with torch.no_grad():
            action = self.actor_local.get_det_action(state)

        action2 = 0
        return action.numpy(), np.array(action2)

    def get_action_algorithm1(self, state):
        # print('state', state)
        print('sim value:', state[18],state[38],state[58])
        """Returns actions for given state as per current policy."""
        if state[18] < 0.6 and state[38] < 0.6 and state[58] < 0.6:
            action1 = 3
            print('都低于0.6')
        elif state[18] >= state[38] and state[18] >= state[58]:
            action1 = 0
        elif state[38] >= state[18] and state[38] >= state[58]:
            action1 = 1
        elif state[58] >= state[18] and state[58] >= state[38]:
            action1 = 2
        else:
            print('都不符合')
            action1 = random.randint(0, 3)
        action2 = 0
        return np.array(action1), np.array(action2)

    def calc_policy_loss(self, states, alpha):
        _, action_probs, log_pis = self.actor_local.evaluate(states)

        q1 = self.critic1(states)   
        q2 = self.critic2(states)
        min_Q = torch.min(q1,q2)
        actor_loss = (action_probs * (alpha.to(self.device) * log_pis - min_Q )).sum(1).mean()
        log_action_pi = torch.sum(log_pis * action_probs, dim=1)
        return actor_loss, log_action_pi
    
    def learn(self, step, experiences, gamma, d=1):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, log_pis = self.calc_policy_loss(states, current_alpha)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Compute alpha loss
        alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            _, action_probs, log_pis = self.actor_local.evaluate(next_states)
            Q_target1_next = self.critic1_target(next_states)
            Q_target2_next = self.critic2_target(next_states)
            Q_target_next = action_probs * (torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * log_pis)

            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * (1 - dones) * Q_target_next.sum(dim=1).unsqueeze(-1)) 
            # print('action_probs', action_probs.shape)
            # print('next_states', next_states.shape)
            # print('Q_target1_next', Q_target1_next.shape)
            # print('Q_target2_next', Q_target2_next.shape)
            # print('Q_target_next', Q_target_next.shape)
            # print('rewards', rewards.shape)
            # print('dones', dones.shape)
        
        # Compute critic loss
        q1 = self.critic1(states)
        # print('q1', type(q1), q1.shape)
        # print('actions', actions.long().shape)
        q2 = self.critic2(states)
        # print(q1)
        # print('-------------------------------------------------------------------------')
        # print('actions.long()', actions.long().shape)
        # print('q1', q1.shape)
        # print('q2', q1.shape)
        q1_ = q1.gather(1, actions.long())
        q2_ = q2.gather(1, actions.long())
        # print('q1_', q1_.shape)
        # print('q2_', q1_.shape)
        # print('Q_targets', Q_targets.shape)

        critic1_loss = 0.5 * F.mse_loss(q1_, Q_targets)
        critic2_loss = 0.5 * F.mse_loss(q2_, Q_targets)
        
        cql1_scaled_loss = torch.logsumexp(q1, dim=1).mean() - q1.mean()
        cql2_scaled_loss = torch.logsumexp(q2, dim=1).mean() - q2.mean()
        
        cql_alpha_loss = torch.FloatTensor([0.0])
        cql_alpha = torch.FloatTensor([0.0])
        if self.with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), min=0.0, max=1000000.0).to(self.device)
            cql1_scaled_loss = cql_alpha * (cql1_scaled_loss - self.target_action_gap)
            cql2_scaled_loss = cql_alpha * (cql2_scaled_loss - self.target_action_gap)

            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss = (- cql1_scaled_loss - cql2_scaled_loss) * 0.5 
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optimizer.step()
        
        total_c1_loss = critic1_loss + cql1_scaled_loss
        total_c2_loss = critic2_loss + cql2_scaled_loss
        
        
        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        total_c1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        total_c2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
        
        return actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(), cql1_scaled_loss.item(), cql2_scaled_loss.item(), current_alpha, cql_alpha_loss.item(), cql_alpha.item()

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
