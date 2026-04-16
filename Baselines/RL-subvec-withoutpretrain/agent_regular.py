import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import numpy as np
import math
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import random


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states_for_ppo = []
        self.logprobs = []
        self.rewards = []
        self.states_for_common = []

    def clear(self):
        del self.actions[:]
        del self.states_for_ppo[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.states_for_common[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim_critic, state_dim_actor, action_dim, actor_node, critic_node, device):
        super(ActorCritic, self).__init__()
        self.device = device
        self.action_dim = action_dim
        # print(state_dim_actor,actor_node,action_dim)
        self.actor = nn.Sequential(
            nn.Linear(state_dim_actor, actor_node),
            nn.Tanh(),
            nn.Linear(actor_node, actor_node),
            nn.Tanh(),
            nn.Linear(actor_node, actor_node),
            nn.Tanh(),
            nn.Linear(actor_node, action_dim),
            nn.Softmax(dim=-1)
        )

        # 通过价值输出动作
        self.critic = nn.Sequential(
            nn.Linear(state_dim_critic, critic_node),
            nn.Tanh(),
            nn.Linear(critic_node, critic_node),
            nn.Tanh(),
            nn.Linear(critic_node, critic_node),
            nn.Tanh(),
            nn.Linear(critic_node, 1)
        )
        # 通过策略评估state+action
    def get_new_state(self, state_for_ppo, task_input):
        att = self.linear_layer(self.transen(task_input.view(1, 1, 768)).view(1, 768))
        new_state = torch.cat([state_for_ppo[0:30], att.squeeze()], 0)

        return new_state

    def get_action(self, state, eval=False):

        uniform_data = [[float(elem) if isinstance(elem, np.ndarray) else float(elem) for elem in sublist] for sublist in state]

        data_array = np.array(uniform_data)
        state = torch.from_numpy(np.squeeze(data_array)).float().to(self.device)
        # print('state_true',state.shape)
        action_probs = self.actor(state)
        print('action_probs', action_probs)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        action2 = 0
        return action.detach().cpu().numpy(), np.array(action2)


    def evaluate(self, state_for_ppo, state_for_common, action, action_oldlogprobs):
        # 最初是一样的，更新后可以输出学习后的动作
        # print(state_for_ppo[0])
        # print(action.shape,'action')
        action_probs = self.actor(state_for_ppo)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        # print(action_logprobs.shape,'action_logprobs')
        dist_entropy = dist.entropy()
        state_for_common_values = self.critic(state_for_common)
        return action_logprobs, state_for_common_values, dist_entropy


class PPO:
    def __init__(self, state_dim_critic, state_dim_actor, action_dim, actor_node, critic_node, lr_actor, lr_critic,
                 gamma, K_epochs, eps_clip, device):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim_critic, state_dim_actor, action_dim, actor_node, critic_node, self.device).to(
            self.device)
        self.policy.load_state_dict(torch.load("model_network.pth"))
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic},
        ])
        self.policy_old = ActorCritic(state_dim_critic, state_dim_actor, action_dim, actor_node, critic_node, self.device).to(
            self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.load_state_dict(torch.load("model_network.pth"))
        # print('载入初始化模型')
        self.MseLoss = nn.MSELoss()

    def generate_unique_random_numbers(self, start, end, count):
        numbers = set()

        while len(numbers) < count:
            number = random.randint(start, end)
            numbers.add(number)

        return list(numbers)

    def learn(self, steps, experiences):
        states, actions, exe_rewards, next_states, dones = experiences
        learn_states_for_ppo = states
        learn_states_for_common = states
        learn_actions = actions
        from torch.distributions import Categorical
        # 生成包含10000个子列表的列表，每个子列表包含4个随机数
        # learn_logprobs = [[random.randint(0, 100) for _ in range(4)] for _ in range(len(states))]
        learn_logprobs = []
        for x in range(len(states)):
            if actions[x] == 0:
                dist = Categorical(torch.tensor([1,0,0,0]).to(self.device))

            elif actions[x] == 1:
                dist = Categorical(torch.tensor([0,1,0,0]).to(self.device))

            elif actions[x] == 2:
                dist = Categorical(torch.tensor([0,0,1,0]).to(self.device))

            elif actions[x] == 3:
                dist = Categorical(torch.tensor([0,0,0,1]).to(self.device))
            action_logprob = dist.log_prob(torch.tensor(actions[x]).to(self.device))
            learn_logprobs.append(action_logprob)

        discounted_reward = 0

        rewards = []
        for reward in reversed(torch.squeeze(exe_rewards)):
            # print(reward, 'reward')
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards 归一化奖励值
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        # print('learn_states_for_ppo', learn_states_for_ppo.shape)
        old_states_for_ppo = learn_states_for_ppo.detach().to(self.device)
        old_states_for_common = learn_states_for_ppo.detach().to(self.device)
        old_actions = torch.squeeze(learn_actions).detach().to(self.device)
        old_logprobs = torch.tensor(learn_logprobs).detach().to(self.device)
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            # 动作一直是这个，学习的是采取这个动作的概率，这个变化了。这两个动作的采取概率在最初学习时是相同的，学习一次网络对于相同的动作采取的概率输出就不同了。
            logprobs, states_for_common_values, dist_entropy = self.policy.evaluate(old_states_for_ppo,
                                                                                    old_states_for_common, old_actions,
                                                                                    old_logprobs)
            # match states_for_common_values tensor dimensions with rewards tensor
            states_for_common_values = torch.squeeze(states_for_common_values)

            ratios = torch.exp(logprobs - old_logprobs.detach())
            # Finding Surrogate Loss替代损失函数
            # 优势函数等于奖励-状态价值函数
            advantages = rewards - states_for_common_values.detach()
            surr1 = ratios * advantages

            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(states_for_common_values,
                                                                 rewards) - 0.01 * dist_entropy
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        # clear buffer
        # 清空记录?
        self.buffer.clear()

        model_policy = self.policy_old
        if steps % 10 == 0 and steps !=0:
            torch.save(model_policy.state_dict(), 'model_network.pth')
            print('模型保存成功')
        # self.policy_old.load_state_dict(torch.load("model_network.pth"))

        return loss

    def get_action(self, state, eval=False):

        # uniform_data = [[float(elem) if isinstance(elem, np.ndarray) else float(elem) for elem in sublist] for sublist in state]

        # data_array = np.array(uniform_data)
        # # mean_val = np.mean(data_array)
        # # std_val = np.std(data_array)
        # # standardized_array = (data_array - mean_val) / std_val

        # state = torch.from_numpy(np.squeeze(data_array)).float().to(self.device)
        # # print('state_true',state.shape)
        # print('sim value:', state[15],state[35],state[55])
        print('sim value:', state[15],state[19],state[23])
        action1, action2 = self.policy.get_action(state)

        return action1, action2

    def save(self, model_path_and_name):
        torch.save(self.policy_old.state_dict(), model_path_and_name)

    def load(self, model_path_and_name):
        self.policy_old.load_state_dict(torch.load(model_path_and_name, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(model_path_and_name, map_location=lambda storage, loc: storage))
