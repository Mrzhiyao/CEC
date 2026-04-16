import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
# from torch.distributions import 
from collections import OrderedDict
import random
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim_critic, state_dim_actor, action_dim, actor_node, critic_node, device):
        super(ActorCritic, self).__init__()
        self.device = device
        self.action_dim = action_dim
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
        action_probs = self.actor(state)
        print('action_probs', action_probs)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        action2 = 0
        return action.detach().cpu().numpy(), np.array(action2)


    def evaluate(self, state_for_ppo, state_for_common, action, action_oldlogprobs):

        action_probs = self.actor(state_for_ppo)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_for_common_values = self.critic(state_for_common)
        return action_logprobs, state_for_common_values, dist_entropy


class PPOBC():
    def __init__(self, state_dim_actor, action_dim, actor_node, lr_actor, lr_critic, device, expert_data):
        self.state_dim = state_dim_actor
        self.ac_dim = action_dim
        self.actor_node = actor_node
        self.device = device
        # self._create_networks()
        self.policy = ActorCritic(state_dim_actor, state_dim_actor, action_dim, actor_node, actor_node, self.device).to(
            self.device)
        # 从配置中设置学习率
        self.optimizer_actor = optim.Adam(self.policy.actor.parameters(),
                                          lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.policy.critic.parameters(),
                                           lr=lr_critic)
        self.policy_old = ActorCritic(state_dim_actor, state_dim_actor, action_dim, actor_node, actor_node, self.device).to(
            self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

        # 从配置中设置超参数
        self.eps_clip = 0.2
        self.gamma = 0.99
        self.lamda = 1
        self.max_grad_norm = 0.5
        self.ppo_update_steps = 50
        self.bc_loss_weight = 2.5
        self.num_epochs = 500
        num_pretrain_epochs = 100
        self.K_epochs = 50
        self.pretrain_bc(expert_data, num_pretrain_epochs)
        print('BC预训练完成')

    def pretrain_bc(self, expert_data, num_pretrain_epochs):
        """预训练行为克隆（BC）"""
        obs, actions, rewards, next_obs, dones = expert_data
        # print(len(obs))
        for epoch in range(num_pretrain_epochs):
            bc_loss = self._compute_bc_loss(obs.detach().to(self.device), actions.detach().to(self.device).squeeze(1).long())
            self.optimizer_actor.zero_grad()
            bc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), max_norm=self.max_grad_norm)
            self.optimizer_actor.step()

    def learn(self, steps, experiences, epoch):
        """训练一个批次数据"""
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

        value_loss_list = []
        combined_actor_loss_list = []
        policy_loss_list = []
        bc_loss_list = []
        for _ in range(self.K_epochs):
            # 评估获得logprobs、状态值和熵
            logprobs, states_for_common_values, dist_entropy = self.policy.evaluate(old_states_for_ppo,
                                                                        old_states_for_common, old_actions,
                                                                        old_logprobs)
            
            states_for_common_values = torch.squeeze(states_for_common_values)

            # 计算策略损失 (PPO Clip Loss)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - states_for_common_values.detach()  # 使用分离的值网络输出
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy.mean()
            policy_loss_list.append(policy_loss)
            # 计算 BC 损失并与 PPO 损失结合
            bc_loss = self._compute_bc_loss(states.detach().to(self.device), actions.detach().to(self.device).squeeze(1).long())
            # print('bc_loss', bc_loss)
            bc_loss_list.append(bc_loss)
            dynamic_bc_loss_weight = self.bc_loss_weight * (1 - epoch / self.num_epochs)
            combined_actor_loss = policy_loss + dynamic_bc_loss_weight * bc_loss
            combined_actor_loss_list.append(combined_actor_loss)
            # 计算值函数损失
            value_loss = self.MseLoss(states_for_common_values, rewards)
            value_loss_list.append(value_loss)
            # 分别更新Actor和Critic ----------------------------
            
            # 第一步：仅更新Actor
            self.optimizer_actor.zero_grad()
            combined_actor_loss.mean().backward(retain_graph=True)  # 保留计算图供Critic使用
            torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), max_norm=0.5)
            self.optimizer_actor.step()
            
            # 第二步：仅更新Critic
            self.optimizer_critic.zero_grad()
            value_loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), max_norm=0.5)
            self.optimizer_critic.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        # clear buffer
        # 清空记录?
        # self.buffer.clear()

        model_policy = self.policy_old
        if steps % 100 == 0 and steps !=0:
            torch.save(model_policy.state_dict(), 'model_network_ppobc.pth')
            print('模型保存成功')

        return  policy_loss_list, bc_loss_list, combined_actor_loss_list, value_loss_list

    def _compute_bc_loss(self, obs, actions):
        """计算行为克隆（BC）损失"""
        pretrain_action = self.policy.actor(obs)
        bc_loss = nn.CrossEntropyLoss()(pretrain_action, actions)
        return bc_loss

    def get_action(self, state, eval=False):
        print('sim value:', state[15],state[19],state[23])
        action1, action2 = self.policy.get_action(state)
        return action1, action2

    def save(self, model_path_and_name):
        torch.save(self.policy_old.state_dict(), model_path_and_name)

    def load(self, model_path_and_name):
        self.policy_old.load_state_dict(torch.load(model_path_and_name, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(model_path_and_name, map_location=lambda storage, loc: storage))

