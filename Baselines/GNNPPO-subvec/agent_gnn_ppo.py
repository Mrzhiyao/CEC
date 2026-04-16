import torch
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
from torch_geometric.data import Data
import torch
import torch.nn as nn
import os 

class GraphFeatureExtractor(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim):
        super().__init__()
        self.node_encoder = torch.nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = torch.nn.Linear(edge_dim, hidden_dim)
        self.global_encoder = torch.nn.Linear(global_dim, hidden_dim)
        
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=hidden_dim, heads=4, concat=False)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=hidden_dim)
        
        self.node_out = torch.nn.Linear(hidden_dim, hidden_dim)
        self.global_out = torch.nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, data):
        x = F.relu(self.node_encoder(data.x))
        edge_attr = F.relu(self.edge_encoder(data.edge_attr))

        if len(data.global_features.shape) == 1:
            global_tensor = data.global_features.unsqueeze(0)
        else:
            global_tensor = data.global_features
        u = F.relu(self.global_encoder(global_tensor))
        
        # 使用索引操作广播全局特征到每个节点
        if isinstance(data, Batch):
            u_nodes = u[data.batch]
        else:
            u_nodes = u.expand(x.size(0), -1)
        
        x = x + u_nodes
        x = F.relu(self.conv1(x, data.edge_index, edge_attr))
        x = F.relu(self.conv2(x, data.edge_index, edge_attr))
        
        node_repr = self.node_out(x)
        global_repr = global_mean_pool(x, data.batch)
        global_repr = self.global_out(global_repr)
        
        return node_repr, global_repr

class PPOPolicyNetwork(torch.nn.Module):
    def __init__(self, node_hidden_dim, global_hidden_dim, action_dim):
        super().__init__()
        # 策略网络（Actor）
        self.actor_node = torch.nn.Linear(node_hidden_dim, 32)
        self.actor_global = torch.nn.Linear(global_hidden_dim, 32)
        self.actor_merge = torch.nn.Linear(64, 32)
        self.actor_out = torch.nn.Linear(32, action_dim)
        
        # 价值网络（Critic）
        self.critic_node = torch.nn.Linear(node_hidden_dim, 32)
        self.critic_global = torch.nn.Linear(global_hidden_dim, 32)
        self.critic_merge = torch.nn.Linear(64, 32)
        self.critic_out = torch.nn.Linear(32, 1)
    
    def forward(self, node_features, global_features, batch=None):
        """
        batch: 形状 [总节点数]，表示每个节点属于哪个图
        """
        if batch is None:
            # 单图模式
            actor_global = F.relu(self.actor_global(global_features))
            critic_global = F.relu(self.critic_global(global_features))
            
            actor_global_expanded = actor_global.expand(node_features.size(0), -1)
            critic_global_expanded = critic_global.expand(node_features.size(0), -1)
        else:
            # 批处理模式
            actor_global = F.relu(self.actor_global(global_features))
            critic_global = F.relu(self.critic_global(global_features))
            
            # 为每个节点获取对应的全局特征
            actor_global_expanded = actor_global[batch]
            critic_global_expanded = critic_global[batch]
        
        # Actor部分：生成策略
        actor_node = F.relu(self.actor_node(node_features))
        actor_combined = torch.cat([actor_node, actor_global_expanded], dim=1)
        actor_hidden = F.relu(self.actor_merge(actor_combined))
        action_logits = self.actor_out(actor_hidden)
        
        # 创建分布并采样动作
        action_dist = Categorical(logits=action_logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        # Critic部分：评估价值
        critic_node = F.relu(self.critic_node(node_features))
        critic_combined = torch.cat([critic_node, critic_global_expanded], dim=1)
        critic_hidden = F.relu(self.critic_merge(critic_combined))
        value = self.critic_out(critic_hidden).squeeze(-1)
        
        return action, log_prob, value, action_dist, action_logits
    
    def evaluate(self, node_features, global_features, actions, batch=None):
        """评估给定动作的对数概率和状态价值"""
        if batch is None:
            # 单图模式
            actor_global = F.relu(self.actor_global(global_features))
            actor_global_expanded = actor_global.expand(node_features.size(0), -1)
        else:
            # 批处理模式
            actor_global = F.relu(self.actor_global(global_features))
            actor_global_expanded = actor_global[batch]
        
        # Actor部分：计算给定动作的对数概率
        actor_node = F.relu(self.actor_node(node_features))
        actor_combined = torch.cat([actor_node, actor_global_expanded], dim=1)

        # print('actor_combined', actor_combined.shape)
        actor_hidden = F.relu(self.actor_merge(actor_combined))
        action_logits = self.actor_out(actor_hidden)
        
        # 创建分布并计算对数概率
        action_dist = Categorical(logits=action_logits)
        logprobs = action_dist.log_prob(actions)
        dist_entropy = action_dist.entropy()
        

        if batch is None:
            # 单图模式
            critic_global = F.relu(self.critic_global(global_features))
            critic_global_expanded = critic_global.expand(node_features.size(0), -1)
        else:
            # 批处理模式
            critic_global = F.relu(self.critic_global(global_features))
            critic_global_expanded = critic_global[batch]

        critic_node = F.relu(self.critic_node(node_features))
        critic_global = F.relu(self.critic_global(global_features))
        critic_combined = torch.cat([critic_node, critic_global_expanded], dim=1)
        critic_hidden = F.relu(self.critic_merge(critic_combined))
        state_for_common_values = self.critic_out(critic_hidden).squeeze(-1)
        return logprobs,state_for_common_values, dist_entropy

LR = 0.01
GAMMA = 0.90
EPISILO = 0.9
MEMORY_CAPACITY = 10000
Q_NETWORK_ITERATION = 100
BATCH_SIZE = 600
action_dim = 4

class GNNPPO:
    def __init__(self, config, state_dim_actor, action_dim, actor_node, lr_actor, device):
        self.node_dim = config.get('node_dim')
        self.edge_dim = config.get('edge_dim')
        self.global_dim = config.get('global_dim')
        self.hidden_dim = config.get('hidden_dim', 512)
        self.action_dim = config.get('action_dim', 4)
        self.lr = config.get('lr', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.eps_clip = config.get('eps_clip', 0.2)
        self.device = device
        self.feature_extractor = GraphFeatureExtractor(
            self.node_dim, self.edge_dim, self.global_dim, self.hidden_dim
        ).to(self.device)

        self.policy = PPOPolicyNetwork(
            self.hidden_dim, self.hidden_dim, self.action_dim
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + list(self.policy.parameters()),
            lr=self.lr
        )
        self.MseLoss = nn.MSELoss()
        # 存储经验
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

        self.K_epochs = 50
    
    def process_graph(self, graph_data):
        """处理单个图数据"""
        if not isinstance(graph_data, Batch):
            graph_data = Batch.from_data_list([graph_data])
        
        # 获取特征表示
        node_features, global_features = self.feature_extractor(graph_data)
        # 为策略网络分离全局特征
        batch_size = len(torch.unique(graph_data.batch))
        global_features = global_features[:batch_size]
        
        return node_features, global_features, graph_data.batch
    
    def get_gnn_data(self, custom_x, custom_edge_index, custom_edge_attr, custom_global_features, custom_y):
        custom_graph = Data(
            x=custom_x,                   # 节点特征 [num_nodes, feature_dim]
            edge_index=custom_edge_index,  # 边索引 [2, num_edges]
            edge_attr=custom_edge_attr,    # 边特征 [num_edges, feature_dim]
            global_features=custom_global_features,  # 全局特征 [1, global_feature_dim]
            y=custom_y                    # 目标标签（可选）
        )

        return custom_graph

    def get_action(self, states, edge_index, edge_feature):
        # """获取图中每个节点的动作/标签"""
        custom_x = []
        custom_edge_index = []
        
        custom_edge_attr = []
        sub_custom_edge_attr = []
        custom_global_features = []
        custom_y = torch.randn(len(states))
        st = time.time()
        for state in states:
            uniform_data = [[float(elem) if isinstance(elem, np.ndarray) else float(elem) for elem in sublist] for sublist in state]

            data_array = np.array(uniform_data)
            state = torch.from_numpy(np.squeeze(data_array)).float().to(self.device)

            custom_x.append(state[6:])

            if custom_global_features == []:
                custom_global_features.append(state[0:6])

        for iiii in range(len(edge_index[0])):
            sub_custom_edge_attr.append(edge_feature[0] [ edge_index[0][iiii]    ])
            sub_custom_edge_attr.append(edge_feature[1] [ edge_index[0][iiii]    ])
            sub_custom_edge_attr.append(edge_feature[2] [ edge_index[0][iiii]    ])
            
            custom_edge_attr.append(sub_custom_edge_attr)
            sub_custom_edge_attr = []

        custom_x_tensor = torch.stack(custom_x)  # 形状 [3, 21]
        custom_global_features_tensor = torch.stack(custom_global_features)

        custom_graph = Data(
            x=custom_x_tensor.to(self.device),                   # 节点特征 [num_nodes, feature_dim]
            edge_index=torch.tensor(edge_index, dtype=torch.long).to(self.device),  # 边索引 [2, num_edges]
            edge_attr=torch.tensor(custom_edge_attr, dtype=torch.float32).to(self.device),    # 边特征 [num_edges, feature_dim]
            global_features=custom_global_features_tensor.to(self.device),  # 全局特征 [1, global_feature_dim]
            y=torch.tensor(custom_y, dtype=torch.float32).to(self.device)                    # 目标标签（可选）
        ).to(self.device)

        node_features, global_features, batch = self.process_graph(custom_graph)

        actions, log_probs, values, _, action_logits = self.policy(node_features, global_features)
        action2 = 0

        with open('compute_time.txt', 'a') as f:  # 以追加模式打开文件  
            f.write(str(time.time()-st) + '\n')  
        gnn_task_state = custom_graph
        
        return actions.cpu().numpy(), np.array(action2), gnn_task_state

    def store_experience(self, rewards, done):
        """存储每个节点的奖励和完成标志"""
        num_nodes = len(self.states[-1].x) if self.states else 0
        self.rewards.extend([rewards] * num_nodes)
        self.dones.extend([done] * num_nodes)
    
    def learn(self, steps, experiences):
        states, actions, exe_rewards, next_states, dones = experiences

        new_action = []
        new_experiences_rewards = []
        insert_index = 0
        for sub_a in actions:
            for subsub_a in sub_a:
                new_action.append(subsub_a)
                new_experiences_rewards.append(exe_rewards[insert_index])
            insert_index = insert_index + 1

        learn_states_for_ppo = states
        learn_states_for_common = states
        learn_actions = new_action
        from torch.distributions import Categorical
        learn_logprobs = []
        for x in range(len(new_action)):
            if new_action[x] == 0:
                dist = Categorical(torch.tensor([1,0,0,0]).to(self.device))

            elif new_action[x] == 1:
                dist = Categorical(torch.tensor([0,1,0,0]).to(self.device))

            elif new_action[x] == 2:
                dist = Categorical(torch.tensor([0,0,1,0]).to(self.device))

            elif new_action[x] == 3:
                dist = Categorical(torch.tensor([0,0,0,1]).to(self.device))
            action_logprob = dist.log_prob(torch.tensor(new_action[x]).to(self.device))
            learn_logprobs.append(action_logprob)

        discounted_reward = 0

        rewards = []
        for reward in reversed(torch.squeeze(torch.tensor(new_experiences_rewards))):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        old_actions = torch.tensor(learn_actions).detach().to(self.device)
        old_logprobs = torch.tensor(learn_logprobs).detach().to(self.device)
        output_loss = []
        output_policy_loss = []
        output_value_loss = []
        for _ in range(self.K_epochs):
            batch_data = Batch.from_data_list(states).to(self.device)
            batch_node_feat, batch_global_feat = self.feature_extractor(batch_data)
            logprobs, states_for_common_values, dist_entropy = self.policy.evaluate(
                batch_node_feat, 
                batch_global_feat, 
                old_actions,
                batch = batch_data.batch
            )
            states_for_common_values = torch.squeeze(states_for_common_values)

            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - states_for_common_values.detach()
            surr1 = ratios * advantages

            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(states_for_common_values,
                                                                 rewards) - 0.01 * dist_entropy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            policy_loss = -torch.min(surr1, surr2)
            value_loss = self.MseLoss(states_for_common_values, rewards)
            

            output_loss.append(loss.mean())
            output_policy_loss.append(policy_loss.mean())
            output_value_loss.append(value_loss.mean())


        if steps % 100 == 0 and steps !=0:
            self.save_model('./model_weight/saved_model_step_' + str(steps) + '.pth')
            print('模型保存成功')
        
        # 最终输出
        loss_policy = torch.stack(output_policy_loss).mean()
        loss_value = torch.stack(output_value_loss).mean()
        loss_total = torch.stack(output_loss).mean()
        
        return loss, loss_policy, loss_value, loss_total
    
    def save_model(self, file_path):
        """保存模型权重到文件"""
        # 创建包含所有模型权重的字典
        state_dict = {
            'feature_extractor': self.feature_extractor.state_dict(),
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        
        torch.save(state_dict, file_path)
        print(f"模型已保存到: {file_path}")
    
    def load_model(self, file_path):
        """从文件加载模型权重"""
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"模型文件不存在: {file_path}")
        
        # 加载状态字典
        state_dict = torch.load(file_path, map_location=self.device)
        
        # 加载特征提取器
        self.feature_extractor.load_state_dict(state_dict['feature_extractor'])
        
        # 加载策略网络
        self.policy.load_state_dict(state_dict['policy'])
        
        # 加载优化器状态（如果需要继续训练）
        self.optimizer.load_state_dict(state_dict['optimizer'])
        
        print(f"模型已从 {file_path} 加载")
        
        # 设置到评估模式
        self.feature_extractor.eval()
        self.policy.eval()