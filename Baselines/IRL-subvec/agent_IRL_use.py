import argparse
import gym
import os
import sys
import pickle
import time
import numpy as np
import math
import torch
from torch import nn, LongTensor
from torch import tensor, zeros, ones

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils_irl import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from models.mlp_discriminator import Discriminator
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent
from utils import collect_random_main_train_test
from buffer_main import ReplayBuffer
from utils_irl.replay_memory import Memory


# ===== 1. 自定义环境接口 =====
class CustomEnvWrapper:
    def __init__(self, state_dim, action_dim, max_steps=1000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,))
        
        if action_dim == 1:  # 离散动作
            self.action_space = gym.spaces.Discrete(2)  # 假设二值动作
        else:  # 连续动作
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,))
            
        self._max_steps = max_steps
        self.reset()
        
    def reset(self):
        self.state = np.random.randn(self.state_dim).astype(np.float32)
        self.step_count = 0
        return self.state.copy()
    
    def step(self, action):
        # 简化环境动态 - 随机状态转移
        next_state = self.state + 0.1 * np.random.randn(self.state_dim)
        done = self.step_count >= self._max_steps
        reward = -np.mean(next_state**2)  # 简化奖励函数
        self.state = next_state.copy()
        self.step_count += 1
        return next_state, reward, done, {}
    
    def seed(self, seed):
        np.random.seed(seed)
        
    def render(self):
        pass  # 可选实现可视化

# ===== 2. 修改主程序 =====
parser = argparse.ArgumentParser(description='PyTorch GAIL example')
parser.add_argument('--env-name', default="GAIL", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--state-dim', type=int, default=27, metavar='N',
                    help='dimensionality of state space (default: 27)')
parser.add_argument('--action-dim', type=int, default=1, metavar='N',
                    help='dimensionality of action space (default: 4)')
parser.add_argument('--expert-traj-path', metavar='G',
                    help='path of the expert trajectories')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                    help='log std for the policy (default: -0.0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='learning rate (default: 3e-4)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--eval-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size for evaluation (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)

"""自定义环境"""
env = CustomEnvWrapper(args.state_dim, args.action_dim)
state_dim = args.state_dim
action_dim = args.action_dim

class SimpleNormalizer:
    def __init__(self, shape, clip=5, device='cpu'):
        self.shape = shape
        self.clip = clip
        self.fix = False
        self.device = device
        
        # 初始化均值和标准差为PyTorch张量
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.std = torch.ones(shape, dtype=torch.float32, device=device)
    
    def __call__(self, x):
        # 确保输入是PyTorch张量
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        # 确保输入在正确的设备上
        if x.device != self.device:
            x = x.to(self.device)
            
        # 使用PyTorch函数进行归一化
        normalized = (x - self.mean) / self.std
        clipped = torch.clamp(normalized, -self.clip, self.clip)
        return clipped
    
    def update(self, data):
        if not self.fix:
            # 确保数据是PyTorch张量并在正确设备上
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float32, device=self.device)
            elif data.device != self.device:
                data = data.to(self.device)
                
            # 计算新的均值和标准差
            self.mean = data.mean(0)
            self.std = data.std(0) + 1e-8



class IRL():
    """docstring for IRL"""
    def __init__(self, state_dim_actor, action_dim, actor_node, lr_actor, device):
        super(IRL, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.running_state = SimpleNormalizer((state_dim,), clip=5)
        is_disc_action = 1  # 离散动作还是连续动作

        """定义actor和critic"""
        if is_disc_action:
            self.policy_net = DiscretePolicy(state_dim, 4).float()  # 转换所有权重为float32
        else:
            self.policy_net = Policy(state_dim, action_dim, log_std=args.log_std).float()  # 转换所有权重为float32
            
        self.value_net = Value(state_dim)
        self.discrim_net = Discriminator(state_dim + action_dim)
        self.discrim_criterion = nn.BCELoss()
        to_device(device, self.policy_net, self.value_net, self.discrim_net, self.discrim_criterion)

        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=args.learning_rate)
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=args.learning_rate)
        self.optimizer_discrim = torch.optim.Adam(self.discrim_net.parameters(), lr=args.learning_rate)

        # optimization epoch number and batch size for PPO
        self.optim_epochs = 10
        self.optim_batch_size = 64
        
    def expert_reward(self, state, action):
        # 确保输入是Tensor且是float32类型
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state = state.to(dtype=torch.float32, device=self.device)
        
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        else:
            action = action.to(dtype=torch.float32, device=self.device)
        
        # 展平张量
        state_flat = state.flatten()
        action_flat = action.flatten()
        state_action = torch.cat([state_flat, action_flat])
        state_action_tensor = state_action.unsqueeze(0)
        
        # 确保判别器模型为float32
        if not hasattr(self, '_discrim_float_fixed'):
            self.discrim_net = self.discrim_net.float()
            self._discrim_float_fixed = True
        
        with torch.no_grad():
            d = self.discrim_net(state_action_tensor)[0]
            return -math.log(d.item())  # 注意：d.item()避免直接计算log的dtype问题


    def get_action(self, state):

        uniform_data = [[float(elem) if isinstance(elem, np.ndarray) else float(elem) for elem in sublist] for sublist in state]
        data_array = np.array(uniform_data, dtype=np.float32)  # 显式指定为float32
        # 转换并确保数据类型
        state_tensor = torch.from_numpy(data_array).squeeze()  # 使用numpy默认dtype
        state_tensor = state_tensor.float().to(self.device)  # 显式转换为float32
        # 1. 规范化输入数据结构
        if isinstance(state, np.ndarray):
            # 如果已经是numpy数组，直接使用
            data_array = state
        else:
            # 扁平化处理嵌套列表/数组结构
            flattened = []
            for item in state:
                if isinstance(item, (list, np.ndarray)):
                    flattened.extend(item)
                else:
                    flattened.append(item)
            
            # 2. 创建 numpy 数组并确保为 float32
            data_array = np.array(flattened, dtype=np.float32)
        
        # 3. 转换为 Tensor 并确保类型一致
        state_tensor = torch.tensor(data_array, dtype=torch.float32, device=self.device)
        
        # 4. 添加批次维度 (如果缺失)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)  # 从 [N] 变为 [1, N]
    
        action = self.policy_net.select_action(state_tensor)[0].cpu().numpy()
        action2 = 0
        return action[0], np.array(action2)


    def collect_samples(self, batch, custom_reward, running_state):
        
        states, actions, rewards, next_states, masks = batch

        log = dict()
        memory = Memory()
        num_steps = 0
        total_reward = 0
        min_reward = 1e6
        max_reward = -1e6
        total_c_reward = 0
        min_c_reward = 1e6
        max_c_reward = -1e6
        num_episodes = 0

        for x in range(len(states)):
            total_reward += rewards[x]
            if running_state is not None:
                next_state = running_state(next_states[x])

            if custom_reward is not None:
                reward = custom_reward(states[x], actions[x])
                total_c_reward += reward
                min_c_reward = min(min_c_reward, reward)
                max_c_reward = max(max_c_reward, reward)

            mask = 1

            memory.push(states[x], actions[x], mask, next_state, reward)

            min_reward = min(min_reward, rewards[x])
            max_reward = max(max_reward, rewards[x])
            num_steps = num_steps + 1


        log['num_steps'] = num_steps
        log['num_episodes'] = num_episodes
        log['total_reward'] = total_reward
        log['avg_reward'] = total_reward / num_episodes
        log['max_reward'] = max_reward
        log['min_reward'] = min_reward
        if custom_reward is not None:
            log['total_c_reward'] = total_c_reward
            log['avg_c_reward'] = total_c_reward / num_steps
            log['max_c_reward'] = max_c_reward
            log['min_c_reward'] = min_c_reward


        return memory, log
    


    def update_params(self, expert_traj, batch, i_iter):

        memory, log = self.collect_samples(batch, self.expert_reward, self.running_state)
        batch_memory = memory.sample()

        state_elements = []
        for item in batch_memory.state:
            if isinstance(item, torch.Tensor):
                # 将 CUDA 张量移动到 CPU 并转换为 NumPy 数组
                state_elements.append(item.detach().cpu().numpy())
            elif isinstance(item, np.ndarray):
                # 已经是 NumPy 数组
                state_elements.append(item)
            else:
                # 其他类型（列表、元组等）
                state_elements.append(np.array(item))

        # 确保所有元素都是 NumPy 数组
        if state_elements:
            states = torch.from_numpy(np.stack(state_elements)).to(dtype).to(device)
        else:
            states = torch.empty(0, dtype=dtype, device=device)

        actions_elements = []
        for item in batch_memory.action:
            if isinstance(item, torch.Tensor):
                # 将 CUDA 张量移动到 CPU 并转换为 NumPy 数组
                actions_elements.append(item.detach().cpu().numpy())
            elif isinstance(item, np.ndarray):
                # 已经是 NumPy 数组
                actions_elements.append(item)
            else:
                # 其他类型（列表、元组等）
                actions_elements.append(np.array(item))

        # 确保所有元素都是 NumPy 数组
        if actions_elements:
            actions = torch.from_numpy(np.stack(actions_elements)).to(dtype).to(device)
        else:
            actions = torch.empty(0, dtype=dtype, device=device)

        rewards_elements = []
        for item in batch_memory.reward:
            if isinstance(item, torch.Tensor):
                # 将 CUDA 张量移动到 CPU 并转换为 NumPy 数组
                rewards_elements.append(item.detach().cpu().numpy())
            elif isinstance(item, np.ndarray):
                # 已经是 NumPy 数组
                rewards_elements.append(item)
            else:
                # 其他类型（列表、元组等）
                rewards_elements.append(np.array(item))

        # 确保所有元素都是 NumPy 数组
        if rewards_elements:
            rewards = torch.from_numpy(np.stack(rewards_elements)).to(dtype).to(device)
        else:
            rewards = torch.empty(0, dtype=dtype, device=device)



        masks_elements = []
        for item in batch_memory.mask:
            if isinstance(item, torch.Tensor):
                # 将 CUDA 张量移动到 CPU 并转换为 NumPy 数组
                masks_elements.append(item.detach().cpu().numpy())
            elif isinstance(item, np.ndarray):
                # 已经是 NumPy 数组
                masks_elements.append(item)
            else:
                # 其他类型（列表、元组等）
                masks_elements.append(np.array(item))

        # 确保所有元素都是 NumPy 数组
        if masks_elements:
            masks = torch.from_numpy(np.stack(masks_elements)).to(dtype).to(device)
        else:
            masks = torch.empty(0, dtype=dtype, device=device)

        # 处理所有输入数据
        states = states.to(dtype=torch.float32)
        actions = actions.to(dtype=torch.float32)
        with torch.no_grad():
            # print(states[0])
            values = self.value_net(states)
            fixed_log_probs = self.policy_net.get_log_prob(states, actions)

        """获取优势估计"""
        advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)

        """更新判别器"""
        for _ in range(1):
            # expert_state_actions = torch.from_numpy(expert_traj).to(dtype).to(device)
            expert_state_actions = expert_traj.to(dtype).to(device)
            g_o = self.discrim_net(torch.cat([states, actions], 1))
            e_o = self.discrim_net(expert_state_actions)
            self.optimizer_discrim.zero_grad()


            # 转换目标标签为 float32
            g_targets = torch.ones((states.shape[0], 1), dtype=torch.float32, device=device)
            e_targets = torch.zeros((expert_traj.shape[0], 1), dtype=torch.float32, device=device)

            # 转换预测值为 float32（确保与目标相同类型）
            g_o = g_o.float() if g_o.dtype == torch.float64 else g_o
            e_o = e_o.float() if e_o.dtype == torch.float64 else e_o

            # 计算损失
            discrim_loss = self.discrim_criterion(g_o, g_targets) + \
                        self.discrim_criterion(e_o, e_targets)
            # discrim_loss = self.discrim_criterion(g_o, ones((states.shape[0], 1), device=device)) + \
            #     self.discrim_criterion(e_o, zeros((expert_traj.shape[0], 1), device=device))
            discrim_loss.backward()
            self.optimizer_discrim.step()

        """执行mini-batch PPO更新"""
        optim_iter_num = int(math.ceil(states.shape[0] / self.optim_batch_size))
        for _ in range(self.optim_epochs):
            perm = np.arange(states.shape[0])
            np.random.shuffle(perm)
            perm = LongTensor(perm).to(device)

            states, actions, returns, advantages, fixed_log_probs = \
                states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), fixed_log_probs[perm].clone()

            for i in range(optim_iter_num):
                ind = slice(i * self.optim_batch_size, min((i + 1) * self.optim_batch_size, states.shape[0]))
                states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                    states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

                value_loss = ppo_step(self.policy_net, self.value_net, self.optimizer_policy, self.optimizer_value, 1, states_b, actions_b, returns_b,
                        advantages_b, fixed_log_probs_b, args.clip_epsilon, args.l2_reg)

        """保存模型"""
        policy_net_cpu = self.policy_net.cpu()
        value_net_cpu = self.value_net.cpu()
        discrim_net_cpu = self.discrim_net.cpu()
        
        torch.save({
            'policy': policy_net_cpu.state_dict(),
            'value': value_net_cpu.state_dict(),
            'discrim': discrim_net_cpu.state_dict(),
            'running_state': self.running_state
        }, f'./trained_models/{args.env_name}_iter_{i_iter}.pth')
        
        # 恢复模型到原设备
        self.policy_net.to(device)
        self.value_net.to(device)
        self.discrim_net.to(device)


        return discrim_loss, value_loss
