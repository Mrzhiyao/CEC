import json
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config
import data.generate_environment as generate_data
import data.dataset as data
import tqdm
import train_bc as train_bc
import models.bc_model as bc_model
import models.q_model as q_model
import models.bcq_model as bcq_model
from data.dataloader import EnvironmentDataset
import torch
import models.hyperparameter as hyperparameter
from buffer_main import ReplayBuffer
from utils import collect_random_main_train_test


def load_model(name, model):
    path = f'./trained_models/{name}.pt'
    model = model
    model.load_state_dict(torch.load(path))
    return model


def run():
    behavior_cloning = bc_model.BehavioralModel()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)

    buffer = ReplayBuffer(buffer_size=100000, batch_size=2296, device=device)
    collect_random_main_train_test(dataset=buffer)
    experiences = buffer.sample()
    batch_state, batch_action, batch_reward, batch_next_state, dones = experiences
    num_samples = len(batch_state)

    indices = torch.randperm(num_samples)

    # 计算划分点
    split_idx = int(num_samples * 0.8)

    # 划分训练集
    train_indices = indices[:split_idx]
    train_batch_state = batch_state[train_indices]
    train_batch_action = batch_action[train_indices]
    train_batch_reward = batch_reward[train_indices]
    train_batch_next_state = batch_next_state[train_indices]
    train_dones = dones[train_indices]

    # 划分测试集
    test_indices = indices[split_idx:]
    test_batch_state = batch_state[test_indices]
    test_batch_action = batch_action[test_indices]
    test_batch_reward = batch_reward[test_indices]
    test_batch_next_state = batch_next_state[test_indices]
    test_dones = dones[test_indices]

    # 重新打包为训练集和测试集的元组
    train_experiences = (train_batch_state, train_batch_action, train_batch_reward, train_batch_next_state, train_dones)
    test_experiences = (test_batch_state, test_batch_action, test_batch_reward, test_batch_next_state, test_dones)

    # 打印结果验证
    print(f"训练集大小: {len(train_indices)}")
    print(f"测试集大小: {len(test_indices)}")
    print(f"总样本数: {len(train_indices) + len(test_indices)}")

    train_bc.train(behavior_cloning, config.get_device(), train_experiences, test_experiences,
                  optimizer=optim.AdamW(behavior_cloning.parameters(), lr=0.0001), criterion=nn.CrossEntropyLoss())

def sets_generation():
    envs = generate_data.load_environments()
    optimal_paths = load_optimal_paths()
    train_data, test_data, val_data = data.train_test_val_split(environments=envs, optimal_paths=optimal_paths)
    data.save_dataset(train_data, 'train_data')
    data.save_dataset(test_data, 'test_data')
    data.save_dataset(val_data, 'val_data')


def data_gen():
    envs = generate_data.load_environments()
    save_optimal_paths(envs)
    return envs


def save_optimal_paths(envs):
    agent_positions_all_envs = []
    for index, env in tqdm.tqdm(enumerate(envs), total=len(envs), desc="calculating optimal paths", unit="Environments"):
        _, agent_positions = data.calculate_optimal_trajectory(env, index)
        agent_positions_all_envs.append((index, sorted(list(set(agent_positions)), key=lambda x: x[1])))
    with open('data' + os.sep + 'optimal_paths.json', 'w') as file:
        json.dump(agent_positions_all_envs, file, indent=2)


def load_optimal_paths():
    with open('data' + os.sep + 'optimal_paths.json', 'r') as file:
        data = json.load(file)
    _, paths = map(list, zip(*data))
    return paths

class BC():
    """docstring for BC"""
    def __init__(self, state_dim_actor, action_dim, actor_node, lr_actor, device):
        self.behavior_cloning = bc_model.BehavioralModel()
        name='final_BC_state_dict'
        self.best_bc_model = load_model(name, self.behavior_cloning)
        self.device = device


    def get_actions(self, state):

        uniform_data = [[float(elem) if isinstance(elem, np.ndarray) else float(elem) for elem in sublist] for sublist in state]

        data_array = np.array(uniform_data)
        state = torch.from_numpy(np.squeeze(data_array)).float().to(self.device)
        action_prob = train_bc.get_action(state, self.best_bc_model, config.get_device())
        action = torch.argmax(action_prob, dim=1)  # 形状 [1]
        # print(action.shape)
        action2 = 0
        return action.detach().cpu().numpy()[0], np.array(action2)


if __name__ == '__main__':
    run()
