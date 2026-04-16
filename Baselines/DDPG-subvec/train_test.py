

import gym
import numpy as np
from collections import deque
import torch
import wandb
import argparse
from buffer_main import ReplayBuffer
from utils import collect_random_main_train_test
import random
from agent_ddpg import DDPG
import env_main
import torch.nn as nn
import time
import pickle
import json
import os
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'shared_program'))
from embedding import embedding

connections.connect("default", host="192.168.2.7", port="19530")
from pymilvus import MilvusClient
client = MilvusClient(
    uri='http://192.168.2.7:19530'
)
collection_name = "crewai_agents1_algorithm1_subvec"
collection = Collection("crewai_agents1_algorithm1_subvec")
search_params = {"metric_type": "IP", "limit": 3,
    "params": {"ef": 250}
}

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="DDPG", help="Run name, default: DDPG")
    parser.add_argument("--env", type=str, default="'MyCustomEnv-v0", help="Gym environment name, default: MyCustomEnv-v0")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes, default: 200")
    parser.add_argument("--buffer_size", type=int, default=100000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=600, help="Batch size, default: 256")
    
    args = parser.parse_args()
    return args

def find_files_containing_name(directory_path, name_part):
    try:
        # 获取目录中的所有文件和文件夹
        items = os.listdir(directory_path)
        
        # 过滤出文件并检查名称是否包含指定部分
        matching_files = [item for item in items if os.path.isfile(os.path.join(directory_path, item)) and name_part in item]
        
        return matching_files
    except Exception as e:
        print("An error occurred: {e}")
        return []
def list_files_sorted_by_creation_time(directory_path):
    try:
        # 获取目录中的所有文件和文件夹
        items = os.listdir(directory_path)
        
        # 过滤掉文件夹，只保留文件
        files = [item for item in items if os.path.isfile(os.path.join(directory_path, item))]
        
        # 获取每个文件的创建时间并排序
        files_with_ctime = [(file, os.path.getctime(os.path.join(directory_path, file))) for file in files]
        files_with_ctime.sort(key=lambda x: x[1])
        
        # 生成按创建时间排序的文件列表
        sorted_files = [file for file, ctime in files_with_ctime]
        
        return sorted_files
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def train(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    env = gym.make('MyCustomEnv-v0')
    env.seed(config.seed)
    env.action_space.seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    steps = 0
    average10 = deque(maxlen=10)
    total_steps = 0
    print('FIRST')
    buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device)
    
    state_dim_critic = 27
    state_dim_actor = 27
    action_dim = 4
    actor_node = 512
    critic_node= 512
    K_epochs = 50         # update policy for K epochs in one ddpg update
    eps_clip = 0.2          # clip parameter for ddpg
    gamma = 0.99            # discount factor
    lr_actor = 0.0001       # learning rate for actor network
    lr_critic = 0.001  


    collect_random_main_train_test(dataset=buffer)

    agent = DDPG(state_dim_actor, action_dim, actor_node, lr_actor, device)
    with wandb.init(project="algorithm1-subvec", name=config.run_name+'training-ddpg', config=config):
        for xx in range(200):
            critic_loss, actor_loss = agent.learn(xx, buffer.sample(), gamma=0.99)
            wandb.log({"critic_loss": critic_loss})
            wandb.log({"actor_loss": actor_loss})
            checkpoint = {
                'actor_local': agent.actor_local.state_dict(),
                'actor_target': agent.actor_target.state_dict(),
                'critic_local': agent.critic_local.state_dict(),
                'critic_target': agent.critic_target.state_dict(),
                'actor_optimizer': agent.actor_optimizer.state_dict(),
                'critic_optimizer': agent.critic_optimizer.state_dict(),
            }
            torch.save(checkpoint, 'ddpg_pretain.pth')
            print(f"模型已保存到 {'ddpg_pretain.pth'}")

if __name__ == "__main__":
    config = get_config()
    train(config)
