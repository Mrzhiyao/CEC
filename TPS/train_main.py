import gym
import numpy as np
from collections import deque
import torch
import wandb
import argparse
from buffer_main import ReplayBuffer
import glob
from utils import collect_random_main_train_test
import random
from agent_regular import PPO
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
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'shared_program'))
from embedding import embedding
connections.connect("default", host="192.168.2.5", port="19530")
from pymilvus import MilvusClient
client = MilvusClient(
    uri='http://192.168.2.5:19530'
)
collection_name = "crewai_agents1_algorithm1_subvec"
collection = Collection("crewai_agents1_algorithm1_subvec")
search_params = {"metric_type": "IP", "limit": 3,
    "params": {"ef": 250}
}
def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="PPO", help="Run name, default: PPO")
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

    buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device)

    state_dim_critic = 27
    state_dim_actor = 27
    action_dim = 4
    actor_node = 512
    critic_node= 512
    K_epochs = 50         # update policy for K epochs in one PPO update
    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor
    lr_actor = 0.0001       # learning rate for actor network
    lr_critic = 0.001  

    collect_random_main_train_test(dataset=buffer)

    agent = PPO(state_dim_critic, state_dim_actor, action_dim, 
            actor_node, critic_node, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device)

    with wandb.init(project="algorithm1-subvec", name=config.run_name+'training-ppo', config=config):
        for xx in range(200):
            loss = agent.learn(xx, buffer.sample())
            rewards = 0
            if xx % 10 == 0:
                print("Episode: {} | Polciy Loss: {} | Steps: {}".format(xx/10, torch.mean(loss), xx,))
                wandb.log({"Loss": torch.mean(loss)})

    with wandb.init(project="algorithm1-subvec", name=config.run_name, config=config):
        _base = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'shared_program', 'program')
        path_result = _os.path.join(_base, 'node_receive_resource', 'final')
        path_time = _os.path.join(_base, 'node_receive_resource', 'time', 'final')
        path_subtask = _os.path.join(_base, 'subtasks')
        
        if config.log_video:
            env = gym.wrappers.Monitor(env, './video', video_callable=lambda x: x%10==0, force=True)

        for i in range(0, config.episodes):

            tasks_list = []
            action1_ist = []
            action2_ist = []
            episode_steps = 0
            done = False
            state, state_relation, task_content_list = env.reset()
            task_time = str(time.time())
                                                                                                                                                                                         
            memory1 = []
            rewards = 0
            state_mem = []
            action_mem = []
            action_cost = []
            for t in range(500):
                actions1 = []
                actions2 = []
                log_probs1 = []
                state_new = []

                for m in range(len(state)):
                    add_state_part1 = state[m][0:2]
                    add_state_part2 = state[m][3:5]
                    add_state_part3 = state[m][6:8]
                    add_state_part4 = state[m][9:]

                    result_state = add_state_part1 + add_state_part2 + add_state_part3 + add_state_part4
                    result_state_add = result_state[0:15] + result_state[15:19] + result_state[35:39] + result_state[55:59]
                    action1, action2 = agent.get_action(result_state_add)
                    actions1.append(action1)
                    action_cost.append(action1)
                    actions2.append(action2)
                    if state_relation == []:
                        if m == len(state)-1:
                            state_append = state[m] 
                            state_append[14] = action_pre
                            state_new.append(state_append)
                        else:
                            state_new.append(state[m])
                            action_pre = action1  
                    else:
                        if m == 0:
                            state_new.append(state[m])
                            action_pre = action1
                        else:
                            if m in state_relation[0]:
                                state_append = state[m] 
                                state_append[14] = action_pre
                                state_new.append(state_append)
                            elif m == len(state)-1:
                                state_append = state[m] 
                                state_append[14] = action_pre
                                state_new.append(state_append)
                            else:
                                state_append = state[m] 
                                action_pre = action1
                                state_new.append(state_append)   

                steps += 1
                state_mem.append(state_new)
                action_mem.append(actions1)
    
                for ttt in range(10):
                    loss = agent.learn(t, buffer.sample())
                next_state, next_relation, next_task_content_list, reward, filename = env.step_new(i*500 + t, state_new, task_content_list, state_relation, actions1, actions2, task_time)

                tasks_list.append(filename)
                action1_ist.append(actions1)
                action2_ist.append(actions2)            
                task_time = str(time.time())
                memory1.append((state_new, actions1, actions2, reward, log_probs1))
                write_content = (state_new, actions1, actions2, log_probs1, state_relation)
                with open('./memorys/memory' + str(t) + '.pkl', 'wb') as file:
                    pickle.dump(write_content, file) 

                list_remove = []
                for n in range(len(state_relation)):
                    if 0 not in state_relation[n]:
                        list_remove.append(n)

                state = next_state
                task_content_list = next_task_content_list
                state_relation = next_relation
                rewards = rewards + reward


                if (t+1) % 10 == 0 and t != 0:
                    list_rewards = []
                    number = 0
                    rewards = 0
                    avg_completion = 0
                    avg_delay = 0
                    add_number = 0
                    for task_file in tasks_list:
                        matching_files_result = find_files_containing_name(path_result, task_file)
                        try:
                            with open(path_result + '/' + matching_files_result[0], 'r') as file:
                                content = file.read()
                            matching_files_question = find_files_containing_name(path_subtask, task_file)
                            with open(path_subtask + '/' + matching_files_question[0], 'r') as file:
                                question = file.read()
                                question_text = question.replace('Task:\n',"").replace('\n','')

                                que_find = client.search(data=[embedding(question_text)], collection_name=collection_name,
                                                        output_fields=["text_answerid", "text"],
                                                        search_params=search_params,
                                                        limit=1)
                                text_answerid = que_find[0][0]['entity']['text_answerid']

                                filter_content = "message_id==" + "\"" + text_answerid + "\""
                                result_sim = client.search(data=[embedding(content)], collection_name=collection_name, filter=filter_content, output_fields=["id", "text"], search_params=search_params,limit=1)
                                completion = result_sim[0][0]['distance']

                            matching_files_time = find_files_containing_name(path_time, task_file)
                            send_numberfile = task_file.split('_')[0]
                            with open(path_subtask + '/' + 'task_send_number_' + send_numberfile + '.txt', 'r') as file:
                                start_time = file.read()
                            with open(path_time + '/' + matching_files_time[0], 'r') as file:
                                end_time = file.read()
                            spend_time = float(end_time) - float(start_time)
                            rewards = rewards + float(completion)*100 - float(spend_time)
                            number = number + 1
                            avg_completion = avg_completion + float(completion)*100
                            avg_delay = avg_delay + float(spend_time)

                            if float(completion) < 0.7:
                                rewards_mem = (-spend_time) * 2 - (0.7-float(completion))*10
                            else:
                                rewards_mem = (-spend_time) * 2
                            
                            if rewards_mem < -250:
                                rewards_mem = -250
                            list_rewards.append(rewards_mem)
                        except:
                            rewards = rewards - 50
                            rewards_mem = -250
                            list_rewards.append(rewards_mem)
                            pass

                        for ad in range(len(state_mem[add_number])):

                            if ad != len(state_mem[add_number])-1:
                                next_states = state_mem[add_number][ad+1]
                            else:
                                next_states = state_mem[add_number][ad]

                            if action_mem[add_number][ad] == 0:
                                next_states[0] = random.uniform(60, 70)
                                next_states[1] = random.uniform(50, 60)
                            elif action_mem[add_number][ad] == 1:
                                next_states[3] = random.uniform(70, 80)
                                next_states[4] = random.uniform(50, 55)
                            elif action_mem[add_number][ad] == 2:
                                next_states[6] = random.uniform(70, 80)
                                next_states[7] = random.uniform(50, 55)
                            add_state = state_mem[add_number][ad] 
                            add_next = next_states
                            add_state_part1 = add_state[0:2]
                            add_state_part2 = add_state[3:5]
                            add_state_part3 = add_state[6:8]
                            add_state_part4 = add_state[9:]

                            result_state = add_state_part1 + add_state_part2 + add_state_part3 + add_state_part4

                            add_next_part1 = add_next[0:2]
                            add_next_part2 = add_next[3:5]
                            add_next_part3 = add_next[6:8]
                            add_next_part4 = add_next[9:]

                            result_next = add_next_part1 + add_next_part2 + add_next_part3 + add_next_part4
                            result_state_new = result_state[0:15] + result_state[15:19] + result_state[35:39] + result_state[55:59]
                            result_next_new = result_next[0:15] + result_next[15:19] + result_next[35:39] + result_next[55:59]

                            buffer.add(state=result_state_new, action=action_mem[add_number][ad], reward=rewards_mem-action_mem[add_number].count(3) * 30, next_state=result_next_new, done=0)

                        add_number = add_number + 1


                    if number !=0 :
                        avg_completion = avg_completion/number
                        avg_delay = avg_delay/number
                    else:
                        avg_completion = 0
                        avg_delay = 300


                    average10.append(rewards/10)
                    completion_ratio = number / 10
                    total_steps += episode_steps
                    print("Episode: {} | Reward: {} | Steps: {}".format(t, rewards,  (t+1) % 10))
                    
                    wandb.log({
                            "Steps": (t+1) % 10,
                            "Reward": rewards/10,
                            "Average10": np.mean(average10),
                            "Completion_ratio": completion_ratio,
                            "Avg Cost": action_cost.count(3) * 5,
                            "Cloud LLM Usage ratio":  action_cost.count(3) / len(action_cost),
                            "Delay": avg_delay,
                            "Task Completion Satisfaction": avg_completion,
                            "Episode": t,
                            "Buffer size": buffer.__len__()})

                    if (i %10 == 0) and config.log_video:
                        mp4list = glob.glob('video/*.mp4')
                        if len(mp4list) > 1:
                            mp4 = mp4list[-2]
                            wandb.log({"gameplays": wandb.Video(mp4, caption='episode: '+str(i-10), fps=4, format="gif"), "Episode": i})

                    tasks_list = []
                    action_cost = []
                    state_mem = []
                    action_mem = []
            break



if __name__ == "__main__":
    config = get_config()
    train(config)
