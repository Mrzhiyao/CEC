

import gym
# import pybullet_envs
import numpy as np
from collections import deque
import torch
import wandb
import argparse
from buffer_main import ReplayBuffer
import glob
from utils import save
import random
from agent import CQLSAC
import env_main
import torch.nn as nn
import time
import pickle
import json
import os
import sys as _sys, os as _os
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from embedding import embedding

connections.connect("default", host="YOUR_EDGE_NODE3_IP", port="19530")
from pymilvus import MilvusClient
client = MilvusClient(
    uri='http://YOUR_EDGE_NODE3_IP:19530'
)
collection_name = "crewai_agents1_algorithm1_subvec"
collection = Collection("crewai_agents1_algorithm1_subvec")
search_params = {"metric_type": "IP", "limit": 3,
    "params": {"ef": 250}
}

def get_config():
    parser = argparse.ArgumentParser(description='greedy_sim')
    parser.add_argument("--run_name", type=str, default="greedy_sim_llm", help="Run name, default: greedy_sim_llm")
    parser.add_argument("--env", type=str, default="'MyCustomEnv-v0", help="Gym environment name, default: MyCustomEnv-v0")
    parser.add_argument("--episodes", type=int, default=2, help="Number of episodes, default: 200")
    parser.add_argument("--buffer_size", type=int, default=100_000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")
    
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
    

    with wandb.init(project="step_by_step_vec", name=config.run_name, config=config):
        _base = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'shared_program', 'program')
        path_result = _os.path.join(_base, 'node_receive_resource', 'final')
        path_time = _os.path.join(_base, 'node_receive_resource', 'time', 'final')
        path_subtask = _os.path.join(_base, 'subtasks')
        agent = CQLSAC(state_size=env.observation_space.shape[0],
                         action_size=env.action_space.n,
                         device=device)

        wandb.watch(agent, log="gradients", log_freq=10)

        buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device)
        
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
            action_cost = []
            for t in range(0, 500):
                actions1 = []
                actions2 = []
                log_probs1 = []
                state_new = []
                for m in range(len(state)):
                    action1, action2 = agent.get_action_algorithm1(state[m])
                    actions1.append(action1)
                    action_cost.append(action1)
                    actions2.append(action2)
                    if m == 0:
                        state_new.append(state[m])
                    else:
                        state_append = state[m] 
                        state_append[17] = action_pre
                        state_new.append(state_append)
                    action_pre = action1
                steps += 1
                next_state, next_relation, next_task_content_list, reward, filename = env.step_new(t, state_new, task_content_list, state_relation, actions1, actions2, task_time)
                tasks_list.append(filename)
                action1_ist.append(actions1)
                action2_ist.append(actions2)
            
                task_time = str(time.time())
                memory1.append((state_new, actions1, actions2, reward, log_probs1))
                write_content = (state_new, actions1, actions2, log_probs1, state_relation)
                with open('./memorys/memory' + str(t+500*i) + '.pkl', 'wb') as file:
                    pickle.dump(write_content, file) 

                list_remove = []
                for n in range(len(state_relation)):
                    if 0 not in state_relation[n]:
                        list_remove.append(n)

                state = next_state
                task_content_list = next_task_content_list
                state_relation = next_relation
                rewards = rewards + reward

                if (t+1) % 10 == 0:
                    number = 0
                    rewards = 0
                    avg_completion = 0
                    avg_delay = 0
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
                        except:
                            rewards = rewards - 50
                            pass

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

if __name__ == "__main__":
    config = get_config()
    train(config)
