import gym
from gym import spaces
import numpy as np
import torch
import random
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'shared_program'))
from program.get_tasks import get_task_reset, get_task
from program.paifa_task import dis_task

def custom_sort_key(element):
    if isinstance(element, list):
        return max(element)
    else:
        return element

class MyCustomEnv(gym.Env):
    def __init__(self):
        super(MyCustomEnv, self).__init__()
        # 定义动作空间和状态空间
        self.action_space = spaces.Discrete(4)  # 例如：动作空间有两个离散动作（0和1）
        self.observation_space = spaces.Box(low=0, high=1, shape=(75,), dtype=np.float32)  # 例如：状态空间是一个4维的向量
        self.task_feature_dim = 5
        self.num_servers = 3
        self.cloud_server = 1
        self.input_number = 0
        self.state = None
        self.steps_beyond_done = None


    def reset(self):
        self.current_task, self.current_task_relation, self.current_task_list = get_task_reset()
        self.input_number = self.input_number + 1
        self.next_task, self.next_task_relation, self.next_task_list = self._generate_task()

        return self.current_task, self.current_task_relation, self.current_task_list

    def _generate_task(self):
        return get_task(self.input_number)

    def step_new(self, sta_task_number, state, task_content_list, state_relation, actions1, actions2, task_time):
        new_task_array = []
        for i in range(len(state_relation)):
            if 0 not in state_relation[i]:
                continue
            if actions2[i] == 1:
                indexes_of_0_and_1 = [index for index, value in enumerate(state_relation[i]) if value in {0, 1}]
                if len(new_task_array) == 0:
                    new_task_array.append(indexes_of_0_and_1)
                else:
                    for t in range(len(new_task_array)):
                        if set(indexes_of_0_and_1) & set(new_task_array[t]) != 0:
                            for u in indexes_of_0_and_1:
                                if u not in new_task_array[t]:
                                    new_task_array[t].append(u)
                        else:
                            new_task_array.append(indexes_of_0_and_1)
        new_task_array_sort = []
        for nr in new_task_array:
            new_task_array_sort.append(sorted(nr))
        my_list = list(range(len(state)))
        for ns in new_task_array_sort:
            min_n = min(ns)
            max_n = max(ns)
            for i in range(min_n,max_n+1):
                my_list.remove(i)

        fina_text = []
        for ns in new_task_array_sort:
            fina_text.append(ns)
        for na in my_list:
            fina_text.append(na)
        sorted_task_list = sorted(fina_text, key=custom_sort_key)
        for i in sorted_task_list:
            if isinstance(i, list):
                for ll in range(min(i), max(i)):
                    if ll not in i:
                        i.append(ll)
        sort_list2 = []
        for i in sorted_task_list:
            if isinstance(i, list):
                i = sorted(i, key=custom_sort_key)
                sort_list2.append(i)
            else:
                sort_list2.append(i)
        sorted_task_list = sort_list2
        action_carry = []
        actions_cho = [0, 0, 0, 0]
        for i in sorted_task_list:
            if isinstance(i, list):
                for ll in range(min(i), max(i) + 1):
                    actions_cho[actions1[ll]] = actions_cho[actions1[ll]] + 1
                # 找到列表中的最大值
                max_value = max(actions_cho)
                # 找到所有等于最大值的索引
                max_indices = [index for index, value in enumerate(actions_cho) if value == max_value]
                # 随机选择一个最大值的索引
                chosen_index = random.choice(max_indices)
                action_carry.append(chosen_index)
            else:
                action_carry.append(actions1[i])
        filename = 'filename'
        cost = 0
        for i in actions1:
            if i == 3:
                cost = cost - 3

        # 计算实际的任务数
        delay = -5 * (len(sorted_task_list))
        reward = (delay + cost)

        self.current_task = self.next_task
        self.current_task_relation = self.next_task_relation
        self.current_task_list = self.next_task_list
        self.input_number = sta_task_number + 2
        self.next_task, self.next_task_relation, self.next_task_list = self._generate_task()

        return self.next_task, self.next_task_relation, self.next_task_list, reward, filename

    def render(self, mode='human'):
        # 渲染环境（可选）
        print(f"State: {self.state}")

    def close(self):
        # 关闭环境（可选）
        pass

# 注册并使用自定义环境
from gym.envs.registration import register

register(
    id='MyCustomEnv-v0',
    entry_point='env_main:MyCustomEnv',  # 这个路径应指向你的环境类
)
