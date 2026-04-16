import numpy as np
import torch
import random
from get_tasks import get_task_reset, get_task
from paifa_task import dis_task

def custom_sort_key(element):
    if isinstance(element, list):
        return max(element)
    else:
        return element

class CustomTaskSchedulingEnv:
    def __init__(self):
        self.task_feature_dim = 81
        self.num_servers = 3
        self.cloud_server = 1
        self.input_number = 0

    def reset(self):
        self.current_task, self.current_task_relation, self.current_task_list = get_task_reset()
        self.input_number = self.input_number + 1
        self.next_task, self.next_task_relation, self.next_task_list = self._generate_task()
        return self.current_task, self.current_task_relation

    def _generate_task(self):
        return get_task(self.input_number)

    def step(self, sta_task_number, state, state_relation, actions1, task_time):
        dis_task(sta_task_number, actions1, task_time)

        cost = 0
        for i in actions1:
            if i == 3:
                cost = cost - 3

        delay = -5
        reward = (delay + cost)

        self.current_task = self.next_task
        self.current_task_relation = self.next_task_relation
        self.current_task_list = self.next_task_list
        self.input_number = sta_task_number + 2
        self.next_task, self.next_task_relation, self.next_task_list = self._generate_task()
        while len(self.next_task[0]) != 81:
            self.next_task, self.next_task_relation, self.next_task_list = self._generate_task()

        return self.current_task, self.current_task_relation, reward
