import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import numpy as np
import math
import copy
import random
from program.get_source import get_source

class CQLSAC(nn.Module):
    """Interacts with and learns from the environment."""
    
    def __init__(self,
                        state_size,
                        action_size,
                        device
                ):

        super(CQLSAC, self).__init__()

        self.device = device

    def get_action_algorithm1(self, state, lunxun_number):
        action1 = -1
        path1 = 'YOUR_RESOURCE_FILE_PATH'
        path2 = 'YOUR_RESOURCE_FILE_PATH'
        path3 = 'YOUR_RESOURCE_FILE_PATH'
        list_resource = []
        o_number = 0
        while 1:
            try:
                x1 = get_source(path1)
                x2 = get_source(path2)
                x3 = get_source(path3)
                break
            except:
                pass
        
        list_resource.append(x1[0])
        list_resource.append(x2[0])
        list_resource.append(x3[0])
        if lunxun_number <3:
            for x in range(lunxun_number, 3):
                if float(list_resource[x])<70:
                    lunxun_number =  lunxun_number + 1
                    action1 = x
                    break
                else:
                    lunxun_number =  lunxun_number + 1
        else:
            action1 = 3

        action2 = 0
        if action1 == -1:
            action1 = 3
        return np.array(action1), np.array(action2), lunxun_number
