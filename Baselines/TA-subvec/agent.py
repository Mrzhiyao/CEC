import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import numpy as np
import math
import copy
import random

class CQLSAC(nn.Module):
    """Interacts with and learns from the environment."""
    
    def __init__(self,
                        state_size,
                        action_size,
                        device
                ):
        super(CQLSAC, self).__init__()
        self.device = device
    
    def get_action_algorithm1(self, state):
        if state[18] < 0.6 and state[38] < 0.6 and state[58] < 0.6:
            action1 = 3
        elif state[18] >= state[38] and state[18] >= state[58]:
            action1 = 0
        elif state[38] >= state[18] and state[38] >= state[58]:
            action1 = 1
        elif state[58] >= state[18] and state[58] >= state[38]:
            action1 = 2
        else:
            action1 = random.randint(0, 3)
        action2 = 0
        return np.array(action1), np.array(action2)

 