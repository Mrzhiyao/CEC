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
        self.state_size = state_size
        self.action_size = action_size

        self.device = device
        
    def get_action_algorithm1(self, state):
        """Returns actions for given state as per current policy."""
        action1 = 3
        action2 = 0
        return np.array(action1), np.array(action2)
