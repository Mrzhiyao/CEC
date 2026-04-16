import numpy as np
import random
import torch
from collections import deque, namedtuple

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        data = [item[0] for item in experiences] 
        uniform_data = [[float(elem) if isinstance(elem, np.ndarray) else float(elem) for elem in sublist] for sublist in data]
        data_array = np.array(uniform_data)
        states = torch.from_numpy(np.squeeze(data_array)).float().to(self.device)
        action_st = np.array([item[1] for item in experiences])
        action_st = np.expand_dims(action_st, axis=1)
        
        actions = torch.from_numpy(action_st).float().to(self.device)
        rewards = torch.from_numpy(np.array([item[2] for item in experiences])).float().to(self.device)

        data = [item[3] for item in experiences]

        uniform_data = [[float(elem) if isinstance(elem, np.ndarray) else float(elem) for elem in sublist] for sublist in data]
        data_array = np.array(uniform_data)
        next_states = torch.from_numpy(np.squeeze(data_array)).float().to(self.device)
        dones = torch.from_numpy(np.array([item[4] for item in experiences])).float().to(self.device)
        rewards = rewards.unsqueeze(-1)
        dones = dones.unsqueeze(-1)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
