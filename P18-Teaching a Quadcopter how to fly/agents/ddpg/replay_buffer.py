import random
from collections import namedtuple, deque

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples"""
    
    def __init__(self, buffer_size, batch_size):
        """
            Initialize a ReplayBuffer object
            Args:
                buffer_size: Maximum size of the buffer
                batch_size: Size of each training batch
        """
        # Internal memory
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done"])
        
    def add(self, state, action, reward, next_state, done):
        """
            Add a new experience to memory
            Args:
                state: Current state
                action: Current action performed
                reward: Reward as a consequence of current state
                next_state: Next state
                done: Boolean indicating the end of an episode
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self, batch_size=64):
        """
            Randomly sample a batch of experience from memory
            Args:
                batch_size: Batch size to sample. Defaults to 64
        """
        return random.sample(self.memory, k=self.batch_size)
    
    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.memory)
    
