import random
from collections import deque
import numpy as np

class MemoryBuffer:
    """
    Store the preprocessed sequence, action, 
    reward, and preprocessed next sequence
    in a queue. Store up to N tuples in replay
    memory, and sample uniformly during updates.
    Overwrite older tuples in the memory updates
    once N is exceeded. 
    """
    def __init__(self, N):
        super().__init__()
        self.memory = deque(maxlen=N)
    
    def store_transition(self, transition: list):
        self.memory.append(transition)

    def random_sample(self, minibatch_size):
        minibatch = random.sample(self.memory, minibatch_size)

        return minibatch
    
    def __len__(self):
        return len(self.memory)

    def __getitem__(self, n):
        return self.memory[n]