import numpy as np
from collections import deque # double ended queue

class Memory():
  def __init__(self, max_size):
    self.buffer = deque(maxlen=max_size)

  def save(self, exp):
    self.buffer.append(exp)

  def sample(self, batch_size):
    sample_size = len(self.buffer)
    index = np.random.choice(np.arange(sample_size), batch_size, False)

    return [self.buffer[i] for i in index]
