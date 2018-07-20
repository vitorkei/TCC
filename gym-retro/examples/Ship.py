import pos_methods as pos
import numpy as np
import time
import queue

SCREEN_UPPER_LIMIT = 18
SCREEN_LOWER_LIMIT = 195
SCREEN_LEFT_LIMIT = 8
SCREEN_RIGHT_LIMIT = 159

SCREEN_HEIGHT = 178
SCREEN_WIDTH = 152

class Ship:
  def __init__(self, obs):
    self.pos = {'color': [240, 128, 128],
                'upperBound': 100,
                'lowerBound': 109,
                'leftBound': 84,
                'rightBound': 88}

  def get_pos():
    return self.pos
