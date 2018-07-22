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
  def __init__(self):
    # Posição inicial da nave
    self.base_pos = {'color': [240, 128, 128],
                'upperBound': 100,
                'lowerBound': 109,
                'leftBound': 84,
                'rightBound': 88}
    self.pos = self.base_pos
    self.blink = False

  def get_pos(self):
    return self.pos

  def update_pos(self, obs, delta):
    print ("blink =", self.blink)
    if not self.blink:
      aux = pos.find_ship(obs, self.pos, delta)
      self.pos = aux
      if aux == None:
        self.blink = True
        self.pos = self.base_pos
    else:
      aux = pos.blink(obs, self.pos['color'])
      if not (aux == None):
        self.blink = False
        self.pos = aux
    #print(self.get_pos())

