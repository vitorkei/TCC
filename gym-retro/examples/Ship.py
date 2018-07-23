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
    self.life_count = 64
    self.blink = False # válido para teleporte e para morte
    self.blink_timer = 0
    self.ast_dist = dict() # distância da nave até cada asteróide

  def get_pos(self):
    return self.pos

  def update_pos(self, obs, delta):
    #print("self.blink_timer =", self.blink_timer)
    #print ("blink =", self.blink)
    if not self.blink:
      aux = pos.find_ship(obs, self.pos, delta)
      self.pos = aux
      if aux == None:
        self.blink = True
        self.pos = self.base_pos
        self.blink_timer = 30
        #input("SUMIU")
    else:
      if self.blink_timer == 0:
        aux = pos.blink(obs, self.pos['color'])
        if not (aux == None):
          self.blink = False
          self.pos = aux
          #input("REAPARECEU")
      else:
        self.blink_timer -= 1
    #print("blink =", self.blink)

  def get_life_count(self):
    return self.life_count

  def set_life_count(self, life_count):
    self.life_count = life_count

  def get_blink_timer(self):
    return self.blink_timer

  def has_died(self):
    self.blink_timer = 57
    self.life_count -= 16

  def get_ast_dist(self):
    return self.ast_dist

  def set_ast_dist(self, asteroids):
    ship_center = [0, 0]
    ship_center[0] = self.pos['lowerBound'] + SCREEN_HEIGHT - self.pos['upperBound']
    ship_center[0] = ship_center[0] % SCREEN_HEIGHT
    ship_center[0] = round(ship_center[0] / 2) + self.pos['upperBound']
    ship_center[0] = ship_center[0] % SCREEN_HEIGHT

    ship_center[1] = self.pos['rightBound'] + SCREEN_WIDTH - self.pos['leftBound']
    ship_center[1] = ship_center[1] % SCREEN_WIDTH
    ship_center[1] = round(ship_center[1] / 2) + self.pos['leftBound']
    ship_center[1] = ship_center[1] % SCREEN_WIDTH
    print(ship_center)
    for ID, asteroid in asteroids.items():
      print("\n", ID, "-", asteroid)
      self.ast_dist[ID] = [0, 0]
      
      print("ast_center_line:")
      # Número da linha (aproximada) em que o centro do asteróide está
      ast_center_line = asteroid['lowerBound'] + SCREEN_HEIGHT - asteroid['upperBound']
      ast_center_line = ast_center_line % SCREEN_HEIGHT
      ast_center_line = round(ast_center_line / 2) + asteroid['upperBound']
      ast_center_line = ast_center_line % SCREEN_HEIGHT

      # Número da coluna (aproximada) em que o centro do asteróide esta
      ast_center_col = asteroid['rightBound'] + SCREEN_WIDTH - asteroid['leftBound']
      ast_center_col = ast_center_col % SCREEN_WIDTH
      ast_center_col = round(ast_center_col / 2) + asteroid['leftBound']
      ast_center_col = ast_center_col % SCREEN_WIDTH

      print(ast_center_line, "|", ast_center_col)

      line_dist_0 = abs(ast_center_line - ship_center[0])
      line_dist_1 = SCREEN_HEIGHT - line_dist_0

      col_dist_0 = abs(ast_center_col - ship_center[1])
      col_dist_1 = SCREEN_WIDTH - col_dist_0

      if line_dist_0 < line_dist_1:
        self.ast_dist[ID][0] = line_dist_0
      else:
        self.ast_dist[ID][0] = line_dist_1

      if col_dist_0 < col_dist_1:
        self.ast_dist[ID][1] = col_dist_0
      else:
        self.ast_dist[ID][1] = col_dist_1
