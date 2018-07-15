import posMethods as pM
import speedMethods as sM
import auxMethods as aM
import numpy as np
import time

SCREEN_UPPER_LIMIT = 18
SCREEN_LOWER_LIMIT = 195
SCREEN_LEFT_LIMIT  = 8
SCREEN_RIGHT_LIMIT = 159

SCREEN_HEIGHT = 178
SCREEN_WIDTH = 152

class Asteroids:
  def __init__(self, obs):
    iniPos = pM.findInitialObjects(obs)

    self.count = len(iniPos)
    self.pos = dict()

    for i in range(self.count):
      self.pos[i] = iniPos[i]
      self.pos[i]['spd'] = [0, 0] # (h_vel, v_vel)

  def get_asteroids(self):
    return self.pos

  # Retorna o limite em volta do asteroide em
  # que será buscada a nova posição do asteroide
  def get_search_bounds(self, asteroid, delta):
    upB = asteroid['upperBound'] - SCREEN_UPPER_LIMIT # step1
    upB += SCREEN_HEIGHT - delta # step2
    upB = upB % SCREEN_HEIGHT # step3
    upB += SCREEN_UPPER_LIMIT # step4

    loB = asteroid['lowerBound'] - SCREEN_UPPER_LIMIT # step1
    loB = (loB + delta + 1) % SCREEN_HEIGHT # step2
    loB += SCREEN_UPPER_LIMIT # step3

    leB = asteroid['leftBound'] - SCREEN_LEFT_LIMIT # step1
    leB += SCREEN_WIDTH - delta # step2
    leB = leB % SCREEN_WIDTH # step3
    leB += SCREEN_LEFT_LIMIT # step4

    riB = asteroid['rightBound'] - SCREEN_LEFT_LIMIT # step1
    riB = (riB + delta + 1) % SCREEN_WIDTH # step2
    riB += SCREEN_LEFT_LIMIT # step3

    return upB, loB, leB, riB
  
  def get_smaller_asteroids(self, obs, old_asteroid):
    upB, loB, leB, riB = self.get_search_bounds(old_asteroid, 5)

    pass 

  # Atualiza a posição dos asteroides
  # Precisa ser chamada em todos os frames
  # que os asteroides aparecem (frames pares)
  def update_pos(self, obs, delta):
    destroyed_asteroids = []
    for ID, elem in self.pos.items():
      color = elem['color']
    
      # Determinação dos novos limites. Vide documentação
      upB, loB, leB, riB = self.get_search_bounds(elem, delta)

      asteroid_destroyed = True
      # alteração VERTICAL/HORIZONTAL
      i = upB
      while i != loB:
        j = leB
        while j != riB:
          if np.array_equal(obs[i][j], color):

            diff = i - elem['upperBound']
            elem['spd'][1] = diff
            #print("v_diff =", diff)

            elem['upperBound'] = i
            elem['lowerBound'] = (loB - SCREEN_UPPER_LIMIT + SCREEN_HEIGHT - (delta+1) + diff) % SCREEN_HEIGHT + SCREEN_UPPER_LIMIT
            
            asteroid_destroyed = False
            break
          j -= SCREEN_LEFT_LIMIT
          j = (j + SCREEN_WIDTH + 1) % SCREEN_WIDTH
          j += SCREEN_LEFT_LIMIT
        if np.array_equal(obs[i][j], color):
          break
        i -= SCREEN_UPPER_LIMIT
        i = (i + SCREEN_HEIGHT + 1) % SCREEN_HEIGHT
        i += SCREEN_UPPER_LIMIT

      j = leB
      while j != riB:
        i = upB
        while i != loB:
          if np.array_equal(obs[i][j], color):

            diff = j - elem['leftBound']
            elem['spd'][0] = diff
            #print("h_diff =", diff)

            elem['leftBound'] = j
            elem['rightBound'] = (riB - SCREEN_LEFT_LIMIT + SCREEN_WIDTH - (delta+1) + diff) % SCREEN_WIDTH + SCREEN_LEFT_LIMIT

            asteroid_destroyed = False
            break
          i -= SCREEN_UPPER_LIMIT
          i = (i + SCREEN_HEIGHT + 1) % SCREEN_HEIGHT
          i += SCREEN_UPPER_LIMIT
        if np.array_equal(obs[i][j], color):
          break
        j -= SCREEN_LEFT_LIMIT
        j = (j + SCREEN_WIDTH + 1) % SCREEN_WIDTH
        j += SCREEN_LEFT_LIMIT
      
      if asteroid_destroyed:
        print("aaaaaa\n")
        destroyed_asteroids.append(ID)
        print(ID, "-", elem, "\n")
    print()

    for i in destroyed_asteroids:
      print("DESTROYED asteroid(", i, ") -", self.pos[i], "\n")
      del self.pos[i]
      destroyed_asteroids.remove(i)
