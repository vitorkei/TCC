import posMethods as pM
import speedMethods as sM
import auxMethods as aM
import numpy as np

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
    self.spd = np.array((0, 0))

    for i in range(self.count):
      self.pos[i] = iniPos[i]

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
    pass 

  # Atualiza a posição dos asteroides
  # Precisa ser chamada em todos os frames
  # que os asteroides aparecem (frames pares)
  def update_pos(self, obs):
    destroyed_asteroids = []
    for ID, elem in self.pos.items():
      color = elem['color']
    
      # Determinação dos novos limites. Vide documentação
      upB, loB, leB, riB = self.get_search_bounds(elem, 2)

      asteroid_destroyed = True
      # alteração VERTICAL/HORIZONTAL
      i = upB
      while i != loB:
        j = leB
        while j != riB:
          if np.array_equal(obs[i][j], color):

            diff = i - elem['upperBound']
            #print("v_diff =", diff)

            elem['upperBound'] = i
            elem['lowerBound'] = (loB - SCREEN_UPPER_LIMIT + SCREEN_HEIGHT - 3 + diff) % SCREEN_HEIGHT + SCREEN_UPPER_LIMIT
            
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
            #print("h_diff =", diff)

            elem['leftBound'] = j
            elem['rightBound'] = (riB - SCREEN_LEFT_LIMIT + SCREEN_WIDTH - 3 + diff) % SCREEN_WIDTH + SCREEN_LEFT_LIMIT

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
        print ("DESTROYED asteroid(", ID, ") =", elem, "\n");
        destroyed_asteroids.append(ID)
      else:
        print(ID, "-", elem, "\n")
    print()

    for i in destroyed_asteroids:
      del self.pos[i]

