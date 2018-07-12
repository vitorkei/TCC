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

  # Atualiza a posição dos asteroides
  # Precisa ser chamada em todos os frames
  # que os asteroides aparecem (frames pares)
  def update_pos(self, obs):
    for ID, elem in self.pos.items():
      color = elem['color']
    
      # Determinação dos novos limites. Vide documentação
      upB = elem['upperBound'] - SCREEN_UPPER_LIMIT # step1
      upB += SCREEN_HEIGHT - 2 # step2
      upB = upB % SCREEN_HEIGHT # step3
      upB += SCREEN_UPPER_LIMIT # step4

      loB = elem['lowerBound'] - SCREEN_UPPER_LIMIT # step1
      loB = (loB + 3) % SCREEN_HEIGHT # step2
      loB += SCREEN_UPPER_LIMIT # step3

      leB = elem['leftBound'] - SCREEN_LEFT_LIMIT # step1
      leB += SCREEN_WIDTH - 2 # step2
      leB = leB % SCREEN_WIDTH # step3
      leB += SCREEN_LEFT_LIMIT # step4

      riB = elem['rightBound'] - SCREEN_LEFT_LIMIT # step1
      riB = (riB + 3) % SCREEN_WIDTH # step2
      riB += SCREEN_LEFT_LIMIT # step3

      # alteração VERTICAL/HORIZONTAL
      i = upB
      while i != loB:
        j = leB
        while j != riB:
          if np.array_equal(obs[i][j], color):

            diff = i - elem['upperBound']

            print("v_diff =", diff)
            elem['upperBound'] = i
            print("loB =", loB, ";", "S_U_L =", SCREEN_UPPER_LIMIT, ";", "S_H =", SCREEN_HEIGHT, ";", "diff =", diff)
            elem['lowerBound'] = (loB - SCREEN_UPPER_LIMIT + SCREEN_HEIGHT - 3 + diff) % SCREEN_HEIGHT + SCREEN_UPPER_LIMIT

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

            print("h_diff =", diff)
            elem['leftBound'] = j
            elem['rightBound'] = (riB - SCREEN_LEFT_LIMIT + SCREEN_WIDTH - 3 + diff) % SCREEN_WIDTH + SCREEN_LEFT_LIMIT

            break
          i -= SCREEN_UPPER_LIMIT
          i = (i + SCREEN_HEIGHT + 1) % SCREEN_HEIGHT
          i += SCREEN_UPPER_LIMIT
        if np.array_equal(obs[i][j], color):
          break
        j -= SCREEN_LEFT_LIMIT
        j = (j + SCREEN_WIDTH + 1) % SCREEN_WIDTH
        j += SCREEN_LEFT_LIMIT
      print(elem, "\n")
    print()
    #  # alteração VERTICAL
    #  # TODO Não considera o caso de o asteroide
    #  # ser destruído
    #  for i in range(upB-2, loB+3):
    #    for j in range(leB-2, riB + 3):
    #      if np.array_equal(obs[i][j], color):
    #        diff = i - elem['upperBound']
    #        print("v_diff =", diff)
    #        elem['upperBound'] += diff
    #        elem['lowerBound'] += diff
    #        if self.spd[0] == 0:
    #          self.spd[0] = diff
    #        break
    #    if np.array_equal(obs[i][j], color):
    #      break
    # 
    #  # alteração HORIZONTAL
    #  # TODO não considera o caso de o asteroide
    #  # ser destruído
    #  for j in range(leB-2, riB+3):
    #    for i in range(upB-2, loB+3):
    #      if np.array_equal(obs[i][j], color):
    #        diff = j - elem['leftBound']
    #        print("h_diff =", diff)
    #        elem['rightBound'] += diff
    #        elem['leftBound'] += diff
    #        if self.spd[1] == 0:
    #          self.spd[1] = diff
    #        break
    #    if np.array_equal(obs[i][j], color):
    #      break

