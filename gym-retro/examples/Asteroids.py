import posMethods as pM
import speedMethods as sM
import auxMethods as aM
import numpy as np

class Asteroids:
  def __init__(self, obs):
    iniPos = pM.findInitialObjects(obs)

    self.count = len(iniPos)
    self.pos = dict()

    for i in range(self.count):
      self.pos[i] = iniPos[i]

  def get_asteroids(self):
    return self.pos

  # Atualiza a posição dos asteroides
  # Precisa ser chamada em todos os frames
  # que os asteroides aparecem (frames pares)
  def update_pos(self, obs):
    for k, elem in self.pos.items():
      color = elem['color']
      upB = elem['upperBound']
      loB = elem['lowerBound']
      leB = elem['leftBound']
      riB = elem['rightBound']

      # alteração VERTICAL
      # TODO Não considera o caso de o asteroide
      # ser destruído
      for i in range(upB-2, loB+3):
        for j in range(leB-2, riB + 3):
          if np.array_equal(obs[i][j], color):
            diff = i - elem['upperBound']
            print("diff =", diff)
            elem['upperBound'] += diff
            elem['lowerBound'] += diff
            break
        if np.array_equal(obs[i][j], color):
          break

