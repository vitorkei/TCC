import posMethods as pM
import speedMethods as sM
import auxMethods as aM
import numpy as np

SCREEN_UPPER_LIMIT = 18
SCREEN_LOWER_LIMIT = 194
SCREEN_LEFT_LIMIT  = 8
SCREEN_RIGHT_LIMIT = 159

SCREEN_HEIGHT = 177
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

      print()
      print(color, "<novo> - <velho>")
      print("upB, loB, leB, riB")
      print(upB, "-", elem['upperBound'], ";",
            loB, "-", elem['lowerBound'], ";",
            leB, "-", elem['leftBound'], ";",
            riB, "-", elem['rightBound'])

      # alteração VERTICAL/HORIZONTAL
      # utilizar um for aninhado como mais embaixo
      # não funciona, pois, se uma volta na tela for
      # dada, o i (ou j) não é inicializado, porque vai
      # de 194 até 45, por exemplo, o que significa que
      # o for sequer começa

    #  if self.spd[0] < 0: # se o asteroid estiver subindo
    #    if elem['upperBound'] - SCREEN_UPPER_LIMIT <= 1:
    #      upB = SCREEN_LOWER_LIMIT
    #      loB = elem['lowerBound']
    #  elif self.spd[0] > 0: # se o asteroid estiver descendo
    #    if SCREEN_LOWER_LIMIT - elem['lowerBound'] <= 1:
    #      loB = SCREEN_UPPER_LIMIT
    #      upB = elem['upperBound']
    #  else: # ainda não se sabe para onde vai
    #    upB = elem['upperBound']
    #    loB = elem['lowerBound']

    #  if self.spd[1] < 0: # se asteroid estiver movendo <-
    #    if elem['leftBound'] - SCREEN_LEFT_LIMIT <= 1:
    #      leB = SCREEN_RIGHT_LIMIT
    #      riB = elem['rightBound']
    #  elif self.spd[1] > 0: # se asteroid estiver movendo ->
    #    if SCREEN_RIGHT_LIMIT - elem['rightBound'] <= 1:
    #      riB = SCREEN_LEFT_LIMIT
    #      leB = elem['leftBound']
    #  else: # ainda não se sabe para onde vai
    #    leB = elem['leftBound']
    #    riB = elem['rightBound']

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

