import posMethods as pM
import speedMethods as sM
import auxMethods as aM
import numpy as np
import time
import queue

SCREEN_UPPER_LIMIT = 18
SCREEN_LOWER_LIMIT = 195
SCREEN_LEFT_LIMIT  = 8
SCREEN_RIGHT_LIMIT = 159

SCREEN_HEIGHT = 178
SCREEN_WIDTH = 152

class Asteroids:
  def __init__(self, obs):
    iniPos = pM.find_objects(obs)

    self.count = len(iniPos)
    #print("num ast =", self.count)
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
 
  # Quando um asteróide é destruído, é preciso ver
  # a posição dos asteróides filhos se é que algum surgiu
  #def update_asteroids(self, obs, old_asteroid):
    #q = queue.Queue()

    # Limites da área em que os "asteróides filhos" serão procurados
    #ID  = old_asteroid[0]
    #upB = old_asteroid[1]
    #loB = old_asteroid[2]
    #leB = old_asteroid[3]
    #riB = old_asteroid[4]
    
    #visited = np.full((210, 160), False)

    #objsPos = []

    # Encontra os asteróides filhos
    # Vide Log (16/Jul/2018) para mais informações
    #for i in range(upB, loB):
      #for j in range(leB, riB):
        #if not visited[i][j]:
          #if sum(obs[i][j]) > 0:
            #objsPos.append(pM.asteroid_BFS(obs, visited, i, j))
    
    #for child_asteroid in objsPos:
      #print("\nold_asteroid:", self.pos[old_asteroid[0]])
      #self.pos[self.count] = child_asteroid
      #self.pos[self.count]['spd'] = self.pos[ID]['spd']
      #print("child_asteroid:", child_asteroid)
      #self.count += 1

      #print("num ast =", self.count)


  # Atualiza a posição dos asteroides
  # Precisa ser chamada em todos os frames
  # que os asteroides aparecem (frames pares)
  def update_pos(self, obs, delta):
    #destroyed_asteroids = []
    for ID, elem in self.pos.items():
      color = elem['color']
    
      # Determinação dos novos limites. Vide documentação
      upB, loB, leB, riB = self.get_search_bounds(elem, delta)

      #asteroid_destroyed = True
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
            
            #asteroid_destroyed = False
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

            #asteroid_destroyed = False
            break
          i -= SCREEN_UPPER_LIMIT
          i = (i + SCREEN_HEIGHT + 1) % SCREEN_HEIGHT
          i += SCREEN_UPPER_LIMIT
        if np.array_equal(obs[i][j], color):
          break
        j -= SCREEN_LEFT_LIMIT
        j = (j + SCREEN_WIDTH + 1) % SCREEN_WIDTH
        j += SCREEN_LEFT_LIMIT
      
      #if asteroid_destroyed:
        #print("One asteroid destroyed!\n")
        #destroyed_asteroids.append([ID, upB, loB, leB, riB])
      #else:
        #print(ID, "-", elem, "\n")
      print(ID, "-", elem, "\n")
    #print()

    #for d_a in destroyed_asteroids:
      #print("Destroyed asteroids:\n(", d_a[0], ") -", self.pos[d_a[0]], "\nupB:", d_a[1], "; loB:", d_a[2], "; leB:", d_a[3], "; riB:", d_a[4])
      #self.update_asteroids(obs, d_a)
      #del self.pos[d_a[0]]
      #destroyed_asteroids.remove(d_a)
