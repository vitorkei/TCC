import time
import numpy as np
import queue

SCREEN_UPPER_LIMIT = 18
SCREEN_LOWER_LIMIT = 194
SCREEN_LEFT_LIMIT  = 8
SCREEN_RIGHT_LIMIT = 159

SCREEN_HEIGHT = 177
SCREEN_WIDTH = 152

# Entrada:
#    obs: observação do ambiente (matriz)
#    visited: define pixels já visitados (matriz)
#    (i, j): coordenadas do primeiro pixel visitado
# Saída: dictionary com:
#    color: cor do asteróide
#    upperBound: número da linha do pixel mais acima (menor)
#    lowerBound: número da linha do pixel mais abaixo (maior)
#    leftBound: número da coluna do pixel mais a esquerda (menor)
#    rightbound: número da coluna do pixel mais a direita (maior)
def asteroidBFS(obs, visited, i, j):
  q = queue.Queue()

  color = obs[i][j].copy()
  #print("\n\n***********\nCOLOR =", color, " | (i, j) = (", i, ",", j, ")")
  visited[i][j] = True
  q.put((i, j))

  upperBound = i
  lowerBound = i
  leftBound = j
  rightBound = j

  while q.qsize() > 0:
    m, n = q.get()

    # upB, loB, leB, riB ajudarão a identificar os novos limites do asteróide
    m_up = m - SCREEN_UPPER_LIMIT
    m_up += SCREEN_HEIGHT - 1
    upB = m_up
    m_up = m_up % SCREEN_HEIGHT
    m_up += SCREEN_UPPER_LIMIT
    
    m_down = m - SCREEN_UPPER_LIMIT
    m_down += SCREEN_HEIGHT + 1
    loB = m_down
    m_down = m_down % SCREEN_HEIGHT
    m_down += SCREEN_UPPER_LIMIT

    n_left = n - SCREEN_LEFT_LIMIT
    n_left += SCREEN_WIDTH - 1
    leB = n_left
    n_left = n_left % SCREEN_WIDTH
    n_left += SCREEN_LEFT_LIMIT

    n_right = n - SCREEN_LEFT_LIMIT
    n_right += SCREEN_WIDTH + 1
    riB = n_right
    n_right = n_right % SCREEN_WIDTH
    n_right += SCREEN_LEFT_LIMIT

    if not visited[m_down][n]: # pixel abaixo
      if np.array_equal(obs[m_down][n], color):
        visited[m_down][n] = True
        q.put((m_down, n))
        if loB > lowerBound + SCREEN_HEIGHT - SCREEN_UPPER_LIMIT:
          #print("lowerBound | loB | m_down =", lowerBound, "|", loB, "|", m_down) 
          lowerBound = m_down

    if not visited[m][n_left]: # pixel a esquerda
      if np.array_equal(obs[m][n_left], color):
        visited[m][n_left] = True
        q.put((m, n_left))
        if leB < leftBound + SCREEN_WIDTH - SCREEN_LEFT_LIMIT:
          #print("leftBound | leB | n_left =", leftBound, "|", leB, "|", n_left)
          leftBound = n_left

    if not visited[m_up][n]: # pixel acima
      if np.array_equal(obs[m_up][n], color):
        visited[m_up][n] = True
        q.put((m_up, n))
        if upB < upperBound + SCREEN_HEIGHT - SCREEN_UPPER_LIMIT:
          #print("upperBound | upB | m_up =", upperBound, "|", upB, "|", m_up)
          upperBound = m_up
    
    if n < SCREEN_RIGHT_LIMIT:
      if not visited[m][n_right]: # pixel a direita
        if np.array_equal(obs[m][n_right], color):
          visited[m][n_right] = True
          q.put((m, n_right))
          if riB > rightBound + SCREEN_WIDTH - SCREEN_LEFT_LIMIT:
            #print("rightBound | riB | n_right =", rightBound, "|", riB, "|", n_right)
            rightBound = n_right
  
  #for m in range (210):
    #for n in range(160):
      #if visited[m][n]:
        #print(m, n)
  #input("esperando...")

  return {'color': color,
          'upperBound': upperBound,
          'lowerBound': lowerBound,
          'leftBound': leftBound,
          'rightBound': rightBound}

# Procura todos os objetos na tela (obs) e retorna uma lista deles
# Entrada:
#   obs: observação do ambiente (matriz)
# Saída:
#   objsPos: lista com a cor, limite inferior, superior, esquerdo e
#            e direito de cada objeto encontrado; eles são vistos
#            como retângulos, mesmo não sendo realmente
# A área extra que não faz realmente parte dos objetos encontrados
# pode ser vista como uma margem de segurança caso o agente se
# aproxime demais
def findInitialObjects(obs):
  visited = np.full((210, 160), False)

  objsPos = []

  for i in range(SCREEN_UPPER_LIMIT, SCREEN_LOWER_LIMIT+1):
    for j in range(SCREEN_LEFT_LIMIT, SCREEN_RIGHT_LIMIT+1):
      if not visited[i][j]:
        if sum(obs[i][j]) > 0:
          objsPos.append(asteroidBFS(obs, visited, i, j))

  return objsPos
  
