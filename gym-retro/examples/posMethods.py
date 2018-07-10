import time
import numpy as np
import queue

SCREEN_UPPER_LIMIT = 18
SCREEN_LOWER_LIMIT = 194
SCREEN_LEFT_LIMIT  = 8
SCREEN_RIGHT_LIMIT = 159

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
  visited[i][j] = True
  q.put((i, j))

  upperBound = i
  lowerBound = i
  leftBound = j
  rightBound = j

  while q.qsize() > 0:
    m, n = q.get()

    if not visited[m+1][n]: # pixel abaixo
      visited[m+1][n] = True # não olha p/ mesmo pixel preto de novo
      if np.array_equal(obs[m+1][n], color):
        q.put((m+1, n))
        if m+1 > lowerBound:
          lowerBound += 1

    if not visited[m][n-1]: # pixel a esquerda
      visited[m][n-1] = True
      if np.array_equal(obs[m][n-1], color):
        q.put((m, n-1))
        if n-1 < leftBound:
          leftBound -= 1

    if not visited[m-1][n]: # pixel acima
      visited[m-1][n] = True
      if np.array_equal(obs[m-1][n], color):
        q.put((m-1, n))
        if m-1 < upperBound:
          upperBound -= 1
    
    if n < SCREEN_RIGHT_LIMIT:
      if not visited[m][n+1]: # pixel a direita
        visited[m][n+1] = True
        if np.array_equal(obs[m][n+1], color):
          q.put((m, n+1))
          if n+1 > rightBound:
            rightBound += 1
  
  return {'color': color,
          'upperBound': upperBound,
          'lowerBound': lowerBound,
          'leftBound': leftBound,
          'rightBound': rightBound}

# Procura todos os objetos na tela (obs) e retorna uma lista deles
# Entrada:
#   obs: observação do ambiente (matriz
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

  for i in range(18, 195):
    for j in range(8, 159):
      if not visited[i][j]:
        if sum(obs[i][j]) > 0:
          objsPos.append(asteroidBFS(obs, visited, i, j))

  return objsPos
  
