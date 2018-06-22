#!/usr/bin/env python

import argparse
import retro
import time
import numpy as np
import queue

############################################################
# Um monte de coisas para o programa funcionar. Veio com o exemplo do gym-retro

parser = argparse.ArgumentParser()
parser.add_argument('game', help='the name or path for the game to run')
parser.add_argument('state', nargs='?', help='the initial state file to load, minus the extension')
parser.add_argument('--scenario', '-s', default='scenario', help='the scenario file to load, minus the extension')
parser.add_argument('--record', '-r', action='store_true', help='record bk 2 movies')
parser.add_argument('--verbose', '-v', action='count', default=1, help='increase verbosity (can be specified multiple times)')
parser.add_argument('--quiet', '-q', action='count', default=0, help='decrease verbosity (can be specified multiple times)')
args = parser.parse_args()
env = retro.make(args.game, args.state or retro.STATE_DEFAULT, scenario=args.scenario, record=args.record)
verbosity = args.verbose - args.quiet

#############################################################

# imprime as coordenadas de todos os pixels não pretos da tela
# no tempo dado (instant). Serve para poder imprimir a tela em
# pontos para analisar se necessário (debugging)
def printCoord(instant, obs):
  if t == instant:
    for i in range(len(obs)):
      for j in range(len(obs[i])):
        if sum(obs[i][j] > 0):
          print(i, j)
  time.sleep(1)

# Realiza uma BFS para retornar a cor do asteroide e os
# limites superior, inferior, da esquerda e da direita
def asteroidBFS(obs, visited, i, j):
  q = queue.Queue()
  # busca os pixels que não foram visitados ainda
  color = obs[i][j].copy() # cor do objeto que estou olhando
  visited[i][j] = True
  q.put((i, j))
  
  # na tela do jogo, a parte de baixo tem índices maiores que a de cima
  # e a parte da direita tem índices maiores que a da esquerda
  # Portanto, upperBound será um número menor que lowerBound e
  # leftBound terá índice menor que rightBound
  upperBound = i
  lowerBound = i
  leftBound  = j
  rightBound = j

  while q.qsize() > 0:
    coord = q.get()
    m = coord[0]
    n = coord[1]

    # verifica quais pixels em volta pertencem ao objeto
    # que estou buscando e não foram visitado ainda
    if not visited[m+1][n]: # abaixo
      if(np.array_equal(obs[m+1][n], color)):
        visited[m+1][n] = True
        q.put((m+1, n))
        if m+1 > lowerBound:
          lowerBound += 1

    if not visited[m][n-1]: # a esquerda
      if(np.array_equal(obs[m][n-1], color)):
        visited[m][n-1] = True
        q.put((m, n-1))
        if n-1 < leftBound:
          leftBound -= 1

    if not visited[m-1][n]: # acima
      if(np.array_equal(obs[m-1][n], color)):
        visited[m-1][n] = True
        q.put((m-1, n))
        if m-1 < upperBound:
          upperBound -= 1

    if not visited[m][n+1]: # a direita
      if(np.array_equal(obs[m][n+1], color)):
        visited[m][n+1] = True
        q.put((m, n+1))
        if n+1 > rightBound:
          rightBound += 1

  return (color, upperBound, lowerBound, leftBound, rightBound)

# Encontra a posição dos objetos quando o jogo inicia, considerando-os
# como retângulos e retornando os limites de cima, de baixo, esquerda
# e direita de cada um desses retângulos
# Retornar uma lista com os limites dos asteroides e suas respectivas
# cores, mas ajudar na identificação
def findInitialObjects(obs):
  visited = np.full((210, 160), False)
  
  # lista que armazena a posição de cada objeto na tela
  # APENAS no momento que esta função é rodada
  # Por causa desta função, todos os objetos são vistos como retângulos

  # objects positions
  objsPos = []
  
  # Percorre a matriz apenas na área de interesse (vide Log)
  # A coluna de índice 160 (ou 159, no programa) não é verificada
  # porque acredito que isso não prejudicará o aprendizado e
  # reduz o número de verificações. Talvez seria possível fazer
  # um caso completamente a parte quando fosse buscar nessa última
  # coluna, mas não acho que seja necessário
  for i in range(18, 195):
    for j in range(8, 159):
      if not visited[i][j]:
        if sum(obs[i][j]) > 0: # busca os pixels não-pretos
          objsPos.append(asteroidBFS(obs, visited, i, j))

  return objsPos

# Retorna a velocidade horizontal do asteroide
def horizontalSpeed(obs, obj):
  color = obj[0]
  for j in range(obj[3] - 3, obj[4] + 4):
    for i in range(obj[1] - 5, obj[2] + 6):
      if np.array_equal(obs[i][j], color):
        return j - obj[3]

# Retorna a velocidade vertical do asteroide
def verticalSpeed(obs, obj):
  color = obj[0]
  for i in range(obj[1] - 5, obj[2] + 6):
    for j in range(obj[3] - 3, obj[4] + 4):
      if np.array_equal(obs[i][j], color):
        return i - obj[1]

# Retorna uma lista cujo primeiro elemento de cada elemento é a cor
# do asteroide, para identificação e uma tupla que determina a
# velocidade do asteroide. Tal tupla diz quantos pixels o asteroide
# se move na horizontal e quantos na vertical a cada 2 frames
# (já que os asteroides intercalam os frames que são renderizados
# na tela com a nave).
# Esta função deve ser chamada na segunda vez que os asteroides
# são renderizados, dado que suas velocidades são constantes
def asteroidsSpeed(objsPos, obs):
  astsSpeed = []
  for obj in objsPos:
    color = obj[0] # "identificador" do asteroide

    # A busca de para onde o asteróid foi é feita em uma pequena área
    # em volta da posição inicial. Como os asteróides tem um movimento
    # vertical maior do que horizontal, o espaço de busca para cima
    # e para baixo é maior (+-5 ao invés de +-3). Esta informação
    # é válida para as funções verticalSpeed() e horizontalSpeed()
    
    hSPD = horizontalSpeed(obs, obj)
    vSPD = verticalSpeed(obs, obj)

    astsSpeed.append((color, hSPD, vSPD))

  return astsSpeed

# "main"
try:
  while True:
    obs = env.reset()
    t = 0 # time
    #printCoord(t, obs)
    #print(findInitialObjects(obs))
    #print("BREAK BREAK BREAK")
    #time.sleep(5)
    astsIniPos = findInitialObjects(obs)

    totrew = 0 # total reward
    while True:
      action = env.action_space.sample()
      obs, rew, done, info = env.step(action)
      t += 1
      if t % 10 == 0:
        if verbosity > 1:
          infostr = ''
          if info:
            infostr = ', info: ' + ', '.join(['%s=%i' % (k, v) for k, v in info.items()])
            print(('t=%i' % t) + infostr)
      env.render()

      if t % 10 == 0:
        astsSPD = asteroidsSpeed(astsIniPos, obs)
        print(t, " -", astsSPD)
      time.sleep(0.01)

      # printCoord(400, obs)
      
      totrew += rew
      if verbosity > 0:
        if rew > 0:
          print('time: %i got reward: %d, current reward: %d' % (t, rew, totrew))
        if rew < 0:
          print('time: %i got penalty: %d, current reward: %d' % (t, rew, totrew))
      if done:
        env.render()
        try:
          if verbosity >= 0:
            print("Done! Total reward: time=%i, reward=%d" % (t, totrew))
            input("press enter to continue")
            print()
          else:
            input("")
        except EOFError:
          exit(0)
        break
except KeyboardInterrupt:
  exit(0)


