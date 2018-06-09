#!/usr/bin/env python

import argparse
import retro
import time
import numpy as np
import queue

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

# imprime as coordenadas de todos os pixels não pretos da tela
# no tempo dado (instant). Serve para poder imprimir a tela em
# pontos para analisar se necessário
def printCoord(instant, obs):
  if t == instant:
    for i in range(len(obs)):
      for j in range(len(obs[i])):
        if sum(obs[i][j] > 0):
          print(i, j)
  time.sleep(1)

# Encontra a posição dos objetos quando o jogo inicia, considerando-os
# como retângulos e retornando os limites de cima, de baixo, esquerda
# e direita de cada um desses retângulos
def findInitialObjects(obs):
  print("0")
  visited = np.full((210, 160), False)
  q = queue.Queue()
  
  print("1")
  # lista que armazena a posição de cada objeto na tela
  # APENAS no momento que esta função é rodada
  # Por causa desta função, todos os objetos são vistos como retângulos
  # objects positions
  objsPos = []
  
  print("2")
  # Percorre a matriz apenas na área de interesse (vide Log)
  # A coluna de índice 160 (ou 159, no programa) não é verificada
  # porque acredito que isso não prejudicará o aprendizado e
  # reduz o número de verificações. Talvez seria possível fazer
  # um caso completamente a parte quando fosse buscar nessa última
  # coluna, mas não acho que seja necessário
  for i in range(18, 195):
    for j in range(8, 159):

      # busca os pixels que não foram visitados ainda
      if not visited[i][j]:
        if sum(obs[i][j]) > 0: # busca os pixels não-pretos
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

            # verifica quais pixels em volta pertenvem ao objeto
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

          # Left, Upper Corner & Right, Lower Corner
          objsPos.append((upperBound, lowerBound, leftBound, rightBound))

  return objsPos

# "main"
try:
  while True:
    obs = env.reset()
    env.render()
    t = 0 # time
    printCoord(t, obs)
    print(findInitialObjects(obs))
    print("BREAK BREAK BREAK")
    time.sleep(5)
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


