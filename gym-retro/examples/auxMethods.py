import time

# Imprime as coordenadas de todos os pixels não pretos
# da tela (obs) e para o programa durante delta segundos
# FUNÇÃO DE DEBUGGING
def printCoord(obs, delta):
  for i in range(len(obs)):
    for j in range(len(obs[i])):
      if sum(obs[i][j] > 0):
        print(i, j)
  time.sleep(delta)
