import time
import numpy as np

SCREEN_UPPER_LIMIT = 18
SCREEN_LOWER_LIMIT = 195
SCREEN_LEFT_LIMIT = 8
SCREEN_RIGHT_LIMIT = 159

SCREEN_HEIGHT = 178
SCREEN_WIDTH = 152

# Imprime as coordenadas de todos os pixels não pretos
# da tela (obs) e para o programa durante delta segundos
# FUNÇÃO DE DEBUGGING
def printCoords(obs, delta):
  if t % 500 == 0:
    for i in range(SCREEN_UPPER_LIMIT, SCREEN_LOWER_LIMIT+1):
      for j in range(SCREEN_LEFT_LIMIT, SCREEN_RIGHT_LIMIT+1):
        if np.sum(obs[i][j]) > 0:
          print(i, j)
    print("t =", t)
    input("Aperte enter para prosseguir...")
