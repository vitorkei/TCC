import numpy as np

# Arquivo que contém funções relacionadas a velocidade
# dos asteroides

# Retorna velocidade horizontal do asteroide
def horizontal_speed(obs, obj, delta):
  color = obj['color']
  for j in range(obj['leftBound'] - 3, obj['rightBound'] + 4):
    for i in range(obj['upperBound'] - 5, obj['lowerBound'] + 6):
      if np.array_equal(obs[i][j], color):
        return (j - obj['rightBound']) / delta

# Retorna velocidade vertical do asteroide
def vertical_speed(obs, obj, delta):
  color = obj['color']
  for i in range(obj['upperBound'] - 5, obj['lowerBound'] + 6):
    for j in range(obj['rightBound'] - 3, obj['leftBound'] + 4):
      if np.array_equal(obs[i][j], color):
        return (i - obj['upperBound']) / delta

# Retorna a velocidade de cada asteroide da tela
# Entrada:
#   iniPos: posição inicial dos asteroides (primeira observação)
#   obs: posição dos objetos no momento observado
#   delta: tempo passado entre iniPos e obs
# Saída: lista de tuplas tais que seus elementos são
#   color: cor que identifica o asteroide
#   hSPD: velocidade horizontal do asteroide
#   vSPD: velocidade vertical do steroide
def asteroidsSpeed(ini_pos, obs, delta):
  asts_speed = []
  for k, v in ini_pos:
    print(ini_pos[obj])
    color = v['color']

    h_SPD = horizontal_speed(obs, v, delta)
    v_SPD = vertical_speed(obs, v, delta)

    asts_speed.append((color, h_SPD, v_SPD))

  return asts_speed
