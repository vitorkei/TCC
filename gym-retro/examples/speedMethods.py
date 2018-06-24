import numpy as np

# Arquivo que contém funções relacionadas a velocidade
# dos asteroides

# Retorna velocidade horizontal do asteroide
def horizontalSpeed(obs, obj, delta):
  color = obj['color']
  for j in range(obj['leftBound'] - 3, obj['rightBound'] + 4):
    for i in range(obj['upperBound'] - 5, obj['lowerBound'] + 6):
      if np.array_equal(obs[i][j], color):
        return (j - obj['rightBound']) / delta

# Retorna velocidade vertical do asteroide
def verticalSpeed(obs, obj, delta):
  color = obj['color']
  for i in range(obj['upperBound'] - 5, obj['lowerBound'] + 6):
    for j in range(obj['rightBound'] - 3, obj['leftBound'] + 4):
      if np.array_equal(obs[i][j], color):
        return (i - obj['upperBound']) / delta

# Retorna a velocidade de cada asteroide da tela
# Entrada:
#   iniPos: posição inicial dos asteroides (primeira observação)
#   obs: posição dos objetos no momento obersvado
#   delta: tempo passado entre iniPos e obs
# Saída: lista de tuplas tais que seus elementos são
#   color: cor que identifica o asteroide
#   hSPD: velocidade horizontal do asteroide
#   vSPD: velocidade vertical do steroide
def asteroidsSpeed(iniPos, obs, delta):
  astsSpeed = []
  for k, v in iniPos:
    print(iniPos[obj])
    color = v['color']

    hSPD = horizontalSpeed(obs, v, delta)
    vSPD = verticalSpeed(obs, v, delta)

    astsSpeed.append((color, hSPD, vSPD))

  return astsSpeed
