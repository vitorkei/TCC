import numpy as np

# Arquivo que contém funções relacionadas a velocidade
# dos asteroides

# Retorna velocidade horizontal do asteroide
def horizontalSpeed(obs, obj):
  color = obj[0]
  for j in range(obj[3] - 3, obj[4] + 4):
    for i in range(obj[1] - 5, obj[2] + 6):
      if np.array_equal(obs[i][j], color):
        return j - obj[3]

# Retorna velocidade vertical do asteroide
def verticalSpeed(obs, obj):
  color = obj[0]
  for i in range(obj[1] - 5, obj[2] + 6):
    for j in range(obj[3] - 3, obj[4] + 4):
      if np.array_equal(obs[i][j], color):
        return i - obj[1]

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
  for obj in iniPos:
    color = obj[0]

    hSPD = horizontalSpeed(obs, obj)
    vSPD = verticalSpeed(obs, obj)

    astsSpeed.append((color, hSPD, vSPD))

  return astsSpeed
