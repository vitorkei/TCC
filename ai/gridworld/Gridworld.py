import numpy as np
import gym

H = -2
F = -0.01
G = 10
P = 1 # player
reward_multiplier = 100
init = (0, 0)

#### 10x10
lake = np.array([[F, F, F, F, F, F, F, F, F, F],
                 [F, F, F, F, H, F, F, F, F, F],
                 [F, F, F, F, F, F, F, F, F, F],
                 [F, F, F, F, H, F, F, F, H, H],
                 [F, F, F, F, F, F, F, F, F, F],
                 [F, F, F, F, F, F, F, F, F, F],
                 [F, F, F, F, F, F, F, F, F, F],
                 [H, F, F, F, F, F, F, F, F, F],
                 [H, H, F, F, F, F, F, F, F, F],
                 [F, F, F, H, F, F, F, F, F, G]])
state_map = {0:  (0, 0), 1:  (0, 1), 2:  (0, 2), 3:  (0, 3), 4:  (0, 4), 5:  (0, 5), 6:  (0, 6), 7:  (0, 7), 8:  (0, 8), 9:  (0, 9),
             10: (1, 0), 11: (1, 1), 12: (1, 2), 13: (1, 3), 14: (1, 4), 15: (1, 5), 16: (1, 6), 17: (1, 7), 18: (1, 8), 19: (1, 9),
             20: (2, 0), 21: (2, 1), 22: (2, 2), 23: (2, 3), 24: (2, 4), 25: (2, 5), 26: (2, 6), 27: (2, 7), 28: (2, 8), 29: (2, 9),
             30: (3, 0), 31: (3, 1), 32: (3, 2), 33: (3, 3), 34: (3, 4), 35: (3, 5), 36: (3, 6), 37: (3, 7), 38: (3, 8), 39: (3, 9),
             40: (4, 0), 41: (4, 1), 42: (4, 2), 43: (4, 3), 44: (4, 4), 45: (4, 5), 46: (4, 6), 47: (4, 7), 48: (4, 8), 49: (4, 9),
             50: (5, 0), 51: (5, 1), 52: (5, 2), 53: (5, 3), 54: (5, 4), 55: (5, 5), 56: (5, 6), 57: (5, 7), 58: (5, 8), 59: (5, 9),
             60: (6, 0), 61: (6, 1), 62: (6, 2), 63: (6, 3), 64: (6, 4), 65: (6, 5), 66: (6, 6), 67: (6, 7), 68: (6, 8), 69: (6, 9),
             70: (7, 0), 71: (7, 1), 72: (7, 2), 73: (7, 3), 74: (7, 4), 75: (7, 5), 76: (7, 6), 77: (7, 7), 78: (7, 8), 79: (7, 9),
             80: (8, 0), 81: (8, 1), 82: (8, 2), 83: (8, 3), 84: (8, 4), 85: (8, 5), 86: (8, 6), 87: (8, 7), 88: (8, 8), 89: (8, 9),
             90: (9, 0), 91: (9, 1), 92: (9, 2), 93: (9, 3), 94: (9, 4), 95: (9, 5), 96: (9, 6), 97: (9, 7), 98: (9, 8), 99: (9, 9),
            }

### 8x8
#lake = np.array([[F, F, F, F, F, F, F, H],
#                 [F, F, F, F, F, F, F, F],
#                 [F, F, F, F, H, F, F, F],
#                 [F, F, F, F, H, F, F, F],
#                 [F, F, F, F, F, F, F, F],
#                 [F, F, F, F, F, F, H, F],
#                 [H, F, F, F, F, F, F, F],
#                 [F, F, H, F, F, F, F, G]])
#state_map = {0:  (0, 0), 1:  (0, 1), 2:  (0, 2), 3:  (0, 3), 4:  (0, 4), 5:  (0, 5), 6:  (0, 6), 7:  (0, 7),
#             8:  (1, 0), 9:  (1, 1), 10: (1, 2), 11: (1, 3), 12: (1, 4), 13: (1, 5), 14: (1, 6), 15: (1, 7),
#             16: (2, 0), 17: (2, 1), 18: (2, 2), 19: (2, 3), 20: (2, 4), 21: (2, 5), 22: (2, 6), 23: (2, 7),
#             24: (3, 0), 25: (3, 1), 26: (3, 2), 27: (3, 3), 28: (3, 4), 29: (3, 5), 30: (3, 6), 31: (3, 7),
#             32: (4, 0), 33: (4, 1), 34: (4, 2), 35: (4, 3), 36: (4, 4), 37: (4, 5), 38: (4, 6), 39: (4, 7),
#             40: (5, 0), 41: (5, 1), 42: (5, 2), 43: (5, 3), 44: (5, 4), 45: (5, 5), 46: (5, 6), 47: (5, 7),
#             48: (6, 0), 49: (6, 1), 50: (6, 2), 51: (6, 3), 52: (6, 4), 53: (6, 5), 54: (6, 6), 55: (6, 7),
#             56: (7, 0), 57: (7, 1), 58: (7, 2), 59: (7, 3), 60: (7, 4), 61: (7, 5), 62: (7, 6), 63: (7, 7)
#            }

### 5x5
#lake = np.array([[F, F, F, F, F],
#                 [F, F, H, F, F],
#                 [F, F, F, H, F],
#                 [H, F, F, F, F],
#                 [F, F, F, F, G]])
#
#state_map = {0:  (0,0),  1: (0,1),  2: (0,2),  3: (0,3),  4: (0,4),
#             5:  (1,0),  6: (1,1),  7: (1,2),  8: (1,3),  9: (1,4),
#             10: (2,0), 11: (2,1), 12: (2,2), 13: (2,3), 14: (2,4),
#             15: (3,0), 16: (3,1), 17: (3,2), 18: (3,3), 19: (3,4),
#             20: (4,0), 21: (4,1), 22: (4,2), 23: (4,3), 24: (4,4)
#            }

class Gridworld():
  # self.env = ambiente do jogo pelo gym
  # self.current_pos = posição do agente no mapa. Uma das tuplas do state_map
  # self.old_pos = posição do agente no estado anterior. Uma das tuplas do state_map
  # self.current_state = Mapa do jogo sem o agente estar marcado. Serve para reiniciar o tabuleiro
  # self.action_size = número de ações possíveis
  def __init__(self): # equivale a env.reset()
    self.env = gym.make('FrozenLake8x8-v0')
    self.current_pos = state_map[self.env.reset()] # env.reset() retorna 0, sendo a posição (0, 0) do mapa de tamanho nxn
    self.old_pos = self.current_pos
    self.current_state = np.copy(lake)
    self.current_state[self.current_pos] = P
    self.action_size = 4
  
  def step(self, action):
    translated_action = np.argmax(action)
    next_state, reward, done, info = self.env.step(translated_action) # Realiza ação

    self.old_pos = self.current_pos # Recebe posição anterior
    self.current_pos = state_map[next_state] # Marca nova posição
    self.current_state[self.old_pos] = F     # Atualiza o estado do jogo
    self.current_state[self.current_pos] = P # com a nova posição do agente
    next_state = np.copy(self.current_state) # Atualiza o estado do jogo
    
    reward = lake[self.current_pos] * reward_multiplier # Altera a recompensa recebida para algo mais significativo
    return next_state, reward, done, info

  def reset(self):
    self.current_state = np.copy(lake)
    self.current_pos = state_map[self.env.reset()]
    self.current_state[self.current_pos] = P
    return self.current_state

  def pos(self):
    return self.current_pos

  def close(self):
    self.env.close()
 
  def n(self):
    return len(lake)
  
  #def grid_world(self):
  #  return self.grid

  #def state_map(self):
  #  return self.state_map

  def state(self):
    return self.current_state

  def render(self):
    self.env.render()
    print(self.current_state)

  def action_space(self):
    return self.env.action_space


