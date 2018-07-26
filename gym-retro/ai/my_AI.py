import tensorflow as tf
import numpy as np
import retro
import matplotlib.pyplot as plt
import random
import warnings

from skimage import transform
from skimage.color import rgb2gray
from collections import deque

############ PREPARAÇÃO ############

warnings.filterwarnings('ignore')

env = retro.make(game='Asteroids-Atari2600')

# Altura e largura já desconsiderando as linhas e colunas em que
# não há informação útil para o aprendizado (linhas e colunas sem
# nada neste caso)
base_height = 190
base_width = 152

new_height = 190 # 110
new_width =  152 # 84
num_channels = 4

########## HIPERPARÂMETRO - PRÉ PROCESSAMENTO ##########
stack_size = 4

########## HIPERPARÂMETROS - MODELO ##########
state_size = [new_height, new_width, stack_size] # Entrada é uma pilha de 4 frames
action_size = env.action_space.n                 # 8 ações possíveis
learning_rate = 0.00025

########## HIPERPARÂMETROS - TREINAMENTO ##########
total_episodes = 50 # número total de episódios para o treinamento
max_steps = 50000   # número máximo de ações tomadas em um episódio
batch_size = 64

########## Parâmetros de exploração para estratégia gulosa epsilon ##########
explore_begin = 1.0  # Probabilidade de se explorar no início
explore_end = 0.01   # Probabilidade mínima de explorar
decay_rate = 0.00001 # Taxa de decaimento exponencial para a probabilidade de exploração

########## HIPERPARÂMETRO - Q LEARNING ##########
gamma = 0.9 # Taxa de desconto

########## HIPERPARÂMETROS - MEMÓRIA ##########
pretrain_length = batch_size # Número de experiências armazenadas na memória quando inicializado pela primeira vez
memory_size = 1000000        # Número de experiências capazes de serem armazenadas na memória

########## FLAGS ##########
training = False       # Mudar para True se quiser treinar o agente
episode_render = False # Mudar para True se quiser ver o ambiente renderizado

########################################################
########################################################
########################################################

stacked_frames = deque([np.zeros((new_height, new_width), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)

print("frame size:", env.observation_space)
print("action size:", env.action_space.n)
possible_actions = np.array(np.identity(env.action_space.n, dtype=int).tolist())
print(possible_actions)

def preprocess_frame(frame):
  gray = rgb2gray(frame)

  cropped_frame = gray[5:-15, 8:-0]

  normalized_frame = cropped_frame/255.0

  preprocessed_frame = transform.resize(cropped_frame, [new_height, new_width])

  return preprocesses_frame

def stack_frames(stacked_frames, state, is_new_episode):
  frame = preprocess_frame(state)

  if is_new_episode:
    stacked_frames = deque([np.zeros((new_height, new_width), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)

    for i in range(stack_size):
      stacked_frames.append(frame)

    stacked_state = np.stack(stacked_frames, axis=2)

  else:
    stacked_frames.append(frame)
    stacked_state = np.stack(stacked_frames, axis=2)

  return stacked_state, stacked_frames

class DQNetwork:
  def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
    self.state_size = state_size
    self.action_size = action_size
    self.learning_rate = learning_rate

    with tf.variable_scope(name):
      self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
      self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
      
      self.target_Q = tf.placeholder(tf.float32, [None], name="target")

      # Determinar o número de convoluções (o que acarreta no número de poolings feitos)
      # Determinar o kernel_size de cada camada de convolução
      # Determinar o pool_kernel de cada camada de pooling
      # Entender o que é o strieds que o tf.layers.conv2d pede
      # Descobrir quantos filters serão necessários (número fixo para um certo tamanho de
      #   matriz ou é variável?)
