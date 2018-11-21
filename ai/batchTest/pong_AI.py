# https://towardsdatascience.com/atari-reinforcement-learning-in-depth-part-1-ddqn-ceaa762a546f

import tensorflow as tf
import numpy as np
#import retro
import gym
import matplotlib.pyplot as plt
import random
import warnings
import time
import math

from skimage import transform
from skimage.color import rgb2gray
from collections import deque # double ended queue

############ PREPARAÇÃO ############

warnings.filterwarnings('ignore')

# Altura e largura já desconsiderando as linhas e colunas em que
# não há informação útil para o aprendizado (linhas e colunas sem
# nada neste caso)
base_height = 190
base_width = 152

new_height_aux = 110
new_height = 84
new_width = 84

#env = retro.make(game='Asteroids-Atari2600')
#env = retro.make(game='Enduro-Atari2600')
#env = retro.make(game='Pong-Atari2600')
env = gym.make('PongDeterministic-v4')
# no Pong do gym, 2 = UP, 3 = DOWN

available_actions = np.array(np.identity(env.action_space.n, dtype=int).tolist())
possible_actions = np.array([[1, 0], [0, 1]])
p_a_count = [0, 0, 0, 0, 0, 0, 0, 0]

########################################################
############ PARÂMETROS E HIPERPARÂMETROS ##############
########################################################

### PRÉ PROCESSAMENTO
stack_size = 4

### MODELO
state_size = [new_height, new_width, stack_size] # Entrada é uma pilha de 4 frames
#action_size = env.action_space.n
action_size = 2 # [1, 0] e [0, 1]
learning_rate = 0.00025

### TREINAMENTO
total_episodes = 500 # número total de episódios para o treinamento
max_steps = 18000   # número máximo de ações tomadas em um episódio
batch_size = 32

max_tau = 10000

#update_gap = 10000

### Parâmetros de exploração para estratégia gulosa epsilon
epsilon_ini = 1.0  # Probabilidade de se explorar no início
epsilon_end = 0.1   # Probabilidade mínima de explorar
decay_rate = 20000 # Taxa de decaimento exponencial para a probabilidade de exploração
decay_limit = 1000000

### Q-LEARNING
gamma = 0.99 # Taxa de desconto

### MEMÓRIA
pretrain_length = 50000 # batch_size # Número de experiências armazenadas na memória quando inicializado pela primeira vez
memory_size = 1000000        # Número de experiências capazes de serem armazenadas na memória

### FLAGS
training = True        # Mudar para True se quiser treinar o agente
episode_render = False # Mudar para True se quiser ver o ambiente renderizado

### ARQUITETURA
conv_filters = [32, 64, 64] # Número de filtros em cada camada de conv2d - ELU
kernel_sizes = [8, 4, 3] # Tamanho do kernel de cada camada de conv2d - ELU
stride_sizes = [4, 2, 1] # Número de strides em cada camada de conv2d - ELU

########################################################
########################################################
########################################################

config = tf.ConfigProto(device_count={'GPU':0}) # utilizar CPU ao invés de GPU
#config = tf.ConfigProto()

def preprocess_frame(frame):
    gray = rgb2gray(frame)
    # cropped_frame = gray[19:-15, 8:] # asteroids 177x152 pixels
    cropped_frame = gray[34:-16, 0:] # pong 161x160 pixels
    preprocessed_frame = transform.resize(cropped_frame, [new_height, new_width])
    #normalized_frame = preprocessed_frame/255.0

    #return normalized_frame
    return preprocessed_frame

stacked_frames = deque([np.zeros((new_height, new_width), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)

# Stacka os frames para a haver a noção de movimento e direção do movimento
def stack_frames(stack_frame, state, is_new_episode):
  frame = preprocess_frame(state)

  if is_new_episode:
    stack_frame = deque([np.zeros((new_height, new_width), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)

    for i in range(stack_size):
      stack_frame.append(frame)

    stacked_state = np.stack(stack_frame, axis=2)

  else:
    stack_frame.append(frame)
    stacked_state = np.stack(stack_frame, axis=2)

  return stacked_state, stack_frame

def translate_action(action):
  if np.array_equal(action, np.array([1, 0])):
    #return np.array([0, 0, 0, 0, 1, 0, 0, 0]) # UP
    return 2
  elif np.array_equal(action, np.array([0, 1])):
    #return np.array([0, 0, 0, 0, 0, 1, 0, 0]) # DOWN
    return 3


class DQNNet:
  def __init__(self, state_size, action_size, learning_rate, name):
    self.state_size = state_size
    self.action_size = action_size
    self.learning_rate = learning_rate
    self.name = name
    c_f = conv_filters
    k_s = kernel_sizes
    s_s = stride_sizes

    with tf.variable_scope(self.name):
      # [None, *state_size] == [None, state_size[0], state_size[1], state_size[2]]
      self.inputs = tf.placeholder(tf.float32, [None, *state_size])
      self.actions = tf.placeholder(tf.float32, [None, self.action_size])
      
      # target_Q = R(s, a) + y * maxQhat(s', a')
      self.target_Q = tf.placeholder(tf.float32, [None])

      # Conv2d -> ELU -> maxpooling2d
      self.conv2d = tf.layers.conv2d(inputs = self.inputs,
                                     filters = c_f[0],
                                     kernel_size = k_s[0],
                                     strides = s_s[0],
                                     activation=tf.nn.relu,
                                     padding = "VALID",
                                     use_bias=False,
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                    )
      #self.elu = tf.nn.elu(self.conv2d)

      for i in range(1, len(conv_filters)):
        self.conv2d = tf.layers.conv2d(#inputs = self.elu,
                                       inputs = self.conv2d,
                                       filters = c_f[i],
                                       kernel_size = k_s[i],
                                       strides = s_s[i],
                                       activation=tf.nn.relu,
                                       padding = "VALID",
                                       use_bias=False,
                                       kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
                                      )
        #self.elu = tf.nn.elu(self.conv2d)

      #self.flatten = tf.layers.flatten(self.elu)
      self.flatten = tf.layers.flatten(self.conv2d)

      self.fc = tf.layers.dense(inputs = self.flatten,
                                units = 256,
                                activation = tf.nn.elu,
                                #kernel_initializer=tf.zeros_initializer()
                                #kernel_initializer=tf.contrib.layers.xavier_initializer()
                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                )

      self.output = tf.layers.dense(inputs = self.fc,
                                    units = self.action_size,
                                    #kernel_initializer=tf.zeros_initializer()
                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                   )
      
      # Q é a previsão do Q-valor
      self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions), axis=1)

      # loss é a diferença entre a previsão do Q-valor e o Q-alvo
      self.loss = tf.losses.huber_loss(labels=self.target_Q, predictions=self.Q)

      self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate,momentum=0.95,epsilon=0.01).minimize(self.loss)


# Reinicia o grafo e inicializa uma instância da classe DQNetwork
tf.reset_default_graph()
DQNetwork = DQNNet(state_size, action_size, learning_rate, name="DQNetwork")
TargetNetwork = DQNNet(state_size, action_size, learning_rate, name="TargetNetwork")

# Memória para que o agente se lembre de como lidou com cenários anteriores
# deque = double ended queue
class Memory():
  def __init__(self, max_size):
    self.buffer = deque(maxlen = max_size)

  def add(self, experience):
    self.buffer.append(experience)

  def sample(self, batch_size):
    buffer_size = len(self.buffer)
    index = np.random.choice(np.arange(buffer_size), size = batch_size, replace = False)

    return [self.buffer[i] for i in index]

memory = Memory(max_size = memory_size)

# Pré popula a memória com ações aleatórias
if training == True:
  print("Pre-treinamentos:")
  for i in range(pretrain_length):
    # se for a primeira ação
    if i == 0:
      state = env.reset()
      state, stacked_frames = stack_frames(stacked_frames, state, True)

    # Recebe o estado resultante da ação tomada, a recompensa, e se
    # o jogo acabou ou não, após tomar um ação aleatória
    choice = random.randrange(0, len(possible_actions)) # pong
    action = possible_actions[choice]
    next_state, reward, done, info = env.step(translate_action(action))
    
    # Se o episódio tiver acabado (se a nave tiver sido destruída 4 vezes)
    if done:
      # Episódio terminado
      next_state = np.zeros(state.shape)

      # Adiciona a experiência à memória
      memory.add((state, action, reward, next_state, done))

      # Inicia um novo episódio
      state = env.reset()

      # Stack the frames
      state, stacked_frames = stack_frames(stacked_frames, state, True)

    # Se o episódio NÃO tiver acabado ainda
    else:
      # Adiciona a experiência à memória
      next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
      memory.add((state, action, reward, next_state, done))

      # Atualiza o estado atual para o próximo
      state = next_state

  print("pre-treinamento terminado")

tf.summary.scalar("Loss", DQNetwork.loss)

def predict_action(epsilon_ini, epsilon_end, decay_rate, decay_step, state):
  # Epsilon-greedy strategy para determinar a ação tomada em cada estado
  # Em alguns casos, a ação tomada será aleatória (exploration) ao invés da
  # que retorna maior recompensa (exploitation)
  exp_exp_tradeoff = np.random.rand() # exploration exploitation tradeoffi
  #explore_probability = epsilon_end + (epsilon_ini - epsilon_end) * np.exp(-decay_rate * decay_step)

  explore_probability = epsilon_end + (epsilon_ini - epsilon_end) * np.exp(-decay_step / decay_rate)

  if (explore_probability > exp_exp_tradeoff): # Realiza uma ação aleatória
    choice = random.randrange(0, len(possible_actions))
    action = possible_actions[choice]
  else: # ou a com maior recompensa imediata
    Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs: state.reshape((1, *state.shape))})
    choice = np.argmax(Qs)
    p_a_count[choice+4] += 1
    action = possible_actions[choice]
    #print("Qs:", Qs, " | action:", action, " | p_a_count:", p_a_count, file=open("mini_output.txt", "a"))

  return action, explore_probability

def update_target_graph():
  from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNewtork")
  to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")

  op_holder = []

  for from_var, to_var in zip(from_vars, to_vars):
    op_holder.append(to_var.assign(from_var))

  return op_holder

saver = tf.train.Saver() # Ajuda a salvar o modelo
total_steps = 0

if training == True:
  print("Iniciando treinamento")
  with tf.Session(config=config) as sess:
  #with tf.Session() as sess:
    # Inicializa as variáveis
    sess.run(tf.global_variables_initializer())

    # Inicializa a taxa de decaimento que será usada na redução do epsilon
    decay_step = 0

    tau = 0

    update_target = update_target_graph()
    sess.run(update_target)

    for episode in range(total_episodes):
      print()
      print(episode, "º episódio de treinamento")
      step = 0
      episode_rewards = []
      state = env.reset()
      state, stacked_frames = stack_frames(stacked_frames, state, True)

      while step < max_steps:
        step += 1
        tau += 1
        decay_step += 1
        action, explore_probability = predict_action(epsilon_ini, epsilon_end, decay_rate, decay_step, state)
        next_state, reward, done, info = env.step(translate_action(action))

        if episode_render:
          env.render()

        episode_rewards.append(reward)

        # Se o jogo acabou (a nave foi destruída 4 vezes)
        if done:
          next_state = np.zeros(state.shape, dtype=np.int)
          next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
          total_steps += step
          print(step, " |", total_steps)
          step = max_steps
          total_reward = np.sum(episode_rewards)
          print(episode, total_reward, explore_probability, loss)
          memory.add((state, action, reward, next_state, done))
        else:
          next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
          memory.add((state, action, reward, next_state, done))
          state = next_state

        # Parte de aprendizado
        batch          = memory.sample(batch_size)
        states_mb      = np.array([each[0] for each in batch], ndmin=3)
        actions_mb     = np.array([each[1] for each in batch])
        rewards_mb     = np.array([each[2] for each in batch])
        next_states_mb = np.array([each[3] for each in batch], ndmin=3)
        dones_mb       = np.array([each[4] for each in batch])

        target_Qs_batch = []

        # recebe Q-value do próximo estado
        Qs_next_state        = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs: next_states_mb})
        Qs_target_next_state = sess.run(TargetNetwork.output, feed_dict={TargetNetwork.inputs: next_states_mb})

        for i in range(len(batch)):
          terminal = dones_mb[i]

          if terminal:
            target_Qs_batch.append(rewards_mb[i])
          else:
            target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
            target_Qs_batch.append(target)

        targets_mb = np.array([each for each in target_Qs_batch])

        loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer], feed_dict = {DQNetwork.inputs: states_mb,
                                                                               DQNetwork.target_Q: targets_mb,
                                                                               DQNetwork.actions: actions_mb})

        if tau > max_tau:
          update_target = update_target_graph()
          sess.run(update_target)
          tau = 0
          print("Model updated!")
     
      #print("steps: ", step)
      save_path = saver.save(sess, "/var/tmp/models/model.ckpt")
      #print("contador de ações tomadas:", p_a_count)

  print("treinamento terminado")

#print("Número de vezes que cada ação foi tomada:", p_a_count)
#p_a_count = [0, 0, 0, 0, 0, 0, 0, 0]
if True:
  with tf.Session(config=config) as sess:
  #with tf.Session() as sess:
    total_test_rewards = []

    saver.restore(sess, "/var/tmp/models/model.ckpt")
    #saver.restore(sess, "~/models/model.ckpt")

    for episode in range(1):
      total_rewards = 0
    
      state = env.reset()
      state, stacked_frames = stack_frames(stacked_frames, state, True)

      print("EPISODE:", episode)

      while True:
        #state = state.reshape((1, *state_size))
        exp_exp_tradeoff = np.random.rand()

        explore_probability = 0.

        if explore_probability > exp_exp_tradeoff:
          choice = random.randrange(0, len(possible_actions))
          action = possible_actions[choice]
        else:
          Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs: state.reshape((1, *state.shape))})
          choice = np.argmax(Qs)
          p_a_count[choice+4] += 1
          action = possible_actions[choice]
          #print(Qs, " -", action, " -", p_a_count, file=open("mini_out.txt", "a"))

        next_state, reward, done, info = env.step(translate_action(action))
        total_rewards += reward

        if done:
          print("SCORE: ", total_rewards)
          break
        else:
          next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
          state = next_state
      #print("Número de vezes que cada ação foi tomada:", p_a_count)

env.close()

print("HIPER PARAMETROS:\n")
print("new_height_aux:", new_height_aux, " | new_height:", new_height, " | new_width:", new_width)
print("possible_actions:\n", possible_actions)
print("learning_rate:", learning_rate)
print("total_episodes:", total_episodes)
print("max_steps:", max_steps)
print("batch_size:", batch_size)
print("max_tau:", max_tau)
print("epsilon_ini:", epsilon_ini)
print("epsilon_end:", epsilon_end)
print("decay_rate:", decay_rate)
print("gamma:", gamma)
print("pretrain_length:", pretrain_length)
print("memory_size:", memory_size)
print("conv_filters:", conv_filters)
print("kernel_sizes:", kernel_sizes)
print("stride_sizes:", stride_sizes)
