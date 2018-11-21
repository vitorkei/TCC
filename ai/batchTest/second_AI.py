# https://towardsdatascience.com/atari-reinforcement-learning-in-depth-part-1-ddqn-ceaa762a546f

import tensorflow as tf
import numpy as np
import retro
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
new_height = 95
new_width =  84
num_channels = 4

#env = retro.make(game='Asteroids-Atari2600')
#env = retro.make(game='Enduro-Atari2600')
env = retro.make(game='Pong-Atari2600')

#movie = retro.Movie('Pong-Atari2600-Start-000000.bk2')
#movie.step()
#env = retro.make(game=movie.get_game(), state=None, use_restricted_actions=retro.ACTIONS_ALL)
#env.initial_state = movie.get_state()
#env.reset()

possible_actions = np.array(np.identity(env.action_space.n, dtype=int).tolist())
available_actions = np.array([[1, 0], [0, 1]])


########################################################
############ PARÂMETROS E HIPERPARÂMETROS ##############
########################################################

### PRÉ PROCESSAMENTO
stack_size = 4

### MODELO
state_size = [new_height, new_width, stack_size] # Entrada é uma pilha de 4 frames
#action_size = env.action_space.n
action_size = len(available_actions)
learning_rate = 0.00025

### TREINAMENTO
total_episodes = 50 # número total de episódios para o treinamento
max_steps = 100000   # número máximo de ações tomadas em um episódio
batch_size = 64

max_tau = 10000

### Parâmetros de exploração para estratégia gulosa epsilon
explore_begin = 1.0  # Probabilidade de se explorar no início
explore_end = 0.1   # Probabilidade mínima de explorar
decay_rate = 0.00001 # Taxa de decaimento exponencial para a probabilidade de exploração
decay_limit = 1000000

### Q-LEARNING
gamma = 0.99 # Taxa de desconto

### MEMÓRIA
pretrain_length = 5000# batch_size # Número de experiências armazenadas na memória quando inicializado pela primeira vez
memory_size = 1000000        # Número de experiências capazes de serem armazenadas na memória

### FLAGS
training = True        # Mudar para True se quiser treinar o agente
episode_render = False # Mudar para True se quiser ver o ambiente renderizado

### ARQUITETURA
conv_filters = [32, 64, 64] # Número de filtros em cada camada de conv2d - ELU
kernel_sizes = [8, 4, 3] # Tamanho do kernel de cada camada de conv2d - ELU
stride_sizes = [4, 2, 1] # Número de strides em cada camada de conv2d - ELU
pool_kernel = [2, 2] # Tamanho do kernel de cada camada de maxpool2d

### PENALIDADE
old_life_count = 0
hit = 1
alive = 2
penalty = -1

#repeat_action = 4

########################################################
########################################################
########################################################


def preprocess_frame(frame):
  gray = rgb2gray(frame)
  preprocessed_frame = transform.resize(gray, [new_height_aux, new_width])
  cropped_frame = preprocessed_frame[12:-3, 0:]
  #normalized_frame = cropped_frame/255.0
  #normalized_frame = preprocessed_frame/255.0
  #return normalized_frame # 84x84x1 frame
  return cropped_frame
  #return preprocessed_frame

  #return normalized_frame

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

class DDDQNNet:
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
      self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
      self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
      
      # target_Q = R(s, a) + y * maxQhat(s', a')
      self.target_Q = tf.placeholder(tf.float32, [None], name="target")

      # Conv2d -> ELU -> maxpooling2d
      self.conv2d = tf.layers.conv2d(inputs = self.inputs_,
                                     filters = c_f[0],
                                     kernel_size = k_s[0],
                                     strides = s_s[0],
                                     padding = "VALID",
                                     #kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
      self.elu = tf.nn.relu(self.conv2d)

      for i in range(1, len(conv_filters)):
        self.conv2d = tf.layers.conv2d(inputs = self.elu,
                                       filters = c_f[i],
                                       kernel_size = k_s[i],
                                       strides = s_s[i],
                                       padding = "VALID",
                                       #kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d())
                                       kernel_initializer = tf.contrib.layers.variance_scaling_initializer())
        self.elu = tf.nn.relu(self.conv2d)

      self.flatten = tf.layers.flatten(self.elu)

      # not dueling
      #self.fc = tf.keras.layers.Dense(256, activation='relu')(self.flatten)
      #self.fc = tf.layers.dense(inputs = self.flatten,
      #                          units = 512,
      #                          activation = tf.nn.relu,
      #                          kernel_initializer=tf.contrib.layers.xavier_initializer()
      #                          )
      # fully-connected layers
      self.value_fc = tf.layers.dense(inputs = self.flatten,
                                units = 512,
                                activation = tf.nn.elu,
                                #activation = tf.nn.elu,
                                kernel_initializer = tf.contrib.layers.xavier_initializer()
                                #kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
                                )
      self.value = tf.layers.dense(inputs = self.value_fc,
                                   units = 1,
                                   activation = None,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer()
                                   #kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                   )

      self.advantage_fc = tf.layers.dense(inputs = self.flatten,
                                          units = 512,
                                          activation = tf.nn.elu,
                                          #activation = tf.nn.elu,
                                          kernel_initializer = tf.contrib.layers.xavier_initializer()
                                          #kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                          )
      self.advantage = tf.layers.dense(inputs = self.advantage_fc,
                                       units = self.action_size,
                                       activation = None,
                                       kernel_initializer = tf.contrib.layers.xavier_initializer()
                                       #kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                       )



      #self.output = tf.keras.layers.Dense(self.action_size)(self.fc)
      #self.output = tf.layers.dense(inputs = self.fc,
      #                              #kernel_initializer = tf.contrib.layers.xavier_initializer(),
      #                              kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),
      #                              units = self.action_size,
      #                              activation=tf.nn.sigmoid)
      
      # dueling
      self.output = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))
      # Q é a previsão do Q-valor
      self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

      # loss é a diferença entre a previsão do Q-valor e o Q-alvo
      #self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
      self.loss = tf.losses.huber_loss(self.Q, self.target_Q)

      self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.95, epsilon=0.01).minimize(self.loss)
      #self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


# Reinicia o grafo e inicializa uma instância da classe DQNetwork
tf.reset_default_graph()
DQNetwork = DDDQNNet(state_size, action_size, learning_rate, name="DQNetwork")
TargetNetwork = DDDQNNet(state_size, action_size, learning_rate, name="TargetNetwork")

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


print("Pre-treinamentos:")
# Pré popula a memória com ações aleatórias
if training == True:
  for i in range(pretrain_length):
    # se for a primeira ação
    if i == 0:
      state = env.reset()
      state, stacked_frames = stack_frames(stacked_frames, state, True)

    # Recebe o estado resultante da ação tomada, a recompensa, e se
    # o jogo acabou ou não, após tomar um ação aleatória
    #choice = random.randrange(0, len(possible_actions))
    #choice = random.choice([0, 4, 5, 6, 7]) # asteroids
    choice = random.randrange(0, len(available_actions)) # pong
    if choice == 0:
      choice = 4
    elif choice == 1:
      choice = 5
    action = possible_actions[choice]
    next_state, reward, done, info = env.step(action)
    #if reward > 0:
     # reward = hit
   # if reward == 0:
    #  reward = alive
    #if info['lives'] < old_life_count: # perdeu vida
    #  reward = penalty

    # Stack the frames
    #next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
#    if count == 100:
#      for i in range(new_height):
#        for j in range(new_width):
#          if False:
#            print(next_state[i][j])
#          else:
#            if next_state[i][j][2] - 0.35179035 < 0.00000001:
#              print(".", end=" ")
#            else:
#              print("0", end=" ")
#        print()
#
#      time.sleep(100)
    
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

# Preparando Tensorboard
#writer = tf.summary.FileWriter("/var/tmp/tensorboard/dqn/1")
tf.summary.scalar("Loss", DQNetwork.loss)
#write_op = tf.summary.merge_all()

def predict_action(explore_begin, explore_end, decay_rate, decay_step, state, actions):
  # Epsilon-greedy strategy para determinar a ação tomada em cada estado
  # Em alguns casos, a ação tomada será aleatória (exploration) ao invés da
  # que retorna maior recompensa (exploitation)
  exp_exp_tradeoff = np.random.rand() # exploration exploitation tradeoff
  explore_probability = explore_end + (explore_begin - explore_end) * np.exp(-decay_rate * decay_step)
  #explore_probability = explore_begin - decay_rate * decay_step
  #if (decay_step > decay_limit or explore_probability < explore_end):
  #  explore_probability = explore_end

  if (explore_probability > exp_exp_tradeoff): # Realiza uma ação aleatória
    #choice = random.randint(0, len(possible_actions)-1)
    #choice = random.choice([0, 4, 5, 6, 7]) # asteroids
    #choice = random.choice([4, 5]) # pong
    choice = random.randrange(0, len(available_actions))
    if choice == 0:
      choice = 4
    elif choice == 1:
      choice = 5
    action = possible_actions[choice]
  else: # ou a com maior recompensa imediata
    Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
    choice = np.argmax(Qs)
    if choice == 0:
      choice = 4
    elif choice == 1:
      choice = 5
    action = possible_actions[choice]

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

print("Iniciando treinamento")
if training == True:
  with tf.Session() as sess:
    # Inicializa as variáveis
    sess.run(tf.global_variables_initializer())

    # Inicializa a taxa de decaimento que será usada na redução do epsilon
    decay_step = 0

    tau = 0

    update_target = update_target_graph()
    sess.run(update_target)

    for episode in range(total_episodes):
      print("\n", episode, "episódio de treinamento")
      step = -1
      episode_rewards = []
      state = env.reset()
      state, stacked_frames = stack_frames(stacked_frames, state, True)
      latest_action = 0

      while step < max_steps:
        step += 1
        tau += 1
        decay_step += 1
        action, explore_probability = predict_action(explore_begin, explore_end, decay_rate, decay_step, state, possible_actions)
        #if step % 4 == 0:
        #  action, explore_probability = predict_action(explore_begin, explore_end, decay_rate, decay_step, state, possible_actions)
        #  latest_action = action
        #else:
        #  action = latest_action
        next_state, reward, done, info = env.step(action)

        if episode_render:
          env.render()

        episode_rewards.append(reward)

        # Se o jogo acabou (a nave foi destruída 4 vezes)
        if done:
          next_state = np.zeros(state.shape)
          next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
          total_steps += step
          print(step, " |", total_steps)
          step = max_steps
          total_reward = np.sum(episode_rewards)
          print(episode, total_reward, explore_probability, loss)
          memory.add((state, action, reward, next_state, done))
          
        #if done:
          #next_state = np.zeros((new_height, new_width), dtype=np.int)
        #  next_state = np.zeros(state.shape)
        #  next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
        #  memory.add((state, action, reward, next_state, done))
        #  state = env.reset()
        #  print("end step:", step)
          #step = max_steps
        #  total_reward = np.sum(episode_rewards)
        #  print(episode, total_reward, explore_probability, loss)
        #  rewards_list.append((episode, total_reward))

        else:
          next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
          memory.add((state, action, reward, next_state, done))
          state = next_state

        #if step % repeat_action != 0:
        #  continue
        # Parte de aprendizado
        batch = memory.sample(batch_size)
        states_mb = np.array([each[0] for each in batch], ndmin=3)
        actions_mb_aux = np.array([each[1] for each in batch])
        actions_mb = []
        for ac in actions_mb_aux:
          if np.array_equal(ac, np.array([0, 0, 0, 0, 1, 0, 0, 0])):
            actions_mb.append([1, 0])
          elif np.array_equal(ac, np.array([0, 0, 0, 0, 0, 1, 0, 0])):
            actions_mb.append([0, 1])
          else:
            print("ação incorreta:", ac)
            choice = random.randrange(0, len(available_actions))
            action = available_actions[choice]
            actions_mb.append(action)
        actions_mb = np.array(actions_mb)
        rewards_mb = np.array([each[2] for each in batch])
        next_states_mb = np.array([each[3] for each in batch], ndmin=3)
        dones_mb = np.array([each[4] for each in batch])

        target_Qs_batch = []

        # recebe Q-value do próximo estado
        Qs_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: next_states_mb})
        Qs_target_next_state = sess.run(TargetNetwork.output, feed_dict={TargetNetwork.inputs_: next_states_mb})

        for i in range(len(batch)):
          terminal = dones_mb[i]
          action = np.argmax(Qs_next_state[i])

          if terminal:
            target_Qs_batch.append(rewards_mb[i])
          else:
            target = rewards_mb[i] + gamma * Qs_target_next_state[i][action]
            target_Qs_batch.append(target)

        targets_mb = np.array([each for each in target_Qs_batch])

        loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer], feed_dict = {DQNetwork.inputs_: states_mb,
                                                                               DQNetwork.target_Q: targets_mb,
                                                                               DQNetwork.actions_: actions_mb})
        #print("aux =", aux)
        #summary = sess.run(write_op, feed_dict = {DQNetwork.inputs_: states_mb,
        #                                          DQNetwork.target_Q: targets_mb,
        #                                          DQNetwork.actions_: actions_mb})
        #writer.add_summary(summary, episode)
        #writer.flush()

        if tau > max_tau:
          update_target = update_target_graph()
          sess.run(update_target)
          tau = 0
          #print("Model updated")
     
      #print("steps: ", step)
      save_path = saver.save(sess, "/var/tmp/models/model.ckpt")

print("treinamento terminado")

if True:
  with tf.Session() as sess:
    total_test_rewards = []

    saver.restore(sess, "/var/tmp/models/model.ckpt")
    #saver.restore(sess, "~/models/model.ckpt")

    for episode in range(10):
      total_rewards = 0
    
      state = env.reset()
      state, stacked_frames = stack_frames(stacked_frames, state, True)

      print("EPISODE:", episode)

      while True:
        exp_exp_tradeoff = np.random.rand()

        explore_probability = 0.001

        if explore_probability > exp_exp_tradeoff:
          #choice = random.randint(0, len(possible_actions)-1)
          #choice = random.choice([0, 4, 5, 6, 7]) # asteroids
          #choice = random.choice([4, 5]) # pong
          choice = random.randrange(0, len(available_actions))
          if choice == 0:
            choice = 4
          elif choice == 1:
            choice = 5
          action = possible_actions[choice]
        else:
          Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
          choice = np.argmax(Qs)
          if choice == 0:
            choice = 4
          elif choice == 1:
            choice = 5
          action = possible_actions[choice]

        next_state, reward, done, info = env.step(action)
        total_rewards += reward

        if done:
          print("SCORE: ", total_rewards)
          break
        else:
          next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
          state = next_state
#    while True:
#      state = state.reshape((1, *state_size))
#      Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state})
#
#      choice = np.argmax(Qs)
#      action = possible_actions[choice]
#      #print(action)
#
#      next_state, reward, done, info = env.step(action)
#      if episode_render:
#        env.render()
#
#      total_rewards += reward
#
#      if done:
#        print("Score:", total_rewards)
#        total_test_rewards.append(total_rewards)
#        break
#
#      next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
#      state = next_state


if False:
  while movie.step():
    keys = []
    for i in range(env.NUM_BUTTONS):
      keys.append(movie.get_key(i))
    _obs, _rew, _done, _info = env.step(keys)

env.close()

print(new_height, "x", new_width)
print(total_episodes)
print(max_steps)
print(batch_size)
print(explore_begin)
print(explore_end)
print(decay_rate)
print(gamma)
print(memory_size)
print(conv_filters)
print(kernel_sizes)
print(stride_sizes)
print(penalty)
print(alive)
