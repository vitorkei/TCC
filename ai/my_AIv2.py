import tensorflow as tf
import numpy as np
import random
import retro
import matplotlib.pyplot as plt
import time
import warnings

from skimage import transform
from skimage.color import rgb2gray
from collections import deque # double ended queue
warnings.filterwarnings('ignore')

#######################################################
#######################################################
#######################################################

base_height = 190 # Altura da tela sem contar partes desnecessárias da moldura
base_width = 152 # Largura da tela sem contar partes desnecessárias da moldura

new_height = 110 # Altura da tela redimensionada
new_width = 84   # Largura da tela redimensionada
num_channels = 4 # número de canais

# Inicializa o jogo e recebe as ações possíveis
env = retro.make(game='Asteroids-Atari2600')
possible_actions = np.array(np.identity(env.action_space.n, dtype=int).tolist())

#######################################################
############ PARÂMETROS E HIPERPARÂMETROS #############
#######################################################

### PRÉ PROCESSAMENTO
stack_size = 4

### MODELO
state_size = [new_height, new_width, stack_size]
action_size = env.action_space.n
learning_rate = 0.00025

### TREINAMENTO
total_episodes = 21
max_steps = 10000
batch_size = 64

### Parâmetros de exploração para estratégia gulosa epsilon
explore_begin = 1.0
explore_end = 0.01
decay_rate = 0.00001

### Q-LEARNING
gamma = 0.9

### MEMÓRIA
pretrain_length = batch_size
memory_size = 100000

### FLAGS
training = True
episode_render = False

### ARQUITETURA
conv_filters = [16, 32, 32]
kernel_sizes = [6, 3, 2]
stride_sizes = [3, 2, 2]

#######################################################
#######################################################
#######################################################

### Pré-processamento
### Serve para reduzir a complexidade (e, portanto, tempo de
### computação) ao ao reduzir o tamanho da tela e utilizar
### apenas tons de cinza
def preprocess_frame(frame):
  #gray = rgb2gray(frame) # tentar usar sem acinzentar antes
  #cropped_frame = gray[5:-15, 8:]
  cropped_frame = frame[5:-15, 8:]
  normalized_frame = cropped_frame/255.0
  preprocessed_frame = transform.resize(normalized_frame, [new_height, new_width])

  return preprocessed_frame

### Inicializa uma double-ended queue
stacked_frames = deque([np.zeros((new_height, new_width), dtype=int) for i in range(stack_size)], maxlen=stack_size)

### Empilha os quadros do jogo para que a inteligência consiga
### ter noção de movimento. Uma imagem não diz nada sobre movimento,
### mas 4 em sequência conseguem
def stack_frames(stacked_frames, state, is_new_episode):
  frame = preprocess_frame(state)

  # Caso seja um episódio novo, limpa a pilha e cria uma nova inicial
  if is_new_episode:
    stacked_frames = deque([np.zeros((new_height, new_width), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)

    for i in range(stack_size);
      stacked_frames.append(frame)

    stacked_state = np.stack(stacked_frames, axis=2)

  # Caso contrário, só empilha o próximo quadro
  else: # not new episode
    stacked_frames.append(frame)
    stacked_state = np.stack(stacked_frames, axis=2)

  return stacked_state, stacked_frames

# Dueling Double Deep Q Learning (Neural Network)
class DDDQNNet:
  def __init__(self, state_size, action_size, learning_rate, name):
    self.state_size = state_size
    self.action_size = action_size
    self.learning_rate = learning_rate
    self.name = name
    c_f = conv_filters
    k_s = kernel_sizes
    s_s = stride_sizes

    # O nome será útil para saber se é a DQNetwork ou TargetNetwork
    with tf.variable_scope(self.name):
      # Se, por exemplo, state_size = [110, 84, 4], então
      # [None, *state_size] = [None, 110, 84, 4]
      self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs_")
      self.ISWeights_ = tf.placeholder(tf.float32, [None, 1], name="IS_weights")
      self.actions_ = tf.placeholder(tf.float32, [None, action_size], name="actions_")
      self.target_Q = tf.placeholder(tf.float32, [None], name="target_Q")

      # Primeira camada de convolução e de ELU
      self.conv2d = tf.layers.conv2d(inputs = self.inputs_,
                                     filters = c_f[0],
                                     kernel_size = k_s[0],
                                     strides = s_s[0],
                                     padding = "VALID",
                                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                     name="conv2d0")
      self.elu = tf.nn.elu(self.conv2d, name="elu0")
      # Todas as outras camadas de convolução e de ELU
      for i in range(1, len(conv_filters)):
        self.conv2d = tf.layers.conv2d(inputs = self.elu,
                                       filters = c_f[i],
                                       kernel_size = k_s[i],
                                       strides = s_s[i],
                                       padding = "VALID",
                                       kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       name=("conv2d"+str(i)))
        self.elu = tf.nn.elu(self.conv2d, name=("elu"+str(i)))
     
      self.flatten = tf.layers.flatten(self.elu)
      #self.flatten = tf.contrib.layers.flatten(self.elu) # usado na v1 da IA

      # value_fc e value calculam V(s)
      self.value_fc = tf.layers.dense(inputs = self.flatten,
                                      units = 512,
                                      activation = tf.nn.elu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="value_fc")
      self.value = tf.layers.dense(inputs = self.value_fc,
                                   units = 1,
                                   activation = None,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                   name="value")

      # advantage_fc e advantage calculam A(s, a)
      self.advantage_fc = tf.layers.dense(inptus = self.flatten,
                                          units = 512,
                                          activation = tf.nn.elu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name="advantage_fc")

      self.advantage = tf.layers.dense(inputs = self.advantage_fc,
                                       units = self.action_size,
                                       activation = None,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       name="advantage")

      # Agrega as camadas
      # Q(s, a) = V(s) + (A(s, a) - 1/|A| * sum A(s, a'))
      self.output = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))
      
      # Q é o Q-valor previsto
      self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

      # A perda é modificada por causa do Prioritized Experience Replay
      self.absolute_errors = tf.abs(self.target_Q - self.Q)
      self.loss = tf.reduce_mean(self.ISWeights_ * tf.squared_difference(self.target_Q, self.Q))
      self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

### Reinicia o gráfico e instancia a DQNetwork e target network
tf.reset_default_graph()
DQNetwork = DDDQNNet(state_size, action_size, learning_rate, name="DQNetwork")
TargetNetwork = DDDQNNet(state_size, action_size, learning_rate, name="TargetNetwork")

### Classe adaptada da versão de Morvan Zhou:
### https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
class SumTree(object):
  data_pointer = 0
  
  # Inicializa a árvore com 0 nós e todos os dado com valor 0
  def __init__(self, capacity):
    self.capacity = capacity
    self.tree = np.zeros(2 * capacity - 1)
    self.data = np.zeros(capacity, dtype=object) # Contém as experiências
  
  # Adiciona as prioridades e as experiências
  def add(self, priority, data):
    tree_index = self.data_pointer + self.capacity - 1
    self.data[self.data_pointer] = data # atualiza o frame data
    self.update(tree_index, priority)   # atualiza a folha
    self.data_pointer += 1              # adiciona 1 ao data_pointer
   
    # Caso tenha passado da capacidade, começa a sobreescrever a partir
    # da primeira experiência
    if self.data_pointer >= self.capacity:
      self.data_pointer = 0

  # Atualiza a prioridade da folha e propaga de forma a mudar a árvore
  def update(self, tree_index, priority):
    change = priority - self.tree[tree_index]
    self.tree[tree_index] = priority
    
    while tree_index != 0:
      tree_index = (tree_index - 1) // 2 # Divisão truncada
      self.tree[tree_index] += change
  
  # Recebe o index da folha, sua prioridade e a experiência associada
  def get_leaf(self, v):
    parent_index = 0
    
    while True:
      left_child_index = 2 * parent_index + 1
      right_child_index = left_child_index + 1
      
      if left_child_index >= len(self.tree):
        leaf_index = parent_index
        break
      else:
        if v <= self.tree[left_child_index]:
          parent_index = left_child_index
        else:
          v -= self.tree[left_child_index]
          parent_index = right_child_index

    data_index = leaf_index - self.capacity + 1

    return leaf_index, self.tree[leaf_index], self.data[data_index]

  def total_priority(self):
    return self.tree[0]

# Prioritized Experience Replay (PER)
class Memory(object):
  # Hiperparâmetros
  PER_e = 0.01 # Evita que experiências tenham prob=0
  PER_a = 0.6  # Para fazer o tradeoff entre usar experiência
               # de maior prioridade e ação aleatória
  PER_b = 0.4  # 
  PER_b_increment_per_sampling = 0.001
  absolute_error_upper = 1.

  def __init__(self, capacity):
    self.tree = SumTree(capacity)

  # Armazena uma nova experiência na árvore
  def store(self, experience):
    max_priority = np.max(self.tree.tree[-self.tree.capacity:])
    if max_priority == 0:
      max_priority = self.absolute_error_upper

    self.tree.add(max_priority, experience)
  
  def sample(self, n):
    memory_b = []
    b_idx       = np.empty((n,), dtype=np.int32)
    b_ISWeights = np.empty((n, 1), dtype=np.float32)

    priority_segment = self.tree.total_priority / n

    self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling]) # max = 1

    p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
    max_weight = (p_min * n) ** (-self.PER_b)
    
    for i in range(n):
      a = priority_segment * i
      b = priority_segment * (i+1)
      value = np.random.uniform(a, b)

      index, priority, data = self.tree.get_leaf(value)
      sampling_probabilities = priority / self.tree.total_priority
      b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b) / max_weight
      b_idx[i] = index
      experience = [data]
      memory_b.append(experience)
    
    return b_idx, memory_b, b_ISWeights

  def batch_update(self, tree_idx, abs_errors):
    abs_errors += self.PER_e
    clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
    ps = np.power(clipped_errors, self.PER_a)

    for ti, p in zip(tree_idx, ps):
      self.tree.update(ti, p)

memory = Memory(memory_size)

for i in range(pretrain_length):
  if i == 0:
    state = env.reset()
    state, stacked_frames = stack_frames(stacked_frames, state, True)

  choice = random.randint(0, len(possible_actions)-1) # escolhe N t.q. a<=N<=b
  action = possible_actions[choice]
  next_state, reward, done, info = env.step(action)

  #env.render()

  #next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

  if done:
    next_state = np.zeros(state.shape)

    memory.store((state, action, reward, next_state, done))

    state = env.reset()
    state, stacked_frames = stack_frames(stacked_frames, state, True)
  else:
    next_state = memory.store((state, action, reward, next_state, done))
    state = next_state

#writer = tf.summary.FileWriter("./tensorboard/dddqn/1")
#tf.summary.scalar("Loss", DQNetwork.loss)
#write_op = tf.summary.merge_all()

def predict_action(explore_begin, explore_end, decay_rate, decay_step, state, actions):
  exp_exp_tradeoff = np.random.rand() # exploration exploitation tradeoff
  
  explore_probability = explore_end + (explore_begin - explore_end) * np.exp(-decay_rate * decay_step)

  if explore_probability > exp_exp_tradeoff:
    choice = random.randint(0, len(possible_actions)-1)
    action = possible_actions[choice]
  else:
    Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state_shape))})
    choice = np.argmax(Qs)
    action = possible_actions[choice]

  return action, explore_probability

def update_target_graph():
  from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")
  to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")
  op_holder = []

  for from_var, to_var in zip(from_vars, to_vars):
    op_holder.append(to_var.assign(from_var))

  return op_holder

saver = tf.train.Saver()
if training == True:
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    decay_step = 0
    tau = 0
    
