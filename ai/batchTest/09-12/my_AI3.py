import tensorflow as tf
import numpy as np
import retro
import matplotlib.pyplot as plt
import random
import warnings

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
new_width =  84
num_channels = 4

env = retro.make(game='Asteroids-Atari2600')
#print("frame size:", env.observation_space)
#print("action size:", env.action_space.n)
possible_actions = np.array(np.identity(env.action_space.n, dtype=int).tolist())
#print(possible_actions)


########################################################
############ PARÂMETROS E HIPERPARÂMETROS ##############
########################################################

### PRÉ PROCESSAMENTO
stack_size = 4

### MODELO
state_size = [new_height, new_width, stack_size] # Entrada é uma pilha de 4 frames
action_size = env.action_space.n                 # 8 ações possíveis
learning_rate = 0.00025

### TREINAMENTO
total_episodes = 25 # número total de episódios para o treinamento
max_steps = 1000000   # número máximo de ações tomadas em um episódio
batch_size = 32

### Parâmetros de exploração para estratégia gulosa epsilon
explore_begin = 1.0  # Probabilidade de se explorar no início
explore_end = 0.1   # Probabilidade mínima de explorar
decay_rate = 0.000001 # Taxa de decaimento exponencial para a probabilidade de exploração

### Q-LEARNING
gamma = 0.4 # Taxa de desconto

### MEMÓRIA
pretrain_length = batch_size # Número de experiências armazenadas na memória quando inicializado pela primeira vez
#pretrain_length = 100
memory_size = 1000000        # Número de experiências capazes de serem armazenadas na memória

### FLAGS
training = True        # Mudar para True se quiser treinar o agente
episode_render = False # Mudar para True se quiser ver o ambiente renderizado

### ARQUITETURA
conv_filters = [16, 32] # Número de filtros em cada camada de conv2d - ELU
kernel_sizes = [8, 4] # Tamanho do kernel de cada camada de conv2d - ELU
stride_sizes = [4, 2] # Número de strides em cada camada de conv2d - ELU
pool_kernel = [3, 2] # Tamanho do kernel de cada camada de maxpool2d

########################################################
########################################################
########################################################


def preprocess_frame(frame):
  gray = rgb2gray(frame)
  cropped_frame = gray[18:-15, 8:]
  normalized_frame = cropped_frame/255.0
  #normalized_frame = gray/255.0
  preprocessed_frame = transform.resize(normalized_frame, [new_height, new_width])
  #preprocessed_frame = preprocessed_frame[7:-19, :]
  return preprocessed_frame # 84x84x1 frame

  #return normalized_frame

stacked_frames = deque([np.zeros((new_height, new_width), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)

# Stacka os frames para a haver a noção de movimento e direção do movimento
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
    c_f = conv_filters
    k_s = kernel_sizes
    s_s = stride_sizes
    p_k = pool_kernel

    with tf.variable_scope(name):
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
                                     padding = "SAME",
                                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
      self.elu = tf.nn.elu(self.conv2d)
      #self.maxpool2d = tf.layers.max_pooling2d(inputs = self.elu,
                                               #pool_size = p_k[0],
                                               #strides = s_s[0])

      for i in range(1, len(conv_filters)):
        self.conv2d = tf.layers.conv2d(#inputs = self.maxpool2d,
                                       inputs = self.elu,
                                       filters = c_f[i],
                                       kernel_size = k_s[i],
                                       strides = s_s[i],
                                       padding = "SAME",
                                       kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d())
        self.elu = tf.nn.elu(self.conv2d)
        #self.maxpool2d = tf.layers.max_pooling2d(inputs = self.elu,
                                                 #pool_size = p_k[i],
                                                 #strides = s_s[i])

      #self.flatten = tf.contrib.layers.flatten(self.maxpool2d)
      self.flatten = tf.contrib.layers.flatten(self.elu)

      # fully connected layer
      self.fc = tf.layers.dense(inputs = self.flatten,
                                units = 256,
                                activation = tf.nn.elu,
                                kernel_initializer = tf.contrib.layers.xavier_initializer())

      self.output = tf.layers.dense(inputs = self.fc,
                                    kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                    units = self.action_size,
                                    activation=None)
      # Q é a previsão do Q-valor
      self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_))

      # loss é a diferença entre a previsão do Q-valor e o Q-alvo
      self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

      self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

# Reinicia o grafo e inicializa uma instância da classe DQNetwork
tf.reset_default_graph()
DQNetwork = DQNetwork(state_size, action_size, learning_rate)

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
for i in range(pretrain_length): 

  # se for a primeira ação
  if i == 0:
    state = env.reset()
    state, stacked_frames = stack_frames(stacked_frames, state, True)

  # Recebe o estado resultante da ação tomada, a recompensa, e se
  # o jogo acabou ou não, após tomar um ação aleatória
  choice = random.randint(0, len(possible_actions)-1)
  action = possible_actions[choice]
  next_state, reward, done, info = env.step(action)
    
  # env.render()

  # Stack the frames
  next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
    
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
    memory.add((state, action, reward, next_state, done))

    # Atualiza o estado atual para o próximo
    state = next_state

# Preparando Tensorboard
writer = tf.summary.FileWriter("./tensorboard/dqn/1")
tf.summary.scalar("Loss", DQNetwork.loss)
write_op = tf.summary.merge_all()

def predict_action(explore_begin, explore_end, decay_rate, decay_step, state, actions):
  # Epsilon-greedy strategy para determinar a ação tomada em cada estado
  # Em alguns casos, a ação tomada será aleatória (exploration) ao invés da
  # que retorna maior recompensa (exploitation)
  exp_exp_tradeoff = np.random.rand() # exploration exploitation tradeoff
  explore_probability = explore_end + (explore_begin - explore_end) * np.exp(-decay_rate * decay_step)

  if (explore_probability > exp_exp_tradeoff): # Realiza uma ação aleatória
    choice = random.randint(0, len(possible_actions)-1)
    action = possible_actions[choice]
  else: # ou a com maior recompensa imediata
    Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
    choice = np.argmax(Qs)
    action = possible_actions[choice]

  return action, explore_probability

saver = tf.train.Saver() # Ajuda a salvar o modelo

rewards_list = []
#highest_score = 0
lowest_loss = 1000.0

if training == True:
  with tf.Session() as sess:
    # Inicializa as variáveis
    sess.run(tf.global_variables_initializer())

    # Inicializa a taxa de decaimento que será usada na redução do epsilon
    decay_step = 0

    for episode in range(total_episodes):
      step = 0
      episode_rewards = []
      state = env.reset()
      state, stacked_frames = stack_frames(stacked_frames, state, True)

      while step < max_steps:
        step += 1
        decay_step += 1
        action, explore_probability = predict_action(explore_begin, explore_end, decay_rate, decay_step, state, possible_actions)
        next_state, reward, done, info = env.step(action)

        if episode_render:
          env.render()

        episode_rewards.append(reward)

        # Se o jogo acabou (a nave foi destruída 4 vezes)
        if done:
          next_state = np.zeros((new_height, new_width), dtype=np.int)
          next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
          step = max_steps
          total_reward = np.sum(episode_rewards)

          print(episode, total_reward, explore_probability, loss)

          rewards_list.append((episode, total_reward))

          memory.add((state, action, reward, next_state, done))
        else:
          next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
          memory.add((state, action, reward, next_state, done))
          state = next_state

        # Parte de aprendizado
        batch = memory.sample(batch_size)
        states_mb = np.array([each[0] for each in batch], ndmin=3)
        actions_mb = np.array([each[1] for each in batch])
        rewards_mb = np.array([each[2] for each in batch])
        next_states_mb = np.array([each[3] for each in batch], ndmin=3)
        dones_mb = np.array([each[4] for each in batch])

        target_Qs_batch = []

        # recebe Q-value do próximo estado
        Qs_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: next_states_mb})

        for i in range(len(batch)):
          terminal = dones_mb[i]

          if terminal:
            target_Qs_batch.append(rewards_mb[i])
          else:
            target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
            target_Qs_batch.append(target)

        targets_mb = np.array([each for each in target_Qs_batch])

        loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer], feed_dict = {DQNetwork.inputs_: states_mb,
                                                                               DQNetwork.target_Q: targets_mb,
                                                                               DQNetwork.actions_: actions_mb})
        summary = sess.run(write_op, feed_dict = {DQNetwork.inputs_: states_mb,
                                                  DQNetwork.target_Q: targets_mb,
                                                  DQNetwork.actions_: actions_mb})
        writer.add_summary(summary, episode)
        writer.flush()
      
      save_path = saver.save(sess, "/var/tmp/models/model.ckpt")
      print(episode, "saved")
      #if loss < lowest_loss:
      #  save_path = saver.save(sess, "/var/tmp/models/model.ckpt")
      #  #highest_score = total_reward
      #  lowest_loss = loss
      #  print(episode, "saved")

#print("rewards list:")
#for reward in rewards_list:
#  print("episode:", reward[0], " - reward:", reward[1])

with tf.Session() as sess:
  total_test_rewards = []

  saver.restore(sess, "/var/tmp/models/model.ckpt")
  #saver.restore(sess, "~/models/model.ckpt")

  for episode in range(1):
    total_rewards = 0
      
    state = env.reset()
    state, stacked_frames = stack_frames(stacked_frames, state, True)

    print("EPISODE:", episode)

    while True:
      state = state.reshape((1, *state_size))
      Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state})

      choice = np.argmax(Qs)
      action = possible_actions[choice]
      print(action)

      next_state, reward, done, info = env.step(action)
      if episode_render:
        env.render()

      total_rewards += reward

      if done:
        print("Score:", total_rewards)
        total_test_rewards.append(total_rewards)
        break

      next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
      state = next_state

  env.close()

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
