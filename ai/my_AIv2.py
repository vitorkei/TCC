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

base_height = 190
base_width = 152

new_height = 110
new_width = 84
num_channels = 4

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

def preprocess_frame(frame):
  #gray = rgb2gray(frame) # tentar usar sem acinzentar antes
  #cropped_frame = gray[5:-15, 8:]
  cropped_frame = frame[5:-15, 8:]
  normalized_frame = cropped_frame/255.0
  preprocessed_frame = transform.resize(normalized_frame, [new_height, new_width])

  return preprocessed_frame

stacked_frames = deque([np.zeros((new_height, new_width), dtype=int) for i in range(stack_size)], maxlen=stack_size)

def stack_frames(stacked_frames, state, is_new_episode):
  frame = preprocess_frame(state)

  if is_new_episode:
    stacked_frames = deque([np.zeros((new_height, new_width), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)

    for i in range(stack_size);
      stacked_frames.append(frame)

    stacked_state = np.stack(stacked_frames, axis=2)

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

    with tf.variable_scope(self.name):
      self.inputs_ = tf.placeholder(tf.float32, [None, *state_size])
      self.ISWeights_ = tf.placeholder(tf.float32, [None, 1])
      self.actions_ = tf.placeholder(tf.float32, [None, action_size])
      self.target_Q = tf.placeholder(tf.float32, [None])

      self.conv2d = tf.layers.conv2d(inputs = self.inputs_,
                                     filters = c_f[0],
                                     kernel_size = k_s[0],
                                     strides = s_s[0],
                                     padding = "VALID",
                                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
      self.elu = tf.nn.elu(self.conv2d)

      for i in range(1, len(conv_filters)):
        self.conv2d = tf.layers.conv2d(inputs = self.elu,
                                       filters = c_f[i],
                                       kernel_size = k_s[i],
                                       strides = s_s[i],
                                       padding = "VALID",
                                       kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        self.elu = tf.nn.elu(self.conv2d)
      
      self.flatten = tf.layers.flatten(self.elu)
      #self.flatten = tf.contrib.layers.flatten(self.elu) # usado na v1 da IA
      self.value_fc = tf.layers.dense(inputs = self.flatten,
                                      units = 512,
                                      activation = tf.nn.elu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())

      self.value = tf.layers.dense(inputs = self.value_fc,
                                   units = 1,
                                   activation = None,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())

      self.advantage_fc = tf.layers.dense(inptus = self.flatten,
                                          units = 512,
                                          activation = tf.nn.elu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())

      self.advantage = tf.layers.dense(inputs = self.advantage_fc,
                                       units = self.action_size,
                                       activation = None,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer())

      self.output = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))
      self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
      self.absolute_errors = tf.abs(self.target_Q - self.Q)
      self.loss = tf.reduce_mean(self.ISWeights_ * tf.squared_difference(self.target_Q, self.Q))
      self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

tf.reset_default_graph()
DQNetwork = DDDQNNet(state_size, action_size, learning_rate, name="DQNetwork")
TargetNetwork = DDDQNNet(state_size, action_size, learning_rate, name="TargetNetwork")
# https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20%28%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay%29.ipynb
