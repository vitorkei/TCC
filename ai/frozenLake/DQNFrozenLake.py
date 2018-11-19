from FrozenLake import *

import tensorflow as tf
import numpy as np
import gym
import random
import warnings
import time
import math

#from skimage import transform
#from skimage.color import rgb2gray
from collections import deque

warnings.filterwarnings('ignore')


env = FrozenLake()
nrow = env.n()
ncol = env.n()

available_actions = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
#stack_size = 4

state_size = [nrow, ncol, 1]
action_size = len(available_actions)
learning_rate = .05 

total_episodes = 2000
max_steps = 200 
batch_size = 200 

max_tau = 200

explore_ini = 0.9
explore_min = 0.4
decay_rate = 2000

gamma = .9

pretrain_length = 200
memory_size = 200

training = True
episode_render = False

conv_filters = [] # 8, 48
kernel_sizes = [] # 2, 2
stride_sizes = [] # 1, 1

config = tf.ConfigProto(device_count={'GPU':0})

class DQNet:
  def __init__(self, state_size, action_size, learning_rate, name):
    self.state_size = state_size
    self.action_size = action_size
    self.learning_rate = learning_rate
    self.name = name
    c_f = conv_filters
    k_s = kernel_sizes
    s_s = stride_sizes

    with tf.variable_scope(self.name):
      self.input = tf.placeholder(tf.float32, [None, *state_size]) # placeholder_0
      self.action = tf.placeholder(tf.float32, [None, action_size])# placeholder_1
      self.target_Q = tf.placeholder(tf.float32, [None])           # placeholder_2

      if len(c_f) > 0:
        self.conv2d = tf.layers.conv2d(inputs = self.input,
                                       filters = c_f[0],
                                       kernel_size = k_s[0],
                                       strides = s_s[0],
                                       #kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
                                       kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d()
                                       #kernel_initializer = tf.zeros_initializer()
                                      )
        self.elu = tf.nn.elu(self.conv2d)

        for i in range(1, len(conv_filters)):
          self.conv2d = tf.layers.conv2d(inputs = self.elu,
                                         filters = c_f[i],
                                         kernel_size = k_s[i],
                                         strides = s_s[i],
                                         kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d()
                                         #kernel_initializer = tf.zeros_initializer()
                                        )

        self.flatten = tf.layers.flatten(self.elu)
        
        #self.fc = tf.layers.dense(inputs = self.flatten,
        #                          units = 48,
        #                          activation = tf.nn.elu,
        #                          #kernel_initializer = tf.contrib.layers.xavier_initializer()
        #                          kernel_initializer = tf.zeros_initializer()
        #                         )
        self.output = tf.layers.dense(inputs = self.flatten,
                                      units = self.action_size,
                                      #kernel_initializer = tf.contrib.layers.xavier_initializer()
                                      kernel_initializer = tf.zeros_initializer()
                                     )
      else:
        self.flatten = tf.layers.flatten(self.input)

        self.output = tf.layers.dense(inputs = self.flatten,
                                      units = self.action_size,
                                      #kernel_initializer = tf.contrib.layers.xavier_initializer()
                                      kernel_initializer = tf.zeros_initializer()
                                     )

      self.Q = tf.reduce_sum(tf.multiply(self.output, self.action), axis=1)
      self.loss = tf.losses.huber_loss(self.Q, self.target_Q)

      #self.optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon = 0.0001).minimize(self.loss)
      self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, momentum=0.1).minimize(self.loss)

tf.reset_default_graph()
DQNetwork = DQNet(state_size, action_size, learning_rate, name="DQNetwork")
TargetNetwork = DQNet(state_size, action_size, learning_rate, name="TargetNetwork")

class Memory():
  def __init__(self, max_size):
    self.buffer = deque(maxlen = max_size)

  def add(self, experience):
    self.buffer.append(experience)

  def sample(self, batch_size):
    buffer_size = len(self.buffer)
    index = np.random.choice(np.arange(buffer_size), size=batch_size, replace = False)

    return [self.buffer[i] for i in index]

memory = Memory(max_size = memory_size)

if training == True:
  print("Pré-treinamento...")
  state = env.reset()
  for i in range(pretrain_length):
    choice = random.randrange(0, len(available_actions))
    action = available_actions[choice]
    next_state, reward, done, info = env.step(action)

    if done:
      next_state = np.zeros(state.shape, dtype=np.int)
      state_3d = state.reshape((state.shape[0], state.shape[1], 1))
      next_state_3d = next_state.reshape((next_state.shape[0], next_state.shape[1], 1))
      memory.add((state_3d, action, reward, next_state_3d, done))
      state = env.reset()
    else:
      state_3d = state.reshape((state.shape[0], state.shape[1], 1))
      next_state_3d = next_state.reshape((next_state.shape[0], next_state.shape[1], 1))
      memory.add((state_3d, action, reward, next_state_3d, done))
      state = next_state

  print("Pré treinamento terminado!!")

def predict_action(explore_ini, explore_min, decay_rate, decay_step, state):
  exp_exp_tradeoff = np.random.rand()
  #explore_probability = explore_min + (explore_ini - explore_min) * np.exp(-decay_rate * decay_step)
  explore_probability = explore_min + (explore_ini - explore_min) * np.exp(-decay_step / decay_rate)

  if explore_probability > exp_exp_tradeoff:
    choice = random.randrange(0, len(available_actions))
    action = available_actions[choice]
  else:
    state_3d = state.reshape((state.shape[0], state.shape[1], 1))
    Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.input: state.reshape((1, *state_3d.shape))})
    #print("\n\nPREDICT_ACTION:", Qs)
    choice = np.argmax(Qs)
    action = available_actions[choice]
  
  return action, explore_probability

def update_target_graph():
  from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")
  to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")

  op_holder = []

  for from_var, to_var in zip(from_vars, to_vars):
    op_holder.append(to_var.assign(from_var))

  return op_holder

saver = tf.train.Saver()
total_steps = 0

success_count = 0
failure_count = 0

if training == True:
  print("Iniciando treinamento...")
  with tf.Session(config=config) as sess:
  #with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    decay_step = 0

    tau = 0

    update_target = update_target_graph()
    sess.run(update_target)

    for episode in range(total_episodes):
      #if episode % 5 == 0:
      #  print()
      #  print(episode, "º episódio de treinamento")
      step = 0
      state = env.reset()

      while step < max_steps:
        step += 1
        tau += 1
        decay_step += 1
        action, explore_probability = predict_action(explore_ini, explore_min, decay_rate, decay_step, state)
        next_state, reward, done, info = env.step(action)

        if done:
          next_state = np.zeros(state.shape, dtype=np.int)
          total_steps += step
          #print(step, " |", total_steps)
          step = max_steps
          #print(episode, total_reward, explore_probability, loss)
          state_3d = state.reshape((state.shape[0], state.shape[1], 1))
          next_state_3d = next_state.reshape((next_state.shape[0], next_state.shape[1], 1))
          memory.add((state_3d, action, reward, next_state_3d, done))
          if reward > 0:
            #print("SUCESSO!")
            success_count += 1
          elif reward < 0:
            failure_count += 1
        else:
          state_3d = state.reshape((state.shape[0], state.shape[1], 1))
          next_state_3d = next_state.reshape((next_state.shape[0], next_state.shape[1], 1))
          memory.add((state_3d, action, reward, next_state_3d, done))
          state = next_state

        batch          = memory.sample(batch_size)
        states_mb      = np.array([each[0] for each in batch])
        actions_mb     = np.array([each[1] for each in batch])
        rewards_mb     = np.array([each[2] for each in batch])
        next_states_mb = np.array([each[3] for each in batch])
        dones_mb       = np.array([each[4] for each in batch])

        target_Qs_batch = []

        Qs_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.input: next_states_mb})
        Qs_target_next_state = sess.run(TargetNetwork.output, feed_dict = {TargetNetwork.input: next_states_mb})

        for i in range(len(batch)):
          terminal = dones_mb[i]

          if terminal:
            target_Qs_batch.append(rewards_mb[i])
          else:
            target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
            target_Qs_batch.append(target)

        targets_mb = np.array([each for each in target_Qs_batch])

        loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer], feed_dict = {DQNetwork.input: states_mb,
                                                                               DQNetwork.target_Q: targets_mb,
                                                                               DQNetwork.action: actions_mb})
        if tau > max_tau:
          update_target = update_target_graph()
          sess.run(update_target)
          tau = 0
          print("Model updated!")
          #print("exp_prob:", explore_probability)
                                                                               

      save_path = saver.save(sess, "/var/tmp/models3/model.ckpt")

  print("Número de sucessos no treinamento:", success_count, "/", total_episodes)
  print("Número de fracassos no treinamento:", failure_count, "/", total_episodes)
  print("Número de time-outs no treinamento:", total_episodes-failure_count-success_count, "/", total_episodes)

with tf.Session(config=config) as sess:
#with tf.Session() as sess:
  total_test_rewards = []

  saver.restore(sess, "/var/tmp/models3/model.ckpt")

  for episode in range(1):
    print(episode+1, "º episódio")
    state = env.reset()
    step = 0
    while step < max_steps:
      #env.render()
      #print()
      #print(env.pos()[0]*ncol+env.pos()[1], "-", env.pos()[0], ",", env.pos()[1], end=" ")
      state_3d = state.reshape((state.shape[0], state.shape[1], 1))
      Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.input: state.reshape((1, *state_3d.shape))})
      #print(Qs)
      choice = np.argmax(Qs)
      action = available_actions[choice]

      next_state, reward, done, info = env.step(action)
      step += 1

      if done:
        print("Reward:", reward)
        break
      else:
        state = next_state
    if step >= max_steps:
      print("Número máximo de passos atingido")

env.close()


