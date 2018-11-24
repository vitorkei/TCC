from DQN import *
from Memory import *

import tensorflow as tf
import numpy as np
import retro
import random

from skimage import transform
from skimage.color import rgb2gray
from collections import deque

# 2 = UP, 3 = DOWN
env    = retro.make('Asteroids-Atari2600')
config = tf.ConfigProto(device_count={'GPU':0}) # CPU instead of GPU

########################################################
################## HYPER PARAMETERS ####################
########################################################

new_height        = 84
new_width         = 84
stack_size        = 4
state_size        = [new_height, new_width, stack_size]
available_actions = np.array([[1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 1]])
action_size       = len(available_actions)

memory_size     = 1000000
pretrain_length = 500
total_episodes  = 100
max_steps       = 100000
max_tau         = 10000

eps_ini    = 1.0
eps_min    = 0.1
decay_rate = 20000

gamma            = 0.99
optimizer_params = [0.00025, 0.95, 0.01] # lr, momentum, epsilon
units            = 512
batch_size       = 64
conv_params      = [[48, 96, 96], # conv filters
                    [8, 4, 3],    # kernel size
                    [4, 2, 1]]    # stride

training = True

########################################################
##################### METHODS ##########################
########################################################

def preprocess(frame):
    gray = rgb2gray(frame)
    cropped_frame = gray[18:194, 8:] # 161x160 pixels
    
    return transform.resize(cropped_frame, [new_height, new_width])

def stack_frames(stack, state, is_new_episode):
  frame = preprocess(state)

  if is_new_episode:
    stack = deque([np.zeros((new_height, new_width), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)

    for i in range(stack_size):
      stack.append(frame)

    stacked_state = np.stack(stack, axis=2)

  else:
    stack.append(frame)
    stacked_state = np.stack(stack, axis=2)

  return stacked_state, stack

def translate_action(action):
  if np.array_equal(action, np.array([1, 0, 0, 0, 0])): # shoot
    return np.array([1, 0, 0, 0, 0, 0, 0, 0])
  elif np.array_equal(action, np.array([0, 1, 0, 0, 0])): # move forward
    return np.array([0, 0, 0, 0, 1, 0, 0, 0])
  elif np.array_equal(action, np.array([0, 0, 1, 0, 0])): # hyperspace / teleport
    return np.array([0, 0, 0, 0, 0, 1, 0, 0])
  elif np.array_equal(action, np.array([0, 0, 0, 1, 0])): # spin counter-clockwise
    return np.array([0, 0, 0, 0, 0, 0, 1, 0])
  elif np.array_equal(action, np.array([0, 0, 0, 0, 1])): # spin clockwise
    return np.array([0, 0, 0, 0, 0, 0, 0, 1])

def choose_action(eps_ini, eps_end, decay_rate, decay_step, state):
  exploit = np.random.rand()
  explore = eps_end + (eps_ini - eps_end) * np.exp(-decay_step / decay_rate)

  if explore > exploit:
    choice = random.randrange(0, len(available_actions))

    return available_actions[choice]

  Q_values = sess.run(DQNetwork.output, feed_dict = {DQNetwork.input: state.reshape((1, *state.shape))})
  choice = np.argmax(Q_values)
  
  return available_actions[choice]

def update_target_network():
  from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNewtork")
  to_vars   = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")

  op_holder = []

  for from_var, to_var in zip(from_vars, to_vars):
    op_holder.append(to_var.assign(from_var))

  return op_holder

########################################################
###################### MISC ############################
########################################################

tf.reset_default_graph()
stacked_frames = deque([np.zeros((new_height, new_width), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)
DQNetwork      = DQNNet(state_size, action_size, optimizer_params, conv_params, units, "DQNetwork")
TargetNetwork  = DQNNet(state_size, action_size, optimizer_params, conv_params, units, "TargetNetwork")
memory         = Memory(memory_size)
saver          = tf.train.Saver()
total_steps    = 0

########################################################
#################### TRAININGS #########################
########################################################

if training == True:
  print("Pretraining...")
  state = env.reset()
  state, stacked_frames = stack_frames(stacked_frames, state, True)

  for i in range(pretrain_length):
    choice = random.randrange(0, len(available_actions))
    action = available_actions[choice]
    next_state, reward, done, info = env.step(translate_action(action))
    
    if done:
      next_state = np.zeros(state.shape)
      memory.add((state, action, reward, next_state, done))
      state = env.reset()
      state, stacked_frames = stack_frames(stacked_frames, state, True)
    else:
      next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
      memory.add((state, action, reward, next_state, done))
      state = next_state

  print("Pretraining finished!!")

if training == True:
  print("Training...")
  with tf.Session(config=config) as sess:
  #with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    decay_step = 0
    tau = 0
    target_network = update_target_network()
    sess.run(target_network)

    for episode in range(total_episodes):
      step = 0
      episode_reward = 0
      state = env.reset()
      state, stacked_frames = stack_frames(stacked_frames, state, True)

      while step < max_steps:
        step       += 1
        tau        += 1
        decay_step += 1

        action = choose_action(eps_ini, eps_min, decay_rate, decay_step, state)
        next_state, reward, done, info = env.step(translate_action(action))
        episode_reward += reward

        if done:
          next_state = np.zeros(state.shape, dtype=np.int)
          next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
          total_steps += step
          print(episode, episode_reward, step, total_steps)
          step = max_steps
          memory.add((state, action, reward, next_state, done))
        else:
          next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
          memory.add((state, action, reward, next_state, done))
          state = next_state

        batch       = memory.sample(batch_size)
        states_mb   = np.array([each[0] for each in batch], ndmin=3)
        actions_mb  = np.array([each[1] for each in batch])
        rewards_mb  = np.array([each[2] for each in batch])
        n_states_mb = np.array([each[3] for each in batch], ndmin=3)
        dones_mb    = np.array([each[4] for each in batch])

        Q_targets_batch   = []
        Q_n_states        = sess.run(DQNetwork.output,     feed_dict = {DQNetwork.input: n_states_mb})

        for i in range(len(batch)):
          terminal = dones_mb[i]

          if terminal:
            Q_targets_batch.append(rewards_mb[i])
          else:
            target = rewards_mb[i] + gamma * np.max(Q_n_states[i])
            Q_targets_batch.append(target)

        targets_mb = np.array([each for each in Q_targets_batch])
        loss, _    = sess.run([DQNetwork.loss, DQNetwork.optimizer], feed_dict = {DQNetwork.input: states_mb,
                                                                                  DQNetwork.Q_target: targets_mb,
                                                                                  DQNetwork.actions: actions_mb})

        if tau > max_tau:
          target_network = update_target_network()
          sess.run(target_network)
          tau = 0
          print("Model updated!")
     
      # save_path = saver.save(sess, "/var/tmp/models_pong/model.ckpt") # rede IME
      save_path = saver.sess(sess, "../models/models_asteroids/model.ckpt")

  print("Training finished!!")

########################################################
####################### PLAY ###########################
########################################################

with tf.Session(config=config) as sess:
#with tf.Session() as sess:
  #saver.restore(sess, "/var/tmp/models_pong/model.ckpt")
  save_path = saver.sess(sess, "../models/models_asteroids/model.ckpt")

  for episode in range(10):
    print("EPISODE:", episode)
    env.seed(episode)
    total_rewards = 0
    state = env.reset()
    state, stacked_frames = stack_frames(stacked_frames, state, True)

    while True:
      Qs     = sess.run(DQNetwork.output, feed_dict = {DQNetwork.input: state.reshape((1, *state.shape))})
      choice = np.argmax(Qs)
      action = available_actions[choice]
      next_state, reward, done, info = env.step(translate_action(action))
      
      total_rewards += reward

      if done:
        print("SCORE:", total_rewards)
        break
      else:
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
        state = next_state

env.close()
