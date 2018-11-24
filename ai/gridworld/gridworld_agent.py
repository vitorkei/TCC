from Gridworld import *
from DQN import *
from Memory import *

import tensorflow as tf
import numpy as np
import gym
import random

from collections import deque

env = Gridworld()
config = tf.ConfigProto(device_count={'GPU':0})

nrow              = env.n()
ncol              = env.n()
state_size        = [nrow, ncol, 1]
available_actions = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
action_size       = len(available_actions)

memory_size     = 200
pretrain_length = 200
total_episodes  = 2000
max_steps       = 200 
max_tau         = 200

eps_ini    = 0.9
eps_min    = 0.4
decay_rate = 200

gamma            = .9
optimizer_params = [.05, .1] # lr, momentum
batch_size       = 200
conv_params      = [[8], # conv filter
                    [2], # kernel size
                    [1]] # stride size

training = True

#######################################################
################## METHODS ############################
#######################################################

def choose_action(eps_ini, eps_min, decay_rate, decay_step, state):
  exploit = np.random.rand()
  explore = eps_min + (eps_ini - eps_min) * np.exp(-decay_step / decay_rate)

  if explore > exploit:
    choice = random.randrange(0, len(available_actions))

    return available_actions[choice]

  state_3d = state.reshape((state.shape[0], state.shape[1], 1))
  Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.input: state.reshape((1, *state_3d.shape))})
  choice = np.argmax(Qs)

  return available_actions[choice]

def update_target_network():
  from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")
  to_vars   = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")

  op_holder = []

  for from_var, to_var in zip(from_vars, to_vars):
    op_holder.append(to_var.assign(from_var))

  return op_holder

#######################################################
##################### MISC ############################
#######################################################

tf.reset_default_graph()
DQNetwork     = DQN(state_size, action_size, optimizer_params, conv_params, "DQNetwork")
TargetNetwork = DQN(state_size, action_size, optimizer_params, conv_params, "TargetNetwork")
memory        = Memory(max_size = memory_size)
saver         = tf.train.Saver()
total_steps   = 0

success_count = 0
failure_count = 0

#######################################################
################## TRAINING ###########################
#######################################################

if training == True:
  print("Pretraining...")
  state = env.reset()
  for i in range(pretrain_length):
    choice = random.randrange(0, len(available_actions))
    action = available_actions[choice]
    next_state, reward, done, info = env.step(action)

    if done:
      next_state    = np.zeros(state.shape, dtype=np.int)
      state_3d      = state.reshape((state.shape[0], state.shape[1], 1))
      next_state_3d = next_state.reshape((next_state.shape[0], next_state.shape[1], 1))
      memory.add((state_3d, action, reward, next_state_3d, done))
      state         = env.reset()
    else:
      state_3d      = state.reshape((state.shape[0], state.shape[1], 1))
      next_state_3d = next_state.reshape((next_state.shape[0], next_state.shape[1], 1))
      memory.add((state_3d, action, reward, next_state_3d, done))
      state         = next_state

  print("Pretraining finished!!")

if training == True:
  print("Training...")
  with tf.Session(config=config) as sess:
  #with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    decay_step = 0
    tau = 0
    update_target = update_target_network()
    sess.run(update_target)

    for episode in range(total_episodes):
      #if episode % 5 == 0:
      #  print()
      #  print(episode, "º episódio de treinamento")
      step  = 0
      state = env.reset()

      while step < max_steps:
        step       += 1
        tau        += 1
        decay_step += 1

        action = choose_action(eps_ini, eps_min, decay_rate, decay_step, state)
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

          print(episode, reward)
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

        batch       = memory.sample(batch_size)
        states_mb   = np.array([each[0] for each in batch])
        actions_mb  = np.array([each[1] for each in batch])
        rewards_mb  = np.array([each[2] for each in batch])
        n_states_mb = np.array([each[3] for each in batch])
        dones_mb    = np.array([each[4] for each in batch])

        Q_targets_batch = []

        Q_n_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.input: n_states_mb})
        #Q_target_n_state = sess.run(TargetNetwork.output, feed_dict = {TargetNetwork.input: n_states_mb})

        for i in range(len(batch)):
          terminal = dones_mb[i]

          if terminal:
            Q_targets_batch.append(rewards_mb[i])
          else:
            target = rewards_mb[i] + gamma * np.max(Q_n_state[i])
            Q_targets_batch.append(target)

        targets_mb = np.array([each for each in Q_targets_batch])

        loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer], feed_dict = {DQNetwork.input: states_mb,
                                                                               DQNetwork.Q_target: targets_mb,
                                                                               DQNetwork.action: actions_mb})
        if tau > max_tau:
          update_target = update_target_network()
          sess.run(update_target)
          tau = 0
          #print("Model updated!")

      # save_path = saver.save(sess, "/var/tmp/models_gridworld/model.ckpt") # rede IME
      save_path = saver.save(sess, "../models/models_gridworld/model.ckpt")

  print("Training finished!!")
  print("Número de sucessos no treinamento:",  success_count, "/", total_episodes)
  print("Número de fracassos no treinamento:", failure_count, "/", total_episodes)
  print("Número de time-outs no treinamento:", total_episodes - failure_count - success_count, "/", total_episodes)

#######################################################
#################### PLAY #############################
#######################################################

with tf.Session(config=config) as sess:
#with tf.Session() as sess:
  total_test_rewards = []

  #saver.restore(sess, "/var/tmp/models_gridworld/model.ckpt")
  save_path = saver.save(sess, "../models/models_gridworld/model.ckpt")

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
