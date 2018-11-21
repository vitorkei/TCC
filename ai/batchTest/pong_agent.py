import tensorflow as tf
import numpy as np
import gym

from DQN import *
from Memory import *
from Aux_Methods import *
from Config import *


env = gym.make('PongDeterministic-v4')
possible_actions = np.array(np.identity(env.action_space.n, dtype=int).tolist())
available_actions = np.array([[1, 0], [0, 1]])

new_size = [84, 84]
crop_size = [34, -16, 0]
stack_size = 4
state_size = new_size.append(stack_size)
action_size = len(available_actions) # Number of AVAILABLE actions (only those allowed to be taken)

max_steps = 18000
batch_size = 32

max_tau = 10000

training_size = 500
pretrain_size = 50000
memory_size = 1000000

# Config
conv_filters = [32, 64, 64]
kernel_sizes = [8, 4, 3]
stride_sizes = [4, 2, 1]
units = 512
learning_rate = 0.00025
momentum = 0.95
epsilon = 0.01
gamma = 0.99

# Choose action
eps_ini = 1.0
eps_min = 0.1
decay = 20000

training = True

### The actions are expressed in a way to the DQN, but differently to the environment
### This is done to improve the learning
def translate_action(action):
  if np.array_equal(action, np.array([1, 0])):
    return 2
  elif np.array_equal(action, np.array([0, 1])):
    return 3

stack = deque([np.zeros(new_size, dtype=np.int) for i in range(stack_size)], maxlen=stack_size)

config = Config(conv_filters,
                kernel_sizes,
                stride_sizes,
                units,
                learning_rate,
                momentum,
                epsilon,
               )

tf.reset_default_graph()
DQNetwork = DQN(state_size, action_size, "DQNetwork", config)
TargetNetwork = DQN(state_size, action_size, "TargetNetwork", config)
memory = Memory(memory_size)

# Pre-training to fill up a bit of the memory
if training == True:
  print("Pre-training....")
  state = env.reset()
  state, stack = stack_frames(stack, state, True, stack_size, crop_size, new_size)

  for i in range(pretrain_size):
    choice = random.randrange(0, len(available_actions))
    action = available_actions[choice]
    next_state, reward, done, info = env.step(translate_action(action))

    if done:
      next_state = np.zeros(state.shape, dtype=np.int)
      memory.save((state, action, reward, next_state, done))
      state = env.reset()
      state, stack = stack_frames(stack, state, True, stack_size, crop_size, new_size)
    else:
      next_state, stack = stack_frames(stack, state, True, stack_size, crop_size, new_size)
      memory.save((state, action, reward, next_state, done))
      state = next_state

  print("Pre-training finished!")

saver = tf.train.Saver()
total_steps = 0

if training == True:
  print("Training...")
  with tf.Session(config=tf.ConfigProto(device_count={'GPU':0})): # Use CPU
    sess.run(tf.global_variables_initializer())
    decay_step = 0
    tau = 0
    update_target = update_target_graph("DQNetwork", "TargetNetwork")
    sess.run(update_target)

    for episode in range(training_size):
      step = 0
      episode_rewards = []
      state = env.reset()
      state, stack = stack_fraes(stack, state, True, stack_size, crop_size, new_size)

      while step < max_steps:
        step += 1
        tau += 1
        decay_step += 1
        action = choose_action(eps_ini, eps_min, decay, decay_step, state, available_actions
        next_state, reward, done, info = env.step(translate_action(action))
        episode_rewards.append(reward)

        if done:
          next_state = np.zeros(state.shape, dtype=np.int)
          next_state, stack = stack_frames(stack, next_state, False, stack_size, crop_size, new_size)
          total_steps += step
          total_reward = np.sum(episode_rewards)
          print(episode, total_reward)
          memory.save((state, action, reward, next_state, done))
        else:
          next_state, stack = stack_frames(stack, next_state, False, stack_size, crop_size, new_size)
          memory.save((state, action, reward, next_state, done))
          state = next_state

        batch       = memory.sample(batch_size)
        states_mb   = np.array([each[0] for each in batch], ndmin=3)
        action_mb   = np.array([each[1] for each in batch])
        rewards_mb  = np.array([each[2] for each in batch])
        n_states_mb = np.array([each[3] for each in batch], ndmin=3)
        dones_mb    = np.array([each[4] for each in batch])

        targe_Q_values_batch = []

        Q_values_next_state        = sess.run(DQNetwork.output,
                                              feed_dict={DQNetwork.input:n_states_mb})
        Q_values_target_next_state = sess.run(TargetNewtork.output,
                                              feed_dict={TargetNetwork.input: n_states_mb})

        for i in range(len(batch)):
          terminal = dones_mb[i]

          if terminal:
            target_Q_values_batch.append(rewards_mb[i])
          else:
            target = rewards_mb[i] + gamma * np.max(Q_values_next_state[i])
            target_Q_values_batch.append(target)

          targets_mb = np.array([each for each in target_Q_values_batch])

          loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                             feed_dict={DQNetwork.input: states_mb,
                                        DQNetwork.target_Q: targets_mb,
                                        DQNetwork.action: actions_mb})

          if tau > max_tau:
            update_target = update_target_graph()
            sess.run(update_target)
            tau = 0
            print("Model updated!")

        save_path = saver.save(sess, "/var/tmp/models/model.ckpt")
  
  print("Training finished!")

with tf.Session(config=tf.ConfigProto(device_count={'GPU':0})): # Use CPU
  saver.restore(sess, "/var/tmp/models/model.ckpt")
  total_rewards = 0
  state = env.reset()
  state, stack = stack_frames(stack, state, True, stack_size, crop_size, new_size)

  while True:
    exploit = np.random.rand()
    epsilon = 0.05

    if epsilon > exploit:
      choice = random.randrange(0, len(available_actions))
      action = available_actions[choice]
    else:
      Q_values = sess.run(DQNetwork.output,
                          feed_dict={DQNetwork.input: state.reshape((1, *state.shape))})
      choice = np.argmax(Q_values)
      action = available_actions[choice]

    next_state, reward, done, info = env.step(translate_action(action))
    total_rewards += reward

    if done:
      print("SCORE:", total_rewards)
      break
    else:
      next_state, stack = stack_frames(stack, next_state, False, stack_size, crop_size, new_size)
      state = next_state

env.close()
