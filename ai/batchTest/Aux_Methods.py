# Métodos auxiliares

import numpy as np
from skimage import transform
from skimage.color import rgb2gray

# Preprocess the frames, converting to scales of gray
# removing parts of the frame and downsampling it.
def preprocess(frame, crop_size, new_size):
  gray = rgb2gray(frame)

  # Both Pong and Asteroids's frame uses column 159
  cropped = gray[crop_size[0]:crop_size[1], crop_size[2]:]

  downsample = transform.resize(cropped, [new_size[0], new_size[1]])

  return downsample

# Crate the frame stack which is sent as input to the DQN.
# Despite the name, the operation made is closer to a queue
# (it even uses a deque!)
def stack_frames(stack, state, is_new_episode, stack_size, crop_size, new_size):
  frame = preprocess(state, crop_size, new_size)

  if is_new_episode:
    stack = deque([np.zeros(new_size, dtype=np.int) for i in range(stack_size)], maxlen=stack_size)
    
    for i in range(stack_size):
      stack.append(frame)

    stacked_state = np.stack(stack, axis=2)
  else:
    stack.append(frame)
    stacked_state = np.stack(stack, axis=2)

  return stacked_state, stack

# Chooses an action according to the explore-exploit dilemma
def choose_action(eps_ini, eps_min, decay, step, state, available_actions, DQNetwork):
  exploit = np.random.rand()
  epsilon = eps_min + (eps_ini - eps_min) * np.exp(-step / decay)

  if epsilon > exploit: # explore chosen over exploit
    choice = random.randrange(0, len(available_actions))
    action = available_actions[choice]
  else: # exploit chosen over explore
    Q_values = sess.run(DQNetwork.output,
                        feed_dict={DQNetwork.input:state.reshape((1, *state.shape))})
    choice = np.argmax(Q_values)
    action = available_actions[choice]

  return action

# Updates the target network (Fixed Target technique)
def update_target_network(DQNetwork, TargetNetwork):
  from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, DQNetwork)
  to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, TargetNetwork)

  op.holder = []

  for from_var, to_var in zip(from_vars, to_vars):
    op.holder.append(to_var.assign(from_var))

  return op_holder
