import tensorflow as tf
import numpy as np
import retro

from skimage import transform
from skimage.color import rgb2gray

import matplotlib.pyplot as plt

from collections import deque

import random

import warnings
warnings.filterwarnings('ignore')

env = retro.make(game='Asteroids-Atari2600')

print("frame size:", env.observation_space)
print("action size:", env.action_space.n)

possible_actions = np.array(np.identity(env.action_space.n, dtype=int).tolist())
print(possible_actions)

def preprocess_frame(frame):
  gray = rgb2gray(frame)

