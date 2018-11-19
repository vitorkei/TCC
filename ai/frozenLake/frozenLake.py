# /usr/lib/python3.7/site-packages/gym/envs/toy_text/frozen_lake.py
# colocar de volta if is_slipery: se quiser ter a aleatoriedade nos movimentos de novo

# http://rll.berkeley.edu/deeprlcoursesp17/docs/hw3.pdf

import gym
import numpy as np
import time

env = gym.make('FrozenLake-v0')

Q = np.zeros([env.observation_space.n, env.action_space.n])
lr = .8
gamma = .95
num_episodes = 1000
reward_list = []
max_steps = 99

# ações válidas:
# 0 = left
# 1 = down
# 2 = right
# 3 = up

for i in range(num_episodes):
  state = env.reset()
  total_reward = 0
  j = 0
  done = False
  while j < max_steps:
    j += 1
    action = np.argmax(Q[state,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
    #print(action, type(action))
    next_state, reward, done, info = env.step(action)

    Q[state,action] = Q[state,action] + lr*(reward + gamma*np.max(Q[next_state,:]) - Q[state,action])
    total_reward += reward
    state = next_state
    if done:
      break
  reward_list.append(total_reward)

print("Total score:", str(sum(reward_list)))
print("Score over time:", str(sum(reward_list)/num_episodes))

print("Tabela de Q-valores final:")
print(Q)
#print("0 = LEFT | 1 = DOWN | 2 = RIGHT | 3 = UP")
#grid = env.reset()
#env.render()
