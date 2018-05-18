import gym
from gym import wrappers

env = gym.make('Asteroids-v0')
obs = env.reset()

for i in range(1000):
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format((i+1)))
        break


env.close()

