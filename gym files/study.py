import gym

env = gym.make('CartPole-v0')
highscore = 0

for i_episode in range(20): # run 20 episodes
    obs = env.reset()
    points = 0 # keep track of the reward of each episode
    while True: # run until episode is done
        env.render()
        action = 1 if obs[2] > 0 else 0 # if angle is positive, move right. Else, move left
        obs, reward, done, info = env.step(action)
        points += reward
        if done:
            if points > highscore: # record high score
                highscore = points
            break
env.close()

