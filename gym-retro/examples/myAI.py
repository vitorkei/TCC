#!/user/bin/env python

import argparse
import retro
import time
import numpy as np
import queue
import sys

import posMethods as pos
import speedMethods as spd
import auxMethods as aux
import Asteroids

########################################################
# Parser retirado do exemplo do Gym-Retro
# Poderia tentar fazer o retro.make() funcionar sem eles,
# mas não é prioridade no momento, então só trouxe junto
# para o código funcionar
parser = argparse.ArgumentParser()
parser.add_argument('state', nargs='?')
parser.add_argument('--scenario', '-s', default='scenario')
parser.add_argument('--record', '-r', action='store_true')
parser.add_argument('--verbose', '-v', action='count', default=1)
parser.add_argument('--quiet', '-q', action='count', default=0)
parser.add_argument("delta", type=int, default=4)

args = parser.parse_args()
env = retro.make('Asteroids-Atari2600', args.state or retro.STATE_DEFAULT, scenario=args.scenario, record=args.record)
verbosity = args.verbose - args.quiet

########################################################

if __name__ == "__main__":
  if (args.delta % 2 != 0): # se for ímpar
    raise ValueError("Delta deve ser um número par maior ou igual a 4 (quatro)\nUsage: python3 myAI.py <delta>")

  delta = args.delta
  obs = env.reset()
  t = 0
  asteroids = Asteroids.Asteroids(obs)
  print("env.reset():")
  for k, elem in asteroids.get_asteroids().items():
    print(elem)

  print("===============")

  totrew = 0 # total reward

  # Vide Log (15/Jul/2018) para entender esta flag
  got_reward = False

  while True:
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    t += 1
    env.render()
    time.sleep(0.01)

    totrew += rew
#    if t % 10 == 0:
#      astsSPD = spd.asteroidsSpeed(asteroids.get_asteroids(), obs, t)
#      print("\nt =", t)
#      for ast in astsSPD:
#        print("color:", ast[0])
#        print("hSPD =", ast[1])
#        print("vSPD =", ast[2])

    if t % args.delta == 0:
      print("t =", t, "| Delta =", delta, "\n")
      if not got_reward:
        print("findInitialObjects(obs):")
        for elem in pos.findInitialObjects(obs):
          print(elem)

        print("\nAsteroids.update_pos(obs):")
        asteroids.update_pos(obs, delta)
       
        # Vide Log (16/Jul/2018) para entender esta verificação
        if delta > args.delta:
          delta = args.delta

        #for k, elem in asteroids.get_asteroids().items():
          #print(asteroids.get_asteroids()[elem])
        # print(k, "-", elem)
        #print()
        print("==============")
        #time.sleep(1.5)
      else:
        print("got_reward = True!! Setando para False...\n")
        got_reward = False
        delta += args.delta

    if rew > 0:
      print("time =", t, "\nReward:", rew, "\n")
      got_reward = True
    elif rew < 0:
      print("time =", t, "\nPenalty:", rew, "\n")
      got_reward = True

    if done:
      env.render()
      print("Done! total reward: time =", t, ", reward =", totrew)
      input("Press enter to close")
      print()
      break

  env.close()

