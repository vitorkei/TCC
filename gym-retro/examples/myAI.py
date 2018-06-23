#!/user/bin/env python

import argparse
import retro
import time
import numpy as np
import queue

#import posMethods as pos
#import speedMethods as spd
#import auxMethods as aux
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

args = parser.parse_args()
env = retro.make('Asteroids-Atari2600', args.state or retro.STATE_DEFAULT, scenario=args.scenario, record=args.record)
verbosity = args.verbose - args.quiet

########################################################

if __name__ == "__main__":
  obs = env.reset()
  t = 0
  asteroids = Asteroids.Asteroids(obs)

  for elem in asteroids.get_asteroids():
    print(elem, "-", asteroids.get_asteroids()[elem])
  #astsIniPos = pos.findInitialObjects(obs)
  

  totrew = 0 # total reward
  
  while True:
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    t += 1
    env.render()
    time.sleep(0.01)

    totrew += rew
    """
    if t % 10 =i= 0:
      astsSPD = spd.asteroidsSpeed(astsIniPos, obs, t)
      print("\nt =", t)
      for ast in astsSPD:
        print("color:", ast[0])
        print("hSPD =", ast[1])
        print("vSPD =", ast[2])
      time.sleep(3)
    """
    if rew > 0:
      print("time =", t, "\nReward:", rew, "\n")
    elif rew < 0:
      print("time =", t, "\nPenalty:", rew, "\n")

    if done:
      env.render()
      print("Done! total reward: time =", t, ", reward =", totrew)
      input("Press enter to close")
      print()
      break

  env.close()

