#!/user/bin/env python

import argparse
import retro
import time
import numpy as np
import queue
import sys

import pos_methods as pos
import speed_methods as spd
import aux_methods as aux
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
parser.add_argument("delta", type=int) # intervalo de frames entre cada busca pela nova posição dos asteróides

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
  #time.sleep(2)

  totrew = 0 # total reward

  # Vide Log (15/Jul/2018) para entender esta flag
  next_action = 0 # 0 = update_pos() - faz update_pos()
                  # 1 = find_objects() - faz find_objects()
                  # 2 = wait - aguarda uma iteração porque uma
                  #            recompensa foi recebida recentemente

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
      print("==============")
      print("t =", t, "| Delta =", delta, "\n")
      if next_action == 0:
        if t > 3166:
          print("find_objects(obs):")
          for elem in pos.find_objects(obs):
            print(elem)

        print("\nAsteroids.update_pos(obs):")
        asteroids.update_pos(obs, delta)
       
        #for k, elem in asteroids.get_asteroids().items():
          #print(asteroids.get_asteroids()[elem])
        # print(k, "-", elem)
        #print()
        #time.sleep(2)
        if t > 3166:
          input("pressione enter para o próximo passo...")

      elif next_action == 1:
        print("next_action == 1\nasteroid.update_asteroids(obs):")
        asteroids.update_asteroids(obs)
        for ID, elem in asteroids.get_asteroids().items():
          print(ID, "-", elem)

        print("find_objects(obs):")
        for elem in pos.find_objects(obs):
          print(elem)
        #print(asteroids.get_asteroids())
        next_action = 0
        print("next_action = 0\n")
        #delta += args.delta

      elif next_action == 2:
        print("next_action == 2")
        next_action = 1
        print("next_action = 1\n")

    if rew > 0:
      print("time =", t, "\nReward:", rew, "\n")
      next_action = 2
    elif rew < 0:
      print("time =", t, "\nPenalty:", rew, "\n")
      next_action = 2

    if done:
      env.render()
      print("Done! total reward: time =", t, ", reward =", totrew)
      input("Press enter to close")
      print()
      break

  env.close()

