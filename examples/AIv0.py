#!/usr/bin/env python

import argparse
import retro
import time
import numpy as np
import queue

import posMethods as pos
import speedMethods as spd
import auxMethods as aux

############################################################
# Um monte de coisas para o programa funcionar. Veio com o exemplo do gym-retro

parser = argparse.ArgumentParser()
parser.add_argument('game', help='the name or path for the game to run')
parser.add_argument('state', nargs='?', help='the initial state file to load, minus the extension')
parser.add_argument('--scenario', '-s', default='scenario', help='the scenario file to load, minus the extension')
parser.add_argument('--record', '-r', action='store_true', help='record bk 2 movies')
parser.add_argument('--verbose', '-v', action='count', default=1, help='increase verbosity (can be specified multiple times)')
parser.add_argument('--quiet', '-q', action='count', default=0, help='decrease verbosity (can be specified multiple times)')
args = parser.parse_args()
env = retro.make(args.game, args.state or retro.STATE_DEFAULT, scenario=args.scenario, record=args.record)
verbosity = args.verbose - args.quiet

#############################################################

# "main"
try:
  while True:
    obs = env.reset()
    t = 0 # time
    #aux.printCoord(t, obs)
    #print(findInitialObjects(obs))
    #print("BREAK BREAK BREAK")
    #time.sleep(5)
    astsIniPos = pos.findInitialObjects(obs)

    totrew = 0 # total reward
    while True:
      action = env.action_space.sample()
      obs, rew, done, info = env.step(action)
      t += 1
      if t % 10 == 0:
        if verbosity > 1:
          infostr = ''
          if info:
            infostr = ', info: ' + ', '.join(['%s=%i' % (k, v) for k, v in info.items()])
            print(('t=%i' % t) + infostr)
      env.render()

      if t % 10 == 0:
        astsSPD = spd.asteroidsSpeed(astsIniPos, obs, t)
        print("\nt =", t)
        for ast in astsSPD:
          print("color:", ast[0])
          print("hSPD =", ast[1])
          print("vSPD =", ast[2])
        time.sleep(5)
      time.sleep(0.01)

      # printCoord(400, obs)
      
      totrew += rew
      if verbosity > 0:
        if rew > 0:
          print('time: %i got reward: %d, current reward: %d' % (t, rew, totrew))
        if rew < 0:
          print('time: %i got penalty: %d, current reward: %d' % (t, rew, totrew))
      if done:
        env.render()
        try:
          if verbosity >= 0:
            print("Done! Total reward: time=%i, reward=%d" % (t, totrew))
            input("press enter to continue")
            print()
          else:
            input("")
        except EOFError:
          exit(0)
        break
except KeyboardInterrupt:
  exit(0)


