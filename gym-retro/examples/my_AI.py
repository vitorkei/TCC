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
import Ship

SCREEN_UPPER_LIMIT = 18
SCREEN_LOWER_LIMIT = 195
SCREEN_LEFT_LIMIT = 8
SCREEN_RIGHT_LIMIT = 159

SCREEN_HEIGHT = 178
SCREEN_WIDTH = 152

########################################################
# Parser retirado do exemplo do Gym-Retro
# Poderia tentar fazer o retro.make() funcionar sem eles,
# mas não é prioridade no momento, então só trouxe junto
# para o código funcionar

parser = argparse.ArgumentParser()
parser.add_argument("--delta", '-d', type=int, default=2, help='intervalo de frames entre verificações (2 = frames sim, frame não)') # intervalo de frames entre cada busca pela nova posição dos asteróides
parser.add_argument("--loss", '-l', type=int, default=-500, help='penalidade por perder uma vida') # penalidade/perda por perder uma vida
parser.add_argument('state', nargs='?')
parser.add_argument('--scenario', '-s', default='scenario')
parser.add_argument('--record', '-r', action='store_true')
parser.add_argument('--verbose', '-v', action='count', default=1)
parser.add_argument('--quiet', '-q', action='count', default=0)
parser.add_argument('--players', '-p', type=int, default=1, help='number of players/agents (default: 1)')
 

args = parser.parse_args()
env = retro.make('Asteroids-Atari2600', args.state or retro.State.DEFAULT, scenario=args.scenario, record=args.record, players=args.players)
verbosity = args.verbose - args.quiet

########################################################

if __name__ == "__main__":
  if args.delta % 2 != 0: # se for ímpar
    raise ValueError("delta must be an even number greater than 0 (zero)")
  if args.loss > 0:
    print("WARNING: loss value is higher than 0 (zero)")

  delta = args.delta
  obs = env.reset()
  t = 0
  sys_tot_rew = 0 # system total reward: pontuação do jogador segundo o sistema
  AI_tot_rew = 0 # AI total reward: pontuação do jogador segundo o entendimento da IA. Vide Log 22/Jul/2018 para mais detalhes
  asteroids = Asteroids.Asteroids(obs)
  ship = Ship.Ship()
  life_count = ship.get_life_count() 

  print("env.reset():")
  for k, elem in asteroids.get_asteroids().items():
    print(elem)
  #print("===============")

  # Vide Log (15/Jul/2018) para entender esta flag
  next_action = 0 # 0 = update_pos() - faz update_pos()
                  # 1 = find_objects() - faz find_objects()
                  # 2 = wait - aguarda uma iteração porque uma
                  #            recompensa foi recebida recentemente

  while True:
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    life_count = info['lives']
    #print("life_count =", life_count)
    t += 1
    env.render()
    time.sleep(0.01)
    
    print("==============")
    print("t =", t, "| Delta =", delta, "\n")
    if t % args.delta == 0:
      if next_action == 0:
        #print("find_objects(obs):")
        #for elem in pos.find_objects(obs):
          #print(elem)

        #print("\nAsteroids.update_pos(obs):")
        asteroids.update_pos(obs, delta)
        #for ID, elem in asteroids.get_asteroids().items():
          #print(ID, "-", elem)
       
      elif next_action == 1:
        #print("next_action == 1\nasteroid.update_asteroids(obs):")
        asteroids.update_asteroids(obs)
        #for ID, elem in asteroids.get_asteroids().items():
          #print(ID, "-", elem)

        #print("find_objects(obs):")
        #for elem in pos.find_objects(obs):
          #print(elem)
        next_action = 0
        #print("next_action = 0\n")

      elif next_action == 2:
        #print("next_action == 2")
        next_action = 1
        #print("next_action = 1\n") 

    else:
      ship.update_pos(obs, delta)
      print(ship.get_pos())
      #input("ship_update_pos() waiting...")

    ship.set_ast_dist(asteroids.get_asteroids())
    #for ID, elem in ship.get_ast_dist().items():
      #print(ID, "-", elem)
    #input("waiting...")

    sys_tot_rew += rew
    if life_count < ship.get_life_count():
      rew = args.loss
      ship.has_died()
      #input("HAS DIED")
    AI_tot_rew += rew

    if rew > 0:
      print("time =", t, "\nReward:", rew, "\n")
      print("sys_tot_rew =", sys_tot_rew)
      print("AI_tot_rew =", AI_tot_rew)
      next_action = 2
    elif rew < 0:
      print("time =", t, "\nPenalty:", rew, "\n")
      print("sys_tot_rew =", sys_tot_rew)
      print("AI_tot_rew =", AI_tot_rew)
      next_action = 2
    if rew != 0:
      print("=======")
      print()

    if done:
      env.render()
      print("Done!\nTime:", t, "\nSystem total reward =", sys_tot_rew, "AI total reward =", AI_tot_rew)
      input("Press enter to close")
      print()
      break
   
    
    #aux.printCoords(obs, t, 500, 0)
    #aux.printCoords(obs, t, 499, 0)
  env.close()

