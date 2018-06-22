import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("coord", help="insira o nome do arquivo de coordenadas para ser impresso, como coord1.txt ou coordenadas0.txt")
args = parser.parse_args()

def puts(text):
  sys.stdout.write(text)
  sys.stdout.flush()

def printSample():
  np.set_printoptions(threshold=np.nan)
  m = np.full((210, 160), "0")
  f = open(args.coord, 'r')
  for line in f:
    a = [int(s) for s in line.split() if s.isdigit()]
    i = a[0]
    j = a[1]
    m[i][j] = "."

  for i in range(210):
    for j in range(160):
      puts(m[i][j])
    print()

printSample()
