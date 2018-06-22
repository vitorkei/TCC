import sys
import numpy as np
#import argparse

def puts(text):
  sys.stdout.write(text)
  sys.stdout.flush()

def main():
  k = -2
  path = './coords/coord'
  path += str(k) + '.txt'

  np.set_printoptions(threshold=np.nan)
  m = np.full((210, 160), "0")

  coords = open(path, 'r')
  
  for line in coords:
    a = [int(s) for s in line.split() if s.isdigit()]
    i = a[0]
    j = a[1]
    m[i][j] = "."
  
  new_path = './coords/test' + str(k)
  new_file = open(new_path, 'w')
  for i in range(210):
    for j in range(160):
      new_file.write(str(m[i][j]))
    new_file.write("\n")

main()
