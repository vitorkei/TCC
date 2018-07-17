import sys
import numpy as np

def puts(text):
    sys.stdout.write(text)
    sys.stdout.flush()

def printSample():
    np.set_printoptions(threshold=np.nan)
    m = np.full((210, 160), "0")
    f = open('teste.txt', 'r')
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
